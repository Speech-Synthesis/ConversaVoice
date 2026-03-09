"""Groq Whisper API client for speech-to-text transcription."""

import os
import io
import logging
import tempfile
from typing import Optional, Callable

import numpy as np
import requests

logger = logging.getLogger(__name__)


class STTError(Exception):
    """Exception raised for STT errors."""
    pass


class GroqWhisperClient:
    """
    Groq Whisper API client for speech-to-text.
    
    Cloud-based alternative to local Whisper model.
    No GPU required, no model download needed.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq Whisper client.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise STTError(
                "GROQ_API_KEY not found. Set it in .env or pass as parameter."
            )
        
        self.sample_rate = 16000
        self.chunk_duration = 5  # seconds per chunk
        self.silence_threshold = 0.01
        self.min_speech_duration = 0.5
        self._client = None
        self._base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com")
        
    def _get_client(self):
        """Lazy load Groq client."""
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise STTError(
                    "Groq package not installed. Run: pip install groq"
                )
        return self._client
    
    def _numpy_to_wav_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert numpy audio array to WAV bytes.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            WAV file as bytes
        """
        import wave
        
        # Ensure audio is in correct format
        if audio_array.dtype != np.int16:
            # Convert float32 [-1, 1] to int16
            audio_array = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_array.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()

    def _transcribe_bytes(self, filename: str, audio_bytes: bytes) -> str:
        """
        Transcribe audio bytes with SDK when available, otherwise use direct HTTP.
        """
        client = self._get_client()

        # Newer Groq SDKs expose audio.transcriptions.
        audio_api = getattr(client, "audio", None)
        transcriptions_api = getattr(audio_api, "transcriptions", None) if audio_api else None
        if transcriptions_api is not None:
            transcription = transcriptions_api.create(
                file=(filename, audio_bytes),
                model="whisper-large-v3",
                response_format="json",
                language="en"
            )
            return transcription.text.strip()

        # groq==0.4.2 does not expose audio APIs; use OpenAI-compatible endpoint directly.
        endpoint = f"{self._base_url.rstrip('/')}/openai/v1/audio/transcriptions"
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            data={
                "model": "whisper-large-v3",
                "response_format": "json",
                "language": "en"
            },
            files={"file": (filename, audio_bytes)},
            timeout=60
        )
        response.raise_for_status()
        payload = response.json()
        text = payload.get("text", "").strip()
        if not text:
            raise STTError("Empty transcription response from Groq API")
        return text
    
    def transcribe_audio(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a numpy audio array using Groq Whisper API.
        
        Args:
            audio_array: Numpy array of audio samples
            sample_rate: Sample rate of the audio
            
        Returns:
            Transcribed text
        """
        try:
            # Convert numpy array to WAV bytes
            wav_bytes = self._numpy_to_wav_bytes(audio_array, sample_rate)
            return self._transcribe_bytes("audio.wav", wav_bytes)
            
        except Exception as e:
            logger.error(f"Groq transcription failed: {e}")
            raise STTError(f"Transcription failed: {e}")
    
    def transcribe_file(self, filepath: str) -> str:
        """
        Transcribe an audio file using Groq Whisper API.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            with open(filepath, "rb") as audio_file:
                audio_bytes = audio_file.read()
            return self._transcribe_bytes(os.path.basename(filepath), audio_bytes)
            
        except Exception as e:
            logger.error(f"Groq file transcription failed: {e}")
            raise STTError(f"File transcription failed: {e}")
    
    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Simple voice activity detection."""
        return np.abs(audio_chunk).mean() > self.silence_threshold
    
    def listen_once(self, timeout: float = 10.0) -> str:
        """
        Listen for a single utterance and return the transcribed text.
        
        Args:
            timeout: Maximum seconds to wait for speech
            
        Returns:
            Transcribed text or empty string if timeout
        """
        import pyaudio
        import time
        
        p = pyaudio.PyAudio()
        
        try:
            default_device = p.get_default_input_device_info()
            logger.info(f"Using microphone: {default_device['name']}")
        except Exception:
            raise STTError("No microphone found!")
        
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        print("[Listening for speech...]")
        
        audio_buffer = []
        start_time = time.time()
        speech_detected = False
        
        try:
            while True:
                if time.time() - start_time > timeout:
                    print("[Timeout - no speech detected]")
                    break
                
                # Read audio chunk
                frames = []
                for _ in range(0, int(self.sample_rate / 1024 * self.chunk_duration)):
                    data = stream.read(1024, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))
                
                audio_chunk = np.concatenate(frames)
                
                # Check for speech
                if self._is_speech(audio_chunk):
                    speech_detected = True
                    audio_buffer.append(audio_chunk)
                elif speech_detected and audio_buffer:
                    # Speech ended, process it
                    full_audio = np.concatenate(audio_buffer)
                    if len(full_audio) / self.sample_rate >= self.min_speech_duration:
                        text = self.transcribe_audio(full_audio, self.sample_rate)
                        return text
                    audio_buffer = []
                    speech_detected = False
        
        except KeyboardInterrupt:
            print("\n[Cancelled]")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
        
        return ""
    
    def start_listening(self, callback: Optional[Callable] = None) -> None:
        """
        Start listening from microphone with VAD.
        
        Args:
            callback: Function to call with transcribed text
        """
        import pyaudio
        
        self._listening = True
        
        p = pyaudio.PyAudio()
        
        try:
            default_device = p.get_default_input_device_info()
            logger.info(f"Using microphone: {default_device['name']}")
        except Exception:
            raise STTError("No microphone found!")
        
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        print("\n[Listening... Press Ctrl+C to stop]\n")
        
        audio_buffer = []
        
        try:
            while self._listening:
                # Read audio chunk
                frames = []
                for _ in range(0, int(self.sample_rate / 1024 * self.chunk_duration)):
                    if not self._listening:
                        break
                    data = stream.read(1024, exception_on_overflow=False)
                    frames.append(np.frombuffer(data, dtype=np.float32))
                
                if not frames:
                    break
                
                audio_chunk = np.concatenate(frames)
                
                # Check for speech
                if self._is_speech(audio_chunk):
                    audio_buffer.append(audio_chunk)
                elif audio_buffer:
                    # Process accumulated speech
                    full_audio = np.concatenate(audio_buffer)
                    
                    if len(full_audio) / self.sample_rate >= self.min_speech_duration:
                        text = self.transcribe_audio(full_audio, self.sample_rate)
                        if text:
                            if callback:
                                callback(text)
                            else:
                                print(f"> {text}")
                    
                    audio_buffer = []
        
        except KeyboardInterrupt:
            print("\n[Stopped]")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    def stop_listening(self) -> None:
        """Stop listening from microphone."""
        self._listening = False
