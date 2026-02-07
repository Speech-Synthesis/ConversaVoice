"""Whisper client for speech-to-text transcription."""

import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*logits_process.*")
warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")
logging.getLogger("transformers").setLevel(logging.ERROR)


class STTError(Exception):
    """Exception raised for STT errors."""
    pass


class WhisperClient:
    """
    Whisper-based speech-to-text client for the orchestrator.

    Wraps Distil-Whisper for real-time microphone transcription
    with voice activity detection (VAD).
    """

    def __init__(self, model_id: str = "distil-whisper/distil-large-v3"):
        """
        Initialize the Whisper client.

        Args:
            model_id: HuggingFace model ID for Whisper
        """
        self.model_id = model_id
        self.pipe = None
        self.device = None
        self.dtype = None
        self.sample_rate = 16000
        self.chunk_duration = 5  # seconds per chunk
        self.silence_threshold = 0.01
        self.min_speech_duration = 0.5
        self._is_loaded = False

    def load_model(self) -> None:
        """Load the Whisper model onto GPU."""
        if self._is_loaded:
            return

        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Loading Whisper model on {self.device}...")

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(self.device)

            processor = AutoProcessor.from_pretrained(self.model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.dtype,
                device=self.device,
            )

            self._is_loaded = True
            print("Whisper model loaded successfully!")

        except Exception as e:
            raise STTError(f"Failed to load Whisper model: {e}")

    def _is_speech(self, audio_chunk) -> bool:
        """Simple voice activity detection."""
        import numpy as np
        return np.abs(audio_chunk).mean() > self.silence_threshold

    def transcribe_audio(self, audio_array, sample_rate: int = 16000) -> str:
        """
        Transcribe a numpy audio array.

        Args:
            audio_array: Numpy array of audio samples
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text
        """
        self.load_model()

        # Resample if needed
        if sample_rate != self.sample_rate:
            import torch
            import torchaudio.functional as F
            audio_tensor = torch.from_numpy(audio_array).float()
            audio_array = F.resample(audio_tensor, sample_rate, self.sample_rate).numpy()

        result = self.pipe({"array": audio_array, "sampling_rate": self.sample_rate})
        return result["text"].strip()

    def start_listening(self, callback=None) -> None:
        """
        Start listening from microphone with VAD.

        Args:
            callback: Function to call with transcribed text
        """
        import pyaudio
        import numpy as np

        self.load_model()
        self._listening = True

        p = pyaudio.PyAudio()

        try:
            default_device = p.get_default_input_device_info()
            print(f"Using microphone: {default_device['name']}")
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

    def listen_once(self, timeout: float = 10.0) -> str:
        """
        Listen for a single utterance and return the transcribed text.

        Args:
            timeout: Maximum seconds to wait for speech

        Returns:
            Transcribed text or empty string if timeout
        """
        import pyaudio
        import numpy as np
        import time

        self.load_model()

        p = pyaudio.PyAudio()

        try:
            default_device = p.get_default_input_device_info()
            print(f"Using microphone: {default_device['name']}")
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
