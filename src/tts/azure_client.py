import os
import requests
import tempfile
import logging
from typing import Optional, Callable, Generator
from src.tts.ssml_builder import ProsodyProfile, SSMLBuilder

logger = logging.getLogger(__name__)

class TTSError(Exception):
    """Exception raised when TTS synthesis fails."""
    def __init__(self, message: str, reason: Optional[str] = None, details: Optional[str] = None):
        self.reason = reason
        self.details = details
        super().__init__(f"{message}. Reason: {reason}. Details: {details}")

class AzureTTSClient:
    """
    Text-to-Speech client using Azure Cognitive Services via REST API.
    Bypasses the C++ SDK to avoid Linux/Render audio driver crashes (ALSA) and local deadlocks.
    """

    def __init__(
        self,
        subscription_key: Optional[str] = None,
        region: Optional[str] = None,
        voice_gender: str = "female"
    ):
        self.subscription_key = subscription_key or os.getenv("AZURE_SPEECH_KEY")
        self.region = region or os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.voice_gender = voice_gender
        
        self.ssml_builder = SSMLBuilder(
            voice=SSMLBuilder.DEFAULT_FEMALE_VOICE if voice_gender == "female" else SSMLBuilder.DEFAULT_MALE_VOICE
        )
        self._available = None

    @property
    def _base_url(self):
        return f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
            
        if not self.subscription_key or not self.region:
            logger.warning("Azure TTS credentials missing")
            self._available = False
            return False
            
        self._available = True
        return True

    def _synthesize_rest(self, ssml: str) -> bytes:
        if not self.is_available():
            raise TTSError("Azure TTS not configured correctly")
            
        headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm',
            'User-Agent': 'ConversaVoice'
        }
        
        try:
            response = requests.post(self._base_url, headers=headers, data=ssml.encode('utf-8'), timeout=15)
            if response.status_code == 200:
                return response.content
            else:
                raise TTSError("Azure TTS REST API failed", str(response.status_code), response.text)
        except requests.exceptions.RequestException as e:
            raise TTSError("Azure TTS network request failed", "NetworkError", str(e))

    def synthesize_to_file(self, text: str, filepath: str, profile: ProsodyProfile = ProsodyProfile.NEUTRAL, **kwargs) -> str:
        ssml = self.ssml_builder.build(text, profile=profile, **kwargs)
        audio_data = self._synthesize_rest(ssml)
        with open(filepath, "wb") as f:
            f.write(audio_data)
        return filepath

    def synthesize_to_file_with_params(self, text: str, filepath: str, style: Optional[str] = None, pitch: Optional[str] = None, rate: Optional[str] = None) -> str:
        ssml = self.ssml_builder.build_from_llm_response(text=text, style=style, pitch=pitch, rate=rate)
        audio_data = self._synthesize_rest(ssml)
        with open(filepath, "wb") as f:
            f.write(audio_data)
        return filepath

    def synthesize_to_bytes(self, text: str, profile: ProsodyProfile = ProsodyProfile.NEUTRAL, **kwargs) -> bytes:
        ssml = self.ssml_builder.build(text, profile=profile, **kwargs)
        return self._synthesize_rest(ssml)
        
    def synthesize_to_bytes_with_params(self, text: str, style: Optional[str] = None, pitch: Optional[str] = None, rate: Optional[str] = None) -> bytes:
        ssml = self.ssml_builder.build_from_llm_response(text=text, style=style, pitch=pitch, rate=rate)
        return self._synthesize_rest(ssml)

    def speak(self, text: str, profile: ProsodyProfile = ProsodyProfile.NEUTRAL, **kwargs) -> None:
        if not self.is_available():
            raise TTSError("Azure TTS is not available")
        audio_data = self._synthesize_rest(self.ssml_builder.build(text, profile=profile, **kwargs))
        self._play_audio_bytes(audio_data)

    def speak_with_llm_params(self, text: str, style: Optional[str] = None, pitch: Optional[str] = None, rate: Optional[str] = None) -> None:
        if not self.is_available():
            raise TTSError("Azure TTS is not available")
        audio_data = self._synthesize_rest(self.ssml_builder.build_from_llm_response(text=text, style=style, pitch=pitch, rate=rate))
        self._play_audio_bytes(audio_data)

    def _play_audio_bytes(self, audio_data: bytes) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            import os, subprocess
            if os.name == "nt":
                os.startfile(temp_path)
            elif os.name == "posix":
                for player in ["aplay", "paplay", "afplay"]:
                    try:
                        subprocess.run([player, temp_path], check=True)
                        break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
        finally:
            if os.name != "nt" and os.path.exists(temp_path):
                os.unlink(temp_path)

    def speak_chunked(
        self,
        text: str,
        style: Optional[str] = None,
        pitch: Optional[str] = None,
        rate: Optional[str] = None,
        on_sentence_start: Optional[Callable[[str, int], None]] = None,
        on_sentence_complete: Optional[Callable[[int], None]] = None
    ) -> None:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for i, sentence in enumerate(sentences):
            if on_sentence_start:
                on_sentence_start(sentence, i)
                
            self.speak_with_llm_params(
                text=sentence,
                style=style,
                pitch=pitch,
                rate=rate
            )
            
            if on_sentence_complete:
                on_sentence_complete(i)

    def synthesize_chunks_generator(
        self,
        text: str,
        style: Optional[str] = None,
        pitch: Optional[str] = None,
        rate: Optional[str] = None
    ) -> Generator[bytes, None, None]:
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            audio_bytes = self.synthesize_to_bytes_with_params(
                text=sentence,
                style=style,
                pitch=pitch,
                rate=rate
            )
            yield audio_bytes
