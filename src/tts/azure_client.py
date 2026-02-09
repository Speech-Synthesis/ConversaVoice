"""
Azure Neural TTS client for ConversaVoice.

Provides speech synthesis with SSML support for emotional prosody.
"""

import os
from typing import Optional, Tuple

import azure.cognitiveservices.speech as speechsdk

from .ssml_builder import SSMLBuilder, ProsodyProfile


class TTSError(Exception):
    """Exception raised when TTS synthesis fails."""

    def __init__(self, message: str, reason: Optional[str] = None, details: Optional[str] = None):
        self.reason = reason
        self.details = details
        super().__init__(message)


def _check_result(result: speechsdk.SpeechSynthesisResult) -> None:
    """
    Check synthesis result and raise TTSError if failed.

    Args:
        result: Speech synthesis result to check

    Raises:
        TTSError: If synthesis was not successful
    """
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return

    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        error_msg = f"Speech synthesis canceled: {cancellation.reason}"

        if cancellation.reason == speechsdk.CancellationReason.Error:
            raise TTSError(
                message=error_msg,
                reason=str(cancellation.error_code),
                details=cancellation.error_details
            )
        raise TTSError(message=error_msg, reason=str(cancellation.reason))

    raise TTSError(
        message=f"Speech synthesis failed with reason: {result.reason}",
        reason=str(result.reason)
    )


class AzureTTSClient:
    """
    Azure Neural TTS client with SSML support.

    Handles speech synthesis with emotional prosody control.
    """

    def __init__(
        self,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        voice: str = "en-US-JennyNeural"
    ):
        """
        Initialize Azure TTS client.

        Args:
            speech_key: Azure Speech API key (defaults to AZURE_SPEECH_KEY env var)
            speech_region: Azure region (defaults to AZURE_SPEECH_REGION env var)
            voice: Azure Neural voice name
        """
        self.speech_key = speech_key or os.getenv("AZURE_SPEECH_KEY")
        self.speech_region = speech_region or os.getenv("AZURE_SPEECH_REGION")

        if not self.speech_key:
            raise ValueError(
                "Azure Speech key not provided. "
                "Set AZURE_SPEECH_KEY environment variable or pass speech_key parameter."
            )

        if not self.speech_region:
            raise ValueError(
                "Azure Speech region not provided. "
                "Set AZURE_SPEECH_REGION environment variable or pass speech_region parameter."
            )

        self.voice = voice
        self.ssml_builder = SSMLBuilder(voice=voice)

        # Initialize speech config
        self._speech_config = self._create_speech_config()

    def _create_speech_config(self) -> speechsdk.SpeechConfig:
        """Create Azure Speech configuration."""
        config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )

        # Use PCM format for direct speaker playback (MP3 doesn't play to speakers)
        config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        )

        return config

    def _create_synthesizer(
        self,
        audio_config: Optional[speechsdk.audio.AudioOutputConfig] = None
    ) -> speechsdk.SpeechSynthesizer:
        """
        Create speech synthesizer.

        Args:
            audio_config: Audio output configuration (None for default speaker)

        Returns:
            Configured speech synthesizer
        """
        if audio_config is None:
            # Default to speaker output
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

        return speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )

    @property
    def config(self) -> speechsdk.SpeechConfig:
        """Get the speech configuration."""
        return self._speech_config

    def speak(
        self,
        text: str,
        profile: ProsodyProfile = ProsodyProfile.NEUTRAL,
        **kwargs
    ) -> speechsdk.SpeechSynthesisResult:
        """
        Synthesize text and play through default speaker.

        Args:
            text: Text to synthesize
            profile: Prosody profile for emotional control
            **kwargs: Additional prosody overrides (pitch, rate, volume, style)

        Returns:
            Speech synthesis result
        """
        ssml = self.ssml_builder.build(text, profile=profile, **kwargs)
        synthesizer = self._create_synthesizer()

        result = synthesizer.speak_ssml_async(ssml).get()
        _check_result(result)
        return result

    def speak_with_llm_params(
        self,
        text: str,
        style: Optional[str] = None,
        pitch: Optional[str] = None,
        rate: Optional[str] = None
    ) -> speechsdk.SpeechSynthesisResult:
        """
        Synthesize using parameters from LLM response.

        Designed to work directly with Groq LLM JSON output.

        Args:
            text: Reply text from LLM
            style: Style from LLM (e.g., "empathetic")
            pitch: Pitch from LLM (e.g., "-5%")
            rate: Rate from LLM (e.g., "0.85")

        Returns:
            Speech synthesis result
        """
        ssml = self.ssml_builder.build_from_llm_response(
            text=text,
            style=style,
            pitch=pitch,
            rate=rate
        )
        synthesizer = self._create_synthesizer()

        result = synthesizer.speak_ssml_async(ssml).get()
        _check_result(result)
        return result

    def synthesize_to_file(
        self,
        text: str,
        filepath: str,
        profile: ProsodyProfile = ProsodyProfile.NEUTRAL,
        **kwargs
    ) -> speechsdk.SpeechSynthesisResult:
        """
        Synthesize text and save to audio file.

        Args:
            text: Text to synthesize
            filepath: Output file path
            profile: Prosody profile for emotional control
            **kwargs: Additional prosody overrides

        Returns:
            Speech synthesis result
        """
        ssml = self.ssml_builder.build(text, profile=profile, **kwargs)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=filepath)
        synthesizer = self._create_synthesizer(audio_config=audio_config)

        result = synthesizer.speak_ssml_async(ssml).get()
        _check_result(result)
        return result

    def synthesize_to_bytes(
        self,
        text: str,
        profile: ProsodyProfile = ProsodyProfile.NEUTRAL,
        **kwargs
    ) -> bytes:
        """
        Synthesize text and return audio bytes.

        Args:
            text: Text to synthesize
            profile: Prosody profile for emotional control
            **kwargs: Additional prosody overrides

        Returns:
            Audio data as bytes

        Raises:
            TTSError: If synthesis fails
        """
        ssml = self.ssml_builder.build(text, profile=profile, **kwargs)

        # Use None audio config to get audio data in result
        audio_config = None
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config,
            audio_config=audio_config
        )

        result = synthesizer.speak_ssml_async(ssml).get()
        _check_result(result)
        return result.audio_data
