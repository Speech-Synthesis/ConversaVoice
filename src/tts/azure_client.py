"""
Azure Neural TTS client for ConversaVoice.

Provides speech synthesis with SSML support for emotional prosody.
"""

import os
from typing import Optional

import azure.cognitiveservices.speech as speechsdk

from .ssml_builder import SSMLBuilder, ProsodyProfile


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

        # Set output format to high-quality audio
        config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
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
