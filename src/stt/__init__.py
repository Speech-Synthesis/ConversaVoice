"""Speech-to-Text module using Distil-Whisper."""

from .whisper_client import WhisperClient, STTError

__all__ = ["WhisperClient", "STTError"]
