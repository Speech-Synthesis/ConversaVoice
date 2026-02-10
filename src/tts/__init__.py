"""
Text-to-Speech module for ConversaVoice.

Provides Azure Neural TTS integration with SSML support
for emotional prosody control and word emphasis.
Also includes Piper TTS as a local fallback.
"""

from .ssml_builder import (
    SSMLBuilder,
    ProsodyProfile,
    ProsodySettings,
    PROSODY_PROFILES,
    EmphasisLevel,
    AZURE_EXPRESS_STYLES,
)
from .azure_client import AzureTTSClient, TTSError
from .piper_client import PiperTTSClient, PiperTTSError

__all__ = [
    "SSMLBuilder",
    "ProsodyProfile",
    "ProsodySettings",
    "PROSODY_PROFILES",
    "EmphasisLevel",
    "AZURE_EXPRESS_STYLES",
    "AzureTTSClient",
    "TTSError",
    "PiperTTSClient",
    "PiperTTSError",
]
