"""
ConversaVoice - Context-aware voice assistant with emotional intelligence.
"""

from .orchestrator import Orchestrator, PipelineResult, PipelineState, OrchestratorError
from .stt import WhisperClient, STTError
from .nlp import SentimentAnalyzer, SentimentResult
from .fallback import FallbackManager, FallbackConfig, ServiceType, ServiceMode

__version__ = "0.1.0"

__all__ = [
    "Orchestrator",
    "PipelineResult",
    "PipelineState",
    "OrchestratorError",
    "WhisperClient",
    "STTError",
    "SentimentAnalyzer",
    "SentimentResult",
    "FallbackManager",
    "FallbackConfig",
    "ServiceType",
    "ServiceMode",
]
