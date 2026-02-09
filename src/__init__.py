"""
ConversaVoice - Context-aware voice assistant with emotional intelligence.
"""

from .orchestrator import Orchestrator, PipelineResult, PipelineState, OrchestratorError
from .stt import WhisperClient, STTError
from .nlp import SentimentAnalyzer, SentimentResult

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
]
