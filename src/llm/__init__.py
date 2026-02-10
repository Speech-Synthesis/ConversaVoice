"""
LLM module for ConversaVoice.
Provides Groq API integration with Llama 3 for intelligent responses.
Also includes Ollama as a local fallback for offline operation.
"""

from .groq_client import GroqClient, GroqConfig, EmotionalResponse
from .ollama_client import OllamaClient, OllamaConfig, OllamaError

__all__ = [
    "GroqClient",
    "GroqConfig",
    "EmotionalResponse",
    "OllamaClient",
    "OllamaConfig",
    "OllamaError",
]
