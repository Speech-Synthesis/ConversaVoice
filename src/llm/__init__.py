"""
LLM module for ConversaVoice.
Provides Groq API integration with Llama 3 for intelligent responses.
"""

from .groq_client import GroqClient, GroqConfig, EmotionalResponse

__all__ = ["GroqClient", "GroqConfig", "EmotionalResponse"]
