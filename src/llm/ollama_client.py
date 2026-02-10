"""
Ollama client for ConversaVoice.

Provides local LLM inference as a fallback when Groq API is unavailable.
Ollama runs models locally for offline operation.
"""

import os
import json
import logging
import requests
from typing import Optional, Generator, Callable
from dataclasses import dataclass

from .groq_client import EmotionalResponse

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    host: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 1024


class OllamaError(Exception):
    """Exception raised when Ollama request fails."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.details = details
        super().__init__(message)


class OllamaClient:
    """
    Local LLM client using Ollama.

    Ollama runs language models locally for offline operation.
    Used as a fallback when Groq API is unavailable.
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Initialize Ollama client.

        Args:
            config: Optional OllamaConfig. If not provided, uses defaults/env vars.
        """
        if config is None:
            config = OllamaConfig(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama3.2")
            )
        self.config = config
        self._available = None

    def is_available(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            True if Ollama server is responding, False otherwise.
        """
        if self._available is not None:
            return self._available

        try:
            response = requests.get(
                f"{self.config.host}/api/tags",
                timeout=5
            )
            self._available = response.status_code == 200
        except requests.RequestException:
            self._available = False

        if not self._available:
            logger.warning("Ollama is not available")

        return self._available

    def list_models(self) -> list[str]:
        """
        List available models in Ollama.

        Returns:
            List of model names.
        """
        try:
            response = requests.get(
                f"{self.config.host}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except requests.RequestException as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for emotional intelligence.

        Uses the same prompt as GroqClient for consistency.
        """
        return """You are ConversaVoice, an emotionally intelligent voice assistant that ACTS differently based on user emotions, not just acknowledges them.

## CORE RULES

1. **USE CONTEXT**: Always reference facts from the conversation. If user said "Python and AI models", your recommendations MUST reflect that (GPU, 16GB+ RAM, etc.).

2. **BE DECISIVE**: You are an intelligent agent, NOT a form. Make informed assumptions rather than asking endless clarifying questions.

3. **EMOTION CHANGES BEHAVIOR**: Each emotional style requires DIFFERENT decision-making, not just different tone.

## STYLE-BASED BEHAVIOR (CRITICAL)

### neutral (default for new conversations, factual queries)
- Ask 1-2 clarifying questions if genuinely needed
- Provide balanced, informative responses

### cheerful (greetings, good news, excitement, task completion)
- Be enthusiastic and action-oriented
- Celebrate progress, encourage next steps
- Keep energy high, be concise

### patient (confusion, complex topics, learning)
- Simplify explanations
- Break down into smaller steps
- Ask AT MOST one clarifying question

### empathetic (frustration, repetition, annoyance, escalation)
- **STOP ASKING QUESTIONS IMMEDIATELY**
- Make reasonable assumptions based on context
- Give a direct, actionable answer NOW
- Acknowledge their frustration briefly, then SOLVE the problem
- If you lack info, make a safe/general recommendation

### de_escalate (anger, high frustration, threats to leave)
- Stay calm and grounded
- Speak slowly and softly
- Focus on resolution, not explanation

## FRUSTRATION ESCALATION POLICY

Detect frustration when user:
- Repeats themselves or asks the same thing differently
- Uses phrases like "just tell me", "why is this so hard", "I already said", "again"
- Shows impatience or annoyance

When frustrated:
1. Do NOT ask more questions
2. Do NOT apologize excessively (one brief acknowledgment max)
3. DO give a concrete answer using available context
4. DO make assumptions if needed - a reasonable guess is better than more questions

## CONTEXT USAGE (MANDATORY)

Before responding, mentally review the conversation:
- What has the user already told you?
- What can you infer from their statements?
- Use this information in your response

Example: If user said "programming with Python and AI", recommend laptops with:
- Dedicated NVIDIA GPU (for ML/AI)
- 16GB+ RAM (for large models)
- Fast SSD (for datasets)
NOT generic specs like "i5, 8GB RAM".

## RESPONSE FORMAT

Always respond with valid JSON:
{
    "reply": "Your response here",
    "style": "neutral|cheerful|patient|empathetic|de_escalate",
    "emphasis_words": ["word1", "word2"]
}

- emphasis_words: Optional list of 1-3 KEY words in your reply that should be stressed/emphasized when spoken. Choose words that:
  - Convey the most important information
  - Are action items or key nouns
  - Help clarify meaning through vocal stress
  - Example: For "I recommend the NVIDIA RTX 4060", emphasize ["NVIDIA", "RTX 4060"]

Only output the JSON object, no additional text."""

    def chat(self, user_message: str, context: Optional[str] = None) -> str:
        """
        Send a message to local LLM and get a response.

        Args:
            user_message: The user's input text.
            context: Optional conversation context/history.

        Returns:
            The assistant's response text.
        """
        if not self.is_available():
            raise OllamaError("Ollama is not available")

        messages = []

        # Add system prompt
        messages.append({
            "role": "system",
            "content": self._get_system_prompt()
        })

        # Add context if provided
        if context:
            messages.append({
                "role": "user",
                "content": f"Previous context: {context}"
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        try:
            response = requests.post(
                f"{self.config.host}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

        except requests.RequestException as e:
            raise OllamaError(f"Ollama request failed: {e}")

    def chat_stream(
        self,
        user_message: str,
        context: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, str]:
        """
        Stream a response from local LLM token by token.

        Args:
            user_message: The user's input text.
            context: Optional conversation context/history.
            on_chunk: Optional callback for each chunk received.

        Yields:
            Individual tokens/chunks as they arrive.

        Returns:
            Complete response text.
        """
        if not self.is_available():
            raise OllamaError("Ollama is not available")

        messages = []

        # Add system prompt
        messages.append({
            "role": "system",
            "content": self._get_system_prompt()
        })

        # Add context if provided
        if context:
            messages.append({
                "role": "user",
                "content": f"Previous context: {context}"
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        try:
            response = requests.post(
                f"{self.config.host}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            token = data["message"]["content"]
                            full_response += token
                            if on_chunk:
                                on_chunk(token)
                            yield token
                    except json.JSONDecodeError:
                        continue

            return full_response

        except requests.RequestException as e:
            raise OllamaError(f"Ollama streaming request failed: {e}")

    def _parse_response(self, raw_response: str) -> EmotionalResponse:
        """
        Parse the LLM response into an EmotionalResponse.

        Args:
            raw_response: Raw text from the LLM.

        Returns:
            Parsed EmotionalResponse object.
        """
        try:
            json_str = raw_response.strip()

            # Find JSON object in response
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx]

            data = json.loads(json_str)

            emphasis_words = data.get("emphasis_words", [])
            if not isinstance(emphasis_words, list):
                emphasis_words = []

            return EmotionalResponse(
                reply=data.get("reply", ""),
                style=data.get("style", "neutral"),
                emphasis_words=emphasis_words,
                raw_response=raw_response
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return EmotionalResponse(
                reply=raw_response,
                style="neutral",
                emphasis_words=[],
                raw_response=raw_response
            )

    def get_emotional_response(
        self,
        user_message: str,
        context: Optional[str] = None
    ) -> EmotionalResponse:
        """
        Get an emotionally-aware response with style label.

        Args:
            user_message: The user's input text.
            context: Optional conversation context/history.

        Returns:
            EmotionalResponse with reply and style label.
        """
        try:
            raw_response = self.chat(user_message, context)
            return self._parse_response(raw_response)
        except Exception as e:
            logger.error(f"Error getting emotional response: {e}")
            return EmotionalResponse(
                reply="I'm sorry, I encountered an issue. Could you please repeat that?",
                style="empathetic",
                raw_response=str(e)
            )

    def get_emotional_response_stream(
        self,
        user_message: str,
        context: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None
    ) -> EmotionalResponse:
        """
        Get an emotionally-aware response with streaming.

        Args:
            user_message: The user's input text.
            context: Optional conversation context/history.
            on_token: Optional callback for each token.

        Returns:
            EmotionalResponse with reply, style, and emphasis_words.
        """
        try:
            tokens = []
            for token in self.chat_stream(user_message, context, on_chunk=on_token):
                tokens.append(token)

            raw_response = "".join(tokens)
            return self._parse_response(raw_response)

        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            return EmotionalResponse(
                reply="I'm sorry, I encountered an issue. Could you please repeat that?",
                style="empathetic",
                emphasis_words=[],
                raw_response=str(e)
            )
