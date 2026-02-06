"""
Groq API client for ConversaVoice.
Provides async interface to Llama 3 for intelligent, emotionally-aware responses.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class GroqConfig:
    """Configuration for Groq API client."""
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class EmotionalResponse:
    """Parsed response with emotional prosody parameters."""
    reply: str
    style: str = "neutral"
    pitch: str = "0%"
    rate: str = "1.0"
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reply": self.reply,
            "style": self.style,
            "pitch": self.pitch,
            "rate": self.rate
        }


class GroqClient:
    """
    Async client for Groq API with Llama 3.

    Provides emotionally intelligent responses with prosody parameters
    for text-to-speech synthesis.
    """

    def __init__(self, config: Optional[GroqConfig] = None):
        """
        Initialize the Groq client.

        Args:
            config: Optional GroqConfig. If not provided, uses environment variables.
        """
        if config is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is required")
            config = GroqConfig(api_key=api_key)

        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy initialization of Groq client."""
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.config.api_key)
        return self._client

    def chat(self, user_message: str, context: Optional[str] = None) -> str:
        """
        Send a message to Llama 3 and get a response.

        Args:
            user_message: The user's input text.
            context: Optional conversation context/history.

        Returns:
            The assistant's response text.
        """
        client = self._get_client()

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

        # Call Groq API
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return response.choices[0].message.content

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for emotional intelligence.

        The prompt instructs Llama 3 to:
        1. Understand user intent and emotion
        2. Generate appropriate responses
        3. Return JSON with prosody parameters for TTS
        """
        return """You are ConversaVoice, an emotionally intelligent voice assistant.

Your task is to:
1. Understand what the user is saying and how they feel
2. Respond with empathy and appropriate emotional tone
3. Return your response in JSON format

Analyze the user's emotional state:
- If they seem frustrated or are repeating themselves, use style "empathetic"
- If they seem confused, use style "patient"
- If they seem happy or excited, use style "cheerful"
- For neutral queries, use style "neutral"

Always respond with valid JSON in this exact format:
{
    "reply": "Your response text here",
    "style": "empathetic|patient|cheerful|neutral",
    "pitch": "-10% to +10%",
    "rate": "0.8 to 1.2"
}

Prosody guidelines:
- empathetic: pitch="-5%", rate="0.85"
- patient: pitch="-3%", rate="0.9"
- cheerful: pitch="+5%", rate="1.1"
- neutral: pitch="0%", rate="1.0"

Important: Only output the JSON object, no additional text."""

    def get_emotional_response(
        self, user_message: str, context: Optional[str] = None
    ) -> EmotionalResponse:
        """
        Get an emotionally-aware response with prosody parameters.

        Args:
            user_message: The user's input text.
            context: Optional conversation context/history.

        Returns:
            EmotionalResponse with reply and prosody parameters.
        """
        try:
            raw_response = self.chat(user_message, context)
            return self._parse_response(raw_response)
        except Exception as e:
            logger.error(f"Error getting emotional response: {e}")
            # Return a safe fallback response
            return EmotionalResponse(
                reply="I'm sorry, I encountered an issue. Could you please repeat that?",
                style="empathetic",
                pitch="-5%",
                rate="0.9",
                raw_response=str(e)
            )

    def _parse_response(self, raw_response: str) -> EmotionalResponse:
        """
        Parse the LLM response into an EmotionalResponse.

        Args:
            raw_response: Raw text from the LLM.

        Returns:
            Parsed EmotionalResponse object.
        """
        try:
            # Try to extract JSON from the response
            # Handle cases where LLM might add extra text
            json_str = raw_response.strip()

            # Find JSON object in response
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx]

            data = json.loads(json_str)

            return EmotionalResponse(
                reply=data.get("reply", ""),
                style=data.get("style", "neutral"),
                pitch=data.get("pitch", "0%"),
                rate=data.get("rate", "1.0"),
                raw_response=raw_response
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # If JSON parsing fails, use the raw response as the reply
            return EmotionalResponse(
                reply=raw_response,
                style="neutral",
                pitch="0%",
                rate="1.0",
                raw_response=raw_response
            )
