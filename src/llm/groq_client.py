"""
Groq API client for ConversaVoice.
Provides async interface to Llama 3 for intelligent, emotionally-aware responses.
Supports streaming for reduced latency.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Generator, Callable
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class GroqConfig:
    """Configuration for Groq API client."""
    api_key: str
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class EmotionalResponse:
    """Parsed response with emotional style label and emphasis words."""
    reply: str
    style: str = "neutral"
    emphasis_words: list = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reply": self.reply,
            "style": self.style,
            "emphasis_words": self.emphasis_words
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

    def chat_stream(
        self,
        user_message: str,
        context: Optional[str] = None,
        on_chunk: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, str]:
        """
        Stream a response from Llama 3 token by token.

        Args:
            user_message: The user's input text.
            context: Optional conversation context/history.
            on_chunk: Optional callback for each chunk received.

        Yields:
            Individual tokens/chunks as they arrive.

        Returns:
            Complete response text.
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

        # Call Groq API with streaming
        stream = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                full_response += token
                if on_chunk:
                    on_chunk(token)
                yield token

        return full_response

    def get_emotional_response_stream(
        self,
        user_message: str,
        context: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None
    ) -> EmotionalResponse:
        """
        Get an emotionally-aware response with streaming.

        Streams tokens for lower perceived latency, then parses
        the complete response for style and emphasis.

        Args:
            user_message: The user's input text.
            context: Optional conversation context/history.
            on_token: Optional callback for each token (for real-time display).

        Returns:
            EmotionalResponse with reply, style, and emphasis_words.
        """
        try:
            # Collect all tokens
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

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for emotional intelligence.

        The prompt instructs Llama 3 to:
        1. Understand user intent and emotion
        2. Generate appropriate responses with behavior changes
        3. Return JSON with reply and style label (prosody fetched from Redis)
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

    def get_emotional_response(
        self, user_message: str, context: Optional[str] = None
    ) -> EmotionalResponse:
        """
        Get an emotionally-aware response with style label.

        Args:
            user_message: The user's input text.
            context: Optional conversation context/history.

        Returns:
            EmotionalResponse with reply and style label.
            Prosody parameters are fetched from Redis based on style.
        """
        try:
            raw_response = self.chat(user_message, context)
            return self._parse_response(raw_response)
        except Exception as e:
            print(f"\n[DEBUG] Groq API Error: {e}")
            logger.error(f"Error getting emotional response: {e}")
            # Return a safe fallback response
            return EmotionalResponse(
                reply="I'm sorry, I encountered an issue. Could you please repeat that?",
                style="empathetic",
                raw_response=str(e)
            )

    def _parse_response(self, raw_response: str) -> EmotionalResponse:
        """
        Parse the LLM response into an EmotionalResponse.

        Args:
            raw_response: Raw text from the LLM.

        Returns:
            Parsed EmotionalResponse object with reply, style, and emphasis_words.
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

            # Extract emphasis_words (default to empty list if not provided)
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
            # If JSON parsing fails, use the raw response as the reply
            return EmotionalResponse(
                reply=raw_response,
                style="neutral",
                emphasis_words=[],
                raw_response=raw_response
            )
