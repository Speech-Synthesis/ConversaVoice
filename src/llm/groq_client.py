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
        2. Generate appropriate responses with behavior changes
        3. Return JSON with prosody parameters for TTS
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
- Prosody: pitch="0%", rate="1.0"

### cheerful (greetings, good news, excitement, task completion)
- Be enthusiastic and action-oriented
- Celebrate progress, encourage next steps
- Keep energy high, be concise
- Prosody: pitch="+5%", rate="1.1"

### patient (confusion, complex topics, learning)
- Simplify explanations
- Break down into smaller steps
- Ask AT MOST one clarifying question
- Prosody: pitch="-3%", rate="0.9"

### empathetic (frustration, repetition, annoyance, escalation)
- **STOP ASKING QUESTIONS IMMEDIATELY**
- Make reasonable assumptions based on context
- Give a direct, actionable answer NOW
- Acknowledge their frustration briefly, then SOLVE the problem
- If you lack info, make a safe/general recommendation
- Prosody: pitch="-5%", rate="0.85"

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
    "style": "neutral|cheerful|patient|empathetic",
    "pitch": "-10% to +10%",
    "rate": "0.8 to 1.2"
}

Only output the JSON object, no additional text."""

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
