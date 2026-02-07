"""
Async Orchestrator for ConversaVoice.

Manages the real-time pipeline: Microphone → Whisper → LLM → TTS → Speaker
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

from .llm import GroqClient
from .memory import RedisClient, VectorStore


class PipelineState(Enum):
    """States of the voice assistant pipeline."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class PipelineResult:
    """Result from a single pipeline cycle."""

    user_input: str
    assistant_response: str
    style: Optional[str] = None
    pitch: Optional[str] = None
    rate: Optional[str] = None
    is_repetition: bool = False
    latency_ms: float = 0.0


class Orchestrator:
    """
    Async orchestrator for the voice assistant pipeline.

    Coordinates: Whisper (STT) → Groq (LLM) → Azure (TTS)
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        on_state_change: Optional[Callable[[PipelineState], None]] = None,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            session_id: Session ID for conversation memory
            on_state_change: Callback when pipeline state changes
            on_transcription: Callback when user speech is transcribed
            on_response: Callback when assistant response is ready
        """
        self.session_id = session_id or self._generate_session_id()
        self.on_state_change = on_state_change
        self.on_transcription = on_transcription
        self.on_response = on_response

        self._state = PipelineState.IDLE
        self._running = False
        self._lock = asyncio.Lock()

        # Components (initialized lazily)
        self._llm_client = None
        self._tts_client = None
        self._redis_client = None
        self._vector_store = None

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if the orchestrator is running."""
        return self._running

    def _set_state(self, state: PipelineState) -> None:
        """Update pipeline state and notify callback."""
        self._state = state
        if self.on_state_change:
            self.on_state_change(state)

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return f"session-{uuid.uuid4().hex[:8]}"

    async def initialize(self) -> None:
        """
        Initialize all pipeline components.

        Call this before running the pipeline.
        """
        # Initialize LLM client
        self._llm_client = GroqClient()

        # Initialize Redis and memory components
        self._redis_client = RedisClient()
        self._redis_client.create_session(self.session_id)
        self._vector_store = VectorStore(self._redis_client)

    async def _get_llm_response(self, user_input: str) -> tuple[str, str, str, str, bool]:
        """
        Get response from LLM with context awareness.

        Args:
            user_input: User's transcribed text

        Returns:
            Tuple of (reply, style, pitch, rate, is_repetition)
        """
        # Check for repetition
        repetition_result = self._vector_store.check_repetition(
            self.session_id,
            user_input
        )
        is_repetition = repetition_result.is_repetition

        # Store the user message and its vector
        self._redis_client.add_message(self.session_id, "user", user_input)
        self._vector_store.store_vector(self.session_id, user_input)

        # Get conversation context
        context = self._redis_client.get_context_string(self.session_id)

        # Build context hint for LLM if user is repeating
        context_hint = ""
        if is_repetition:
            context_hint = " The user seems to be repeating themselves - respond with extra patience."

        # Get LLM response
        response = self._llm_client.get_emotional_response(
            user_input,
            context=context + context_hint
        )

        # Store assistant response
        self._redis_client.add_message(self.session_id, "assistant", response.reply)

        return (
            response.reply,
            response.style,
            response.pitch,
            response.rate,
            is_repetition
        )

    async def shutdown(self) -> None:
        """
        Shutdown the orchestrator and cleanup resources.
        """
        self._running = False
        self._set_state(PipelineState.IDLE)

    async def process_text(self, text: str) -> PipelineResult:
        """
        Process text input through the pipeline (skip STT).

        Useful for testing or text-based interaction.

        Args:
            text: User input text

        Returns:
            Pipeline result with response and metadata
        """
        start_time = time.perf_counter()

        async with self._lock:
            self._set_state(PipelineState.PROCESSING)

            # Notify transcription callback
            if self.on_transcription:
                self.on_transcription(text)

            # Get LLM response
            reply, style, pitch, rate, is_repetition = await self._get_llm_response(text)

            # Notify response callback
            if self.on_response:
                self.on_response(reply)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            self._set_state(PipelineState.IDLE)

            return PipelineResult(
                user_input=text,
                assistant_response=reply,
                style=style,
                pitch=pitch,
                rate=rate,
                is_repetition=is_repetition,
                latency_ms=latency_ms
            )

    async def process_audio(self, audio_data: bytes) -> PipelineResult:
        """
        Process audio input through the full pipeline.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Pipeline result with response and metadata
        """
        pass  # Will be implemented in subsequent commits

    async def run_interactive(self) -> None:
        """
        Run the orchestrator in interactive mode.

        Continuously listens for voice input and responds.
        Press Ctrl+C to stop.
        """
        pass  # Will be implemented in subsequent commits
