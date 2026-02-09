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
from .tts import AzureTTSClient, TTSError
from .stt import WhisperClient, STTError


class OrchestratorError(Exception):
    """Exception raised when the orchestrator encounters an error."""

    def __init__(self, message: str, component: Optional[str] = None):
        self.component = component
        super().__init__(message)


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
        self._stt_client = None

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

        Raises:
            OrchestratorError: If any component fails to initialize
        """
        try:
            # Initialize LLM client
            self._llm_client = GroqClient()
        except Exception as e:
            raise OrchestratorError(f"Failed to initialize LLM: {e}", component="llm")

        try:
            # Initialize Redis and memory components
            self._redis_client = RedisClient()
            self._redis_client.create_session(self.session_id)
            self._redis_client.init_prosody_profiles()  # Initialize prosody profiles
            self._vector_store = VectorStore(self._redis_client)
        except Exception as e:
            raise OrchestratorError(f"Failed to initialize Redis: {e}", component="memory")

        try:
            # Initialize TTS client
            self._tts_client = AzureTTSClient()
        except Exception as e:
            raise OrchestratorError(f"Failed to initialize TTS: {e}", component="tts")

    async def initialize_stt(self) -> None:
        """
        Initialize the STT component (Whisper).

        Call this only if using voice input mode.

        Raises:
            OrchestratorError: If STT fails to initialize
        """
        try:
            self._stt_client = WhisperClient()
            # Load model in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._stt_client.load_model)
        except Exception as e:
            raise OrchestratorError(f"Failed to initialize STT: {e}", component="stt")

    async def _get_llm_response(self, user_input: str) -> tuple[str, str, bool]:
        """
        Get response from LLM with context awareness.

        Args:
            user_input: User's transcribed text

        Returns:
            Tuple of (reply, style, is_repetition)
        """
        # Check for repetition (also stores the embedding)
        repetition_result = self._vector_store.check_repetition(
            self.session_id,
            user_input
        )
        is_repetition = repetition_result.is_repetition

        # Update context labels (first_time, repetition, frustration)
        context_labels = self._redis_client.update_context_labels(
            self.session_id,
            is_repetition=is_repetition
        )

        # Store the user message in conversation history
        self._redis_client.add_message(self.session_id, "user", user_input)

        # Get conversation context
        context = self._redis_client.get_context_string(self.session_id)

        # Get context hint based on labels (first_time, repetition, frustration)
        context_hint = self._redis_client.get_context_hint(self.session_id)
        if context_hint:
            context = f"{context}\n\n[Context: {context_hint}]"

        # Get LLM response
        response = self._llm_client.get_emotional_response(
            user_input,
            context=context
        )

        # Store assistant response
        self._redis_client.add_message(self.session_id, "assistant", response.reply)

        return (
            response.reply,
            response.style,
            is_repetition
        )

    async def _speak(self, text: str, style: Optional[str] = None) -> None:
        """
        Synthesize and speak text using TTS.

        Fetches prosody parameters from Redis based on style label.

        Args:
            text: Text to speak
            style: Emotional style label (e.g., "empathetic")
        """
        self._set_state(PipelineState.SPEAKING)

        # Fetch prosody from Redis based on style
        prosody = self._redis_client.get_prosody(style or "neutral")

        # Run TTS in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._tts_client.speak_with_llm_params(
                text=text,
                style=style,
                pitch=prosody.get("pitch", "0%"),
                rate=prosody.get("rate", "1.0")
            )
        )

    async def shutdown(self) -> None:
        """
        Shutdown the orchestrator and cleanup resources.
        """
        self._running = False
        self._set_state(PipelineState.IDLE)

    async def process_text(self, text: str, speak: bool = True) -> PipelineResult:
        """
        Process text input through the pipeline (skip STT).

        Useful for testing or text-based interaction.

        Args:
            text: User input text
            speak: Whether to speak the response (default True)

        Returns:
            Pipeline result with response and metadata

        Raises:
            OrchestratorError: If processing fails
        """
        start_time = time.perf_counter()

        async with self._lock:
            try:
                self._set_state(PipelineState.PROCESSING)

                # Notify transcription callback
                if self.on_transcription:
                    self.on_transcription(text)

                # Get LLM response (style label only, prosody fetched from Redis)
                reply, style, is_repetition = await self._get_llm_response(text)

                # Notify response callback
                if self.on_response:
                    self.on_response(reply)

                # Speak the response (prosody fetched from Redis in _speak)
                if speak:
                    try:
                        await self._speak(reply, style)
                    except TTSError as e:
                        print(f"  [TTS Warning: {e}]")

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                self._set_state(PipelineState.IDLE)

                return PipelineResult(
                    user_input=text,
                    assistant_response=reply,
                    style=style,
                    is_repetition=is_repetition,
                    latency_ms=latency_ms
                )

            except Exception as e:
                self._set_state(PipelineState.ERROR)
                raise OrchestratorError(f"Pipeline error: {e}")

    async def process_voice(self, timeout: float = 10.0, speak: bool = True) -> Optional[PipelineResult]:
        """
        Listen for voice input and process through the pipeline.

        Args:
            timeout: Maximum seconds to wait for speech
            speak: Whether to speak the response (default True)

        Returns:
            Pipeline result with response and metadata, or None if no speech detected

        Raises:
            OrchestratorError: If processing fails
        """
        if not self._stt_client:
            raise OrchestratorError("STT not initialized. Call initialize_stt() first.", component="stt")

        self._set_state(PipelineState.LISTENING)

        # Listen for speech in a thread pool
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                None,
                lambda: self._stt_client.listen_once(timeout=timeout)
            )
        except STTError as e:
            self._set_state(PipelineState.ERROR)
            raise OrchestratorError(f"STT error: {e}", component="stt")

        if not text:
            self._set_state(PipelineState.IDLE)
            return None

        # Notify transcription callback
        if self.on_transcription:
            self.on_transcription(text)

        # Process through the rest of the pipeline
        return await self.process_text(text, speak=speak)

    async def run_interactive(self, use_voice: bool = False) -> None:
        """
        Run the orchestrator in interactive mode.

        Args:
            use_voice: If True, use microphone input (Whisper STT)
                       If False, use text input mode
        """
        self._running = True
        mode = "Voice" if use_voice else "Text"
        print("\n" + "=" * 50)
        print(f"ConversaVoice - Interactive Mode ({mode})")
        print("=" * 50)
        print(f"Session ID: {self.session_id}")
        if use_voice:
            print("Speak into your microphone. Press Ctrl+C to stop.")
        else:
            print("Type 'quit' or 'exit' to stop.")
        print("=" * 50 + "\n")

        await self.initialize()

        if use_voice:
            await self.initialize_stt()

        while self._running:
            try:
                if use_voice:
                    # Voice input mode
                    print("\n[Listening...]")
                    result = await self.process_voice(timeout=15.0)

                    if result is None:
                        print("[No speech detected, try again]")
                        continue

                    print(f"\nYou: {result.user_input}")
                else:
                    # Text input mode
                    self._set_state(PipelineState.LISTENING)
                    user_input = input("\nYou: ").strip()

                    if not user_input:
                        continue

                    if user_input.lower() in ("quit", "exit", "q"):
                        print("\nGoodbye!")
                        break

                    result = await self.process_text(user_input)

                # Display result info
                print(f"\nAssistant: {result.assistant_response}")
                print(f"  [style: {result.style}, latency: {result.latency_ms:.0f}ms]")

                if result.is_repetition:
                    print("  [Detected: User is repeating]")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Shutting down...")
                break

        await self.shutdown()
