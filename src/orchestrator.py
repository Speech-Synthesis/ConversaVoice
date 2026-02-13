"""
Async Orchestrator for ConversaVoice.

Manages the real-time pipeline: Microphone → Whisper → LLM → TTS → Speaker
Supports automatic fallback to local services when cloud APIs fail.
"""

import asyncio
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable
from enum import Enum

from .llm import GroqClient, OllamaClient
from .memory import RedisClient, VectorStore
from .tts import AzureTTSClient, TTSError, PiperTTSClient, PiperTTSError
from .stt import WhisperClient, STTError
from .nlp import SentimentAnalyzer
from .fallback import FallbackManager, ServiceType, ServiceMode, FallbackConfig

logger = logging.getLogger(__name__)


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
    pitch: Optional[str] = None
    rate: Optional[str] = None


class Orchestrator:
    """
    Async orchestrator for the voice assistant pipeline.

    Coordinates: Whisper (STT) → Groq (LLM) → Azure (TTS)
    Supports automatic fallback to local services (Ollama, Piper).
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        on_state_change: Optional[Callable[[PipelineState], None]] = None,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        fallback_config: Optional[FallbackConfig] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            session_id: Session ID for conversation memory
            on_state_change: Callback when pipeline state changes
            on_transcription: Callback when user speech is transcribed
            on_response: Callback when assistant response is ready
            on_token: Callback for each token in streaming mode
            fallback_config: Configuration for fallback behavior
        """
        self.session_id = session_id or self._generate_session_id()
        self.on_state_change = on_state_change
        self.on_transcription = on_transcription
        self.on_response = on_response
        self.on_token = on_token

        self._state = PipelineState.IDLE
        self._running = False
        self._lock = asyncio.Lock()

        # Fallback manager for cloud/local switching
        self._fallback_manager = FallbackManager(fallback_config)
        self._fallback_manager.set_mode_change_callback(self._on_service_mode_change)

        # Cloud components (initialized lazily)
        self._llm_client = None
        self._tts_client = None
        self._redis_client = None
        self._vector_store = None
        self._stt_client = None
        self._sentiment_analyzer = None

        # Local fallback components
        self._local_llm_client = None
        self._local_tts_client = None

        # Background recording state
        self._recording = False
        self._audio_buffer = []
        self._audio_stream = None
        self._pyaudio_instance = None

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

    def _on_service_mode_change(self, service_type: ServiceType, mode: ServiceMode) -> None:
        """Handle service mode changes (cloud to local or vice versa)."""
        logger.info(f"Service {service_type.value} switched to {mode.value}")
        if self.on_state_change:
            # Briefly signal state change for mode switch awareness
            pass  # Could add a callback for mode changes if needed

    def get_fallback_status(self) -> dict:
        """Get current fallback status for all services."""
        return self._fallback_manager.get_summary()

    async def initialize(self) -> None:
        """
        Initialize all pipeline components.

        Call this before running the pipeline.
        Initializes both cloud and local fallback clients when available.

        Raises:
            OrchestratorError: If any component fails to initialize
        """
        # Initialize cloud LLM client
        try:
            self._llm_client = GroqClient()
            self._fallback_manager.set_cloud_available(ServiceType.LLM, True)
        except Exception as e:
            logger.warning(f"Failed to initialize cloud LLM: {e}")
            self._fallback_manager.set_cloud_available(ServiceType.LLM, False)

        # Initialize local LLM fallback (Ollama)
        try:
            self._local_llm_client = OllamaClient()
            if self._local_llm_client.is_available():
                self._fallback_manager.set_local_available(ServiceType.LLM, True)
                logger.info("Local LLM (Ollama) available as fallback")
            else:
                self._local_llm_client = None
        except Exception as e:
            logger.debug(f"Local LLM not available: {e}")
            self._local_llm_client = None

        # Ensure at least one LLM is available
        if self._llm_client is None and self._local_llm_client is None:
            raise OrchestratorError("No LLM available (cloud or local)", component="llm")

        # Initialize Redis and memory components
        try:
            self._redis_client = RedisClient()
            self._redis_client.create_session(self.session_id)
            self._redis_client.init_prosody_profiles()
            self._vector_store = VectorStore(self._redis_client)
        except Exception as e:
            raise OrchestratorError(f"Failed to initialize Redis: {e}", component="memory")

        # Initialize cloud TTS client
        try:
            self._tts_client = AzureTTSClient()
            self._fallback_manager.set_cloud_available(ServiceType.TTS, True)
        except Exception as e:
            logger.warning(f"Failed to initialize cloud TTS: {e}")
            self._fallback_manager.set_cloud_available(ServiceType.TTS, False)

        # Initialize local TTS fallback (Piper)
        try:
            self._local_tts_client = PiperTTSClient()
            if self._local_tts_client.is_available():
                self._fallback_manager.set_local_available(ServiceType.TTS, True)
                logger.info("Local TTS (Piper) available as fallback")
            else:
                self._local_tts_client = None
        except Exception as e:
            logger.debug(f"Local TTS not available: {e}")
            self._local_tts_client = None

        # Ensure at least one TTS is available
        if self._tts_client is None and self._local_tts_client is None:
            raise OrchestratorError("No TTS available (cloud or local)", component="tts")

        # Initialize sentiment analyzer (lightweight, no external deps required)
        self._sentiment_analyzer = SentimentAnalyzer()

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

    def _get_external_context(self) -> str:
        """
        Get external context information (date, time, day of week).

        Returns:
            Formatted string with current date/time context.
        """
        now = datetime.now()
        day_name = now.strftime("%A")
        date_str = now.strftime("%B %d, %Y")
        time_str = now.strftime("%I:%M %p")

        # Time of day context
        hour = now.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"

        return f"Current time: {time_str} ({time_of_day}), {day_name}, {date_str}"

    def _get_active_llm_client(self):
        """Get the currently active LLM client based on fallback status."""
        if self._fallback_manager.should_use_local(ServiceType.LLM):
            return self._local_llm_client or self._llm_client
        return self._llm_client or self._local_llm_client

    async def _get_llm_response(self, user_input: str) -> tuple[str, str, bool]:
        """
        Get response from LLM with context awareness and fallback support.

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

        # Analyze sentiment for emotion detection
        detected_emotion = None
        if self._sentiment_analyzer:
            detected_emotion = self._sentiment_analyzer.get_emotion_for_context(user_input)

        # Update context labels (first_time, repetition, frustration)
        context_labels = self._redis_client.update_context_labels(
            self.session_id,
            is_repetition=is_repetition,
            detected_emotion=detected_emotion
        )

        # Detect and store user preferences from message
        detected_prefs = self._redis_client.detect_preferences_from_message(user_input)
        if detected_prefs:
            self._redis_client.set_user_preferences(self.session_id, detected_prefs)

        # Store the user message in conversation history
        self._redis_client.add_message(self.session_id, "user", user_input)

        # Get conversation context
        context = self._redis_client.get_context_string(self.session_id)

        # Add external context (date/time)
        external_context = self._get_external_context()
        context = f"[{external_context}]\n\n{context}"

        # Get context hint based on labels (first_time, repetition, frustration)
        context_hint = self._redis_client.get_context_hint(self.session_id)
        if context_hint:
            context = f"{context}\n\n[Context: {context_hint}]"

        # Add user preferences hint
        prefs_hint = self._redis_client.get_preferences_hint(self.session_id)
        if prefs_hint:
            context = f"{context}\n\n[User Preferences: {prefs_hint}]"

        # Get LLM response with fallback support
        response = None
        llm_client = self._get_active_llm_client()

        try:
            response = llm_client.get_emotional_response(user_input, context=context)
            self._fallback_manager.report_success(ServiceType.LLM)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            self._fallback_manager.report_failure(ServiceType.LLM, str(e))

            # Try fallback if available
            fallback_client = self._local_llm_client if llm_client == self._llm_client else self._llm_client
            if fallback_client:
                try:
                    response = fallback_client.get_emotional_response(user_input, context=context)
                    self._fallback_manager.report_success(ServiceType.LLM)
                except Exception as e2:
                    logger.error(f"LLM fallback also failed: {e2}")
                    self._fallback_manager.report_failure(ServiceType.LLM, str(e2))
                    raise

        if response is None:
            raise OrchestratorError("No LLM response received", component="llm")

        # Store assistant response
        self._redis_client.add_message(self.session_id, "assistant", response.reply)

        return (
            response.reply,
            response.style,
            is_repetition
        )

    async def _get_llm_response_stream(
        self,
        user_input: str,
        on_token: Optional[Callable[[str], None]] = None
    ) -> tuple[str, str, bool]:
        """
        Get response from LLM with streaming for lower latency.
        Supports fallback to local LLM.

        Args:
            user_input: User's transcribed text
            on_token: Callback for each token received

        Returns:
            Tuple of (reply, style, is_repetition)
        """
        # Check for repetition (also stores the embedding)
        repetition_result = self._vector_store.check_repetition(
            self.session_id,
            user_input
        )
        is_repetition = repetition_result.is_repetition

        # Analyze sentiment for emotion detection
        detected_emotion = None
        if self._sentiment_analyzer:
            detected_emotion = self._sentiment_analyzer.get_emotion_for_context(user_input)

        # Update context labels (first_time, repetition, frustration)
        self._redis_client.update_context_labels(
            self.session_id,
            is_repetition=is_repetition,
            detected_emotion=detected_emotion
        )

        # Detect and store user preferences from message
        detected_prefs = self._redis_client.detect_preferences_from_message(user_input)
        if detected_prefs:
            self._redis_client.set_user_preferences(self.session_id, detected_prefs)

        # Store the user message in conversation history
        self._redis_client.add_message(self.session_id, "user", user_input)

        # Get conversation context
        context = self._redis_client.get_context_string(self.session_id)

        # Add external context (date/time)
        external_context = self._get_external_context()
        context = f"[{external_context}]\n\n{context}"

        # Get context hint based on labels
        context_hint = self._redis_client.get_context_hint(self.session_id)
        if context_hint:
            context = f"{context}\n\n[Context: {context_hint}]"

        # Add user preferences hint
        prefs_hint = self._redis_client.get_preferences_hint(self.session_id)
        if prefs_hint:
            context = f"{context}\n\n[User Preferences: {prefs_hint}]"

        # Get LLM response with streaming and fallback support
        loop = asyncio.get_event_loop()
        llm_client = self._get_active_llm_client()
        response = None

        try:
            response = await loop.run_in_executor(
                None,
                lambda: llm_client.get_emotional_response_stream(
                    user_input,
                    context=context,
                    on_token=on_token
                )
            )
            self._fallback_manager.report_success(ServiceType.LLM)
        except Exception as e:
            logger.warning(f"LLM streaming failed: {e}")
            self._fallback_manager.report_failure(ServiceType.LLM, str(e))

            # Try fallback
            fallback_client = self._local_llm_client if llm_client == self._llm_client else self._llm_client
            if fallback_client:
                try:
                    response = await loop.run_in_executor(
                        None,
                        lambda: fallback_client.get_emotional_response_stream(
                            user_input,
                            context=context,
                            on_token=on_token
                        )
                    )
                    self._fallback_manager.report_success(ServiceType.LLM)
                except Exception as e2:
                    logger.error(f"LLM fallback streaming also failed: {e2}")
                    self._fallback_manager.report_failure(ServiceType.LLM, str(e2))
                    raise

        if response is None:
            raise OrchestratorError("No LLM response received", component="llm")

        # Store assistant response
        self._redis_client.add_message(self.session_id, "assistant", response.reply)

        return (
            response.reply,
            response.style,
            is_repetition
        )

    def _get_active_tts_client(self):
        """Get the currently active TTS client based on fallback status."""
        if self._fallback_manager.should_use_local(ServiceType.TTS):
            return self._local_tts_client or self._tts_client
        return self._tts_client or self._local_tts_client

    async def _speak(self, text: str, style: Optional[str] = None) -> None:
        """
        Synthesize and speak text using TTS with fallback support.

        Fetches prosody parameters from Redis based on style label.

        Args:
            text: Text to speak
            style: Emotional style label (e.g., "empathetic")
        """
        self._set_state(PipelineState.SPEAKING)

        # Fetch prosody from Redis based on style
        prosody = self._redis_client.get_prosody(style or "neutral")

        # Get active TTS client
        tts_client = self._get_active_tts_client()
        loop = asyncio.get_event_loop()

        try:
            await loop.run_in_executor(
                None,
                lambda: tts_client.speak_with_llm_params(
                    text=text,
                    style=style,
                    pitch=prosody.get("pitch", "0%"),
                    rate=prosody.get("rate", "1.0")
                )
            )
            self._fallback_manager.report_success(ServiceType.TTS)
        except (TTSError, PiperTTSError) as e:
            logger.warning(f"TTS failed: {e}")
            self._fallback_manager.report_failure(ServiceType.TTS, str(e))

            # Try fallback
            fallback_client = self._local_tts_client if tts_client == self._tts_client else self._tts_client
            if fallback_client:
                try:
                    await loop.run_in_executor(
                        None,
                        lambda: fallback_client.speak_with_llm_params(
                            text=text,
                            style=style,
                            pitch=prosody.get("pitch", "0%"),
                            rate=prosody.get("rate", "1.0")
                        )
                    )
                    self._fallback_manager.report_success(ServiceType.TTS)
                except Exception as e2:
                    logger.error(f"TTS fallback also failed: {e2}")
                    self._fallback_manager.report_failure(ServiceType.TTS, str(e2))
                    raise
            else:
                raise

    def start_recording_background(self) -> None:
        """
        Start background audio recording for Streamlit.
        
        Call this when the user clicks the microphone button.
        Audio is buffered until stop_recording_background() is called.
        
        Raises:
            OrchestratorError: If STT not initialized or recording fails
        """
        if not self._stt_client:
            raise OrchestratorError("STT not initialized. Call initialize_stt() first.", component="stt")
        
        if self._recording:
            return  # Already recording
        
        import pyaudio
        import numpy as np
        
        self._audio_buffer = []
        self._recording = True
        
        try:
            self._pyaudio_instance = pyaudio.PyAudio()
            
            # Open audio stream
            self._audio_stream = self._pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self._stt_client.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self._audio_callback
            )
            
            self._audio_stream.start_stream()
            logger.info("[Background recording started]")
            
        except Exception as e:
            self._recording = False
            if self._pyaudio_instance:
                self._pyaudio_instance.terminate()
                self._pyaudio_instance = None
            raise OrchestratorError(f"Failed to start recording: {e}", component="stt")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream to buffer audio data."""
        import numpy as np
        import pyaudio
        
        if self._recording:
            audio_chunk = np.frombuffer(in_data, dtype=np.float32)
            self._audio_buffer.append(audio_chunk)
        
        return (in_data, pyaudio.paContinue)
    
    def stop_recording_background(self) -> str:
        """
        Stop background recording and transcribe the captured audio.
        
        Returns:
            Transcribed text from the recording
            
        Raises:
            OrchestratorError: If transcription fails
        """
        if not self._recording:
            return ""
        
        import numpy as np
        
        self._recording = False
        
        try:
            # Stop and close the stream
            if self._audio_stream:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
                self._audio_stream = None
            
            if self._pyaudio_instance:
                self._pyaudio_instance.terminate()
                self._pyaudio_instance = None
            
            # Process the buffered audio
            if not self._audio_buffer:
                logger.info("[No audio captured]")
                return ""
            
            full_audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []
            
            # Check if there's enough audio
            duration = len(full_audio) / self._stt_client.sample_rate
            if duration < self._stt_client.min_speech_duration:
                logger.info(f"[Audio too short: {duration:.2f}s]")
                return ""
            
            # Transcribe
            logger.info(f"[Transcribing {duration:.2f}s of audio...]")
            text = self._stt_client.transcribe_audio(full_audio, self._stt_client.sample_rate)
            logger.info(f"[Transcribed: {text}]")
            return text
            
        except Exception as e:
            self._audio_buffer = []
            raise OrchestratorError(f"Failed to process recording: {e}", component="stt")

    async def shutdown(self) -> None:
        """
        Shutdown the orchestrator and cleanup resources.
        """
        self._running = False
        
        # Stop any ongoing recording
        if self._recording:
            try:
                self.stop_recording_background()
            except Exception as e:
                logger.warning(f"Error stopping recording during shutdown: {e}")
        
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

                # Get prosody parameters for UI display
                prosody = self._redis_client.get_prosody(style or "neutral")
                pitch = prosody.get("pitch", "0%")
                rate = prosody.get("rate", "1.0")

                # Speak the response (prosody fetched from Redis in _speak)
                if speak:
                    try:
                        await self._speak(reply, style)
                    except (TTSError, PiperTTSError) as e:
                        self._redis_client.record_error(self.session_id, "tts")
                        logger.warning(f"TTS Warning: {e}")

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                self._set_state(PipelineState.IDLE)

                return PipelineResult(
                    user_input=text,
                    assistant_response=reply,
                    style=style,
                    is_repetition=is_repetition,
                    latency_ms=latency_ms,
                    pitch=pitch,
                    rate=rate
                )

            except Exception as e:
                self._set_state(PipelineState.ERROR)
                self._redis_client.record_error(self.session_id, "pipeline")
                raise OrchestratorError(f"Pipeline error: {e}")

    async def process_text_stream(
        self,
        text: str,
        speak: bool = True,
        on_token: Optional[Callable[[str], None]] = None
    ) -> PipelineResult:
        """
        Process text input with streaming LLM response.

        Shows tokens as they arrive for lower perceived latency.

        Args:
            text: User input text
            speak: Whether to speak the response (default True)
            on_token: Callback for each token (overrides self.on_token)

        Returns:
            Pipeline result with response and metadata

        Raises:
            OrchestratorError: If processing fails
        """
        start_time = time.perf_counter()
        token_callback = on_token or self.on_token

        async with self._lock:
            try:
                self._set_state(PipelineState.PROCESSING)

                # Notify transcription callback
                if self.on_transcription:
                    self.on_transcription(text)

                # Get LLM response with streaming
                reply, style, is_repetition = await self._get_llm_response_stream(
                    text,
                    on_token=token_callback
                )

                # Notify response callback (full response)
                if self.on_response:
                    self.on_response(reply)

                # Get prosody parameters for UI display
                prosody = self._redis_client.get_prosody(style or "neutral")
                pitch = prosody.get("pitch", "0%")
                rate = prosody.get("rate", "1.0")

                # Speak the response
                if speak:
                    try:
                        await self._speak(reply, style)
                    except (TTSError, PiperTTSError) as e:
                        self._redis_client.record_error(self.session_id, "tts")
                        logger.warning(f"TTS Warning: {e}")

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                self._set_state(PipelineState.IDLE)

                return PipelineResult(
                    user_input=text,
                    assistant_response=reply,
                    style=style,
                    is_repetition=is_repetition,
                    latency_ms=latency_ms,
                    pitch=pitch,
                    rate=rate
                )

            except Exception as e:
                self._set_state(PipelineState.ERROR)
                self._redis_client.record_error(self.session_id, "pipeline")
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
            self._redis_client.record_error(self.session_id, "stt")
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
                print(f"  [style: {result.style}, pitch: {result.pitch}, rate: {result.rate}, latency: {result.latency_ms:.0f}ms]")

                if result.is_repetition:
                    print("  [Detected: User is repeating]")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Shutting down...")
                break

        await self.shutdown()