"""
Refactored Orchestrator for ConversaVoice.
Uses SessionManager and ResponseProcessor for modularity.
"""

import asyncio
import time
import logging
from typing import Optional, Callable

from .llm import GroqClient, OllamaClient
from .memory import RedisClient, VectorStore
from .tts import AzureTTSClient, PiperTTSClient
from .stt import WhisperClient, STTError
from .nlp import SentimentAnalyzer
from .fallback import FallbackManager, ServiceType, ServiceMode, FallbackConfig
from .orchestrator_models import PipelineState, PipelineResult
from .session_manager import SessionManager
from .response_processor import ResponseProcessor

logger = logging.getLogger(__name__)

class OrchestratorError(Exception):
    """Exception raised when the orchestrator encounters an error."""
    def __init__(self, message: str, component: Optional[str] = None):
        self.component = component
        super().__init__(message)

class Orchestrator:
    """
    Async orchestrator for the voice assistant pipeline.
    Coordinates: STT → Session Management → Response Processing → TTS
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
        self.session_id = session_id or self._generate_session_id()
        self.on_state_change = on_state_change
        self.on_transcription = on_transcription
        self.on_response = on_response
        self.on_token = on_token

        self._state = PipelineState.IDLE
        self._running = False
        self._lock = asyncio.Lock()

        # Components
        self._fallback_manager = FallbackManager(fallback_config)
        self._fallback_manager.set_mode_change_callback(self._on_service_mode_change)
        
        self.session_manager = None
        self.response_processor = None
        self._stt_client = None
        
        # Clients (stored for processor initialization)
        self._llm_client = None
        self._local_llm_client = None
        self._tts_client = None
        self._local_tts_client = None
        self._redis_client = None
        self._vector_store = None
        self._sentiment_analyzer = None

        # Background recording state
        self._recording = False
        self._audio_buffer = []
        self._audio_stream = None
        self._pyaudio_instance = None

    @property
    def state(self) -> PipelineState:
        return self._state

    def _set_state(self, state: PipelineState) -> None:
        self._state = state
        if self.on_state_change:
            self.on_state_change(state)

    def _generate_session_id(self) -> str:
        import uuid
        return f"session-{uuid.uuid4().hex[:8]}"

    def _on_service_mode_change(self, service_type: ServiceType, mode: ServiceMode) -> None:
        logger.info(f"Service {service_type.value} switched to {mode.value}")

    def get_fallback_status(self) -> dict:
        return self._fallback_manager.get_summary()

    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            self._llm_client = GroqClient()
            self._fallback_manager.set_cloud_available(ServiceType.LLM, True)
        except Exception as e:
            logger.warning(f"Cloud LLM init failed: {e}")
            self._fallback_manager.set_cloud_available(ServiceType.LLM, False)

        try:
            self._local_llm_client = OllamaClient()
            if self._local_llm_client.is_available():
                self._fallback_manager.set_local_available(ServiceType.LLM, True)
            else:
                self._local_llm_client = None
        except:
            self._local_llm_client = None

        try:
            self._redis_client = RedisClient()
            self._redis_client.create_session(self.session_id)
            self._redis_client.init_prosody_profiles()
            self._vector_store = VectorStore(self._redis_client)
        except Exception as e:
            raise OrchestratorError(f"Redis init failed: {e}", component="memory")

        try:
            self._tts_client = AzureTTSClient()
            self._fallback_manager.set_cloud_available(ServiceType.TTS, True)
        except Exception as e:
            logger.warning(f"Cloud TTS init failed: {e}")
            self._fallback_manager.set_cloud_available(ServiceType.TTS, False)

        try:
            self._local_tts_client = PiperTTSClient()
            if self._local_tts_client.is_available():
                self._fallback_manager.set_local_available(ServiceType.TTS, True)
            else:
                self._local_tts_client = None
        except:
            self._local_tts_client = None

        self._sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize helper managers
        self.session_manager = SessionManager(
            self.session_id, 
            self._redis_client, 
            self._vector_store, 
            self._sentiment_analyzer
        )
        
        self.response_processor = ResponseProcessor(
            self._fallback_manager,
            self._llm_client,
            self._local_llm_client,
            self._tts_client,
            self._local_tts_client
        )

    async def initialize_stt(self) -> None:
        import os
        stt_backend = os.getenv("STT_BACKEND", "groq").lower()
        try:
            if stt_backend == "groq":
                from .stt import GroqWhisperClient
                self._stt_client = GroqWhisperClient()
            else:
                self._stt_client = WhisperClient()
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._stt_client.load_model)
        except Exception as e:
            raise OrchestratorError(f"STT init failed: {e}", component="stt")

    async def process_text(self, text: str, speak: bool = True) -> PipelineResult:
        start_time = time.perf_counter()
        async with self._lock:
            try:
                self._set_state(PipelineState.PROCESSING)
                if self.on_transcription:
                    self.on_transcription(text)

                # 1. Prepare context
                context, is_repetition = self.session_manager.prepare_context(text)

                # 2. Get LLM response
                reply, style = await self.response_processor.get_llm_response(text, context)
                self.session_manager.add_assistant_response(reply)

                if self.on_response:
                    self.on_response(reply)

                # 3. Handle TTS
                prosody = self.session_manager.get_prosody(style)
                if speak:
                    try:
                        await self.response_processor.speak(reply, style, prosody)
                    except Exception as e:
                        self._redis_client.record_error(self.session_id, "tts")
                        logger.warning(f"TTS Error: {e}")

                latency_ms = (time.perf_counter() - start_time) * 1000
                self._set_state(PipelineState.IDLE)

                return PipelineResult(
                    user_input=text,
                    assistant_response=reply,
                    style=style,
                    is_repetition=is_repetition,
                    latency_ms=latency_ms,
                    pitch=prosody.get("pitch", "0%"),
                    rate=prosody.get("rate", "1.0")
                )
            except Exception as e:
                self._set_state(PipelineState.ERROR)
                self._redis_client.record_error(self.session_id, "pipeline")
                raise OrchestratorError(f"Pipeline error: {e}")

    async def process_text_stream(self, text: str, speak: bool = True, on_token: Optional[Callable[[str], None]] = None) -> PipelineResult:
        start_time = time.perf_counter()
        token_callback = on_token or self.on_token
        async with self._lock:
            try:
                self._set_state(PipelineState.PROCESSING)
                if self.on_transcription:
                    self.on_transcription(text)

                # 1. Prepare context
                context, is_repetition = self.session_manager.prepare_context(text)

                # 2. Get LLM response stream
                reply, style = await self.response_processor.get_llm_response_stream(text, context, on_token=token_callback)
                self.session_manager.add_assistant_response(reply)

                if self.on_response:
                    self.on_response(reply)

                # 3. Handle TTS
                prosody = self.session_manager.get_prosody(style)
                if speak:
                    try:
                        await self.response_processor.speak(reply, style, prosody)
                    except Exception as e:
                        self._redis_client.record_error(self.session_id, "tts")
                        logger.warning(f"TTS Error: {e}")

                latency_ms = (time.perf_counter() - start_time) * 1000
                self._set_state(PipelineState.IDLE)

                return PipelineResult(
                    user_input=text,
                    assistant_response=reply,
                    style=style,
                    is_repetition=is_repetition,
                    latency_ms=latency_ms,
                    pitch=prosody.get("pitch", "0%"),
                    rate=prosody.get("rate", "1.0")
                )
            except Exception as e:
                self._set_state(PipelineState.ERROR)
                self._redis_client.record_error(self.session_id, "pipeline")
                raise OrchestratorError(f"Pipeline error: {e}")

    async def process_voice(self, timeout: float = 10.0, speak: bool = True) -> Optional[PipelineResult]:
        if not self._stt_client:
            raise OrchestratorError("STT not initialized", component="stt")
        self._set_state(PipelineState.LISTENING)
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(None, lambda: self._stt_client.listen_once(timeout=timeout))
        except Exception as e:
            self._set_state(PipelineState.ERROR)
            self._redis_client.record_error(self.session_id, "stt")
            raise OrchestratorError(f"STT error: {e}", component="stt")

        if not text:
            self._set_state(PipelineState.IDLE)
            return None

        return await self.process_text(text, speak=speak)

    async def shutdown(self) -> None:
        self._running = False
        if self._recording:
            try:
                self.stop_recording_background()
            except:
                pass
        self._set_state(PipelineState.IDLE)

    # Audio recording methods (keep as is for now)
    def start_recording_background(self) -> None:
        if not self._stt_client:
            raise OrchestratorError("STT not initialized", component="stt")
        if self._recording: return
        import pyaudio
        self._audio_buffer = []
        self._recording = True
        try:
            self._pyaudio_instance = pyaudio.PyAudio()
            self._audio_stream = self._pyaudio_instance.open(
                format=pyaudio.paFloat32, channels=1, rate=self._stt_client.sample_rate,
                input=True, frames_per_buffer=1024, stream_callback=self._audio_callback
            )
            self._audio_stream.start_stream()
        except Exception as e:
            self._recording = False
            if self._pyaudio_instance: self._pyaudio_instance.terminate()
            raise OrchestratorError(f"Recording failed: {e}", component="stt")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        import numpy as np
        import pyaudio
        if self._recording:
            self._audio_buffer.append(np.frombuffer(in_data, dtype=np.float32))
        return (in_data, pyaudio.paContinue)

    def stop_recording_background(self) -> str:
        if not self._recording: return ""
        import numpy as np
        self._recording = False
        try:
            if self._audio_stream:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            if self._pyaudio_instance: self._pyaudio_instance.terminate()
            if not self._audio_buffer: return ""
            full_audio = np.concatenate(self._audio_buffer)
            self._audio_buffer = []
            return self._stt_client.transcribe_audio(full_audio, self._stt_client.sample_rate)
        except Exception as e:
            self._audio_buffer = []
            raise OrchestratorError(f"Recording process failed: {e}", component="stt")
