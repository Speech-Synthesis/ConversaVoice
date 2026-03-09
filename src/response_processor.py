import asyncio
import logging
from typing import Optional, Callable
from .fallback import FallbackManager, ServiceType
from .tts import TTSError, PiperTTSError

logger = logging.getLogger(__name__)

class ResponseProcessor:
    """
    Handles LLM and TTS processing with fallback support.
    """
    
    def __init__(self, fallback_manager: FallbackManager, llm_client, local_llm_client=None, tts_client=None, local_tts_client=None):
        self.fallback_manager = fallback_manager
        self._llm_client = llm_client
        self._local_llm_client = local_llm_client
        self._tts_client = tts_client
        self._local_tts_client = local_tts_client

    def _get_active_llm_client(self):
        """Get the currently active LLM client based on fallback status."""
        if self.fallback_manager.should_use_local(ServiceType.LLM):
            return self._local_llm_client or self._llm_client
        return self._llm_client or self._local_llm_client

    def _get_active_tts_client(self):
        """Get the currently active TTS client based on fallback status."""
        if self.fallback_manager.should_use_local(ServiceType.TTS):
            return self._local_tts_client or self._tts_client
        return self._tts_client or self._local_tts_client

    async def _get_llm_call(self, user_input: str, context: str, stream: bool = False, on_token: Optional[Callable[[str], None]] = None):
        """Shared logic for LLM calls with fallback."""
        llm_client = self._get_active_llm_client()
        loop = asyncio.get_event_loop()
        
        try:
            if stream:
                response = await loop.run_in_executor(
                    None,
                    lambda: llm_client.get_emotional_response_stream(
                        user_input,
                        context=context,
                        on_token=on_token
                    )
                )
            else:
                response = await loop.run_in_executor(
                    None, 
                    lambda: llm_client.get_emotional_response(user_input, context=context)
                )
            self.fallback_manager.report_success(ServiceType.LLM)
            return response
        except Exception as e:
            logger.warning(f"LLM {'streaming ' if stream else ''}failed: {e}")
            self.fallback_manager.report_failure(ServiceType.LLM, str(e))
            
            # Try fallback
            fallback_client = self._local_llm_client if llm_client == self._llm_client else self._llm_client
            if fallback_client:
                try:
                    if stream:
                        response = await loop.run_in_executor(
                            None,
                            lambda: fallback_client.get_emotional_response_stream(
                                user_input,
                                context=context,
                                on_token=on_token
                            )
                        )
                    else:
                        response = await loop.run_in_executor(
                            None,
                            lambda: fallback_client.get_emotional_response(user_input, context=context)
                        )
                    self.fallback_manager.report_success(ServiceType.LLM)
                    return response
                except Exception as e2:
                    logger.error(f"LLM fallback {'streaming ' if stream else ''}failed: {e2}")
                    self.fallback_manager.report_failure(ServiceType.LLM, str(e2))
                    raise
            else:
                raise

    async def get_llm_response(self, user_input: str, context: str) -> tuple[str, str]:
        """Get LLM response with context awareness and fallback."""
        response = await self._get_llm_call(user_input, context, stream=False)
        if response is None:
            raise Exception("No LLM response received")
        return response.reply, response.style

    async def get_llm_response_stream(self, user_input: str, context: str, on_token: Optional[Callable[[str], None]] = None) -> tuple[str, str]:
        """Get streaming LLM response."""
        response = await self._get_llm_call(user_input, context, stream=True, on_token=on_token)
        if response is None:
            raise Exception("No LLM response received")
        return response.reply, response.style

    async def speak(self, text: str, style: str, prosody: dict) -> None:
        """Synthesize and speak text."""
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
            self.fallback_manager.report_success(ServiceType.TTS)
        except (TTSError, PiperTTSError) as e:
            logger.warning(f"TTS failed: {e}")
            self.fallback_manager.report_failure(ServiceType.TTS, str(e))
            
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
                    self.fallback_manager.report_success(ServiceType.TTS)
                except Exception as e2:
                    logger.error(f"TTS fallback failed: {e2}")
                    self.fallback_manager.report_failure(ServiceType.TTS, str(e2))
                    raise
            else:
                raise
