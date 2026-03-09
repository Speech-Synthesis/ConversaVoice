"""Service layer for orchestrator management."""

import os
import sys
import asyncio
import logging
from typing import Dict, Optional
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.orchestrator import Orchestrator, PipelineResult

logger = logging.getLogger(__name__)


class OrchestratorService:
    """
    Service layer for managing orchestrator instances.
    
    Maintains a pool of orchestrator instances per session.
    """
    
    def __init__(self):
        """Initialize the orchestrator service."""
        self._orchestrators: Dict[str, Orchestrator] = {}
        self._lock = asyncio.Lock()
    
    async def get_orchestrator(self, session_id: str) -> Orchestrator:
        """
        Get or create an orchestrator for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Orchestrator instance for the session
        """
        async with self._lock:
            if session_id not in self._orchestrators:
                logger.info(f"Creating new orchestrator for session: {session_id}")
                orchestrator = Orchestrator(session_id=session_id)
                await orchestrator.initialize()
                self._orchestrators[session_id] = orchestrator
            
            return self._orchestrators[session_id]
    
    async def transcribe_audio(
        self, 
        audio_data: bytes, 
        session_id: str
    ) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio file bytes
            session_id: Session ID
            
        Returns:
            Transcribed text
        """
        orchestrator = await self.get_orchestrator(session_id)
        
        # Save audio to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        try:
            # Ensure STT is initialized
            if not orchestrator._stt_client:
                await orchestrator.initialize_stt()
            
            # Transcribe the audio file
            text = orchestrator._stt_client.transcribe_file(temp_audio_path)
            logger.info(f"Transcribed audio for session {session_id}: {text}")
            return text
        finally:
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    async def process_chat(
        self, 
        text: str, 
        session_id: str
    ) -> PipelineResult:
        """
        Process chat text through LLM.
        
        Args:
            text: User input text
            session_id: Session ID
            
        Returns:
            Pipeline result with response and metadata
        """
        orchestrator = await self.get_orchestrator(session_id)
        result = await orchestrator.process_text(text, speak=False)
        logger.info(f"Processed chat for session {session_id}: {result.assistant_response[:50]}...")
        return result
    
    async def synthesize_speech(
        self,
        text: str,
        style: Optional[str] = None,
        pitch: Optional[str] = None,
        rate: Optional[str] = None,
        voice_gender: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            style: Emotional style label
            pitch: Pitch adjustment
            rate: Speech rate adjustment
            voice_gender: Voice gender (male/female)
            session_id: Optional session ID for context
            
        Returns:
            Path to the generated audio file
        """
        # Use a default session if none provided
        if not session_id:
            session_id = "tts-default"
        
        orchestrator = await self.get_orchestrator(session_id)
        
        # Create temp file for audio output
        import tempfile
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_path = temp_audio.name
        temp_audio.close()
        
        # Get TTS client
        tts_client = orchestrator._tts_client
        
        # Update TTS Client gender if provided
        if voice_gender:
            tts_client.voice_gender = voice_gender
            tts_client.ssml_builder.voice = tts_client.ssml_builder.DEFAULT_MALE_VOICE if voice_gender == "male" else tts_client.ssml_builder.DEFAULT_FEMALE_VOICE
            
        # Build SSML with parameters
        ssml = tts_client.ssml_builder.build_from_llm_response(
            text=text,
            style=style,
            pitch=pitch,
            rate=rate
        )
        
        # Synthesize to file (blocking call wrapped in thread)
        import azure.cognitiveservices.speech as speechsdk
        
        def _speak():
            audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=tts_client._speech_config,
                audio_config=audio_config
            )
            return synthesizer.speak_ssml_async(ssml).get()

        loop = asyncio.get_running_loop()
        synthesis_result = await loop.run_in_executor(None, _speak)
        
        if synthesis_result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.error(f"TTS synthesis failed: {synthesis_result.reason}")
            raise Exception("Failed to synthesize speech")
        
        logger.info(f"Synthesized speech to: {audio_path}")
        return audio_path
    
    async def get_health_status(self) -> dict:
        """
        Get health status of all services.
        """
        temp_session = "health-check"
        try:
            # Check if we already have one
            async with self._lock:
                if temp_session in self._orchestrators:
                    orchestrator = self._orchestrators[temp_session]
                else:
                    # Initialize lazily without waiting for all connections immediately if possible
                    # Or just return basic info here to avoid timeout
                    return {
                        "status": "Starting up",
                        "stt": "unknown",
                        "llm": "unknown",
                        "tts": "unknown",
                        "fallback_status": {}
                    }
                    
            fallback_status = orchestrator.get_fallback_status()
            
            return {
                "stt": "healthy" if orchestrator._stt_client else "not_initialized",
                "llm": "healthy" if orchestrator._llm_client else "not_initialized",
                "tts": "healthy" if orchestrator._tts_client else "not_initialized",
                "fallback_status": fallback_status
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "stt": "error",
                "llm": "error",
                "tts": "error",
                "error": str(e)
            }
    
    async def cleanup_session(self, session_id: str):
        """
        Clean up a session and its orchestrator.
        
        Args:
            session_id: Session ID to clean up
        """
        async with self._lock:
            if session_id in self._orchestrators:
                logger.info(f"Cleaning up session: {session_id}")
                orchestrator = self._orchestrators[session_id]
                await orchestrator.shutdown()
                del self._orchestrators[session_id]


# Global service instance
orchestrator_service = OrchestratorService()
