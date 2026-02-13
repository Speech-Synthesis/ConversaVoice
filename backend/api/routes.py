"""FastAPI routes for ConversaVoice API."""

import os
import logging
import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse

from .models import (
    TranscribeResponse,
    ChatRequest,
    ChatResponse,
    SynthesizeRequest,
    SynthesizeResponse,
    SessionResponse,
    HealthResponse
)
from ..services.orchestrator_service import orchestrator_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of all services.
    """
    try:
        services = await orchestrator_service.get_health_status()
        return HealthResponse(
            status="healthy",
            services=services
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session", response_model=SessionResponse)
async def create_session():
    """
    Create a new session.
    
    Returns a new session ID.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Created new session: {session_id}")
    
    return SessionResponse(
        session_id=session_id,
        created_at=datetime.now().isoformat()
    )


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    session_id: Optional[str] = Form(None, description="Session ID")
):
    """
    Transcribe audio file to text.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        session_id: Optional session ID for context
        
    Returns:
        Transcribed text
    """
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    try:
        # Read audio data
        audio_data = await audio.read()
        logger.info(f"Received audio file: {audio.filename}, size: {len(audio_data)} bytes")
        
        # Transcribe
        text = await orchestrator_service.transcribe_audio(audio_data, session_id)
        
        return TranscribeResponse(
            text=text,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat text through LLM.
    
    Args:
        request: Chat request with text and optional session ID
        
    Returns:
        LLM response with style and prosody metadata
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Process through LLM
        result = await orchestrator_service.process_chat(request.text, session_id)
        
        return ChatResponse(
            response=result.assistant_response,
            style=result.style,
            pitch=result.pitch,
            rate=result.rate,
            is_repetition=result.is_repetition,
            latency_ms=result.latency_ms,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text.
    
    Args:
        request: Synthesis request with text and optional style/prosody
        
    Returns:
        URL to download the synthesized audio
    """
    try:
        # Synthesize speech
        audio_path = await orchestrator_service.synthesize_speech(
            text=request.text,
            style=request.style,
            pitch=request.pitch,
            rate=request.rate
        )
        
        # Return URL to download the audio
        # The audio file will be served by the /audio/{filename} endpoint
        filename = os.path.basename(audio_path)
        
        return SynthesizeResponse(
            audio_url=f"/api/audio/{filename}"
        )
    except Exception as e:
        logger.error(f"Speech synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Download synthesized audio file.
    
    Args:
        filename: Audio filename
        
    Returns:
        Audio file
    """
    import tempfile
    audio_path = os.path.join(tempfile.gettempdir(), filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=filename
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and clean up resources.
    
    Args:
        session_id: Session ID to delete
    """
    try:
        await orchestrator_service.cleanup_session(session_id)
        return {"status": "success", "message": f"Session {session_id} deleted"}
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Session cleanup failed: {str(e)}")
