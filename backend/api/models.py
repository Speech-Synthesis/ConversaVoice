"""Pydantic models for API request/response validation."""

from typing import Optional
from pydantic import BaseModel, Field


class TranscribeRequest(BaseModel):
    """Request model for audio transcription."""
    session_id: Optional[str] = Field(None, description="Session ID for context")


class TranscribeResponse(BaseModel):
    """Response model for audio transcription."""
    text: str = Field(..., description="Transcribed text from audio")
    session_id: str = Field(..., description="Session ID")


class ChatRequest(BaseModel):
    """Request model for chat/LLM interaction."""
    text: str = Field(..., description="User input text")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")


class ChatResponse(BaseModel):
    """Response model for chat/LLM interaction."""
    response: str = Field(..., description="Assistant's response text")
    style: Optional[str] = Field(None, description="Emotional style label")
    pitch: Optional[str] = Field(None, description="Pitch adjustment")
    rate: Optional[str] = Field(None, description="Speech rate adjustment")
    is_repetition: bool = Field(False, description="Whether response is a repetition")
    latency_ms: float = Field(0.0, description="Processing latency in milliseconds")
    session_id: str = Field(..., description="Session ID")


class SynthesizeRequest(BaseModel):
    """Request model for text-to-speech synthesis."""
    text: str = Field(..., description="Text to synthesize")
    style: Optional[str] = Field(None, description="Emotional style label")
    pitch: Optional[str] = Field(None, description="Pitch adjustment")
    rate: Optional[str] = Field(None, description="Speech rate adjustment")


class SynthesizeResponse(BaseModel):
    """Response model for text-to-speech synthesis."""
    audio_url: str = Field(..., description="URL to download the synthesized audio")


class SessionResponse(BaseModel):
    """Response model for session information."""
    session_id: str = Field(..., description="Session ID")
    created_at: Optional[str] = Field(None, description="Session creation timestamp")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    services: dict = Field(..., description="Status of individual services")
