"""
Data models for the Conversation Simulation system.

Defines Pydantic models for scenarios, personas, emotional states,
and session tracking.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from datetime import datetime


class EmotionState(str, Enum):
    """Possible emotional states for customer personas."""

    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    ANXIOUS = "anxious"
    CONFUSED = "confused"
    NEUTRAL = "neutral"
    HOPEFUL = "hopeful"
    SATISFIED = "satisfied"
    DELIGHTED = "delighted"


class PersonalityTrait(str, Enum):
    """Personality traits that affect customer behavior."""

    IMPATIENT = "impatient"
    PATIENT = "patient"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    ANALYTICAL = "analytical"
    EMOTIONAL = "emotional"
    DEMANDING = "demanding"
    EASYGOING = "easygoing"


class DifficultyLevel(str, Enum):
    """Scenario difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class PersonaConfig(BaseModel):
    """Configuration for a customer persona."""

    name: str = Field(default="Customer", description="Customer's name")
    emotion_start: EmotionState = Field(
        default=EmotionState.NEUTRAL,
        description="Initial emotional state"
    )
    personality: PersonalityTrait = Field(
        default=PersonalityTrait.PATIENT,
        description="Primary personality trait"
    )
    goal: str = Field(
        ...,
        description="What the customer wants to achieve"
    )
    patience_level: int = Field(
        default=5,
        ge=1,
        le=10,
        description="How patient the customer is (1-10)"
    )
    escalation_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="How many poor responses before escalation"
    )


class ScenarioConfig(BaseModel):
    """Complete scenario configuration."""

    scenario_id: str = Field(..., description="Unique scenario identifier")
    title: str = Field(..., description="Human-readable scenario title")
    description: str = Field(..., description="Brief scenario description")
    category: str = Field(
        default="general",
        description="Scenario category (e.g., 'complaints', 'billing', 'support')"
    )
    persona: PersonaConfig = Field(..., description="Customer persona configuration")
    background_context: str = Field(
        ...,
        description="Background story and context for the scenario"
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Scenario difficulty level"
    )
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Criteria for successful resolution"
    )
    common_mistakes: List[str] = Field(
        default_factory=list,
        description="Common mistakes trainees make in this scenario"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for filtering scenarios"
    )
    estimated_duration_minutes: int = Field(
        default=5,
        ge=1,
        le=30,
        description="Estimated scenario duration in minutes"
    )


class EmotionTransition(BaseModel):
    """Represents a transition in emotional state."""

    from_state: EmotionState
    to_state: EmotionState
    trigger: str = Field(..., description="What caused the transition")
    timestamp: datetime = Field(default_factory=datetime.now)
    turn_number: int = Field(default=0)


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""

    turn_number: int
    role: str = Field(..., description="'trainee' or 'customer'")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    emotion_state: Optional[EmotionState] = None
    detected_sentiment: Optional[str] = None
    detected_techniques: List[str] = Field(
        default_factory=list,
        description="Communication techniques detected (empathy, acknowledgment, etc.)"
    )


class SimulationSession(BaseModel):
    """Complete simulation session data."""

    session_id: str
    scenario_id: str
    scenario_title: str
    trainee_id: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = Field(default="active", description="'active', 'completed', 'abandoned'")

    # Conversation data
    turns: List[ConversationTurn] = Field(default_factory=list)
    emotion_transitions: List[EmotionTransition] = Field(default_factory=list)

    # Final state
    final_emotion: Optional[EmotionState] = None
    resolution_achieved: bool = False

    # Metadata
    total_turns: int = Field(default=0)
    duration_seconds: Optional[float] = None


class AnalysisResult(BaseModel):
    """Post-session analysis result."""

    session_id: str
    scenario_id: str

    # Scores (1-10)
    empathy_score: int = Field(ge=1, le=10)
    de_escalation_score: int = Field(ge=1, le=10)
    communication_clarity_score: int = Field(ge=1, le=10)
    problem_solving_score: int = Field(ge=1, le=10)
    efficiency_score: int = Field(ge=1, le=10)
    overall_score: int = Field(ge=1, le=10)

    # Outcomes
    de_escalation_success: bool
    resolution_achieved: bool
    customer_satisfaction_predicted: str = Field(
        description="'low', 'medium', 'high'"
    )

    # Detailed feedback
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    specific_feedback: List[str] = Field(default_factory=list)
    recommended_training: List[str] = Field(default_factory=list)

    # Metrics
    turn_count: int
    duration_seconds: float
    emotion_changes: int

    # Raw data
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
