"""FastAPI routes for Conversation Simulation system."""

import logging
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

from src.simulation import (
    ScenarioEngine,
    ScenarioError,
    get_scenario_engine,
    SimulationController,
    SimulationError,
    get_simulation_controller,
    SessionTracker,
    get_session_tracker,
    AnalysisEngine,
    AnalysisError,
    get_analysis_engine,
    DifficultyLevel,
    EmotionState,
    get_voice_analyzer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/simulation", tags=["simulation"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ScenarioSummary(BaseModel):
    """Summary of a scenario for listing."""
    scenario_id: str
    title: str
    description: str
    category: str
    difficulty: str
    persona_emotion: str
    persona_personality: str
    estimated_duration: int
    tags: List[str]


class StartSimulationRequest(BaseModel):
    """Request to start a simulation."""
    scenario_id: str = Field(..., description="ID of the scenario to run")
    trainee_id: Optional[str] = Field(None, description="Optional trainee identifier")


class StartSimulationResponse(BaseModel):
    """Response after starting a simulation."""
    session_id: str
    scenario_id: str
    scenario_title: str
    customer_name: str
    initial_emotion: str
    opening_message: str
    prosody: dict


class TraineeInputRequest(BaseModel):
    """Request with trainee's response."""
    session_id: str = Field(..., description="Current simulation session ID")
    message: str = Field(..., description="Trainee's response to the customer")


class SimulationTurnResponse(BaseModel):
    """Response from a simulation turn."""
    customer_message: str
    emotion_state: str
    emotion_changed: bool
    previous_emotion: Optional[str]
    prosody: dict
    turn_number: int
    detected_techniques: List[str]
    detected_issues: List[str]
    # Conversation completion tracking
    conversation_complete: bool = False
    approaching_end: bool = False
    ending_type: Optional[str] = None
    goodbye_message: Optional[str] = None


class EndSimulationRequest(BaseModel):
    """Request to end a simulation."""
    session_id: str
    resolution_achieved: bool = Field(False, description="Whether issue was resolved")


class SessionSummary(BaseModel):
    """Summary of a simulation session."""
    session_id: str
    scenario_id: str
    scenario_title: str
    trainee_id: Optional[str]
    status: str
    total_turns: int
    duration_seconds: Optional[float]
    final_emotion: Optional[str]
    resolution_achieved: bool
    emotion_changes: int


class AnalysisResponse(BaseModel):
    """Analysis result response."""
    session_id: str
    scenario_id: str
    overall_score: int
    empathy_score: int
    de_escalation_score: int
    communication_clarity_score: int
    problem_solving_score: int
    efficiency_score: int
    de_escalation_success: bool
    resolution_achieved: bool
    customer_satisfaction_predicted: str
    strengths: List[str]
    areas_for_improvement: List[str]
    specific_feedback: List[str]
    recommended_training: List[str]
    turn_count: int
    duration_seconds: float
    emotion_changes: int


class QuickScoreResponse(BaseModel):
    """Quick score without full LLM analysis."""
    empathy_score: int
    de_escalation_score: int
    efficiency_score: int
    overall_score: int
    emotion_improvement: int
    techniques_used: List[str]
    resolution_achieved: bool


class VoiceAnalysisResponse(BaseModel):
    """Voice analysis result."""
    analysis_success: bool
    primary_emotion: str
    secondary_emotion: Optional[str] = None
    emotion_confidence: float
    delivery_scores: Dict[str, int]
    acoustic_features: Dict[str, float]
    error_message: Optional[str] = None


# ============================================================================
# Active Simulation State (per-session)
# ============================================================================

# Store active controllers by session_id
_active_simulations: dict[str, SimulationController] = {}


def get_active_controller(session_id: str) -> SimulationController:
    """Get active simulation controller for session."""
    if session_id not in _active_simulations:
        raise HTTPException(
            status_code=404,
            detail=f"No active simulation found for session: {session_id}"
        )
    return _active_simulations[session_id]


# ============================================================================
# Scenario Endpoints
# ============================================================================

@router.get("/scenarios", response_model=List[ScenarioSummary])
async def list_scenarios(
    category: Optional[str] = Query(None, description="Filter by category"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty (easy, medium, hard, expert)"),
):
    """
    List available simulation scenarios.

    Optionally filter by category or difficulty level.
    """
    try:
        engine = get_scenario_engine()

        # Convert difficulty string to enum if provided
        difficulty_enum = None
        if difficulty:
            try:
                difficulty_enum = DifficultyLevel(difficulty.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid difficulty: {difficulty}. Must be easy, medium, hard, or expert."
                )

        summaries = engine.list_scenario_summaries(
            category=category,
            difficulty=difficulty_enum
        )

        return [ScenarioSummary(**s) for s in summaries]

    except ScenarioError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scenarios/{scenario_id}")
async def get_scenario(scenario_id: str):
    """Get detailed information about a specific scenario."""
    try:
        engine = get_scenario_engine()
        scenario = engine.get_scenario(scenario_id)

        return {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "description": scenario.description,
            "category": scenario.category,
            "difficulty": scenario.difficulty.value,
            "persona": {
                "name": scenario.persona.name,
                "emotion_start": scenario.persona.emotion_start.value,
                "personality": scenario.persona.personality.value,
                "goal": scenario.persona.goal,
                "patience_level": scenario.persona.patience_level,
            },
            "background_context": scenario.background_context,
            "success_criteria": scenario.success_criteria,
            "common_mistakes": scenario.common_mistakes,
            "estimated_duration_minutes": scenario.estimated_duration_minutes,
            "tags": scenario.tags,
        }

    except ScenarioError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/categories")
async def list_categories():
    """Get list of all scenario categories."""
    engine = get_scenario_engine()
    return {"categories": engine.get_categories()}


# ============================================================================
# Simulation Control Endpoints
# ============================================================================

@router.post("/start", response_model=StartSimulationResponse)
async def start_simulation(request: StartSimulationRequest):
    """
    Start a new simulation session.

    Returns the session ID and customer's opening message.
    """
    try:
        # Create new controller for this simulation
        controller = SimulationController()

        # Start the simulation
        session = controller.start_simulation(
            scenario_id=request.scenario_id,
            trainee_id=request.trainee_id,
        )

        # Get opening message
        opening = controller.get_opening_message()

        # Store controller for this session
        _active_simulations[session.session_id] = controller

        # Get scenario for customer name
        scenario = controller.current_scenario

        logger.info(f"Started simulation: {session.session_id} for scenario: {request.scenario_id}")

        return StartSimulationResponse(
            session_id=session.session_id,
            scenario_id=session.scenario_id,
            scenario_title=session.scenario_title,
            customer_name=scenario.persona.name,
            initial_emotion=opening.emotion_state.value,
            opening_message=opening.customer_message,
            prosody=opening.prosody,
        )

    except ScenarioError as e:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {e}")
    except SimulationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/respond", response_model=SimulationTurnResponse)
async def process_trainee_response(request: TraineeInputRequest):
    """
    Process trainee's response and get customer's reply.

    Returns the customer's response with updated emotional state.
    """
    try:
        controller = get_active_controller(request.session_id)

        # Process the trainee's input
        response = await controller.process_trainee_input_async(request.message)

        # Extract completion status
        completion_status = response.completion_status
        approaching_end = completion_status.approaching_end if completion_status else False
        ending_type = completion_status.ending_type.value if completion_status and completion_status.ending_type else None

        return SimulationTurnResponse(
            customer_message=response.customer_message,
            emotion_state=response.emotion_state.value,
            emotion_changed=response.emotion_changed,
            previous_emotion=response.previous_emotion.value if response.previous_emotion else None,
            prosody=response.prosody,
            turn_number=response.turn_number,
            detected_techniques=response.trainee_analysis.detected_techniques,
            detected_issues=response.trainee_analysis.detected_issues,
            conversation_complete=response.conversation_complete,
            approaching_end=approaching_end,
            ending_type=ending_type if ending_type != "none" else None,
            goodbye_message=response.goodbye_message,
        )

    except HTTPException:
        raise
    except SimulationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/end", response_model=SessionSummary)
async def end_simulation(request: EndSimulationRequest):
    """
    End the current simulation session.

    Returns session summary and triggers analysis.
    """
    try:
        controller = get_active_controller(request.session_id)

        # End the simulation
        session = controller.end_simulation(
            resolution_achieved=request.resolution_achieved,
            reason="completed"
        )

        # Save session to tracker (with error handling)
        try:
            tracker = get_session_tracker()
            tracker.save_session(session)
        except Exception as save_error:
            logger.warning(f"Failed to save session to tracker: {save_error}")
            # Continue anyway - session data is still in memory

        # Remove from active simulations
        if request.session_id in _active_simulations:
            del _active_simulations[request.session_id]

        logger.info(f"Ended simulation: {request.session_id}")

        return SessionSummary(
            session_id=session.session_id,
            scenario_id=session.scenario_id,
            scenario_title=session.scenario_title,
            trainee_id=session.trainee_id,
            status=session.status,
            total_turns=session.total_turns,
            duration_seconds=session.duration_seconds,
            final_emotion=session.final_emotion.value if session.final_emotion else None,
            resolution_achieved=session.resolution_achieved,
            emotion_changes=len(session.emotion_transitions),
        )

    except HTTPException:
        raise
    except SimulationError as e:
        logger.error(f"Simulation error ending session: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to end simulation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/status/{session_id}")
async def get_simulation_status(session_id: str):
    """Get current status of an active simulation."""
    try:
        controller = get_active_controller(session_id)
        return controller.get_session_summary()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analysis Endpoints
# ============================================================================

@router.get("/analysis/{session_id}", response_model=AnalysisResponse)
async def get_session_analysis(session_id: str):
    """
    Get full LLM-powered analysis of a completed session.

    This may take a few seconds as it uses the LLM for evaluation.
    """
    try:
        # Get session from tracker
        tracker = get_session_tracker()
        session = tracker.get_session(session_id)

        if not session:
            # Session not in tracker - maybe it was just completed
            # Try to return a basic response
            logger.warning(f"Session {session_id} not found in tracker")
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}. It may have expired or not been saved."
            )

        if session.status == "active":
            raise HTTPException(status_code=400, detail="Cannot analyze active session. End it first.")

        # Run analysis
        engine = get_analysis_engine()
        result = engine.analyze_session(session)

        return AnalysisResponse(
            session_id=result.session_id,
            scenario_id=result.scenario_id,
            overall_score=result.overall_score,
            empathy_score=result.empathy_score,
            de_escalation_score=result.de_escalation_score,
            communication_clarity_score=result.communication_clarity_score,
            problem_solving_score=result.problem_solving_score,
            efficiency_score=result.efficiency_score,
            de_escalation_success=result.de_escalation_success,
            resolution_achieved=result.resolution_achieved,
            customer_satisfaction_predicted=result.customer_satisfaction_predicted,
            strengths=result.strengths,
            areas_for_improvement=result.areas_for_improvement,
            specific_feedback=result.specific_feedback,
            recommended_training=result.recommended_training,
            turn_count=result.turn_count,
            duration_seconds=result.duration_seconds,
            emotion_changes=result.emotion_changes,
        )

    except HTTPException:
        raise
    except AnalysisError as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/analysis/{session_id}/quick", response_model=QuickScoreResponse)
async def get_quick_score(session_id: str):
    """
    Get quick score without full LLM analysis.

    Faster but less detailed than full analysis.
    """
    try:
        tracker = get_session_tracker()
        session = tracker.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        engine = get_analysis_engine()
        score = engine.get_quick_score(session)

        return QuickScoreResponse(**score)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis/{session_id}/report")
async def get_analysis_report(session_id: str, format: str = Query("text", regex="^(text|markdown|json)$")):
    """
    Get formatted analysis report.

    Available formats: text, markdown, json
    """
    try:
        tracker = get_session_tracker()
        session = tracker.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        engine = get_analysis_engine()
        result = engine.analyze_session(session)

        if format == "json":
            return result.model_dump()
        else:
            report = engine.generate_summary_report(result)
            return {"format": format, "report": report}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-voice", response_model=VoiceAnalysisResponse)
async def analyze_voice(audio: UploadFile = File(...)):
    """
    Analyze trainee voice for emotion and delivery quality.

    Accepts audio file (WAV format preferred) and returns:
    - Detected voice emotion (calm, stressed, confident, etc.)
    - Delivery scores (calmness, confidence, empathy, pace, clarity)
    - Acoustic features (pitch, energy, speaking rate, etc.)
    """
    try:
        # Read audio data
        audio_data = await audio.read()

        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Analyze voice
        analyzer = get_voice_analyzer()
        result = analyzer.analyze(audio_data)

        # Build response
        return VoiceAnalysisResponse(
            analysis_success=result.analysis_success,
            primary_emotion=result.primary_emotion.value,
            secondary_emotion=result.secondary_emotion.value if result.secondary_emotion else None,
            emotion_confidence=result.emotion_confidence,
            delivery_scores={
                "calmness": result.delivery_scores.calmness,
                "confidence": result.delivery_scores.confidence,
                "empathy": result.delivery_scores.empathy,
                "pace": result.delivery_scores.pace,
                "clarity": result.delivery_scores.clarity,
                "overall": result.delivery_scores.overall,
            },
            acoustic_features={
                "pitch_mean": result.features.pitch_mean,
                "pitch_std": result.features.pitch_std,
                "energy_mean": result.features.energy_mean,
                "speaking_rate": result.features.speaking_rate,
                "pause_ratio": result.features.pause_ratio,
                "duration_seconds": result.features.duration_seconds,
            },
            error_message=result.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice analysis failed: {str(e)}")


# ============================================================================
# Session History Endpoints
# ============================================================================

@router.get("/sessions", response_model=List[SessionSummary])
async def list_sessions(
    trainee_id: Optional[str] = Query(None, description="Filter by trainee ID"),
    scenario_id: Optional[str] = Query(None, description="Filter by scenario ID"),
    limit: int = Query(20, ge=1, le=100, description="Max sessions to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """List completed simulation sessions."""
    try:
        tracker = get_session_tracker()
        sessions = tracker.list_sessions(
            trainee_id=trainee_id,
            scenario_id=scenario_id,
            limit=limit,
            offset=offset,
        )

        return [SessionSummary(**s) for s in sessions]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/transcript")
async def get_session_transcript(
    session_id: str,
    format: str = Query("text", regex="^(text|markdown|json)$")
):
    """Get the full transcript of a session."""
    try:
        tracker = get_session_tracker()
        transcript = tracker.export_session_transcript(session_id, format=format)

        return {"format": format, "transcript": transcript}

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/trainee/{trainee_id}/stats")
async def get_trainee_stats(trainee_id: str):
    """Get aggregate statistics for a trainee."""
    try:
        tracker = get_session_tracker()
        stats = tracker.get_trainee_stats(trainee_id)
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
