"""
Simulation module for ConversaVoice.

Provides conversation simulation capabilities for training
customer-facing professionals through roleplay scenarios.
"""

from .models import (
    EmotionState,
    PersonalityTrait,
    DifficultyLevel,
    PersonaConfig,
    ScenarioConfig,
    EmotionTransition,
    ConversationTurn,
    SimulationSession,
    AnalysisResult,
)
from .scenarios import (
    ScenarioEngine,
    ScenarioError,
    get_scenario_engine,
)
from .persona import (
    PersonaStateEngine,
    ResponseAnalysis,
    analyze_trainee_response,
    EMOTION_PROSODY_MAP,
)
from .controller import (
    SimulationController,
    SimulationResponse,
    SimulationError,
    get_simulation_controller,
)
from .session_tracker import (
    SessionTracker,
    SessionTrackerError,
    get_session_tracker,
)
from .analysis import (
    AnalysisEngine,
    AnalysisError,
    get_analysis_engine,
)
from .conversation_flow import (
    ConversationFlowManager,
    CompletionStatus,
    EndingType,
    get_flow_manager,
    remove_flow_manager,
)
from .voice_analyzer import (
    VoiceAnalyzer,
    VoiceAnalysis,
    VoiceEmotion,
    AcousticFeatures,
    DeliveryScores,
    get_voice_analyzer,
)

__all__ = [
    # Models
    "EmotionState",
    "PersonalityTrait",
    "DifficultyLevel",
    "PersonaConfig",
    "ScenarioConfig",
    "EmotionTransition",
    "ConversationTurn",
    "SimulationSession",
    "AnalysisResult",
    # Scenario Engine
    "ScenarioEngine",
    "ScenarioError",
    "get_scenario_engine",
    # Persona State Engine
    "PersonaStateEngine",
    "ResponseAnalysis",
    "analyze_trainee_response",
    "EMOTION_PROSODY_MAP",
    # Simulation Controller
    "SimulationController",
    "SimulationResponse",
    "SimulationError",
    "get_simulation_controller",
    # Session Tracker
    "SessionTracker",
    "SessionTrackerError",
    "get_session_tracker",
    # Analysis Engine
    "AnalysisEngine",
    "AnalysisError",
    "get_analysis_engine",
    # Conversation Flow
    "ConversationFlowManager",
    "CompletionStatus",
    "EndingType",
    "get_flow_manager",
    "remove_flow_manager",
    # Voice Analyzer
    "VoiceAnalyzer",
    "VoiceAnalysis",
    "VoiceEmotion",
    "AcousticFeatures",
    "DeliveryScores",
    "get_voice_analyzer",
]
