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
]
