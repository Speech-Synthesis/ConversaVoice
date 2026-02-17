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
]
