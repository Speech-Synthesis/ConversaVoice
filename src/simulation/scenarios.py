"""
Scenario Engine for the Conversation Simulation system.

Handles loading, validating, and managing simulation scenarios
from JSON configuration files.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from .models import ScenarioConfig, DifficultyLevel

logger = logging.getLogger(__name__)


class ScenarioError(Exception):
    """Exception raised for scenario-related errors."""

    def __init__(self, message: str, scenario_id: Optional[str] = None):
        self.scenario_id = scenario_id
        super().__init__(message)


class ScenarioEngine:
    """
    Engine for loading and managing simulation scenarios.

    Scenarios are loaded from JSON files in a configurable directory.
    Supports filtering, validation, and dynamic reloading.
    """

    DEFAULT_SCENARIOS_DIR = "scenarios"

    def __init__(self, scenarios_dir: Optional[str] = None):
        """
        Initialize the scenario engine.

        Args:
            scenarios_dir: Path to scenarios directory.
                          Defaults to 'scenarios' in project root.
        """
        if scenarios_dir:
            self.scenarios_dir = Path(scenarios_dir)
        else:
            # Default to project root / scenarios
            project_root = Path(__file__).parent.parent.parent
            self.scenarios_dir = project_root / self.DEFAULT_SCENARIOS_DIR

        self._scenarios: Dict[str, ScenarioConfig] = {}
        self._loaded = False

    def load_scenarios(self, force_reload: bool = False) -> int:
        """
        Load all scenarios from the scenarios directory.

        Args:
            force_reload: If True, reload even if already loaded.

        Returns:
            Number of scenarios loaded.

        Raises:
            ScenarioError: If scenarios directory doesn't exist.
        """
        if self._loaded and not force_reload:
            return len(self._scenarios)

        if not self.scenarios_dir.exists():
            logger.warning(f"Scenarios directory not found: {self.scenarios_dir}")
            # Create the directory
            self.scenarios_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created scenarios directory: {self.scenarios_dir}")
            return 0

        self._scenarios.clear()
        loaded_count = 0

        for filepath in self.scenarios_dir.glob("*.json"):
            try:
                scenario = self._load_scenario_file(filepath)
                self._scenarios[scenario.scenario_id] = scenario
                loaded_count += 1
                logger.info(f"Loaded scenario: {scenario.scenario_id}")
            except Exception as e:
                logger.error(f"Failed to load scenario from {filepath}: {e}")

        self._loaded = True
        logger.info(f"Loaded {loaded_count} scenarios from {self.scenarios_dir}")
        return loaded_count

    def _load_scenario_file(self, filepath: Path) -> ScenarioConfig:
        """
        Load and validate a single scenario file.

        Args:
            filepath: Path to the JSON scenario file.

        Returns:
            Validated ScenarioConfig object.

        Raises:
            ScenarioError: If file is invalid or fails validation.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate using Pydantic
            scenario = ScenarioConfig(**data)
            return scenario

        except json.JSONDecodeError as e:
            raise ScenarioError(f"Invalid JSON in {filepath}: {e}")
        except Exception as e:
            raise ScenarioError(f"Failed to parse scenario {filepath}: {e}")

    def get_scenario(self, scenario_id: str) -> ScenarioConfig:
        """
        Get a scenario by ID.

        Args:
            scenario_id: Unique scenario identifier.

        Returns:
            ScenarioConfig for the requested scenario.

        Raises:
            ScenarioError: If scenario not found.
        """
        self.load_scenarios()

        if scenario_id not in self._scenarios:
            raise ScenarioError(
                f"Scenario not found: {scenario_id}",
                scenario_id=scenario_id
            )

        return self._scenarios[scenario_id]

    def list_scenarios(
        self,
        category: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None,
        tags: Optional[List[str]] = None
    ) -> List[ScenarioConfig]:
        """
        List all scenarios, optionally filtered.

        Args:
            category: Filter by category.
            difficulty: Filter by difficulty level.
            tags: Filter by tags (scenarios must have ALL specified tags).

        Returns:
            List of matching scenarios.
        """
        self.load_scenarios()

        scenarios = list(self._scenarios.values())

        if category:
            scenarios = [s for s in scenarios if s.category == category]

        if difficulty:
            scenarios = [s for s in scenarios if s.difficulty == difficulty]

        if tags:
            scenarios = [
                s for s in scenarios
                if all(tag in s.tags for tag in tags)
            ]

        return scenarios

    def get_categories(self) -> List[str]:
        """Get list of all unique categories."""
        self.load_scenarios()
        categories = set(s.category for s in self._scenarios.values())
        return sorted(categories)

    def get_scenario_summary(self, scenario_id: str) -> Dict[str, Any]:
        """
        Get a summary of a scenario suitable for UI display.

        Args:
            scenario_id: Scenario identifier.

        Returns:
            Dictionary with scenario summary.
        """
        scenario = self.get_scenario(scenario_id)

        return {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "description": scenario.description,
            "category": scenario.category,
            "difficulty": scenario.difficulty.value,
            "persona_emotion": scenario.persona.emotion_start.value,
            "persona_personality": scenario.persona.personality.value,
            "estimated_duration": scenario.estimated_duration_minutes,
            "tags": scenario.tags
        }

    def list_scenario_summaries(
        self,
        category: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None
    ) -> List[Dict[str, Any]]:
        """
        Get summaries of all scenarios for UI listing.

        Args:
            category: Optional category filter.
            difficulty: Optional difficulty filter.

        Returns:
            List of scenario summary dictionaries.
        """
        scenarios = self.list_scenarios(category=category, difficulty=difficulty)
        return [self.get_scenario_summary(s.scenario_id) for s in scenarios]

    def validate_scenario(self, scenario_data: Dict[str, Any]) -> ScenarioConfig:
        """
        Validate scenario data without saving.

        Args:
            scenario_data: Dictionary with scenario configuration.

        Returns:
            Validated ScenarioConfig.

        Raises:
            ScenarioError: If validation fails.
        """
        try:
            return ScenarioConfig(**scenario_data)
        except Exception as e:
            raise ScenarioError(f"Scenario validation failed: {e}")

    def add_scenario(self, scenario: ScenarioConfig, save: bool = True) -> None:
        """
        Add a new scenario.

        Args:
            scenario: ScenarioConfig to add.
            save: If True, save to disk.

        Raises:
            ScenarioError: If scenario ID already exists.
        """
        self.load_scenarios()

        if scenario.scenario_id in self._scenarios:
            raise ScenarioError(
                f"Scenario already exists: {scenario.scenario_id}",
                scenario_id=scenario.scenario_id
            )

        self._scenarios[scenario.scenario_id] = scenario

        if save:
            self._save_scenario(scenario)

    def _save_scenario(self, scenario: ScenarioConfig) -> None:
        """Save scenario to JSON file."""
        filepath = self.scenarios_dir / f"{scenario.scenario_id}.json"

        # Ensure directory exists
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(scenario.model_dump(), f, indent=2, default=str)

        logger.info(f"Saved scenario to: {filepath}")

    @property
    def scenario_count(self) -> int:
        """Get number of loaded scenarios."""
        self.load_scenarios()
        return len(self._scenarios)

    def reload(self) -> int:
        """Force reload all scenarios."""
        return self.load_scenarios(force_reload=True)


# Global scenario engine instance
_scenario_engine: Optional[ScenarioEngine] = None


def get_scenario_engine() -> ScenarioEngine:
    """Get the global scenario engine instance."""
    global _scenario_engine
    if _scenario_engine is None:
        _scenario_engine = ScenarioEngine()
    return _scenario_engine
