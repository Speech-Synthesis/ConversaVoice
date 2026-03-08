"""
Persona State Engine for the Conversation Simulation system.

Manages the emotional state machine for customer personas,
handling transitions based on trainee responses and mapping
emotions to TTS prosody parameters.
"""

import logging
import re
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .models import (
    EmotionState,
    PersonalityTrait,
    PersonaConfig,
    EmotionTransition,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Trainee Response Analysis
# ============================================================================

# Empathy indicators - phrases that show understanding and care
EMPATHY_INDICATORS = [
    "i understand",
    "i can see",
    "that must be",
    "i hear you",
    "i appreciate",
    "thank you for sharing",
    "i'm sorry to hear",
    "i'm sorry you're",
    "that sounds",
    "i can imagine",
    "completely understand",
    "totally understand",
    "i get it",
    "that's frustrating",
    "that's disappointing",
    "i would feel the same",
    "in your situation",
    "from your perspective",
]

# Acknowledgment indicators - recognizing the customer's issue
ACKNOWLEDGMENT_INDICATORS = [
    "you're right",
    "that's correct",
    "absolutely",
    "definitely",
    "of course",
    "i see what you mean",
    "valid point",
    "valid concern",
    "legitimate concern",
    "i acknowledge",
    "you have every right",
    "that makes sense",
]

# Ownership indicators - taking responsibility
OWNERSHIP_INDICATORS = [
    "let me help",
    "i will",
    "i'll take care",
    "i'll handle",
    "i'll make sure",
    "my responsibility",
    "i own this",
    "leave it to me",
    "i've got this",
    "let me fix",
    "let me resolve",
    "i'm going to",
    "i can do",
]

# Solution-oriented indicators
SOLUTION_INDICATORS = [
    "here's what i can do",
    "my suggestion",
    "i recommend",
    "option",
    "alternative",
    "solution",
    "resolve this",
    "fix this",
    "next step",
    "action plan",
    "let me offer",
    "i can provide",
    "would it help if",
    "how about",
]

# Dismissive indicators - phrases that invalidate or ignore
DISMISSIVE_INDICATORS = [
    "that's not possible",
    "nothing i can do",
    "not my fault",
    "not our fault",
    "policy says",
    "policy doesn't allow",
    "you should have",
    "you need to understand",
    "calm down",
    "relax",
    "it's not that bad",
    "there's no way",
    "can't help",
    "won't be able",
    "that's just how",
    "nothing we can do",
    "you're wrong",
    "that's incorrect",
    "actually,",  # often used condescendingly
]

# Defensive indicators
DEFENSIVE_INDICATORS = [
    "but we",
    "however, we",
    "technically",
    "to be fair",
    "in our defense",
    "it's not like",
    "we always",
    "we never",
    "that's not true",
    "i didn't say",
    "that's not what",
]


@dataclass
class ResponseAnalysis:
    """Analysis of a trainee's response."""

    empathy_score: float = 0.0  # 0-1
    acknowledgment_score: float = 0.0  # 0-1
    ownership_score: float = 0.0  # 0-1
    solution_score: float = 0.0  # 0-1
    dismissive_score: float = 0.0  # 0-1
    defensive_score: float = 0.0  # 0-1

    detected_techniques: List[str] = field(default_factory=list)
    detected_issues: List[str] = field(default_factory=list)

    @property
    def positive_score(self) -> float:
        """Combined positive impact score."""
        return (
            self.empathy_score * 0.35 +
            self.acknowledgment_score * 0.20 +
            self.ownership_score * 0.25 +
            self.solution_score * 0.20
        )

    @property
    def negative_score(self) -> float:
        """Combined negative impact score."""
        return (
            self.dismissive_score * 0.6 +
            self.defensive_score * 0.4
        )

    @property
    def net_impact(self) -> float:
        """Net emotional impact (-1 to +1)."""
        return min(1.0, max(-1.0, self.positive_score - self.negative_score))


def analyze_trainee_response(text: str) -> ResponseAnalysis:
    """
    Analyze a trainee's response for communication techniques.

    Args:
        text: The trainee's response text.

    Returns:
        ResponseAnalysis with scores and detected techniques.
    """
    text_lower = text.lower()
    analysis = ResponseAnalysis()

    # Check empathy indicators
    empathy_matches = [ind for ind in EMPATHY_INDICATORS if ind in text_lower]
    if empathy_matches:
        analysis.empathy_score = min(1.0, len(empathy_matches) * 0.4)
        analysis.detected_techniques.append("empathy")

    # Check acknowledgment indicators
    ack_matches = [ind for ind in ACKNOWLEDGMENT_INDICATORS if ind in text_lower]
    if ack_matches:
        analysis.acknowledgment_score = min(1.0, len(ack_matches) * 0.5)
        analysis.detected_techniques.append("acknowledgment")

    # Check ownership indicators
    own_matches = [ind for ind in OWNERSHIP_INDICATORS if ind in text_lower]
    if own_matches:
        analysis.ownership_score = min(1.0, len(own_matches) * 0.4)
        analysis.detected_techniques.append("ownership")

    # Check solution indicators
    sol_matches = [ind for ind in SOLUTION_INDICATORS if ind in text_lower]
    if sol_matches:
        analysis.solution_score = min(1.0, len(sol_matches) * 0.4)
        analysis.detected_techniques.append("solution-oriented")

    # Check dismissive indicators
    dismiss_matches = [ind for ind in DISMISSIVE_INDICATORS if ind in text_lower]
    if dismiss_matches:
        analysis.dismissive_score = min(1.0, len(dismiss_matches) * 0.5)
        analysis.detected_issues.append("dismissive language")

    # Check defensive indicators
    def_matches = [ind for ind in DEFENSIVE_INDICATORS if ind in text_lower]
    if def_matches:
        analysis.defensive_score = min(1.0, len(def_matches) * 0.4)
        analysis.detected_issues.append("defensive language")

    return analysis


# ============================================================================
# Emotion State Machine
# ============================================================================

# Emotion ordering from most negative to most positive
EMOTION_ORDER = [
    EmotionState.ANGRY,
    EmotionState.FRUSTRATED,
    EmotionState.ANXIOUS,
    EmotionState.CONFUSED,
    EmotionState.NEUTRAL,
    EmotionState.HOPEFUL,
    EmotionState.SATISFIED,
    EmotionState.DELIGHTED,
]

# Map emotions to their index for comparison
EMOTION_INDEX = {state: idx for idx, state in enumerate(EMOTION_ORDER)}


# Prosody mapping for customer emotions (for TTS)
EMOTION_PROSODY_MAP: Dict[EmotionState, Dict[str, str]] = {
    EmotionState.ANGRY: {
        "style": "angry",
        "pitch": "+10%",
        "rate": "1.15",
        "volume": "loud",
    },
    EmotionState.FRUSTRATED: {
        "style": "disgruntled",
        "pitch": "+5%",
        "rate": "1.1",
        "volume": "medium",
    },
    EmotionState.ANXIOUS: {
        "style": "fearful",
        "pitch": "+5%",
        "rate": "1.05",
        "volume": "medium",
    },
    EmotionState.CONFUSED: {
        "style": "empathetic",  # Sounds uncertain
        "pitch": "0%",
        "rate": "0.95",
        "volume": "medium",
    },
    EmotionState.NEUTRAL: {
        "style": "neutral",
        "pitch": "0%",
        "rate": "1.0",
        "volume": "medium",
    },
    EmotionState.HOPEFUL: {
        "style": "hopeful",
        "pitch": "+3%",
        "rate": "1.0",
        "volume": "medium",
    },
    EmotionState.SATISFIED: {
        "style": "friendly",
        "pitch": "+5%",
        "rate": "1.0",
        "volume": "medium",
    },
    EmotionState.DELIGHTED: {
        "style": "cheerful",
        "pitch": "+8%",
        "rate": "1.05",
        "volume": "medium",
    },
}


class PersonaStateEngine:
    """
    Manages the emotional state machine for a customer persona.

    Tracks emotional state, handles transitions based on trainee responses,
    and provides prosody parameters for TTS.
    """

    def __init__(
        self,
        persona_config: PersonaConfig,
        on_transition: Optional[Callable[[EmotionTransition], None]] = None,
    ):
        """
        Initialize the persona state engine.

        Args:
            persona_config: Configuration for the customer persona.
            on_transition: Optional callback when emotion state changes.
        """
        self.config = persona_config
        self.on_transition = on_transition

        # Current state
        self._current_emotion = persona_config.emotion_start
        self._turn_count = 0
        self._poor_response_count = 0  # For escalation tracking
        self._transitions: List[EmotionTransition] = []

        # Personality modifiers
        self._patience_modifier = (persona_config.patience_level - 5) / 10  # -0.4 to +0.5
        self._escalation_threshold = persona_config.escalation_threshold

    @property
    def current_emotion(self) -> EmotionState:
        """Get current emotional state."""
        return self._current_emotion

    @property
    def turn_count(self) -> int:
        """Get current turn count."""
        return self._turn_count

    @property
    def transitions(self) -> List[EmotionTransition]:
        """Get list of all emotion transitions."""
        return self._transitions.copy()

    @property
    def prosody(self) -> Dict[str, str]:
        """Get current prosody parameters for TTS."""
        return EMOTION_PROSODY_MAP.get(
            self._current_emotion,
            EMOTION_PROSODY_MAP[EmotionState.NEUTRAL]
        )

    def get_emotion_index(self, emotion: EmotionState) -> int:
        """Get the position of an emotion in the scale (0=most negative)."""
        return EMOTION_INDEX.get(emotion, 4)  # Default to neutral

    def process_trainee_response(
        self,
        response_text: str,
        external_sentiment: Optional[str] = None
    ) -> Tuple[EmotionState, ResponseAnalysis]:
        """
        Process a trainee response and update emotional state.

        Args:
            response_text: The trainee's response.
            external_sentiment: Optional sentiment from external analyzer.

        Returns:
            Tuple of (new_emotion_state, response_analysis)
        """
        self._turn_count += 1

        # Analyze the response
        analysis = analyze_trainee_response(response_text)

        # Calculate state change
        old_emotion = self._current_emotion
        new_emotion = self._calculate_next_emotion(analysis)

        # Record transition if changed
        if new_emotion != old_emotion:
            transition = EmotionTransition(
                from_state=old_emotion,
                to_state=new_emotion,
                trigger=self._get_transition_trigger(analysis),
                timestamp=datetime.now(),
                turn_number=self._turn_count
            )
            self._transitions.append(transition)

            if self.on_transition:
                self.on_transition(transition)

            logger.info(
                f"Emotion transition: {old_emotion.value} → {new_emotion.value} "
                f"(trigger: {transition.trigger})"
            )

        self._current_emotion = new_emotion

        # Track poor responses for escalation
        if analysis.net_impact < -0.2:
            self._poor_response_count += 1
        elif analysis.net_impact > 0.2:
            self._poor_response_count = max(0, self._poor_response_count - 1)

        return new_emotion, analysis

    def _calculate_next_emotion(self, analysis: ResponseAnalysis) -> EmotionState:
        """
        Calculate the next emotional state based on response analysis.

        Uses the net impact score and personality modifiers to determine
        how much the emotion should shift.
        """
        current_index = self.get_emotion_index(self._current_emotion)
        net_impact = analysis.net_impact

        # Apply personality modifier
        # Patient customers are harder to anger, impatient ones are easier
        adjusted_impact = net_impact + (self._patience_modifier * 0.2)

        # Calculate index shift
        # Positive impact moves toward satisfaction, negative toward anger
        if adjusted_impact > 0.3:
            # Significant positive - move 1-2 steps toward positive
            shift = 1 if adjusted_impact < 0.6 else 2
        elif adjusted_impact < -0.3:
            # Significant negative - move 1-2 steps toward negative
            shift = -1 if adjusted_impact > -0.6 else -2
        else:
            # Minor impact - might shift 1 or stay same
            if adjusted_impact > 0.15:
                shift = 1
            elif adjusted_impact < -0.15:
                shift = -1
            else:
                shift = 0

        # Check for escalation threshold
        if self._poor_response_count >= self._escalation_threshold:
            # Force escalation if too many poor responses
            shift = min(shift, -1)

        # Apply shift with bounds
        new_index = max(0, min(len(EMOTION_ORDER) - 1, current_index + shift))

        return EMOTION_ORDER[new_index]

    def _get_transition_trigger(self, analysis: ResponseAnalysis) -> str:
        """Generate a description of what triggered the transition."""
        triggers = []

        if analysis.detected_techniques:
            triggers.append(f"positive: {', '.join(analysis.detected_techniques)}")
        if analysis.detected_issues:
            triggers.append(f"issues: {', '.join(analysis.detected_issues)}")

        if not triggers:
            if analysis.net_impact > 0:
                triggers.append("generally positive response")
            elif analysis.net_impact < 0:
                triggers.append("generally negative response")
            else:
                triggers.append("neutral response")

        return "; ".join(triggers)

    def force_emotion(self, emotion: EmotionState, reason: str = "manual override") -> None:
        """
        Force a specific emotional state (for testing or special scenarios).

        Args:
            emotion: The emotion to set.
            reason: Reason for the override.
        """
        old_emotion = self._current_emotion

        if emotion != old_emotion:
            transition = EmotionTransition(
                from_state=old_emotion,
                to_state=emotion,
                trigger=reason,
                timestamp=datetime.now(),
                turn_number=self._turn_count
            )
            self._transitions.append(transition)

            if self.on_transition:
                self.on_transition(transition)

        self._current_emotion = emotion

    def get_emotion_description(self) -> str:
        """Get a description of current emotional state for LLM context."""
        descriptions = {
            EmotionState.ANGRY: "You are ANGRY. Speak firmly and directly. Show visible frustration. You might raise your voice or use short, clipped sentences.",
            EmotionState.FRUSTRATED: "You are FRUSTRATED. You're exasperated but trying to stay composed. Sigh occasionally. Express disappointment.",
            EmotionState.ANXIOUS: "You are ANXIOUS. You're worried about the outcome. Ask clarifying questions. Express uncertainty about the resolution.",
            EmotionState.CONFUSED: "You are CONFUSED. You don't fully understand what's being explained. Ask for clarification. Express uncertainty.",
            EmotionState.NEUTRAL: "You are NEUTRAL. You're neither upset nor pleased. Respond matter-of-factly. Listen to what's being offered.",
            EmotionState.HOPEFUL: "You are becoming HOPEFUL. The conversation is improving. Show cautious optimism while still wanting confirmation.",
            EmotionState.SATISFIED: "You are SATISFIED. Your concerns are being addressed. Express appreciation but confirm the solution works.",
            EmotionState.DELIGHTED: "You are DELIGHTED. The service exceeded expectations. Express genuine gratitude and positive surprise.",
        }
        return descriptions.get(self._current_emotion, descriptions[EmotionState.NEUTRAL])

    def get_state_summary(self) -> Dict:
        """Get a summary of the current persona state."""
        return {
            "current_emotion": self._current_emotion.value,
            "turn_count": self._turn_count,
            "poor_response_count": self._poor_response_count,
            "transition_count": len(self._transitions),
            "prosody": self.prosody,
        }

    def reset(self) -> None:
        """Reset the state engine to initial state."""
        self._current_emotion = self.config.emotion_start
        self._turn_count = 0
        self._poor_response_count = 0
        self._transitions.clear()
