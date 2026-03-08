"""
Conversation Flow Manager for the Simulation System.

Handles natural conversation ending detection based on:
- Customer emotion state progression
- Satisfaction indicators in customer responses
- Realistic behavior patterns for different customer types
"""

import re
import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from enum import Enum

from .models import EmotionState

logger = logging.getLogger(__name__)


class EndingType(Enum):
    """Types of conversation endings."""
    SATISFIED_GOODBYE = "satisfied_goodbye"      # Happy resolution
    RELUCTANT_ACCEPTANCE = "reluctant_acceptance"  # Grudging acceptance (angry customers)
    FRUSTRATED_EXIT = "frustrated_exit"          # Customer gives up
    NATURAL_CONCLUSION = "natural_conclusion"    # Neutral wrap-up
    NONE = "none"                                # Not ending yet


@dataclass
class CompletionStatus:
    """Status of conversation completion."""
    is_complete: bool = False
    approaching_end: bool = False
    ending_type: EndingType = EndingType.NONE
    confidence: float = 0.0
    turns_at_positive_emotion: int = 0
    suggested_goodbye: Optional[str] = None


# Phrases indicating customer satisfaction/acceptance
SATISFACTION_PHRASES = [
    r"\bthank(?:s| you)\b",
    r"\bthat(?:'s| is) (?:great|perfect|helpful|good)\b",
    r"\bi appreciate\b",
    r"\byou(?:'ve| have) been helpful\b",
    r"\bthat works\b",
    r"\bsounds good\b",
    r"\bexcellent\b",
    r"\bwonderful\b",
    r"\bfantastic\b",
]

# Phrases indicating grudging acceptance (for angry customers)
RELUCTANT_PHRASES = [
    r"\bfine\b",
    r"\balright\b",
    r"\bokay\b",
    r"\bi guess\b",
    r"\bwhatever\b",
    r"\bi suppose\b",
    r"\bif you say so\b",
    r"\blet(?:'s| us) see\b",
    r"\bi hope (?:this|it) works\b",
]

# Phrases indicating frustrated exit
EXIT_PHRASES = [
    r"\bforget it\b",
    r"\bnever mind\b",
    r"\bi(?:'ll| will) go elsewhere\b",
    r"\bcancel (?:my |the )?(?:account|service|subscription)\b",
    r"\bi(?:'m| am) done\b",
    r"\bthis is useless\b",
    r"\bwaste of (?:my )?time\b",
    r"\bi(?:'ll| will) call back\b",
    r"\bspeak to (?:a |your )?(?:manager|supervisor)\b",
]

# Goodbye phrases (can combine with any ending type)
GOODBYE_PHRASES = [
    r"\bgoodbye\b",
    r"\bbye\b",
    r"\bhave a (?:good|nice) day\b",
    r"\btake care\b",
    r"\bthat(?:'s| is) all\b",
    r"\bi(?:'m| am) good\b",
    r"\bnothing else\b",
]

# Goodbye messages for different ending types and emotions
GOODBYE_MESSAGES = {
    EndingType.SATISFIED_GOODBYE: {
        EmotionState.DELIGHTED: [
            "This has been wonderful, thank you so much! You've been incredibly helpful. Goodbye!",
            "Perfect! I really appreciate all your help today. Have a great day!",
            "Excellent! Thank you for going above and beyond. Take care!",
        ],
        EmotionState.SATISFIED: [
            "Great, that solves my problem. Thanks for your help. Goodbye!",
            "Alright, I'm happy with that solution. Thank you!",
            "That works for me. I appreciate your assistance. Have a good day!",
        ],
        EmotionState.HOPEFUL: [
            "Okay, I'm feeling better about this now. Thank you for explaining everything.",
            "Thanks, I'll give that a try. Hopefully it works out!",
            "Alright, that gives me some hope. I appreciate your time.",
        ],
    },
    EndingType.RELUCTANT_ACCEPTANCE: {
        EmotionState.NEUTRAL: [
            "Fine, let's see if this actually works this time.",
            "Okay, I'll accept that. But I'm not fully convinced yet.",
            "Alright. I hope I don't have to call back about this.",
        ],
        EmotionState.HOPEFUL: [
            "Okay, I guess that's reasonable. We'll see how it goes.",
            "Fine. I'm still not happy about this whole situation, but thank you.",
            "Alright, I'll give you the benefit of the doubt.",
        ],
        EmotionState.FRUSTRATED: [
            "Whatever, fine. Just make sure it's actually done this time.",
            "Okay. This better work or I'm calling back.",
            "I guess that'll have to do. Not happy, but I don't have more time for this.",
        ],
    },
    EndingType.FRUSTRATED_EXIT: {
        EmotionState.ANGRY: [
            "You know what? Forget it. I'll take my business elsewhere.",
            "This is ridiculous. I want to speak to a manager.",
            "I'm done. I'll find a company that actually cares about customers.",
        ],
        EmotionState.FRUSTRATED: [
            "Never mind, I'll figure it out myself. This was a waste of time.",
            "Forget it. I'll just cancel and go with a competitor.",
            "I give up. Clearly this isn't getting resolved today.",
        ],
    },
}


class ConversationFlowManager:
    """
    Manages conversation flow and detects natural ending points.

    Different customer types have different ending behaviors:
    - Angry: Takes 5-8 good exchanges, may end reluctantly
    - Frustrated: 3-5 exchanges, needs acknowledgment
    - Neutral: 2-3 exchanges, quick to accept solutions
    """

    # Minimum turns before each emotion type can naturally end
    MIN_TURNS_FOR_ENDING = {
        EmotionState.ANGRY: 5,       # Angry customers don't forgive easily
        EmotionState.FRUSTRATED: 3,
        EmotionState.ANXIOUS: 3,
        EmotionState.CONFUSED: 2,
        EmotionState.NEUTRAL: 2,
        EmotionState.HOPEFUL: 1,
        EmotionState.SATISFIED: 1,
        EmotionState.DELIGHTED: 1,
    }

    # Required consecutive turns at positive emotion before ending
    POSITIVE_TURNS_REQUIRED = {
        EmotionState.ANGRY: 3,       # Needs sustained improvement
        EmotionState.FRUSTRATED: 2,
        EmotionState.ANXIOUS: 2,
        EmotionState.CONFUSED: 1,
        EmotionState.NEUTRAL: 1,
        EmotionState.HOPEFUL: 1,
        EmotionState.SATISFIED: 1,
        EmotionState.DELIGHTED: 1,
    }

    # Emotions considered positive for ending
    POSITIVE_EMOTIONS = {
        EmotionState.NEUTRAL,
        EmotionState.HOPEFUL,
        EmotionState.SATISFIED,
        EmotionState.DELIGHTED,
    }

    def __init__(self, starting_emotion: EmotionState):
        """
        Initialize the flow manager.

        Args:
            starting_emotion: The customer's initial emotion.
        """
        self.starting_emotion = starting_emotion
        self.turn_count = 0
        self.consecutive_positive_turns = 0
        self.emotion_history: List[EmotionState] = [starting_emotion]
        self.last_customer_message = ""
        self._completion_status = CompletionStatus()

    def update(
        self,
        current_emotion: EmotionState,
        customer_message: str,
        trainee_message: str,
    ) -> CompletionStatus:
        """
        Update flow state and check for natural ending.

        Args:
            current_emotion: Current customer emotion.
            customer_message: The customer's latest response.
            trainee_message: The trainee's input that triggered this response.

        Returns:
            CompletionStatus indicating if/how conversation should end.
        """
        self.turn_count += 1
        self.emotion_history.append(current_emotion)
        self.last_customer_message = customer_message

        # Track consecutive positive emotions
        if current_emotion in self.POSITIVE_EMOTIONS:
            self.consecutive_positive_turns += 1
        else:
            self.consecutive_positive_turns = 0

        # Check for completion
        status = self._check_completion(current_emotion, customer_message)
        self._completion_status = status

        logger.debug(
            f"Flow update: turn={self.turn_count}, "
            f"emotion={current_emotion.value}, "
            f"positive_streak={self.consecutive_positive_turns}, "
            f"complete={status.is_complete}"
        )

        return status

    def _check_completion(
        self,
        emotion: EmotionState,
        message: str
    ) -> CompletionStatus:
        """Check if conversation should naturally end."""
        message_lower = message.lower()

        # Check minimum turns based on starting emotion
        min_turns = self.MIN_TURNS_FOR_ENDING.get(self.starting_emotion, 2)
        if self.turn_count < min_turns:
            return CompletionStatus(
                approaching_end=self._is_approaching_end(emotion),
                turns_at_positive_emotion=self.consecutive_positive_turns,
            )

        # Check for explicit goodbye phrases
        has_goodbye = any(
            re.search(pattern, message_lower)
            for pattern in GOODBYE_PHRASES
        )

        # Check for satisfied goodbye
        if self._check_satisfied_goodbye(emotion, message_lower, has_goodbye):
            return CompletionStatus(
                is_complete=True,
                ending_type=EndingType.SATISFIED_GOODBYE,
                confidence=0.9,
                turns_at_positive_emotion=self.consecutive_positive_turns,
                suggested_goodbye=self._get_goodbye_message(
                    EndingType.SATISFIED_GOODBYE, emotion
                ),
            )

        # Check for reluctant acceptance (common for initially angry customers)
        if self._check_reluctant_acceptance(emotion, message_lower, has_goodbye):
            return CompletionStatus(
                is_complete=True,
                ending_type=EndingType.RELUCTANT_ACCEPTANCE,
                confidence=0.85,
                turns_at_positive_emotion=self.consecutive_positive_turns,
                suggested_goodbye=self._get_goodbye_message(
                    EndingType.RELUCTANT_ACCEPTANCE, emotion
                ),
            )

        # Check for frustrated exit (customer gives up)
        if self._check_frustrated_exit(emotion, message_lower):
            return CompletionStatus(
                is_complete=True,
                ending_type=EndingType.FRUSTRATED_EXIT,
                confidence=0.9,
                suggested_goodbye=self._get_goodbye_message(
                    EndingType.FRUSTRATED_EXIT, emotion
                ),
            )

        # Check for natural conclusion (long conversation with resolution)
        if self._check_natural_conclusion(emotion, message_lower, has_goodbye):
            return CompletionStatus(
                is_complete=True,
                ending_type=EndingType.NATURAL_CONCLUSION,
                confidence=0.8,
                turns_at_positive_emotion=self.consecutive_positive_turns,
            )

        # Not complete yet
        return CompletionStatus(
            approaching_end=self._is_approaching_end(emotion),
            turns_at_positive_emotion=self.consecutive_positive_turns,
        )

    def _check_satisfied_goodbye(
        self,
        emotion: EmotionState,
        message: str,
        has_goodbye: bool
    ) -> bool:
        """Check for satisfied customer goodbye."""
        # Must be in positive emotion
        if emotion not in {EmotionState.SATISFIED, EmotionState.DELIGHTED, EmotionState.HOPEFUL}:
            return False

        # Check for satisfaction phrases
        has_satisfaction = any(
            re.search(pattern, message)
            for pattern in SATISFACTION_PHRASES
        )

        # Need satisfaction phrase or goodbye with positive emotion
        if has_satisfaction and has_goodbye:
            return True

        # Or sustained positive emotion with any satisfaction phrase
        required_positive = self.POSITIVE_TURNS_REQUIRED.get(self.starting_emotion, 2)
        if self.consecutive_positive_turns >= required_positive and has_satisfaction:
            return True

        return False

    def _check_reluctant_acceptance(
        self,
        emotion: EmotionState,
        message: str,
        has_goodbye: bool
    ) -> bool:
        """Check for reluctant acceptance (common for angry customers)."""
        # Only applies if started angry/frustrated
        if self.starting_emotion not in {EmotionState.ANGRY, EmotionState.FRUSTRATED}:
            return False

        # Must have improved at least somewhat
        if emotion in {EmotionState.ANGRY}:
            return False

        # Check for reluctant phrases
        has_reluctance = any(
            re.search(pattern, message)
            for pattern in RELUCTANT_PHRASES
        )

        # Reluctant phrase with some improvement
        if has_reluctance and self.consecutive_positive_turns >= 1:
            return True

        # Or goodbye with neutral/hopeful emotion after initially angry
        if has_goodbye and emotion in {EmotionState.NEUTRAL, EmotionState.HOPEFUL}:
            return True

        return False

    def _check_frustrated_exit(
        self,
        emotion: EmotionState,
        message: str
    ) -> bool:
        """Check for frustrated customer exit."""
        # Must still be in negative emotion
        if emotion not in {EmotionState.ANGRY, EmotionState.FRUSTRATED}:
            return False

        # Check for exit phrases
        has_exit = any(
            re.search(pattern, message)
            for pattern in EXIT_PHRASES
        )

        # Exit phrase or long conversation with no improvement
        if has_exit:
            return True

        # Long conversation (10+ turns) with still angry/frustrated
        if self.turn_count >= 10 and self.consecutive_positive_turns == 0:
            return True

        return False

    def _check_natural_conclusion(
        self,
        emotion: EmotionState,
        message: str,
        has_goodbye: bool
    ) -> bool:
        """Check for natural conversation conclusion."""
        # Need substantial conversation
        if self.turn_count < 4:
            return False

        # Goodbye with any non-negative emotion
        if has_goodbye and emotion not in {EmotionState.ANGRY, EmotionState.FRUSTRATED}:
            return True

        # Long sustained positive emotion
        required = self.POSITIVE_TURNS_REQUIRED.get(self.starting_emotion, 2) + 1
        if self.consecutive_positive_turns >= required:
            return True

        return False

    def _is_approaching_end(self, emotion: EmotionState) -> bool:
        """Check if conversation is approaching natural end."""
        # Approaching if in positive emotion and getting close to threshold
        if emotion in self.POSITIVE_EMOTIONS:
            required = self.POSITIVE_TURNS_REQUIRED.get(self.starting_emotion, 2)
            if self.consecutive_positive_turns >= required - 1:
                return True

        return False

    def _get_goodbye_message(
        self,
        ending_type: EndingType,
        emotion: EmotionState
    ) -> Optional[str]:
        """Get appropriate goodbye message for ending type and emotion."""
        import random

        type_messages = GOODBYE_MESSAGES.get(ending_type, {})
        emotion_messages = type_messages.get(emotion, [])

        if emotion_messages:
            return random.choice(emotion_messages)

        # Fallback: try adjacent emotions
        for fallback_emotion in type_messages:
            return random.choice(type_messages[fallback_emotion])

        return None

    def get_status(self) -> CompletionStatus:
        """Get current completion status."""
        return self._completion_status

    def should_generate_goodbye(self) -> bool:
        """Check if customer should say goodbye in next response."""
        return self._completion_status.approaching_end or self._completion_status.is_complete

    def get_goodbye_instruction(self) -> Optional[str]:
        """Get instruction for LLM to generate goodbye."""
        status = self._completion_status

        if status.is_complete:
            if status.ending_type == EndingType.SATISFIED_GOODBYE:
                return (
                    "The customer is satisfied. Generate a genuine thank-you and goodbye. "
                    "Express appreciation for the help received."
                )
            elif status.ending_type == EndingType.RELUCTANT_ACCEPTANCE:
                return (
                    "The customer is reluctantly accepting the solution. Generate a grudging "
                    "acceptance - not fully happy but willing to end the conversation."
                )
            elif status.ending_type == EndingType.FRUSTRATED_EXIT:
                return (
                    "The customer is frustrated and wants to end the conversation negatively. "
                    "They may threaten to leave or ask for escalation."
                )
        elif status.approaching_end:
            return (
                "The conversation is nearing resolution. The customer should start "
                "wrapping up but may have one more question or confirmation."
            )

        return None


# Global manager registry (one per session)
_flow_managers: Dict[str, ConversationFlowManager] = {}


def get_flow_manager(
    session_id: str,
    starting_emotion: Optional[EmotionState] = None
) -> ConversationFlowManager:
    """Get or create flow manager for a session."""
    if session_id not in _flow_managers:
        if starting_emotion is None:
            starting_emotion = EmotionState.NEUTRAL
        _flow_managers[session_id] = ConversationFlowManager(starting_emotion)
    return _flow_managers[session_id]


def remove_flow_manager(session_id: str) -> None:
    """Remove flow manager when session ends."""
    if session_id in _flow_managers:
        del _flow_managers[session_id]
