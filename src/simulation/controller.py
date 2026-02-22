"""
Simulation Controller for the Conversation Simulation system.

Main orchestrator that coordinates scenario loading, persona management,
LLM roleplay responses, and session tracking for training simulations.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime
from dataclasses import dataclass

from .models import (
    EmotionState,
    ScenarioConfig,
    ConversationTurn,
    SimulationSession,
    EmotionTransition,
)
from .scenarios import ScenarioEngine, ScenarioError, get_scenario_engine
from .persona import PersonaStateEngine, ResponseAnalysis, analyze_trainee_response
from .conversation_flow import (
    ConversationFlowManager,
    CompletionStatus,
    EndingType,
    get_flow_manager,
    remove_flow_manager,
)

logger = logging.getLogger(__name__)


class SimulationError(Exception):
    """Exception raised for simulation-related errors."""

    def __init__(self, message: str, session_id: Optional[str] = None):
        self.session_id = session_id
        super().__init__(message)


@dataclass
class SimulationResponse:
    """Response from a simulation turn."""

    customer_message: str
    emotion_state: EmotionState
    emotion_changed: bool
    previous_emotion: Optional[EmotionState]
    trainee_analysis: ResponseAnalysis
    prosody: Dict[str, str]
    turn_number: int
    raw_llm_response: Optional[str] = None
    # Conversation completion tracking
    conversation_complete: bool = False
    completion_status: Optional[CompletionStatus] = None
    goodbye_message: Optional[str] = None


class SimulationController:
    """
    Main controller for conversation simulations.

    Orchestrates the full simulation pipeline:
    Trainee Input → Analysis → Persona State Update → LLM Roleplay → Response

    The AI acts as the CUSTOMER persona, NOT as an assistant.
    """

    def __init__(
        self,
        scenario_engine: Optional[ScenarioEngine] = None,
        llm_client: Optional[Any] = None,
        on_emotion_change: Optional[Callable[[EmotionTransition], None]] = None,
        on_turn_complete: Optional[Callable[[SimulationResponse], None]] = None,
    ):
        """
        Initialize the simulation controller.

        Args:
            scenario_engine: ScenarioEngine instance (uses global if not provided).
            llm_client: LLM client (GroqClient or OllamaClient). Auto-initialized if not provided.
            on_emotion_change: Callback when customer emotion changes.
            on_turn_complete: Callback when a conversation turn completes.
        """
        self.scenario_engine = scenario_engine or get_scenario_engine()
        self._llm_client = llm_client
        self.on_emotion_change = on_emotion_change
        self.on_turn_complete = on_turn_complete

        # Current simulation state
        self._session: Optional[SimulationSession] = None
        self._scenario: Optional[ScenarioConfig] = None
        self._persona_engine: Optional[PersonaStateEngine] = None
        self._flow_manager: Optional[ConversationFlowManager] = None
        self._is_active = False

    @property
    def is_active(self) -> bool:
        """Check if a simulation is currently active."""
        return self._is_active

    @property
    def current_session(self) -> Optional[SimulationSession]:
        """Get the current simulation session."""
        return self._session

    @property
    def current_scenario(self) -> Optional[ScenarioConfig]:
        """Get the current scenario."""
        return self._scenario

    @property
    def current_emotion(self) -> Optional[EmotionState]:
        """Get current customer emotion state."""
        if self._persona_engine:
            return self._persona_engine.current_emotion
        return None

    def _get_llm_client(self):
        """Get or initialize the LLM client."""
        if self._llm_client is None:
            try:
                from ..llm import GroqClient
                self._llm_client = GroqClient()
                logger.info("Initialized Groq LLM client for simulation")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                try:
                    from ..llm import OllamaClient
                    self._llm_client = OllamaClient()
                    logger.info("Initialized Ollama LLM client for simulation")
                except Exception as e2:
                    raise SimulationError(f"No LLM client available: {e2}")
        return self._llm_client

    def start_simulation(
        self,
        scenario_id: str,
        trainee_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> SimulationSession:
        """
        Start a new simulation session.

        Args:
            scenario_id: ID of the scenario to run.
            trainee_id: Optional identifier for the trainee.
            session_id: Optional custom session ID.

        Returns:
            The initialized SimulationSession.

        Raises:
            SimulationError: If scenario not found or simulation already active.
        """
        if self._is_active:
            raise SimulationError("A simulation is already active. End it first.")

        # Load scenario
        try:
            self._scenario = self.scenario_engine.get_scenario(scenario_id)
        except ScenarioError as e:
            raise SimulationError(f"Failed to load scenario: {e}")

        # Generate session ID
        if not session_id:
            import uuid
            session_id = f"sim-{uuid.uuid4().hex[:12]}"

        # Initialize session
        self._session = SimulationSession(
            session_id=session_id,
            scenario_id=scenario_id,
            scenario_title=self._scenario.title,
            trainee_id=trainee_id,
            start_time=datetime.now(),
            status="active",
        )

        # Initialize persona engine
        self._persona_engine = PersonaStateEngine(
            persona_config=self._scenario.persona,
            on_transition=self._handle_emotion_transition,
        )

        # Initialize conversation flow manager
        self._flow_manager = get_flow_manager(
            session_id=session_id,
            starting_emotion=self._scenario.persona.emotion_start,
        )

        self._is_active = True

        logger.info(
            f"Started simulation: {session_id} | "
            f"Scenario: {scenario_id} | "
            f"Initial emotion: {self._persona_engine.current_emotion.value}"
        )

        return self._session

    def _handle_emotion_transition(self, transition: EmotionTransition) -> None:
        """Handle emotion state transitions."""
        if self._session:
            self._session.emotion_transitions.append(transition)

        if self.on_emotion_change:
            self.on_emotion_change(transition)

    def get_opening_message(self) -> SimulationResponse:
        """
        Get the customer's opening message to start the conversation.

        Returns:
            SimulationResponse with the opening customer message.

        Raises:
            SimulationError: If no active simulation.
        """
        if not self._is_active or not self._scenario:
            raise SimulationError("No active simulation")

        # Generate opening message using LLM
        llm_client = self._get_llm_client()

        system_prompt = self._build_system_prompt(is_opening=True)
        user_prompt = "Generate your opening message as the customer. Start the conversation based on your situation and emotional state."

        try:
            response = llm_client.chat(user_prompt, context=system_prompt)
            customer_message = self._extract_message(response)
        except Exception as e:
            logger.error(f"Failed to generate opening message: {e}")
            # Fallback opening based on emotion
            customer_message = self._get_fallback_opening()

        # Record the turn
        turn = ConversationTurn(
            turn_number=0,
            role="customer",
            content=customer_message,
            timestamp=datetime.now(),
            emotion_state=self._persona_engine.current_emotion,
        )
        self._session.turns.append(turn)

        return SimulationResponse(
            customer_message=customer_message,
            emotion_state=self._persona_engine.current_emotion,
            emotion_changed=False,
            previous_emotion=None,
            trainee_analysis=ResponseAnalysis(),
            prosody=self._persona_engine.prosody,
            turn_number=0,
            raw_llm_response=response if 'response' in dir() else None,
        )

    def process_trainee_input(self, trainee_message: str) -> SimulationResponse:
        """
        Process trainee input and generate customer response.

        Args:
            trainee_message: The trainee's response to the customer.

        Returns:
            SimulationResponse with customer's reply and updated state.

        Raises:
            SimulationError: If no active simulation.
        """
        if not self._is_active:
            raise SimulationError("No active simulation")

        turn_number = len(self._session.turns)
        previous_emotion = self._persona_engine.current_emotion

        # Analyze trainee response
        analysis = analyze_trainee_response(trainee_message)

        # Record trainee turn
        trainee_turn = ConversationTurn(
            turn_number=turn_number,
            role="trainee",
            content=trainee_message,
            timestamp=datetime.now(),
            detected_techniques=analysis.detected_techniques,
        )
        self._session.turns.append(trainee_turn)

        # Update persona emotional state
        new_emotion, _ = self._persona_engine.process_trainee_response(trainee_message)
        emotion_changed = new_emotion != previous_emotion

        # Generate customer response
        llm_client = self._get_llm_client()
        system_prompt = self._build_system_prompt()
        context = self._build_conversation_context()

        try:
            response = llm_client.chat(trainee_message, context=f"{system_prompt}\n\n{context}")
            customer_message = self._extract_message(response)
        except Exception as e:
            logger.error(f"Failed to generate customer response: {e}")
            customer_message = self._get_fallback_response()

        # Record customer turn
        customer_turn = ConversationTurn(
            turn_number=turn_number + 1,
            role="customer",
            content=customer_message,
            timestamp=datetime.now(),
            emotion_state=new_emotion,
        )
        self._session.turns.append(customer_turn)

        # Update session
        self._session.total_turns = len(self._session.turns)
        self._session.final_emotion = new_emotion

        # Update conversation flow and check for natural ending
        completion_status = None
        conversation_complete = False
        goodbye_message = None

        if self._flow_manager:
            completion_status = self._flow_manager.update(
                current_emotion=new_emotion,
                customer_message=customer_message,
                trainee_message=trainee_message,
            )
            conversation_complete = completion_status.is_complete
            goodbye_message = completion_status.suggested_goodbye

        # Build response
        sim_response = SimulationResponse(
            customer_message=customer_message,
            emotion_state=new_emotion,
            emotion_changed=emotion_changed,
            previous_emotion=previous_emotion if emotion_changed else None,
            trainee_analysis=analysis,
            prosody=self._persona_engine.prosody,
            turn_number=turn_number + 1,
            raw_llm_response=response,
            conversation_complete=conversation_complete,
            completion_status=completion_status,
            goodbye_message=goodbye_message,
        )

        if self.on_turn_complete:
            self.on_turn_complete(sim_response)

        return sim_response

    async def process_trainee_input_async(self, trainee_message: str) -> SimulationResponse:
        """Async version of process_trainee_input."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.process_trainee_input,
            trainee_message
        )

    def end_simulation(
        self,
        resolution_achieved: bool = False,
        reason: str = "completed"
    ) -> SimulationSession:
        """
        End the current simulation.

        Args:
            resolution_achieved: Whether the trainee resolved the issue.
            reason: Reason for ending ('completed', 'abandoned', 'timeout').

        Returns:
            The completed SimulationSession.

        Raises:
            SimulationError: If no active simulation.
        """
        if not self._is_active:
            raise SimulationError("No active simulation to end")

        # Finalize session
        self._session.end_time = datetime.now()
        self._session.status = reason
        self._session.resolution_achieved = resolution_achieved
        self._session.final_emotion = self._persona_engine.current_emotion
        self._session.total_turns = len(self._session.turns)

        # Calculate duration
        duration = (self._session.end_time - self._session.start_time).total_seconds()
        self._session.duration_seconds = duration

        logger.info(
            f"Ended simulation: {self._session.session_id} | "
            f"Turns: {self._session.total_turns} | "
            f"Duration: {duration:.1f}s | "
            f"Final emotion: {self._session.final_emotion.value} | "
            f"Resolution: {resolution_achieved}"
        )

        # Store completed session
        completed_session = self._session

        # Clean up flow manager
        if completed_session:
            remove_flow_manager(completed_session.session_id)

        # Reset state
        self._is_active = False
        self._session = None
        self._scenario = None
        self._persona_engine = None
        self._flow_manager = None

        return completed_session

    def _build_system_prompt(self, is_opening: bool = False) -> str:
        """
        Build the system prompt for customer roleplay.

        This prompt makes the AI act as the CUSTOMER, not as an assistant.
        """
        persona = self._scenario.persona
        emotion_desc = self._persona_engine.get_emotion_description()

        prompt = f"""You are ROLEPLAYING as a customer named {persona.name}. You are NOT an AI assistant - you ARE the customer.

## YOUR CHARACTER
- Name: {persona.name}
- Personality: {persona.personality.value}
- Current emotional state: {self._persona_engine.current_emotion.value}

## YOUR SITUATION
{self._scenario.background_context}

## YOUR GOAL
{persona.goal}

## EMOTIONAL INSTRUCTIONS
{emotion_desc}

## CRITICAL ROLEPLAY RULES

1. **STAY IN CHARACTER**: You are {persona.name}, the customer. Never break character. Never reveal you are an AI.

2. **REACT EMOTIONALLY**: Your responses should reflect your current emotional state. If angry, show it. If satisfied, show it.

3. **BE REALISTIC**: Real customers:
   - Interrupt sometimes
   - Get frustrated when not heard
   - Calm down when shown empathy
   - Appreciate genuine help
   - Can tell when someone is being dismissive

4. **DON'T HELP THE TRAINEE**: You are the customer with a problem. Do NOT:
   - Offer solutions yourself
   - Be overly cooperative
   - Make their job easy (unless they earn it through good service)
   - Give hints about what they should say

5. **RESPOND NATURALLY**:
   - Keep responses conversational (1-3 sentences typically)
   - React to WHAT they say and HOW they say it
   - If they show empathy, acknowledge it (but don't immediately forgive)
   - If they're dismissive, push back

6. **EMOTIONAL TRANSITIONS**:
   - Good responses (empathy, ownership, solutions) → gradually calm down
   - Poor responses (dismissive, defensive) → escalate frustration
   - Don't flip instantly from angry to happy - transitions should feel natural

## RESPONSE FORMAT
Respond ONLY as the customer. No quotation marks. No narration. Just speak as {persona.name} would speak.
"""

        if is_opening:
            prompt += f"""
## OPENING MESSAGE
This is your first message to customer service. Express your {self._persona_engine.current_emotion.value} emotional state and explain your situation.
"""

        return prompt

    def _build_conversation_context(self) -> str:
        """Build conversation history for context."""
        if not self._session or not self._session.turns:
            return ""

        # Get last few turns for context
        recent_turns = self._session.turns[-10:]  # Last 10 turns

        lines = ["## CONVERSATION SO FAR"]
        for turn in recent_turns:
            role = "Customer" if turn.role == "customer" else "Service Rep"
            lines.append(f"{role}: {turn.content}")

        return "\n".join(lines)

    def _extract_message(self, llm_response: str) -> str:
        """Extract the customer message from LLM response."""
        # Clean up response - remove any JSON formatting if present
        message = llm_response.strip()

        # Remove any accidental roleplay markers
        prefixes_to_remove = [
            "Customer:", "CUSTOMER:", "customer:",
            f"{self._scenario.persona.name}:",
            "Response:", "Message:",
        ]
        for prefix in prefixes_to_remove:
            if message.startswith(prefix):
                message = message[len(prefix):].strip()

        # Remove quotes if wrapped
        if message.startswith('"') and message.endswith('"'):
            message = message[1:-1]

        return message

    def _get_fallback_opening(self) -> str:
        """Get a fallback opening message based on emotion."""
        emotion = self._persona_engine.current_emotion
        fallbacks = {
            EmotionState.ANGRY: "I've been waiting forever to talk to someone! This is absolutely unacceptable!",
            EmotionState.FRUSTRATED: "Hi, I really need some help here. I've been having issues and I'm getting frustrated.",
            EmotionState.ANXIOUS: "Hello, I'm calling because I'm worried about my account. Can someone help me?",
            EmotionState.CONFUSED: "Hi, I'm a bit confused about something and hoping you can clarify.",
            EmotionState.NEUTRAL: "Hello, I'm calling about an issue I need help with.",
        }
        return fallbacks.get(emotion, fallbacks[EmotionState.NEUTRAL])

    def _get_fallback_response(self) -> str:
        """Get a fallback response based on current emotion."""
        emotion = self._persona_engine.current_emotion
        fallbacks = {
            EmotionState.ANGRY: "That's not good enough! I need a real solution here!",
            EmotionState.FRUSTRATED: "I hear what you're saying, but this still doesn't solve my problem.",
            EmotionState.NEUTRAL: "Okay, I understand. What are the next steps?",
            EmotionState.HOPEFUL: "That sounds promising. Can you tell me more about how this will work?",
            EmotionState.SATISFIED: "Thank you, that's really helpful. I appreciate you taking care of this.",
        }
        return fallbacks.get(emotion, "I see. Please continue.")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current or last session."""
        session = self._session
        if not session:
            return {"error": "No session available"}

        return {
            "session_id": session.session_id,
            "scenario_id": session.scenario_id,
            "scenario_title": session.scenario_title,
            "status": session.status,
            "total_turns": session.total_turns,
            "current_emotion": self._persona_engine.current_emotion.value if self._persona_engine else session.final_emotion.value if session.final_emotion else "unknown",
            "emotion_transitions": len(session.emotion_transitions),
            "resolution_achieved": session.resolution_achieved,
            "duration_seconds": session.duration_seconds,
        }


# Global controller instance
_simulation_controller: Optional[SimulationController] = None


def get_simulation_controller() -> SimulationController:
    """Get the global simulation controller instance."""
    global _simulation_controller
    if _simulation_controller is None:
        _simulation_controller = SimulationController()
    return _simulation_controller
