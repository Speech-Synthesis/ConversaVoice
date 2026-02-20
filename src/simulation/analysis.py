"""
Post-Session Analysis Engine for the Conversation Simulation system.

Evaluates trainee performance using LLM-based analysis of conversation
transcripts, generating structured feedback and scores.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .models import (
    SimulationSession,
    AnalysisResult,
    EmotionState,
)

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Exception raised for analysis errors."""
    pass


# Emotion improvement scoring
EMOTION_SCORES = {
    EmotionState.ANGRY: 1,
    EmotionState.FRUSTRATED: 2,
    EmotionState.ANXIOUS: 3,
    EmotionState.CONFUSED: 4,
    EmotionState.NEUTRAL: 5,
    EmotionState.HOPEFUL: 7,
    EmotionState.SATISFIED: 9,
    EmotionState.DELIGHTED: 10,
}


class AnalysisEngine:
    """
    Analyzes completed simulation sessions using LLM evaluation.

    Generates comprehensive feedback including:
    - Skill scores (empathy, de-escalation, clarity, etc.)
    - Strengths and areas for improvement
    - Specific actionable feedback
    - Training recommendations
    """

    def __init__(self, llm_client=None):
        """
        Initialize the analysis engine.

        Args:
            llm_client: LLM client for analysis. Auto-initialized if not provided.
        """
        self._llm_client = llm_client

    def _get_llm_client(self):
        """Get or initialize LLM client."""
        if self._llm_client is None:
            try:
                from ..llm import GroqClient
                self._llm_client = GroqClient()
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
                try:
                    from ..llm import OllamaClient
                    self._llm_client = OllamaClient()
                except Exception as e2:
                    raise AnalysisError(f"No LLM available for analysis: {e2}")
        return self._llm_client

    def analyze_session(self, session: SimulationSession) -> AnalysisResult:
        """
        Perform comprehensive analysis of a simulation session.

        Args:
            session: The completed SimulationSession to analyze.

        Returns:
            AnalysisResult with scores and feedback.

        Raises:
            AnalysisError: If analysis fails.
        """
        if session.status == "active":
            raise AnalysisError("Cannot analyze an active session. End it first.")

        # Calculate basic metrics
        metrics = self._calculate_metrics(session)

        # Get LLM analysis
        llm_analysis = self._get_llm_analysis(session)

        # Combine metrics and LLM analysis
        result = self._build_analysis_result(session, metrics, llm_analysis)

        logger.info(
            f"Analyzed session {session.session_id}: "
            f"Overall score: {result.overall_score}/10"
        )

        return result

    def _calculate_metrics(self, session: SimulationSession) -> Dict[str, Any]:
        """Calculate quantitative metrics from session data."""
        metrics = {}

        # Basic counts
        metrics["turn_count"] = session.total_turns
        metrics["duration_seconds"] = session.duration_seconds or 0
        metrics["emotion_changes"] = len(session.emotion_transitions)

        # Emotion trajectory analysis
        if session.emotion_transitions:
            start_emotion = session.emotion_transitions[0].from_state
        else:
            # Try to get from first customer turn
            customer_turns = [t for t in session.turns if t.role == "customer"]
            start_emotion = customer_turns[0].emotion_state if customer_turns and customer_turns[0].emotion_state else EmotionState.NEUTRAL

        end_emotion = session.final_emotion or EmotionState.NEUTRAL

        start_score = EMOTION_SCORES.get(start_emotion, 5)
        end_score = EMOTION_SCORES.get(end_emotion, 5)

        metrics["emotion_improvement"] = end_score - start_score
        metrics["start_emotion"] = start_emotion.value
        metrics["end_emotion"] = end_emotion.value

        # De-escalation success
        metrics["de_escalation_success"] = end_score > start_score

        # Efficiency (fewer turns = more efficient for resolution)
        if session.resolution_achieved:
            # Good resolution in few turns is efficient
            if metrics["turn_count"] <= 6:
                metrics["efficiency_rating"] = "excellent"
            elif metrics["turn_count"] <= 10:
                metrics["efficiency_rating"] = "good"
            elif metrics["turn_count"] <= 15:
                metrics["efficiency_rating"] = "average"
            else:
                metrics["efficiency_rating"] = "needs_improvement"
        else:
            metrics["efficiency_rating"] = "not_resolved"

        # Analyze trainee turns for techniques used
        trainee_turns = [t for t in session.turns if t.role == "trainee"]
        all_techniques = []
        for turn in trainee_turns:
            all_techniques.extend(turn.detected_techniques)

        metrics["techniques_used"] = list(set(all_techniques))
        metrics["empathy_instances"] = all_techniques.count("empathy")
        metrics["solution_instances"] = all_techniques.count("solution-oriented")

        return metrics

    def _get_llm_analysis(self, session: SimulationSession) -> Dict[str, Any]:
        """Get detailed analysis from LLM."""
        llm_client = self._get_llm_client()

        # Build transcript for analysis
        transcript = self._build_transcript_for_analysis(session)

        # Build analysis prompt
        prompt = self._build_analysis_prompt(session, transcript)

        try:
            response = llm_client.chat(prompt)
            return self._parse_llm_analysis(response)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            # Return default analysis on failure
            return self._get_fallback_analysis(session)

    def _build_transcript_for_analysis(self, session: SimulationSession) -> str:
        """Build a formatted transcript for LLM analysis."""
        lines = []
        for turn in session.turns:
            role = "Customer" if turn.role == "customer" else "Trainee"
            emotion = f" ({turn.emotion_state.value})" if turn.emotion_state else ""
            lines.append(f"{role}{emotion}: {turn.content}")
        return "\n".join(lines)

    def _build_analysis_prompt(self, session: SimulationSession, transcript: str) -> str:
        """Build the analysis prompt for LLM."""
        return f"""You are an expert customer service trainer analyzing a training simulation.

## SCENARIO
Title: {session.scenario_title}
Initial Customer Emotion: {session.emotion_transitions[0].from_state.value if session.emotion_transitions else 'unknown'}
Final Customer Emotion: {session.final_emotion.value if session.final_emotion else 'unknown'}
Resolution Achieved: {session.resolution_achieved}
Total Turns: {session.total_turns}

## CONVERSATION TRANSCRIPT
{transcript}

## ANALYSIS INSTRUCTIONS

Analyze the trainee's performance and provide a detailed evaluation. Score each category from 1-10.

Evaluate:
1. **Empathy Score**: Did the trainee acknowledge emotions, show understanding, use empathetic language?
2. **De-escalation Score**: Did they calm the customer down? Use appropriate techniques?
3. **Communication Clarity Score**: Were explanations clear? Did they avoid jargon? Was information organized?
4. **Problem Solving Score**: Did they offer solutions? Take ownership? Follow through?
5. **Efficiency Score**: Was the conversation appropriately paced? Not too long or rushed?

Also provide:
- 2-3 specific strengths demonstrated
- 2-3 specific areas for improvement
- 2-3 specific pieces of feedback with examples from the conversation
- 1-2 recommended training topics

## RESPONSE FORMAT

Respond with ONLY a valid JSON object in this exact format:
{{
    "empathy_score": <1-10>,
    "de_escalation_score": <1-10>,
    "communication_clarity_score": <1-10>,
    "problem_solving_score": <1-10>,
    "efficiency_score": <1-10>,
    "strengths": ["strength 1", "strength 2"],
    "areas_for_improvement": ["area 1", "area 2"],
    "specific_feedback": ["feedback 1 with example", "feedback 2 with example"],
    "recommended_training": ["training topic 1"],
    "customer_satisfaction_predicted": "low|medium|high"
}}
"""

    def _parse_llm_analysis(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response."""
        try:
            # Extract JSON from response
            response = response.strip()

            # Find JSON object
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM analysis JSON: {e}")
            return self._get_fallback_analysis(None)

    def _get_fallback_analysis(self, session: Optional[SimulationSession]) -> Dict[str, Any]:
        """Generate fallback analysis when LLM fails."""
        return {
            "empathy_score": 5,
            "de_escalation_score": 5,
            "communication_clarity_score": 5,
            "problem_solving_score": 5,
            "efficiency_score": 5,
            "strengths": ["Completed the simulation"],
            "areas_for_improvement": ["Analysis unavailable - please review manually"],
            "specific_feedback": ["Unable to generate detailed feedback at this time"],
            "recommended_training": ["General customer service skills"],
            "customer_satisfaction_predicted": "medium"
        }

    def _build_analysis_result(
        self,
        session: SimulationSession,
        metrics: Dict[str, Any],
        llm_analysis: Dict[str, Any]
    ) -> AnalysisResult:
        """Combine all analysis into final AnalysisResult."""

        # Get scores from LLM analysis
        empathy = llm_analysis.get("empathy_score", 5)
        de_escalation = llm_analysis.get("de_escalation_score", 5)
        clarity = llm_analysis.get("communication_clarity_score", 5)
        problem_solving = llm_analysis.get("problem_solving_score", 5)
        efficiency = llm_analysis.get("efficiency_score", 5)

        # Calculate overall score (weighted average)
        overall = round(
            empathy * 0.25 +
            de_escalation * 0.25 +
            clarity * 0.20 +
            problem_solving * 0.20 +
            efficiency * 0.10
        )

        return AnalysisResult(
            session_id=session.session_id,
            scenario_id=session.scenario_id,
            empathy_score=empathy,
            de_escalation_score=de_escalation,
            communication_clarity_score=clarity,
            problem_solving_score=problem_solving,
            efficiency_score=efficiency,
            overall_score=overall,
            de_escalation_success=metrics.get("de_escalation_success", False),
            resolution_achieved=session.resolution_achieved,
            customer_satisfaction_predicted=llm_analysis.get("customer_satisfaction_predicted", "medium"),
            strengths=llm_analysis.get("strengths", []),
            areas_for_improvement=llm_analysis.get("areas_for_improvement", []),
            specific_feedback=llm_analysis.get("specific_feedback", []),
            recommended_training=llm_analysis.get("recommended_training", []),
            turn_count=metrics.get("turn_count", session.total_turns),
            duration_seconds=metrics.get("duration_seconds", 0),
            emotion_changes=metrics.get("emotion_changes", 0),
            analysis_timestamp=datetime.now(),
        )

    def get_quick_score(self, session: SimulationSession) -> Dict[str, Any]:
        """
        Get a quick score without full LLM analysis.

        Useful for real-time feedback or when LLM is unavailable.

        Args:
            session: The simulation session to score.

        Returns:
            Dictionary with quick scores.
        """
        metrics = self._calculate_metrics(session)

        # Calculate quick scores based on metrics
        empathy_score = min(10, 5 + metrics.get("empathy_instances", 0))

        # De-escalation based on emotion improvement
        improvement = metrics.get("emotion_improvement", 0)
        if improvement >= 3:
            de_escalation_score = 9
        elif improvement >= 1:
            de_escalation_score = 7
        elif improvement == 0:
            de_escalation_score = 5
        else:
            de_escalation_score = 3

        # Efficiency based on turn count
        turns = metrics.get("turn_count", 10)
        if turns <= 6:
            efficiency_score = 9
        elif turns <= 10:
            efficiency_score = 7
        elif turns <= 15:
            efficiency_score = 5
        else:
            efficiency_score = 3

        overall = round((empathy_score + de_escalation_score + efficiency_score) / 3)

        return {
            "empathy_score": empathy_score,
            "de_escalation_score": de_escalation_score,
            "efficiency_score": efficiency_score,
            "overall_score": overall,
            "emotion_improvement": improvement,
            "techniques_used": metrics.get("techniques_used", []),
            "resolution_achieved": session.resolution_achieved,
        }

    def generate_summary_report(self, result: AnalysisResult) -> str:
        """
        Generate a human-readable summary report.

        Args:
            result: The AnalysisResult to summarize.

        Returns:
            Formatted summary string.
        """
        grade = self._score_to_grade(result.overall_score)

        lines = [
            "=" * 60,
            "SIMULATION PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"Session: {result.session_id}",
            f"Scenario: {result.scenario_id}",
            f"Date: {result.analysis_timestamp.strftime('%Y-%m-%d %H:%M')}",
            "",
            "-" * 60,
            "OVERALL PERFORMANCE",
            "-" * 60,
            f"Overall Grade: {grade} ({result.overall_score}/10)",
            f"Resolution Achieved: {'Yes' if result.resolution_achieved else 'No'}",
            f"De-escalation Success: {'Yes' if result.de_escalation_success else 'No'}",
            f"Predicted Customer Satisfaction: {result.customer_satisfaction_predicted.upper()}",
            "",
            "-" * 60,
            "SKILL SCORES",
            "-" * 60,
            f"  Empathy:              {self._score_bar(result.empathy_score)} {result.empathy_score}/10",
            f"  De-escalation:        {self._score_bar(result.de_escalation_score)} {result.de_escalation_score}/10",
            f"  Communication:        {self._score_bar(result.communication_clarity_score)} {result.communication_clarity_score}/10",
            f"  Problem Solving:      {self._score_bar(result.problem_solving_score)} {result.problem_solving_score}/10",
            f"  Efficiency:           {self._score_bar(result.efficiency_score)} {result.efficiency_score}/10",
            "",
            "-" * 60,
            "STRENGTHS",
            "-" * 60,
        ]

        for strength in result.strengths:
            lines.append(f"  + {strength}")

        lines.extend([
            "",
            "-" * 60,
            "AREAS FOR IMPROVEMENT",
            "-" * 60,
        ])

        for area in result.areas_for_improvement:
            lines.append(f"  - {area}")

        lines.extend([
            "",
            "-" * 60,
            "SPECIFIC FEEDBACK",
            "-" * 60,
        ])

        for i, feedback in enumerate(result.specific_feedback, 1):
            lines.append(f"  {i}. {feedback}")

        lines.extend([
            "",
            "-" * 60,
            "RECOMMENDED TRAINING",
            "-" * 60,
        ])

        for training in result.recommended_training:
            lines.append(f"  * {training}")

        lines.extend([
            "",
            "-" * 60,
            "SESSION METRICS",
            "-" * 60,
            f"  Total Turns: {result.turn_count}",
            f"  Duration: {result.duration_seconds:.1f} seconds",
            f"  Emotion Changes: {result.emotion_changes}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _score_to_grade(self, score: int) -> str:
        """Convert numeric score to letter grade."""
        if score >= 9:
            return "A"
        elif score >= 8:
            return "B+"
        elif score >= 7:
            return "B"
        elif score >= 6:
            return "C+"
        elif score >= 5:
            return "C"
        elif score >= 4:
            return "D"
        else:
            return "F"

    def _score_bar(self, score: int) -> str:
        """Generate a visual score bar."""
        filled = "#" * score
        empty = "-" * (10 - score)
        return f"[{filled}{empty}]"


# Global analysis engine instance
_analysis_engine: Optional[AnalysisEngine] = None


def get_analysis_engine() -> AnalysisEngine:
    """Get the global analysis engine instance."""
    global _analysis_engine
    if _analysis_engine is None:
        _analysis_engine = AnalysisEngine()
    return _analysis_engine
