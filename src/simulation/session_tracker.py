"""
Session Tracker for the Conversation Simulation system.

Handles persistence, retrieval, and export of simulation sessions
using Redis for storage.
"""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from .models import (
    SimulationSession,
    ConversationTurn,
    EmotionTransition,
    EmotionState,
)

logger = logging.getLogger(__name__)


class SessionTrackerError(Exception):
    """Exception raised for session tracking errors."""
    pass


class SessionTracker:
    """
    Tracks and persists simulation sessions.

    Uses Redis for storage with automatic expiration.
    Supports session retrieval, listing, and export.
    """

    # Redis key prefixes
    SESSION_PREFIX = "simulation:session:"
    SESSION_LIST_KEY = "simulation:sessions"
    TRAINEE_SESSIONS_PREFIX = "simulation:trainee:"

    # Default TTL for sessions (7 days)
    DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60

    def __init__(self, redis_client=None, ttl_seconds: Optional[int] = None):
        """
        Initialize the session tracker.

        Args:
            redis_client: RedisClient instance. Auto-initialized if not provided.
            ttl_seconds: Time-to-live for sessions in seconds.
        """
        self._redis_client = redis_client
        self.ttl_seconds = ttl_seconds or self.DEFAULT_TTL_SECONDS

    def _get_redis(self):
        """Get or initialize Redis client."""
        if self._redis_client is None:
            from ..memory import RedisClient
            self._redis_client = RedisClient()
        return self._redis_client

    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for a session."""
        return f"{self.SESSION_PREFIX}{session_id}"

    def _trainee_key(self, trainee_id: str) -> str:
        """Generate Redis key for trainee's session list."""
        return f"{self.TRAINEE_SESSIONS_PREFIX}{trainee_id}"

    def save_session(self, session: SimulationSession) -> bool:
        """
        Save a simulation session to Redis.

        Args:
            session: The SimulationSession to save.

        Returns:
            True if saved successfully.
        """
        redis = self._get_redis()
        key = self._session_key(session.session_id)

        try:
            # Serialize session to JSON
            session_data = self._serialize_session(session)
            redis.client.set(key, json.dumps(session_data))
            redis.client.expire(key, self.ttl_seconds)

            # Add to global session list
            redis.client.rpush(
                self.SESSION_LIST_KEY,
                session.session_id
            )

            # Add to trainee's session list if trainee_id exists
            if session.trainee_id:
                redis.client.rpush(
                    self._trainee_key(session.trainee_id),
                    session.session_id
                )

            logger.info(f"Saved session: {session.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            raise SessionTrackerError(f"Failed to save session: {e}")

    def get_session(self, session_id: str) -> Optional[SimulationSession]:
        """
        Retrieve a simulation session by ID.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            SimulationSession or None if not found.
        """
        redis = self._get_redis()
        key = self._session_key(session_id)

        try:
            data = redis.client.get(key)
            if not data:
                return None

            session_data = json.loads(data)
            return self._deserialize_session(session_data)

        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    def update_session(self, session: SimulationSession) -> bool:
        """
        Update an existing session.

        Args:
            session: The updated SimulationSession.

        Returns:
            True if updated successfully.
        """
        return self.save_session(session)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a simulation session.

        Args:
            session_id: The session ID to delete.

        Returns:
            True if deleted successfully.
        """
        redis = self._get_redis()
        key = self._session_key(session_id)

        try:
            # Get session to find trainee_id
            session = self.get_session(session_id)

            # Delete session
            redis.client.delete(key)

            # Remove from global list
            redis.client.lrem(self.SESSION_LIST_KEY, 0, session_id)

            # Remove from trainee list if applicable
            if session and session.trainee_id:
                redis.client.lrem(
                    self._trainee_key(session.trainee_id),
                    0,
                    session_id
                )

            logger.info(f"Deleted session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def list_sessions(
        self,
        trainee_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List simulation sessions with optional filters.

        Args:
            trainee_id: Filter by trainee ID.
            scenario_id: Filter by scenario ID.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.

        Returns:
            List of session summaries.
        """
        redis = self._get_redis()

        try:
            # Get session IDs
            if trainee_id:
                session_ids = redis.client.lrange(
                    self._trainee_key(trainee_id),
                    0, -1
                )
            else:
                session_ids = redis.client.lrange(
                    self.SESSION_LIST_KEY,
                    0, -1
                )

            # Reverse to get most recent first
            session_ids = list(reversed(session_ids))

            # Apply offset and limit
            session_ids = session_ids[offset:offset + limit]

            # Get session summaries
            summaries = []
            for session_id in session_ids:
                session = self.get_session(session_id)
                if session:
                    # Apply scenario filter if specified
                    if scenario_id and session.scenario_id != scenario_id:
                        continue

                    summaries.append(self._get_session_summary(session))

            return summaries

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def get_trainee_stats(self, trainee_id: str) -> Dict[str, Any]:
        """
        Get statistics for a trainee across all their sessions.

        Args:
            trainee_id: The trainee's ID.

        Returns:
            Dictionary with trainee statistics.
        """
        sessions = self.list_sessions(trainee_id=trainee_id, limit=100)

        if not sessions:
            return {
                "trainee_id": trainee_id,
                "total_sessions": 0,
                "message": "No sessions found"
            }

        total_sessions = len(sessions)
        completed_sessions = [s for s in sessions if s["status"] == "completed"]
        resolutions = [s for s in completed_sessions if s.get("resolution_achieved")]

        total_turns = sum(s.get("total_turns", 0) for s in completed_sessions)
        total_duration = sum(s.get("duration_seconds", 0) for s in completed_sessions)

        return {
            "trainee_id": trainee_id,
            "total_sessions": total_sessions,
            "completed_sessions": len(completed_sessions),
            "resolution_rate": len(resolutions) / len(completed_sessions) if completed_sessions else 0,
            "average_turns": total_turns / len(completed_sessions) if completed_sessions else 0,
            "average_duration_seconds": total_duration / len(completed_sessions) if completed_sessions else 0,
            "scenarios_attempted": list(set(s["scenario_id"] for s in sessions)),
        }

    def export_session_transcript(
        self,
        session_id: str,
        format: str = "text"
    ) -> str:
        """
        Export a session transcript.

        Args:
            session_id: The session ID to export.
            format: Export format ('text', 'json', 'markdown').

        Returns:
            Formatted transcript string.
        """
        session = self.get_session(session_id)
        if not session:
            raise SessionTrackerError(f"Session not found: {session_id}")

        if format == "json":
            return json.dumps(self._serialize_session(session), indent=2, default=str)
        elif format == "markdown":
            return self._format_transcript_markdown(session)
        else:
            return self._format_transcript_text(session)

    def _format_transcript_text(self, session: SimulationSession) -> str:
        """Format session as plain text transcript."""
        lines = [
            f"SIMULATION TRANSCRIPT",
            f"=" * 50,
            f"Session ID: {session.session_id}",
            f"Scenario: {session.scenario_title}",
            f"Trainee: {session.trainee_id or 'Anonymous'}",
            f"Date: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {session.duration_seconds:.1f}s" if session.duration_seconds else "In progress",
            f"Status: {session.status}",
            f"Resolution: {'Yes' if session.resolution_achieved else 'No'}",
            f"=" * 50,
            f"",
            f"CONVERSATION:",
            f"-" * 50,
        ]

        for turn in session.turns:
            role = "CUSTOMER" if turn.role == "customer" else "TRAINEE"
            emotion = f" [{turn.emotion_state.value}]" if turn.emotion_state else ""
            timestamp = turn.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{timestamp}] {role}{emotion}:")
            lines.append(f"  {turn.content}")
            lines.append("")

        if session.emotion_transitions:
            lines.extend([
                f"-" * 50,
                f"EMOTION CHANGES:",
            ])
            for transition in session.emotion_transitions:
                lines.append(
                    f"  Turn {transition.turn_number}: "
                    f"{transition.from_state.value} -> {transition.to_state.value} "
                    f"({transition.trigger})"
                )

        return "\n".join(lines)

    def _format_transcript_markdown(self, session: SimulationSession) -> str:
        """Format session as markdown transcript."""
        lines = [
            f"# Simulation Transcript",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Session ID | `{session.session_id}` |",
            f"| Scenario | {session.scenario_title} |",
            f"| Trainee | {session.trainee_id or 'Anonymous'} |",
            f"| Date | {session.start_time.strftime('%Y-%m-%d %H:%M:%S')} |",
            f"| Duration | {session.duration_seconds:.1f}s |" if session.duration_seconds else "| Duration | In progress |",
            f"| Status | {session.status} |",
            f"| Resolution | {'Yes' if session.resolution_achieved else 'No'} |",
            f"",
            f"## Conversation",
            f"",
        ]

        for turn in session.turns:
            role = "**Customer**" if turn.role == "customer" else "**Trainee**"
            emotion = f" _{turn.emotion_state.value}_" if turn.emotion_state else ""
            lines.append(f"{role}{emotion}:")
            lines.append(f"> {turn.content}")
            lines.append("")

        if session.emotion_transitions:
            lines.extend([
                f"## Emotion Transitions",
                f"",
                f"| Turn | From | To | Trigger |",
                f"|------|------|-----|---------|",
            ])
            for transition in session.emotion_transitions:
                lines.append(
                    f"| {transition.turn_number} | "
                    f"{transition.from_state.value} | "
                    f"{transition.to_state.value} | "
                    f"{transition.trigger} |"
                )

        return "\n".join(lines)

    def _serialize_session(self, session: SimulationSession) -> Dict[str, Any]:
        """Serialize a SimulationSession to a dictionary."""
        return {
            "session_id": session.session_id,
            "scenario_id": session.scenario_id,
            "scenario_title": session.scenario_title,
            "trainee_id": session.trainee_id,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "status": session.status,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "role": t.role,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat(),
                    "emotion_state": t.emotion_state.value if t.emotion_state else None,
                    "detected_sentiment": t.detected_sentiment,
                    "detected_techniques": t.detected_techniques,
                }
                for t in session.turns
            ],
            "emotion_transitions": [
                {
                    "from_state": et.from_state.value,
                    "to_state": et.to_state.value,
                    "trigger": et.trigger,
                    "timestamp": et.timestamp.isoformat(),
                    "turn_number": et.turn_number,
                }
                for et in session.emotion_transitions
            ],
            "final_emotion": session.final_emotion.value if session.final_emotion else None,
            "resolution_achieved": session.resolution_achieved,
            "total_turns": session.total_turns,
            "duration_seconds": session.duration_seconds,
        }

    def _deserialize_session(self, data: Dict[str, Any]) -> SimulationSession:
        """Deserialize a dictionary to SimulationSession."""
        turns = [
            ConversationTurn(
                turn_number=t["turn_number"],
                role=t["role"],
                content=t["content"],
                timestamp=datetime.fromisoformat(t["timestamp"]),
                emotion_state=EmotionState(t["emotion_state"]) if t.get("emotion_state") else None,
                detected_sentiment=t.get("detected_sentiment"),
                detected_techniques=t.get("detected_techniques", []),
            )
            for t in data.get("turns", [])
        ]

        transitions = [
            EmotionTransition(
                from_state=EmotionState(et["from_state"]),
                to_state=EmotionState(et["to_state"]),
                trigger=et["trigger"],
                timestamp=datetime.fromisoformat(et["timestamp"]),
                turn_number=et["turn_number"],
            )
            for et in data.get("emotion_transitions", [])
        ]

        return SimulationSession(
            session_id=data["session_id"],
            scenario_id=data["scenario_id"],
            scenario_title=data["scenario_title"],
            trainee_id=data.get("trainee_id"),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            status=data["status"],
            turns=turns,
            emotion_transitions=transitions,
            final_emotion=EmotionState(data["final_emotion"]) if data.get("final_emotion") else None,
            resolution_achieved=data.get("resolution_achieved", False),
            total_turns=data.get("total_turns", len(turns)),
            duration_seconds=data.get("duration_seconds"),
        )

    def _get_session_summary(self, session: SimulationSession) -> Dict[str, Any]:
        """Get a summary of a session for listing."""
        return {
            "session_id": session.session_id,
            "scenario_id": session.scenario_id,
            "scenario_title": session.scenario_title,
            "trainee_id": session.trainee_id,
            "start_time": session.start_time.isoformat(),
            "status": session.status,
            "total_turns": session.total_turns,
            "duration_seconds": session.duration_seconds,
            "final_emotion": session.final_emotion.value if session.final_emotion else None,
            "resolution_achieved": session.resolution_achieved,
            "emotion_changes": len(session.emotion_transitions),
        }


# Global tracker instance
_session_tracker: Optional[SessionTracker] = None


def get_session_tracker() -> SessionTracker:
    """Get the global session tracker instance."""
    global _session_tracker
    if _session_tracker is None:
        _session_tracker = SessionTracker()
    return _session_tracker
