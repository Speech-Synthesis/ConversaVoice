"""
Redis client for ConversaVoice.
Provides connection handling and basic operations for conversation memory.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redis client wrapper for conversation memory.

    Handles connection management and provides methods for
    storing and retrieving conversation data.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0
    ):
        """
        Initialize Redis client.

        Args:
            host: Redis host. Defaults to REDIS_HOST env var or 'localhost'.
            port: Redis port. Defaults to REDIS_PORT env var or 6379.
            db: Redis database number. Defaults to 0.
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db
        self._client = None

    def _get_client(self):
        """Lazy initialization of Redis client."""
        if self._client is None:
            import redis
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True
            )
            # Test connection
            try:
                self._client.ping()
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except redis.ConnectionError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._client

    @property
    def client(self):
        """Get the Redis client instance."""
        return self._get_client()

    def is_connected(self) -> bool:
        """Check if Redis connection is alive."""
        try:
            self._get_client().ping()
            return True
        except Exception:
            return False

    def close(self):
        """Close the Redis connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Redis connection closed")

    # Session Management Methods

    def _session_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"session:{session_id}"

    def _history_key(self, session_id: str) -> str:
        """Generate Redis key for conversation history."""
        return f"history:{session_id}"

    def create_session(self, session_id: str, ttl: int = 3600) -> bool:
        """
        Create a new conversation session with metadata.

        Args:
            session_id: Unique session identifier.
            ttl: Time-to-live in seconds. Defaults to 1 hour.

        Returns:
            True if session created successfully.
        """
        import time
        key = self._session_key(session_id)
        self.client.hset(key, mapping={
            "created": "true",
            "turn_count": "0",
            "start_time": str(int(time.time())),
            "error_count": "0",
            "last_activity": str(int(time.time()))
        })
        self.client.expire(key, ttl)
        logger.info(f"Created session: {session_id}")
        return True

    def add_message(self, session_id: str, role: str, content: str) -> int:
        """
        Add a message to conversation history.

        Args:
            session_id: Session identifier.
            role: Message role ('user' or 'assistant').
            content: Message content.

        Returns:
            Number of messages in history.
        """
        import json
        import time
        key = self._history_key(session_id)
        message = json.dumps({"role": role, "content": content})
        length = self.client.rpush(key, message)

        # Update turn count and last activity
        session_key = self._session_key(session_id)
        self.client.hincrby(session_key, "turn_count", 1)
        self.client.hset(session_key, "last_activity", str(int(time.time())))

        return length

    def get_history(self, session_id: str, limit: int = 15) -> list:
        """
        Get recent conversation history.

        Args:
            session_id: Session identifier.
            limit: Maximum number of messages to retrieve.

        Returns:
            List of message dictionaries.
        """
        import json
        key = self._history_key(session_id)
        messages = self.client.lrange(key, -limit, -1)
        return [json.loads(msg) for msg in messages]

    def get_context_string(self, session_id: str, limit: int = 15) -> str:
        """
        Get conversation history as a formatted string for LLM context.

        Args:
            session_id: Session identifier.
            limit: Maximum number of messages.

        Returns:
            Formatted context string.
        """
        history = self.get_history(session_id, limit)
        if not history:
            return ""

        lines = []
        for msg in history:
            role = msg["role"].capitalize()
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session and its history.

        Args:
            session_id: Session identifier.

        Returns:
            True if cleared successfully.
        """
        self.client.delete(self._session_key(session_id))
        self.client.delete(self._history_key(session_id))
        self.client.delete(self._context_key(session_id))
        logger.info(f"Cleared session: {session_id}")
        return True

    # Session Metadata Methods

    def record_error(self, session_id: str, error_type: str = "general") -> int:
        """
        Record an error in the session.

        Args:
            session_id: Session identifier.
            error_type: Type of error (e.g., "tts", "llm", "stt").

        Returns:
            Total error count for the session.
        """
        session_key = self._session_key(session_id)
        count = self.client.hincrby(session_key, "error_count", 1)
        self.client.hset(session_key, "last_error", error_type)
        logger.warning(f"Session {session_id} error #{count}: {error_type}")
        return count

    def get_session_metadata(self, session_id: str) -> dict:
        """
        Get full session metadata.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with session metadata including duration, turn_count, errors.
        """
        import time
        session_key = self._session_key(session_id)
        data = self.client.hgetall(session_key)

        if not data:
            return {}

        # Calculate session duration
        start_time = int(data.get("start_time", 0))
        last_activity = int(data.get("last_activity", start_time))
        current_time = int(time.time())

        return {
            "session_id": session_id,
            "turn_count": int(data.get("turn_count", 0)),
            "error_count": int(data.get("error_count", 0)),
            "start_time": start_time,
            "last_activity": last_activity,
            "duration_seconds": last_activity - start_time if start_time else 0,
            "idle_seconds": current_time - last_activity if last_activity else 0,
            "last_error": data.get("last_error", None)
        }

    def get_session_summary(self, session_id: str) -> str:
        """
        Get a human-readable session summary.

        Args:
            session_id: Session identifier.

        Returns:
            Formatted summary string.
        """
        meta = self.get_session_metadata(session_id)
        if not meta:
            return "No session data available."

        duration = meta["duration_seconds"]
        mins, secs = divmod(duration, 60)

        summary = f"Session: {meta['turn_count']} turns, {mins}m {secs}s"
        if meta["error_count"] > 0:
            summary += f", {meta['error_count']} errors"

        return summary

    # Prosody Profile Methods

    def _prosody_key(self, style: str) -> str:
        """Generate Redis key for prosody profile."""
        return f"prosody:{style}"

    def init_prosody_profiles(self) -> None:
        """
        Initialize default prosody profiles in Redis.

        Profiles are based on Conversia.md specifications.
        Call this once on startup to ensure profiles exist.
        """
        default_profiles = {
            "neutral": {"pitch": "0%", "rate": "1.0", "volume": "medium"},
            "cheerful": {"pitch": "+5%", "rate": "1.1", "volume": "medium"},
            "patient": {"pitch": "-5%", "rate": "0.9", "volume": "medium"},
            "empathetic": {"pitch": "-5%", "rate": "0.85", "volume": "medium"},
            "de_escalate": {"pitch": "-10%", "rate": "0.8", "volume": "soft"},
        }

        for style, params in default_profiles.items():
            key = self._prosody_key(style)
            # Only set if not already exists (preserve custom values)
            if not self.client.exists(key):
                self.client.hset(key, mapping=params)
                logger.info(f"Initialized prosody profile: {style}")

    def get_prosody(self, style: str) -> dict:
        """
        Get prosody parameters for a given style.

        Args:
            style: Emotion style label (e.g., "empathetic", "patient").

        Returns:
            Dict with pitch, rate, and volume. Falls back to neutral if not found.
        """
        key = self._prosody_key(style)
        params = self.client.hgetall(key)

        if not params:
            # Fallback to neutral if style not found
            logger.warning(f"Prosody profile '{style}' not found, using neutral")
            return {"pitch": "0%", "rate": "1.0", "volume": "medium"}

        return params

    def set_prosody(self, style: str, pitch: str, rate: str, volume: str = "medium") -> bool:
        """
        Set or update a prosody profile.

        Args:
            style: Emotion style label.
            pitch: Pitch adjustment (e.g., "-5%", "+10%").
            rate: Speech rate (e.g., "0.85", "1.1").
            volume: Volume level (e.g., "soft", "medium", "loud").

        Returns:
            True if set successfully.
        """
        key = self._prosody_key(style)
        self.client.hset(key, mapping={
            "pitch": pitch,
            "rate": rate,
            "volume": volume
        })
        logger.info(f"Updated prosody profile: {style}")
        return True

    # Context Labels Methods

    def _context_key(self, session_id: str) -> str:
        """Generate Redis key for context labels."""
        return f"context:{session_id}"

    def get_context_labels(self, session_id: str) -> dict:
        """
        Get all context labels for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with context labels (interaction_type, emotion, turn_count, repetition_count).
        """
        key = self._context_key(session_id)
        labels = self.client.hgetall(key)

        if not labels:
            # Return defaults for new session
            return {
                "interaction_type": "first_time",
                "emotion": "neutral",
                "turn_count": "0",
                "repetition_count": "0"
            }

        return labels

    def set_context_label(self, session_id: str, label: str, value: str) -> bool:
        """
        Set a specific context label.

        Args:
            session_id: Session identifier.
            label: Label name (e.g., "emotion", "interaction_type").
            value: Label value.

        Returns:
            True if set successfully.
        """
        key = self._context_key(session_id)
        self.client.hset(key, label, value)
        return True

    def update_context_labels(
        self,
        session_id: str,
        is_repetition: bool = False,
        detected_emotion: str = None
    ) -> dict:
        """
        Update context labels based on current interaction state.

        Automatically determines interaction_type based on turn count and repetition.

        Args:
            session_id: Session identifier.
            is_repetition: Whether the current input is a repetition.
            detected_emotion: Emotion detected from sentiment analysis (optional).

        Returns:
            Updated context labels dict.
        """
        key = self._context_key(session_id)
        session_key = self._session_key(session_id)

        # Get current turn count
        turn_count = self.client.hget(session_key, "turn_count") or "0"
        turn_count = int(turn_count)

        # Get current repetition count
        current_rep_count = self.client.hget(key, "repetition_count") or "0"
        current_rep_count = int(current_rep_count)

        # Determine interaction type
        if turn_count == 0:
            interaction_type = "first_time"
        elif is_repetition:
            interaction_type = "repetition"
            current_rep_count += 1
        else:
            interaction_type = "continuing"

        # Determine emotion (frustration from repeated repetitions)
        if detected_emotion:
            emotion = detected_emotion
        elif current_rep_count >= 2:
            emotion = "frustrated"
        elif is_repetition:
            emotion = "confused"
        else:
            emotion = "neutral"

        # Update all labels
        labels = {
            "interaction_type": interaction_type,
            "emotion": emotion,
            "turn_count": str(turn_count),
            "repetition_count": str(current_rep_count)
        }

        self.client.hset(key, mapping=labels)
        logger.debug(f"Updated context labels for {session_id}: {labels}")

        return labels

    def get_context_hint(self, session_id: str) -> str:
        """
        Get a formatted context hint string for LLM.

        Args:
            session_id: Session identifier.

        Returns:
            Formatted hint string describing the user's context state.
        """
        labels = self.get_context_labels(session_id)

        hints = []

        # Interaction type hint
        if labels.get("interaction_type") == "first_time":
            hints.append("This is the user's first message in this session.")
        elif labels.get("interaction_type") == "repetition":
            rep_count = labels.get("repetition_count", "1")
            hints.append(f"The user is repeating themselves (repeat #{rep_count}).")

        # Emotion hint
        emotion = labels.get("emotion", "neutral")
        if emotion == "frustrated":
            hints.append("The user appears frustrated - be direct and helpful.")
        elif emotion == "confused":
            hints.append("The user seems confused - explain clearly and patiently.")
        elif emotion == "angry":
            hints.append("The user is angry - stay calm and focus on resolution.")

        return " ".join(hints) if hints else ""

    # User Preferences Methods

    def _preferences_key(self, session_id: str) -> str:
        """Generate Redis key for user preferences."""
        return f"preferences:{session_id}"

    def get_user_preferences(self, session_id: str) -> dict:
        """
        Get all user preferences for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Dict with user preferences (preferred_style, verbosity, name, etc.).
        """
        key = self._preferences_key(session_id)
        prefs = self.client.hgetall(key)

        if not prefs:
            return {
                "preferred_style": "neutral",
                "verbosity": "normal",
                "name": None,
                "interests": None,
                "communication_style": "balanced"
            }

        return prefs

    def set_user_preference(
        self,
        session_id: str,
        key: str,
        value: str
    ) -> bool:
        """
        Set a specific user preference.

        Args:
            session_id: Session identifier.
            key: Preference key (e.g., "preferred_style", "verbosity", "name").
            value: Preference value.

        Returns:
            True if set successfully.
        """
        pref_key = self._preferences_key(session_id)
        self.client.hset(pref_key, key, value)
        logger.debug(f"Set preference {key}={value} for session {session_id}")
        return True

    def set_user_preferences(
        self,
        session_id: str,
        preferences: dict
    ) -> bool:
        """
        Set multiple user preferences at once.

        Args:
            session_id: Session identifier.
            preferences: Dict of preference key-value pairs.

        Returns:
            True if set successfully.
        """
        if not preferences:
            return False

        pref_key = self._preferences_key(session_id)
        self.client.hset(pref_key, mapping=preferences)
        logger.debug(f"Set preferences for session {session_id}: {preferences}")
        return True

    def detect_preferences_from_message(self, message: str) -> dict:
        """
        Detect user preferences from their message.

        Looks for patterns like:
        - "My name is X" -> name
        - "Call me X" -> name
        - "I prefer brief answers" -> verbosity
        - "Be more detailed" -> verbosity

        Args:
            message: User message to analyze.

        Returns:
            Dict of detected preferences (may be empty).
        """
        import re
        detected = {}
        message_lower = message.lower()

        # Detect name
        name_patterns = [
            r"my name is (\w+)",
            r"call me (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, message_lower)
            if match:
                detected["name"] = match.group(1).capitalize()
                break

        # Detect verbosity preference
        if any(word in message_lower for word in ["brief", "short", "concise", "quick"]):
            detected["verbosity"] = "brief"
        elif any(word in message_lower for word in ["detailed", "explain", "thorough", "elaborate"]):
            detected["verbosity"] = "detailed"

        # Detect communication style
        if any(word in message_lower for word in ["formal", "professional"]):
            detected["communication_style"] = "formal"
        elif any(word in message_lower for word in ["casual", "friendly", "relaxed"]):
            detected["communication_style"] = "casual"

        return detected

    def get_preferences_hint(self, session_id: str) -> str:
        """
        Get a formatted preferences hint string for LLM.

        Args:
            session_id: Session identifier.

        Returns:
            Formatted hint string describing user preferences.
        """
        prefs = self.get_user_preferences(session_id)
        hints = []

        # Name hint
        name = prefs.get("name")
        if name:
            hints.append(f"User's name is {name}.")

        # Verbosity hint
        verbosity = prefs.get("verbosity", "normal")
        if verbosity == "brief":
            hints.append("User prefers brief, concise answers.")
        elif verbosity == "detailed":
            hints.append("User prefers detailed explanations.")

        # Communication style hint
        style = prefs.get("communication_style", "balanced")
        if style == "formal":
            hints.append("Use formal, professional language.")
        elif style == "casual":
            hints.append("Use casual, friendly language.")

        # Interests hint
        interests = prefs.get("interests")
        if interests:
            hints.append(f"User interests: {interests}.")

        return " ".join(hints) if hints else ""
