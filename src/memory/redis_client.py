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
        Create a new conversation session.

        Args:
            session_id: Unique session identifier.
            ttl: Time-to-live in seconds. Defaults to 1 hour.

        Returns:
            True if session created successfully.
        """
        key = self._session_key(session_id)
        self.client.hset(key, mapping={"created": "true", "turn_count": "0"})
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
        key = self._history_key(session_id)
        message = json.dumps({"role": role, "content": content})
        length = self.client.rpush(key, message)

        # Update turn count
        session_key = self._session_key(session_id)
        self.client.hincrby(session_key, "turn_count", 1)

        return length

    def get_history(self, session_id: str, limit: int = 10) -> list:
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

    def get_context_string(self, session_id: str, limit: int = 5) -> str:
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
        logger.info(f"Cleared session: {session_id}")
        return True
