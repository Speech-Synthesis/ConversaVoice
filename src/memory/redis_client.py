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
