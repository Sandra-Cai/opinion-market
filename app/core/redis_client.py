"""
Redis client utility for consistent Redis connection management
"""

import redis.asyncio as redis
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global Redis client instance
_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client instance with proper error handling
    Returns None if Redis is not available (for development/testing)
    """
    global _redis_client

    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host="localhost",
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Test connection
            await _redis_client.ping()
            logger.info("Redis client connected successfully")

        except Exception as e:
            logger.warning(
                f"Redis connection failed: {e}. Using mock client for development."
            )
            _redis_client = None

    return _redis_client


async def close_redis_client():
    """Close Redis client connection"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis client connection closed")
