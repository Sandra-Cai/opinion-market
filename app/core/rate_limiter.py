"""
Rate limiting utilities for API protection
"""

import time
from typing import Dict, Optional
from collections import defaultdict, deque
import asyncio
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter implementation"""

    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        async with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]

            # Remove old requests outside time window
            while client_requests and client_requests[0] <= now - self.time_window:
                client_requests.popleft()

            # Check if under limit
            if len(client_requests) < self.max_requests:
                client_requests.append(now)
                return True

            return False

    async def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        async with self.lock:
            now = time.time()
            client_requests = self.requests[client_id]

            # Remove old requests
            while client_requests and client_requests[0] <= now - self.time_window:
                client_requests.popleft()

            return max(0, self.max_requests - len(client_requests))


class DistributedRateLimiter:
    """Redis-based distributed rate limiter"""

    def __init__(self, redis_client, max_requests: int = 100, time_window: int = 60):
        self.redis = redis_client
        self.max_requests = max_requests
        self.time_window = time_window

    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed using Redis"""
        try:
            key = f"rate_limit:{client_id}"
            current = await self.redis.get(key)

            if current is None:
                await self.redis.setex(key, self.time_window, 1)
                return True

            current_count = int(current)
            if current_count < self.max_requests:
                await self.redis.incr(key)
                return True

            return False
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            return True  # Fail open

    async def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests from Redis"""
        try:
            key = f"rate_limit:{client_id}"
            current = await self.redis.get(key)

            if current is None:
                return self.max_requests

            current_count = int(current)
            return max(0, self.max_requests - current_count)
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            return self.max_requests


# Global rate limiter instances
rate_limiter = RateLimiter()
