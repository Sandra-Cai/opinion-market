"""
Advanced caching utilities for performance optimization
"""

import time
import json
import hashlib
from typing import Any, Optional, Dict, Union
import asyncio
import logging

logger = logging.getLogger(__name__)


class MemoryCache:
    """High-performance in-memory cache with TTL support"""

    def __init__(self, max_size: int = 10000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = asyncio.Lock()

    def _generate_key(self, key: Union[str, tuple]) -> str:
        """Generate cache key from input"""
        if isinstance(key, tuple):
            key_str = json.dumps(key, sort_keys=True)
        else:
            key_str = str(key)

        return hashlib.md5(key_str.encode()).hexdigest()

    async def get(self, key: Union[str, tuple]) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._generate_key(key)

        async with self.lock:
            if cache_key not in self.cache:
                return None

            entry = self.cache[cache_key]

            # Check TTL
            if time.time() > entry["expires_at"]:
                del self.cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]
                return None

            # Update access time for LRU
            self.access_times[cache_key] = time.time()

            return entry["value"]

    async def set(
        self, key: Union[str, tuple], value: Any, ttl: Optional[int] = None
    ) -> None:
        """Set value in cache"""
        cache_key = self._generate_key(key)
        ttl = ttl or self.default_ttl

        async with self.lock:
            # Remove oldest entries if at capacity
            if len(self.cache) >= self.max_size:
                await self._evict_oldest()

            self.cache[cache_key] = {
                "value": value,
                "expires_at": time.time() + ttl,
                "created_at": time.time(),
            }
            self.access_times[cache_key] = time.time()

    async def delete(self, key: Union[str, tuple]) -> bool:
        """Delete value from cache"""
        cache_key = self._generate_key(key)

        async with self.lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
                if cache_key in self.access_times:
                    del self.access_times[cache_key]
                return True
            return False

    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self.lock:
            self.cache.clear()
            self.access_times.clear()

    async def _evict_oldest(self) -> None:
        """Evict oldest accessed entry"""
        if not self.access_times:
            return

        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self.lock:
            now = time.time()
            active_entries = sum(
                1 for entry in self.cache.values() if entry["expires_at"] > now
            )

            return {
                "total_entries": len(self.cache),
                "active_entries": active_entries,
                "max_size": self.max_size,
                "hit_rate": getattr(self, "_hit_rate", 0.0),
            }


class CacheDecorator:
    """Decorator for caching function results"""

    def __init__(self, cache: MemoryCache, ttl: int = 300):
        self.cache = cache
        self.ttl = ttl

    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))

            # Try to get from cache
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await self.cache.set(cache_key, result, self.ttl)

            return result

        return wrapper


# Global cache instances
memory_cache = MemoryCache()
cache_decorator = CacheDecorator(memory_cache)
