"""
Advanced caching system for Opinion Market
Provides multi-layer caching with Redis, in-memory, and database caching
"""

import json
import pickle
import hashlib
import time
from typing import Any, Optional, Dict, List, Union, Callable
from functools import wraps
import redis
from redis.exceptions import RedisError
import asyncio
from datetime import datetime, timedelta
from app.core.config import settings
from app.core.database import get_redis_client
from app.core.logging import log_system_metric


class CacheKey:
    """Cache key builder with namespacing"""
    
    def __init__(self, namespace: str = "opinion_market"):
        self.namespace = namespace
    
    def build(self, *parts: str, **kwargs) -> str:
        """Build cache key from parts and parameters"""
        key_parts = [self.namespace] + list(parts)
        
        if kwargs:
            # Sort kwargs for consistent key generation
            sorted_kwargs = sorted(kwargs.items())
            key_parts.extend([f"{k}:{v}" for k, v in sorted_kwargs])
        
        return ":".join(str(part) for part in key_parts)
    
    def hash(self, data: str) -> str:
        """Create hash of data for key generation"""
        return hashlib.md5(data.encode()).hexdigest()


class CacheSerializer:
    """Handle serialization/deserialization for different data types"""
    
    @staticmethod
    def serialize(data: Any) -> bytes:
        """Serialize data for storage"""
        if isinstance(data, (str, int, float, bool)):
            return json.dumps(data).encode()
        elif isinstance(data, (dict, list)):
            return json.dumps(data, default=str).encode()
        else:
            return pickle.dumps(data)
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize data from storage"""
        try:
            # Try JSON first
            return json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Fallback to pickle
                return pickle.loads(data)
            except (pickle.PickleError, EOFError):
                # Return raw bytes as last resort
                return data


class CacheStats:
    """Track cache statistics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.start_time = time.time()
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def uptime(self) -> float:
        """Get cache uptime in seconds"""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "uptime": self.uptime
        }


class MemoryCache:
    """In-memory cache with TTL support"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.stats = CacheStats()
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, data in self.cache.items()
            if data.get("expires_at", 0) < current_time
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self._cleanup_expired()
        
        if key in self.cache:
            data = self.cache[key]
            if data.get("expires_at", 0) > time.time():
                self.access_times[key] = time.time()
                self.stats.hits += 1
                return data["value"]
            else:
                del self.cache[key]
                del self.access_times[key]
        
        self.stats.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        self._cleanup_expired()
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        expires_at = time.time() + ttl if ttl else None
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time()
        }
        self.access_times[key] = time.time()
        self.stats.sets += 1
        return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            self.stats.deletes += 1
            return True
        return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            **self.stats.to_dict(),
            "size": len(self.cache),
            "max_size": self.max_size
        }


class RedisCache:
    """Redis-based cache with advanced features"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or get_redis_client()
        self.serializer = CacheSerializer()
        self.stats = CacheStats()
    
    def _handle_error(self, error: Exception, operation: str):
        """Handle Redis errors"""
        self.stats.errors += 1
        log_system_metric("cache_error", 1, {"operation": operation, "error": str(error)})
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                self.stats.hits += 1
                return self.serializer.deserialize(data)
            else:
                self.stats.misses += 1
                return None
        except RedisError as e:
            self._handle_error(e, "get")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            data = self.serializer.serialize(value)
            if ttl:
                result = self.redis_client.setex(key, ttl, data)
            else:
                result = self.redis_client.set(key, data)
            
            self.stats.sets += 1
            return bool(result)
        except RedisError as e:
            self._handle_error(e, "set")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            result = self.redis_client.delete(key)
            self.stats.deletes += 1
            return bool(result)
        except RedisError as e:
            self._handle_error(e, "delete")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                result = self.redis_client.delete(*keys)
                self.stats.deletes += len(keys)
                return result
            return 0
        except RedisError as e:
            self._handle_error(e, "delete_pattern")
            return 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except RedisError as e:
            self._handle_error(e, "exists")
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get TTL for key"""
        if not self.redis_client:
            return -1
        
        try:
            return self.redis_client.ttl(key)
        except RedisError as e:
            self._handle_error(e, "get_ttl")
            return -1
    
    def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """Increment numeric value"""
        if not self.redis_client:
            return None
        
        try:
            result = self.redis_client.incr(key, amount)
            if ttl and result == amount:  # First increment
                self.redis_client.expire(key, ttl)
            return result
        except RedisError as e:
            self._handle_error(e, "increment")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.stats.to_dict()
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    "redis_memory_used": info.get("used_memory_human"),
                    "redis_connected_clients": info.get("connected_clients"),
                    "redis_total_commands": info.get("total_commands_processed"),
                })
            except RedisError:
                pass
        
        return stats


class MultiLayerCache:
    """Multi-layer cache with L1 (memory) and L2 (Redis)"""
    
    def __init__(self, memory_size: int = 1000):
        self.l1_cache = MemoryCache(memory_size)
        self.l2_cache = RedisCache()
        self.key_builder = CacheKey()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 first, then L2)"""
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats.hits += 1
            return value
        
        # Try L2 cache
        value = self.l2_cache.get(key)
        if value is not None:
            self.stats.hits += 1
            # Store in L1 for faster access
            self.l1_cache.set(key, value)
            return value
        
        self.stats.misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in both cache layers"""
        # Set in L1 cache
        l1_success = self.l1_cache.set(key, value, ttl)
        
        # Set in L2 cache
        l2_success = self.l2_cache.set(key, value, ttl)
        
        if l1_success or l2_success:
            self.stats.sets += 1
            return True
        return False
    
    def delete(self, key: str) -> bool:
        """Delete value from both cache layers"""
        l1_success = self.l1_cache.delete(key)
        l2_success = self.l2_cache.delete(key)
        
        if l1_success or l2_success:
            self.stats.deletes += 1
            return True
        return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        # Clear L1 cache (simple approach)
        self.l1_cache.clear()
        
        # Delete from L2 cache
        deleted = self.l2_cache.delete_pattern(pattern)
        self.stats.deletes += deleted
        return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics"""
        return {
            **self.stats.to_dict(),
            "l1_cache": self.l1_cache.get_stats(),
            "l2_cache": self.l2_cache.get_stats()
        }


# Global cache instance
cache = MultiLayerCache() if settings.ENABLE_CACHING else None


def cached(ttl: int = None, key_prefix: str = "", cache_instance: MultiLayerCache = None):
    """Cache decorator for functions"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not cache_instance:
                return await func(*args, **kwargs)
            
            # Build cache key
            key_parts = [key_prefix, func.__name__]
            if args:
                key_parts.extend(str(arg) for arg in args)
            if kwargs:
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            
            cache_key = cache_instance.key_builder.build(*key_parts)
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not cache_instance:
                return func(*args, **kwargs)
            
            # Build cache key
            key_parts = [key_prefix, func.__name__]
            if args:
                key_parts.extend(str(arg) for arg in args)
            if kwargs:
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            
            cache_key = cache_instance.key_builder.build(*key_parts)
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def cache_invalidate(pattern: str = None, keys: List[str] = None, cache_instance: MultiLayerCache = None):
    """Invalidate cache entries"""
    if not cache_instance:
        return
    
    if pattern:
        cache_instance.delete_pattern(pattern)
    elif keys:
        for key in keys:
            cache_instance.delete(key)


def cache_warm_up(warm_up_funcs: List[Callable], cache_instance: MultiLayerCache = None):
    """Warm up cache with predefined functions"""
    if not cache_instance:
        return
    
    for func in warm_up_funcs:
        try:
            if asyncio.iscoroutinefunction(func):
                asyncio.create_task(func())
            else:
                func()
        except Exception as e:
            log_system_metric("cache_warmup_error", 1, {"function": func.__name__, "error": str(e)})


# Cache utility functions
def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    if cache:
        return cache.get_stats()
    return {"status": "disabled"}


def clear_all_cache():
    """Clear all cache entries"""
    if cache:
        cache.l1_cache.clear()
        cache.l2_cache.delete_pattern("opinion_market:*")


def cache_health_check() -> Dict[str, Any]:
    """Check cache health"""
    health = {
        "status": "healthy",
        "l1_cache": "unknown",
        "l2_cache": "unknown"
    }
    
    if not cache:
        health["status"] = "disabled"
        return health
    
    try:
        # Test L1 cache
        test_key = "health_check_l1"
        cache.l1_cache.set(test_key, "test", 10)
        if cache.l1_cache.get(test_key) == "test":
            health["l1_cache"] = "healthy"
        else:
            health["l1_cache"] = "unhealthy"
        cache.l1_cache.delete(test_key)
        
        # Test L2 cache
        test_key = "health_check_l2"
        cache.l2_cache.set(test_key, "test", 10)
        if cache.l2_cache.get(test_key) == "test":
            health["l2_cache"] = "healthy"
        else:
            health["l2_cache"] = "unhealthy"
        cache.l2_cache.delete(test_key)
        
        if health["l1_cache"] == "unhealthy" and health["l2_cache"] == "unhealthy":
            health["status"] = "unhealthy"
        
    except Exception as e:
        health["status"] = "unhealthy"
        health["error"] = str(e)
    
    return health


# Export commonly used functions
__all__ = [
    "CacheKey",
    "CacheSerializer",
    "MemoryCache",
    "RedisCache",
    "MultiLayerCache",
    "cache",
    "cached",
    "cache_invalidate",
    "cache_warm_up",
    "get_cache_stats",
    "clear_all_cache",
    "cache_health_check",
]
