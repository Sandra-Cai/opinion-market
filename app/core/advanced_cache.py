"""
Advanced Caching System
Provides multi-backend caching with TTL, invalidation, and performance monitoring
"""

import asyncio
import json
import time
import hashlib
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class CacheBackend(Enum):
    """Cache backend types"""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"

class CacheStrategy(Enum):
    """Cache invalidation strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    MANUAL = "manual"

@dataclass
class CacheEntry:
    """Represents a cache entry"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.tags is None:
            self.tags = []
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate the size of the cached value in bytes"""
        try:
            return len(pickle.dumps(self.value))
        except:
            return len(str(self.value).encode('utf-8'))
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self):
        """Update access information"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()

class CacheBackendInterface(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None) -> bool:
        """Set a value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass

class MemoryCacheBackend(CacheBackendInterface):
    """In-memory cache backend"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self._lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a value from cache"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    self.current_memory_bytes -= entry.size_bytes
                    self.stats["misses"] += 1
                    return None
                
                entry.touch()
                self.stats["hits"] += 1
                return entry
            else:
                self.stats["misses"] += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None) -> bool:
        """Set a value in cache"""
        try:
            with self._lock:
                # Remove existing entry if it exists
                if key in self.cache:
                    old_entry = self.cache[key]
                    self.current_memory_bytes -= old_entry.size_bytes
                
                # Create new entry
                expires_at = None
                if ttl:
                    expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.utcnow(),
                    expires_at=expires_at,
                    tags=tags or []
                )
                
                # Check memory limits
                if entry.size_bytes > self.max_memory_bytes:
                    logger.warning(f"Cache entry too large: {entry.size_bytes} bytes")
                    return False
                
                # Evict entries if necessary
                await self._evict_if_needed(entry.size_bytes)
                
                self.cache[key] = entry
                self.current_memory_bytes += entry.size_bytes
                self.stats["sets"] += 1
                
                return True
        except Exception as e:
            logger.error(f"Error setting cache entry: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from cache"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                del self.cache[key]
                self.current_memory_bytes -= entry.size_bytes
                self.stats["deletes"] += 1
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.current_memory_bytes = 0
            return True
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    self.current_memory_bytes -= entry.size_bytes
                    return False
                return True
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "backend": "memory",
                "entries": len(self.cache),
                "max_size": self.max_size,
                "current_memory_bytes": self.current_memory_bytes,
                "max_memory_bytes": self.max_memory_bytes,
                "memory_usage_percent": (self.current_memory_bytes / self.max_memory_bytes * 100) if self.max_memory_bytes > 0 else 0,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "sets": self.stats["sets"],
                "deletes": self.stats["deletes"],
                "evictions": self.stats["evictions"]
            }
    
    async def _evict_if_needed(self, new_entry_size: int):
        """Evict entries if cache limits are exceeded"""
        # Check size limit
        while len(self.cache) >= self.max_size:
            await self._evict_lru()
        
        # Check memory limit
        while self.current_memory_bytes + new_entry_size > self.max_memory_bytes:
            await self._evict_lru()
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
        entry = self.cache[lru_key]
        del self.cache[lru_key]
        self.current_memory_bytes -= entry.size_bytes
        self.stats["evictions"] += 1

class RedisCacheBackend(CacheBackendInterface):
    """Redis cache backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    async def _get_redis(self):
        """Get Redis connection"""
        if self.redis is None:
            try:
                import redis.asyncio as redis
                self.redis = redis.from_url(self.redis_url)
                await self.redis.ping()
            except ImportError:
                logger.error("Redis not available - install redis package")
                return None
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                return None
        return self.redis
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get a value from cache"""
        redis = await self._get_redis()
        if not redis:
            return None
        
        try:
            data = await redis.get(f"cache:{key}")
            if data:
                entry_data = json.loads(data)
                entry = CacheEntry(
                    key=key,
                    value=pickle.loads(entry_data["value"]),
                    created_at=datetime.fromisoformat(entry_data["created_at"]),
                    expires_at=datetime.fromisoformat(entry_data["expires_at"]) if entry_data["expires_at"] else None,
                    access_count=entry_data["access_count"],
                    last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                    tags=entry_data["tags"]
                )
                entry.touch()
                self.stats["hits"] += 1
                return entry
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            self.stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None) -> bool:
        """Set a value in cache"""
        redis = await self._get_redis()
        if not redis:
            return False
        
        try:
            expires_at = None
            if ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                tags=tags or []
            )
            
            entry_data = {
                "value": pickle.dumps(value).hex(),
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed.isoformat(),
                "tags": entry.tags
            }
            
            await redis.setex(f"cache:{key}", ttl or 3600, json.dumps(entry_data))
            self.stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from cache"""
        redis = await self._get_redis()
        if not redis:
            return False
        
        try:
            result = await redis.delete(f"cache:{key}")
            self.stats["deletes"] += 1
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        redis = await self._get_redis()
        if not redis:
            return False
        
        try:
            keys = await redis.keys("cache:*")
            if keys:
                await redis.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache"""
        redis = await self._get_redis()
        if not redis:
            return False
        
        try:
            return await redis.exists(f"cache:{key}") > 0
        except Exception as e:
            logger.error(f"Error checking Redis cache existence: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        redis = await self._get_redis()
        if not redis:
            return {"backend": "redis", "error": "Redis not available"}
        
        try:
            keys = await redis.keys("cache:*")
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "backend": "redis",
                "entries": len(keys),
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "sets": self.stats["sets"],
                "deletes": self.stats["deletes"]
            }
        except Exception as e:
            logger.error(f"Error getting Redis cache stats: {e}")
            return {"backend": "redis", "error": str(e)}

class AdvancedCache:
    """Advanced caching system with multiple backends and strategies"""
    
    def __init__(self, primary_backend: CacheBackend = CacheBackend.MEMORY, 
                 fallback_backend: Optional[CacheBackend] = None,
                 redis_url: str = "redis://localhost:6379"):
        self.primary_backend = self._create_backend(primary_backend, redis_url)
        self.fallback_backend = self._create_backend(fallback_backend, redis_url) if fallback_backend else None
        self.key_generators: Dict[str, Callable] = {}
        self.cache_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "backend_failures": 0
        }
    
    def _create_backend(self, backend_type: CacheBackend, redis_url: str) -> CacheBackendInterface:
        """Create a cache backend instance"""
        if backend_type == CacheBackend.MEMORY:
            return MemoryCacheBackend()
        elif backend_type == CacheBackend.REDIS:
            return RedisCacheBackend(redis_url)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
    
    def register_key_generator(self, name: str, generator: Callable):
        """Register a key generator function"""
        self.key_generators[name] = generator
    
    def generate_key(self, name: str, *args, **kwargs) -> str:
        """Generate a cache key using registered generator"""
        if name in self.key_generators:
            return self.key_generators[name](*args, **kwargs)
        else:
            # Default key generation
            key_data = f"{name}:{args}:{sorted(kwargs.items())}"
            return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        self.cache_stats["total_requests"] += 1
        
        # Try primary backend
        try:
            entry = await self.primary_backend.get(key)
            if entry:
                self.cache_stats["cache_hits"] += 1
                return entry.value
        except Exception as e:
            logger.error(f"Primary backend error: {e}")
            self.cache_stats["backend_failures"] += 1
        
        # Try fallback backend
        if self.fallback_backend:
            try:
                entry = await self.fallback_backend.get(key)
                if entry:
                    self.cache_stats["cache_hits"] += 1
                    return entry.value
            except Exception as e:
                logger.error(f"Fallback backend error: {e}")
                self.cache_stats["backend_failures"] += 1
        
        self.cache_stats["cache_misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None) -> bool:
        """Set a value in cache"""
        success = False
        
        # Set in primary backend
        try:
            if await self.primary_backend.set(key, value, ttl, tags):
                success = True
        except Exception as e:
            logger.error(f"Primary backend set error: {e}")
            self.cache_stats["backend_failures"] += 1
        
        # Set in fallback backend
        if self.fallback_backend:
            try:
                await self.fallback_backend.set(key, value, ttl, tags)
            except Exception as e:
                logger.error(f"Fallback backend set error: {e}")
                self.cache_stats["backend_failures"] += 1
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete a value from cache"""
        success = False
        
        # Delete from primary backend
        try:
            if await self.primary_backend.delete(key):
                success = True
        except Exception as e:
            logger.error(f"Primary backend delete error: {e}")
            self.cache_stats["backend_failures"] += 1
        
        # Delete from fallback backend
        if self.fallback_backend:
            try:
                await self.fallback_backend.delete(key)
            except Exception as e:
                logger.error(f"Fallback backend delete error: {e}")
                self.cache_stats["backend_failures"] += 1
        
        return success
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        success = True
        
        # Clear primary backend
        try:
            await self.primary_backend.clear()
        except Exception as e:
            logger.error(f"Primary backend clear error: {e}")
            success = False
            self.cache_stats["backend_failures"] += 1
        
        # Clear fallback backend
        if self.fallback_backend:
            try:
                await self.fallback_backend.clear()
            except Exception as e:
                logger.error(f"Fallback backend clear error: {e}")
                success = False
                self.cache_stats["backend_failures"] += 1
        
        return success
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache"""
        # Check primary backend
        try:
            if await self.primary_backend.exists(key):
                return True
        except Exception as e:
            logger.error(f"Primary backend exists error: {e}")
            self.cache_stats["backend_failures"] += 1
        
        # Check fallback backend
        if self.fallback_backend:
            try:
                if await self.fallback_backend.exists(key):
                    return True
            except Exception as e:
                logger.error(f"Fallback backend exists error: {e}")
                self.cache_stats["backend_failures"] += 1
        
        return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        primary_stats = await self.primary_backend.get_stats()
        fallback_stats = await self.fallback_backend.get_stats() if self.fallback_backend else None
        
        total_requests = self.cache_stats["total_requests"]
        hit_rate = (self.cache_stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "primary_backend": primary_stats,
            "fallback_backend": fallback_stats,
            "total_requests": total_requests,
            "cache_hits": self.cache_stats["cache_hits"],
            "cache_misses": self.cache_stats["cache_misses"],
            "hit_rate": hit_rate,
            "backend_failures": self.cache_stats["backend_failures"]
        }
    
    def cache_result(self, ttl: Optional[int] = None, tags: List[str] = None, key_generator: str = None):
        """Decorator to cache function results"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_generator and key_generator in self.key_generators:
                    cache_key = self.key_generators[key_generator](func.__name__, *args, **kwargs)
                else:
                    cache_key = self.generate_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                await self.set(cache_key, result, ttl, tags)
                
                return result
            return wrapper
        return decorator

# Global cache instance
advanced_cache = AdvancedCache()

# Convenience functions
async def get_cache(key: str) -> Optional[Any]:
    """Get a value from cache"""
    return await advanced_cache.get(key)

async def set_cache(key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None) -> bool:
    """Set a value in cache"""
    return await advanced_cache.set(key, value, ttl, tags)

async def delete_cache(key: str) -> bool:
    """Delete a value from cache"""
    return await advanced_cache.delete(key)

async def clear_cache() -> bool:
    """Clear all cache entries"""
    return await advanced_cache.clear()

async def cache_exists(key: str) -> bool:
    """Check if a key exists in cache"""
    return await advanced_cache.exists(key)

async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return await advanced_cache.get_stats()
