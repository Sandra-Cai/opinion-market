"""
Advanced Distributed Caching System
Provides sophisticated caching with multiple backends and intelligent invalidation
"""

import asyncio
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from collections import defaultdict, deque
import pickle
import zlib

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import memcached
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False

logger = logging.getLogger(__name__)

class CacheBackend(Enum):
    """Cache backend types"""
    MEMORY = "memory"
    REDIS = "redis"
    MEMCACHED = "memcached"
    HYBRID = "hybrid"

class CacheStrategy(Enum):
    """Cache invalidation strategies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    WRITE_AROUND = "write_around"

class CacheEvent(Enum):
    """Cache events"""
    HIT = "hit"
    MISS = "miss"
    SET = "set"
    DELETE = "delete"
    INVALIDATE = "invalidate"
    EXPIRE = "expire"

@dataclass
class CacheEntry:
    """Represents a cache entry"""
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl: Optional[timedelta] = None
    hits: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    tags: Set[str] = field(default_factory=set)
    size_bytes: int = 0
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return datetime.utcnow() > (self.timestamp + self.ttl)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl.total_seconds() if self.ttl else None,
            "hits": self.hits,
            "last_accessed": self.last_accessed.isoformat(),
            "tags": list(self.tags),
            "size_bytes": self.size_bytes,
            "compressed": self.compressed,
            "metadata": self.metadata
        }

@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    invalidations: int = 0
    expires: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0

class MemoryCacheBackend:
    """In-memory cache backend"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque()
        self._lock = threading.RLock()
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry"""
        with self._lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.access_order.remove(key)
                self.stats.expires += 1
                return None
            
            # Update access info
            entry.hits += 1
            entry.last_accessed = datetime.utcnow()
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.stats.hits += 1
            return entry
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None, 
                  tags: Optional[Set[str]] = None, compress: bool = False) -> bool:
        """Set cache entry"""
        try:
            # Serialize and compress if needed
            serialized_value = self._serialize_value(value, compress)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=serialized_value,
                ttl=ttl,
                tags=tags or set(),
                compressed=compress,
                size_bytes=len(str(serialized_value))
            )
            
            with self._lock:
                # Check memory limits
                if not self._check_memory_limits(entry):
                    return False
                
                # Remove old entry if exists
                if key in self.cache:
                    old_entry = self.cache[key]
                    self.stats.total_size_bytes -= old_entry.size_bytes
                    if key in self.access_order:
                        self.access_order.remove(key)
                
                # Add new entry
                self.cache[key] = entry
                self.access_order.append(key)
                self.stats.sets += 1
                self.stats.total_size_bytes += entry.size_bytes
                self.stats.entry_count = len(self.cache)
                
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache entry: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                
                self.stats.deletes += 1
                self.stats.total_size_bytes -= entry.size_bytes
                self.stats.entry_count = len(self.cache)
                return True
            
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.stats = CacheStats()
            return True
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats.hits + self.stats.misses
            if total_requests > 0:
                self.stats.hit_rate = self.stats.hits / total_requests
                self.stats.miss_rate = self.stats.misses / total_requests
            
            return self.stats
    
    def _serialize_value(self, value: Any, compress: bool = False) -> Any:
        """Serialize value for storage"""
        try:
            if compress:
                serialized = pickle.dumps(value)
                compressed = zlib.compress(serialized)
                return compressed
            else:
                return value
        except:
            return value
    
    def _deserialize_value(self, value: Any, compressed: bool = False) -> Any:
        """Deserialize value from storage"""
        try:
            if compressed:
                decompressed = zlib.decompress(value)
                return pickle.loads(decompressed)
            else:
                return value
        except:
            return value
    
    def _check_memory_limits(self, entry: CacheEntry) -> bool:
        """Check if adding entry would exceed memory limits"""
        # Check entry count limit
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        # Check memory limit
        if self.stats.total_size_bytes + entry.size_bytes > self.max_memory_bytes:
            self._evict_oldest()
        
        return True
    
    def _evict_oldest(self):
        """Evict oldest cache entry"""
        if self.access_order:
            oldest_key = self.access_order.popleft()
            if oldest_key in self.cache:
                entry = self.cache[oldest_key]
                del self.cache[oldest_key]
                self.stats.total_size_bytes -= entry.size_bytes
                self.stats.entry_count = len(self.cache)

class RedisCacheBackend:
    """Redis cache backend"""
    
    def __init__(self, redis_url: str, key_prefix: str = "cache:", 
                 default_ttl: int = 3600, compress: bool = True):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available")
        
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.compress = compress
        self.redis_client = redis.Redis.from_url(redis_url)
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry from Redis"""
        try:
            full_key = f"{self.key_prefix}{key}"
            data = self.redis_client.get(full_key)
            
            if data is None:
                self.stats.misses += 1
                return None
            
            # Deserialize data
            entry_data = json.loads(data)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=entry_data["value"],
                timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                ttl=timedelta(seconds=entry_data["ttl"]) if entry_data["ttl"] else None,
                hits=entry_data["hits"],
                last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                tags=set(entry_data["tags"]),
                size_bytes=entry_data["size_bytes"],
                compressed=entry_data["compressed"],
                metadata=entry_data["metadata"]
            )
            
            # Update hits
            entry.hits += 1
            entry.last_accessed = datetime.utcnow()
            
            # Update in Redis
            await self.set(key, entry.value, entry.ttl, entry.tags, entry.compressed)
            
            self.stats.hits += 1
            return entry
            
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None, 
                  tags: Optional[Set[str]] = None, compress: bool = None) -> bool:
        """Set cache entry in Redis"""
        try:
            if compress is None:
                compress = self.compress
            
            # Serialize value
            serialized_value = self._serialize_value(value, compress)
            
            # Create entry data
            entry_data = {
                "value": serialized_value,
                "timestamp": datetime.utcnow().isoformat(),
                "ttl": ttl.total_seconds() if ttl else None,
                "hits": 0,
                "last_accessed": datetime.utcnow().isoformat(),
                "tags": list(tags or set()),
                "size_bytes": len(str(serialized_value)),
                "compressed": compress,
                "metadata": {}
            }
            
            # Store in Redis
            full_key = f"{self.key_prefix}{key}"
            ttl_seconds = int(ttl.total_seconds()) if ttl else self.default_ttl
            
            self.redis_client.setex(
                full_key, 
                ttl_seconds, 
                json.dumps(entry_data)
            )
            
            self.stats.sets += 1
            return True
            
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cache entry from Redis"""
        try:
            full_key = f"{self.key_prefix}{key}"
            result = self.redis_client.delete(full_key)
            
            if result:
                self.stats.deletes += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries from Redis"""
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
            
            self.stats = CacheStats()
            return True
            
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        try:
            pattern = f"{self.key_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            self.stats.entry_count = len(keys)
            
            total_requests = self.stats.hits + self.stats.misses
            if total_requests > 0:
                self.stats.hit_rate = self.stats.hits / total_requests
                self.stats.miss_rate = self.stats.misses / total_requests
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error getting Redis cache stats: {e}")
            return self.stats
    
    def _serialize_value(self, value: Any, compress: bool = False) -> Any:
        """Serialize value for Redis storage"""
        try:
            if compress:
                serialized = pickle.dumps(value)
                compressed = zlib.compress(serialized)
                return compressed.hex()  # Convert to hex string for JSON
            else:
                return value
        except:
            return value

class AdvancedDistributedCache:
    """Advanced distributed cache with multiple backends"""
    
    def __init__(self, primary_backend: CacheBackend = CacheBackend.MEMORY, 
                 secondary_backend: Optional[CacheBackend] = None,
                 redis_url: Optional[str] = None):
        self.primary_backend = primary_backend
        self.secondary_backend = secondary_backend
        self.redis_url = redis_url
        
        # Initialize backends
        self.backends: Dict[CacheBackend, Union[MemoryCacheBackend, RedisCacheBackend]] = {}
        self._initialize_backends()
        
        # Cache warming and invalidation
        self.cache_warmers: Dict[str, Callable] = {}
        self.invalidation_rules: Dict[str, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    def _initialize_backends(self):
        """Initialize cache backends"""
        # Primary backend
        if self.primary_backend == CacheBackend.MEMORY:
            self.backends[CacheBackend.MEMORY] = MemoryCacheBackend()
        elif self.primary_backend == CacheBackend.REDIS and self.redis_url:
            self.backends[CacheBackend.REDIS] = RedisCacheBackend(self.redis_url)
        
        # Secondary backend
        if self.secondary_backend:
            if self.secondary_backend == CacheBackend.MEMORY:
                self.backends[CacheBackend.MEMORY] = MemoryCacheBackend()
            elif self.secondary_backend == CacheBackend.REDIS and self.redis_url:
                self.backends[CacheBackend.REDIS] = RedisCacheBackend(self.redis_url)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            # Try primary backend first
            primary_backend = self.backends.get(self.primary_backend)
            if primary_backend:
                entry = await primary_backend.get(key)
                if entry:
                    self.stats.hits += 1
                    return entry.value
            
            # Try secondary backend if available
            if self.secondary_backend:
                secondary_backend = self.backends.get(self.secondary_backend)
                if secondary_backend:
                    entry = await secondary_backend.get(key)
                    if entry:
                        # Promote to primary backend
                        await self.set(key, entry.value, entry.ttl, entry.tags)
                        self.stats.hits += 1
                        return entry.value
            
            self.stats.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None, 
                  tags: Optional[Set[str]] = None, compress: bool = False) -> bool:
        """Set value in cache"""
        try:
            success = True
            
            # Set in primary backend
            primary_backend = self.backends.get(self.primary_backend)
            if primary_backend:
                success &= await primary_backend.set(key, value, ttl, tags, compress)
            
            # Set in secondary backend if available
            if self.secondary_backend:
                secondary_backend = self.backends.get(self.secondary_backend)
                if secondary_backend:
                    success &= await secondary_backend.set(key, value, ttl, tags, compress)
            
            if success:
                self.stats.sets += 1
                
                # Update tag index
                if tags:
                    for tag in tags:
                        self.tag_index[tag].add(key)
            
            return success
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            success = True
            
            # Delete from primary backend
            primary_backend = self.backends.get(self.primary_backend)
            if primary_backend:
                success &= await primary_backend.delete(key)
            
            # Delete from secondary backend if available
            if self.secondary_backend:
                secondary_backend = self.backends.get(self.secondary_backend)
                if secondary_backend:
                    success &= await secondary_backend.delete(key)
            
            if success:
                self.stats.deletes += 1
                
                # Remove from tag index
                for tag, keys in self.tag_index.items():
                    keys.discard(key)
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            success = True
            
            for backend in self.backends.values():
                success &= await backend.clear()
            
            if success:
                self.stats = CacheStats()
                self.tag_index.clear()
            
            return success
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate cache entries by tags"""
        try:
            keys_to_invalidate = set()
            
            for tag in tags:
                if tag in self.tag_index:
                    keys_to_invalidate.update(self.tag_index[tag])
            
            invalidated_count = 0
            for key in keys_to_invalidate:
                if await self.delete(key):
                    invalidated_count += 1
            
            self.stats.invalidations += invalidated_count
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Error invalidating by tags: {e}")
            return 0
    
    async def warm_cache(self, warmer_name: str, *args, **kwargs) -> bool:
        """Warm cache using registered warmer"""
        try:
            if warmer_name not in self.cache_warmers:
                logger.warning(f"Cache warmer '{warmer_name}' not found")
                return False
            
            warmer_func = self.cache_warmers[warmer_name]
            result = await warmer_func(*args, **kwargs)
            
            logger.info(f"Cache warmed using '{warmer_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Error warming cache: {e}")
            return False
    
    def register_cache_warmer(self, name: str, warmer_func: Callable):
        """Register a cache warmer function"""
        self.cache_warmers[name] = warmer_func
        logger.info(f"Registered cache warmer: {name}")
    
    def add_invalidation_rule(self, pattern: str, tags: List[str]):
        """Add cache invalidation rule"""
        self.invalidation_rules[pattern] = tags
        logger.info(f"Added invalidation rule: {pattern} -> {tags}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            backend_stats = {}
            
            for backend_type, backend in self.backends.items():
                stats = await backend.get_stats()
                backend_stats[backend_type.value] = {
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "sets": stats.sets,
                    "deletes": stats.deletes,
                    "hit_rate": stats.hit_rate,
                    "miss_rate": stats.miss_rate,
                    "entry_count": stats.entry_count,
                    "total_size_bytes": stats.total_size_bytes
                }
            
            return {
                "overall": {
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "sets": self.stats.sets,
                    "deletes": self.stats.deletes,
                    "invalidations": self.stats.invalidations,
                    "hit_rate": self.stats.hit_rate,
                    "miss_rate": self.stats.miss_rate
                },
                "backends": backend_stats,
                "tag_index_size": len(self.tag_index),
                "cache_warmers": list(self.cache_warmers.keys()),
                "invalidation_rules": len(self.invalidation_rules)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

# Global distributed cache instance
distributed_cache = AdvancedDistributedCache()

# Convenience functions
async def cache_get(key: str) -> Optional[Any]:
    """Get value from distributed cache"""
    return await distributed_cache.get(key)

async def cache_set(key: str, value: Any, ttl: Optional[timedelta] = None, 
                   tags: Optional[Set[str]] = None, compress: bool = False) -> bool:
    """Set value in distributed cache"""
    return await distributed_cache.set(key, value, ttl, tags, compress)

async def cache_delete(key: str) -> bool:
    """Delete value from distributed cache"""
    return await distributed_cache.delete(key)

async def cache_clear() -> bool:
    """Clear all cache entries"""
    return await distributed_cache.clear()

async def cache_invalidate_by_tags(tags: Set[str]) -> int:
    """Invalidate cache entries by tags"""
    return await distributed_cache.invalidate_by_tags(tags)

def register_cache_warmer(name: str, warmer_func: Callable):
    """Register a cache warmer function"""
    distributed_cache.register_cache_warmer(name, warmer_func)

def add_invalidation_rule(pattern: str, tags: List[str]):
    """Add cache invalidation rule"""
    distributed_cache.add_invalidation_rule(pattern, tags)
