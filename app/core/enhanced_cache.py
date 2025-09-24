"""
Enhanced Caching System with Intelligent Cache Management
Provides multi-level caching with automatic invalidation and optimization

Features:
- LRU eviction with intelligent policies
- Tag-based invalidation
- Compression for memory optimization
- Distributed cache clustering support
- Advanced analytics and monitoring
- Performance benchmarking
- Smart eviction algorithms
"""

import asyncio
import logging
import json
import hashlib
import pickle
import zlib
import gzip
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import threading
import time
import statistics
from enum import Enum
import traceback
import functools
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """Base exception for cache operations"""
    pass


class CacheCompressionError(CacheError):
    """Exception raised during compression/decompression"""
    pass


class CacheSerializationError(CacheError):
    """Exception raised during serialization/deserialization"""
    pass


class CacheMemoryError(CacheError):
    """Exception raised when memory limits are exceeded"""
    pass


def cache_operation_retry(max_retries: int = 3, delay: float = 0.1):
    """Decorator for retrying cache operations with exponential backoff"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (CacheCompressionError, CacheSerializationError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Cache operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Cache operation failed after {max_retries + 1} attempts: {e}")
                        raise
                except Exception as e:
                    # Don't retry on other exceptions
                    logger.error(f"Non-retryable cache error: {e}")
                    raise
            
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def safe_cache_operation(func):
    """Decorator for safe cache operations with error handling"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except CacheError as e:
            logger.error(f"Cache error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}\n{traceback.format_exc()}")
            raise CacheError(f"Unexpected error in {func.__name__}: {e}") from e
    return wrapper


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based
    HYBRID = "hybrid"  # Combination of policies


class CompressionLevel(Enum):
    """Compression levels for cache entries"""
    NONE = 0
    FAST = 1
    DEFAULT = 6
    MAX = 9


@dataclass
class CacheAnalytics:
    """Advanced cache analytics"""
    total_requests: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_rate: float = 0.0
    compression_ratio: float = 0.0
    average_access_time: float = 0.0
    memory_efficiency: float = 0.0
    cache_efficiency_score: float = 0.0
    top_keys: List[Tuple[str, int]] = field(default_factory=list)
    access_patterns: Dict[str, int] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Enhanced cache entry data structure"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    size_bytes: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    priority: int = 0  # Higher priority = less likely to be evicted
    cost: float = 0.0  # Cost to recreate this entry
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Enhanced cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    compressed_size: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    compression_ratio: float = 0.0
    memory_efficiency: float = 0.0
    average_access_time: float = 0.0
    total_requests: int = 0
    cache_efficiency_score: float = 0.0


class EnhancedCache:
    """Enhanced caching system with intelligent management"""
    
    def __init__(
        self, 
        max_size: int = 1000, 
        default_ttl: int = 3600,
        eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID,
        compression_level: CompressionLevel = CompressionLevel.DEFAULT,
        enable_analytics: bool = True,
        max_memory_mb: int = 100
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self.compression_level = compression_level
        self.enable_analytics = enable_analytics
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.analytics = CacheAnalytics() if enable_analytics else None
        self.lock = threading.RLock()
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = 300  # 5 minutes
        
        # Performance tracking
        self.access_times: List[float] = []
        self.compression_stats = {"total_compressed": 0, "total_original": 0}
        
        # Clustering support
        self.cluster_nodes: Dict[str, Any] = {}
        self.is_clustered = False
        
    async def start_cleanup(self):
        """Start automatic cache cleanup"""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Cache cleanup started")
            
    async def stop_cleanup(self):
        """Stop automatic cache cleanup"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
            logger.info("Cache cleanup stopped")
            
    async def _cleanup_loop(self):
        """Automatic cleanup loop"""
        while True:
            try:
                await self._cleanup_expired()
                await self._cleanup_lru()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(self.cleanup_interval)
                
    async def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        with self.lock:
            for key, entry in self.cache.items():
                if entry.expires_at and entry.expires_at <= current_time:
                    expired_keys.append(key)
                    
        for key in expired_keys:
            await self.delete(key)
            logger.debug(f"Expired cache entry removed: {key}")
            
    async def _cleanup_lru(self):
        """Remove least recently used entries if cache is full"""
        with self.lock:
            while len(self.cache) > self.max_size:
                # Remove least recently used entry
                key, entry = self.cache.popitem(last=False)
                self.stats.evictions += 1
                self.stats.total_size -= entry.size_bytes
                self.stats.compressed_size -= entry.compressed_size
                logger.debug(f"LRU cache entry evicted: {key}")
    
    async def _evict_entries_by_memory(self):
        """Evict entries based on memory usage and eviction policy"""
        with self.lock:
            while self.stats.compressed_size > self.max_memory_bytes and self.cache:
                if self.eviction_policy == EvictionPolicy.LRU:
                    key, entry = self.cache.popitem(last=False)
                elif self.eviction_policy == EvictionPolicy.LFU:
                    # Find least frequently used entry
                    key, entry = min(self.cache.items(), key=lambda x: x[1].access_count)
                    del self.cache[key]
                elif self.eviction_policy == EvictionPolicy.SIZE:
                    # Find largest entry
                    key, entry = max(self.cache.items(), key=lambda x: x[1].compressed_size)
                    del self.cache[key]
                elif self.eviction_policy == EvictionPolicy.HYBRID:
                    # Hybrid: prioritize by cost/access ratio, then by size
                    def eviction_score(item):
                        entry = item[1]
                        if entry.access_count == 0:
                            return float('inf')  # Never accessed, evict first
                        cost_per_access = entry.cost / entry.access_count
                        size_factor = entry.compressed_size / 1024  # Normalize size
                        return cost_per_access / size_factor
                    
                    key, entry = min(self.cache.items(), key=eviction_score)
                    del self.cache[key]
                else:
                    # Default to LRU
                    key, entry = self.cache.popitem(last=False)
                
                self.stats.evictions += 1
                self.stats.total_size -= entry.size_bytes
                self.stats.compressed_size -= entry.compressed_size
                logger.debug(f"Memory-based eviction: {key} (policy: {self.eviction_policy.value})")
                
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of cached value"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size estimate
    
    @cache_operation_retry(max_retries=2)
    def _compress_value(self, value: Any) -> Tuple[bytes, float]:
        """Compress cache value and return compressed data with ratio"""
        try:
            # Serialize the value
            if isinstance(value, (str, bytes)):
                data = value.encode('utf-8') if isinstance(value, str) else value
            else:
                try:
                    data = pickle.dumps(value)
                except (pickle.PicklingError, TypeError) as e:
                    raise CacheSerializationError(f"Failed to serialize value: {e}") from e
            
            original_size = len(data)
            
            if self.compression_level == CompressionLevel.NONE or original_size < 1024:
                return data, 1.0
            
            # Compress using gzip
            try:
                compressed = gzip.compress(data, compresslevel=self.compression_level.value)
                compressed_size = len(compressed)
                ratio = compressed_size / original_size if original_size > 0 else 1.0
                
                # Only use compression if it actually saves space
                if ratio < 0.9:  # At least 10% savings
                    return compressed, ratio
                else:
                    return data, 1.0
            except Exception as e:
                raise CacheCompressionError(f"Failed to compress data: {e}") from e
                
        except (CacheSerializationError, CacheCompressionError):
            raise
        except Exception as e:
            logger.warning(f"Unexpected compression error: {e}")
            try:
                return pickle.dumps(value), 1.0
            except Exception as fallback_e:
                raise CacheSerializationError(f"Fallback serialization failed: {fallback_e}") from fallback_e
    
    @cache_operation_retry(max_retries=2)
    def _decompress_value(self, compressed_data: bytes, compression_ratio: float) -> Any:
        """Decompress cache value"""
        try:
            if compression_ratio >= 1.0:
                # Not compressed
                try:
                    return pickle.loads(compressed_data)
                except (pickle.UnpicklingError, TypeError) as e:
                    raise CacheSerializationError(f"Failed to deserialize uncompressed data: {e}") from e
            else:
                # Decompress
                try:
                    decompressed = gzip.decompress(compressed_data)
                    return pickle.loads(decompressed)
                except (gzip.BadGzipFile, OSError) as e:
                    raise CacheCompressionError(f"Failed to decompress data: {e}") from e
                except (pickle.UnpicklingError, TypeError) as e:
                    raise CacheSerializationError(f"Failed to deserialize decompressed data: {e}") from e
        except (CacheSerializationError, CacheCompressionError):
            raise
        except Exception as e:
            logger.error(f"Unexpected decompression error: {e}")
            raise CacheError(f"Decompression failed: {e}") from e
    
    def _update_analytics(self, operation: str, key: str = None, access_time: float = None):
        """Update analytics data"""
        if not self.enable_analytics or not self.analytics:
            return
            
        self.analytics.total_requests += 1
        
        if access_time is not None:
            self.access_times.append(access_time)
            # Keep only last 1000 access times for performance
            if len(self.access_times) > 1000:
                self.access_times = self.access_times[-1000:]
            self.analytics.average_access_time = statistics.mean(self.access_times)
        
        if key:
            self.analytics.access_patterns[key] = self.analytics.access_patterns.get(key, 0) + 1
        
        # Update compression stats
        if self.compression_stats["total_original"] > 0:
            self.analytics.compression_ratio = (
                self.compression_stats["total_compressed"] / 
                self.compression_stats["total_original"]
            )
        
        # Calculate cache efficiency score
        hit_rate = self.stats.hits / max(self.stats.total_requests, 1)
        memory_efficiency = 1.0 - (self.stats.compressed_size / max(self.stats.total_size, 1))
        self.analytics.cache_efficiency_score = (hit_rate * 0.6 + memory_efficiency * 0.4) * 100
            
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    @safe_cache_operation
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with enhanced analytics and error handling"""
        start_time = time.time()
        
        try:
            with self.lock:
                if key in self.cache:
                    entry = self.cache[key]
                    
                    # Check if expired
                    if entry.expires_at and entry.expires_at <= datetime.now():
                        del self.cache[key]
                        self.stats.misses += 1
                        self.stats.total_size -= entry.size_bytes
                        self.stats.compressed_size -= entry.compressed_size
                        self._update_analytics("miss_expired", key, time.time() - start_time)
                        return None
                        
                    # Update access info
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    
                    self.stats.hits += 1
                    self.stats.total_requests += 1
                    
                    # Decompress if needed
                    if entry.compression_ratio < 1.0:
                        try:
                            value = self._decompress_value(entry.value, entry.compression_ratio)
                        except (CacheCompressionError, CacheSerializationError) as e:
                            # Decompression failed, remove corrupted entry
                            logger.error(f"Failed to decompress cache entry {key}: {e}")
                            del self.cache[key]
                            self.stats.misses += 1
                            self.stats.total_size -= entry.size_bytes
                            self.stats.compressed_size -= entry.compressed_size
                            self._update_analytics("miss_decompression_failed", key, time.time() - start_time)
                            return None
                    else:
                        value = entry.value
                    
                    self._update_analytics("hit", key, time.time() - start_time)
                    return value
                else:
                    self.stats.misses += 1
                    self.stats.total_requests += 1
                    self._update_analytics("miss_not_found", key, time.time() - start_time)
                    return None
        except Exception as e:
            logger.error(f"Error getting cache entry {key}: {e}")
            self.stats.misses += 1
            self.stats.total_requests += 1
            self._update_analytics("miss_error", key, time.time() - start_time)
            raise
                
    @safe_cache_operation
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        tags: List[str] = None,
        priority: int = 0,
        cost: float = 0.0,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Set value in cache with enhanced features and error handling"""
        try:
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl:
                expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
            
            # Compress the value with error handling
            try:
                compressed_data, compression_ratio = self._compress_value(value)
            except (CacheCompressionError, CacheSerializationError) as e:
                logger.error(f"Failed to compress value for key {key}: {e}")
                return False
            
            original_size = self._calculate_size(value)
            compressed_size = len(compressed_data)
            
            # Check memory limits before adding
            if compressed_size > self.max_memory_bytes:
                raise CacheMemoryError(f"Entry size {compressed_size} exceeds memory limit {self.max_memory_bytes}")
            
            # Update compression stats
            self.compression_stats["total_original"] += original_size
            self.compression_stats["total_compressed"] += compressed_size
            
            entry = CacheEntry(
                key=key,
                value=compressed_data,
                created_at=datetime.now(),
                expires_at=expires_at,
                tags=tags or [],
                size_bytes=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                priority=priority,
                cost=cost,
                metadata=metadata or {}
            )
            
            with self.lock:
                # Remove existing entry if it exists
                if key in self.cache:
                    old_entry = self.cache[key]
                    self.stats.total_size -= old_entry.size_bytes
                    self.stats.compressed_size -= old_entry.compressed_size
                    
                self.cache[key] = entry
                self.stats.total_size += original_size
                self.stats.compressed_size += compressed_size
                self.stats.entry_count = len(self.cache)
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                # Check memory limits and evict if necessary
                if self.stats.compressed_size > self.max_memory_bytes:
                    try:
                        await self._evict_entries_by_memory()
                    except Exception as e:
                        logger.error(f"Failed to evict entries: {e}")
                        # Remove the entry we just added if eviction failed
                        del self.cache[key]
                        self.stats.total_size -= original_size
                        self.stats.compressed_size -= compressed_size
                        self.stats.entry_count = len(self.cache)
                        raise CacheMemoryError("Failed to free memory for new entry")
                
            self._update_analytics("set", key)
            return True
            
        except (CacheMemoryError, CacheCompressionError, CacheSerializationError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error setting cache entry {key}: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                del self.cache[key]
                self.stats.total_size -= entry.size_bytes
                self.stats.entry_count = len(self.cache)
                return True
            return False
            
    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete cache entries by tags"""
        deleted_count = 0
        
        with self.lock:
            keys_to_delete = []
            for key, entry in self.cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(key)
                    
        for key in keys_to_delete:
            if await self.delete(key):
                deleted_count += 1
                
        logger.info(f"Deleted {deleted_count} cache entries with tags: {tags}")
        return deleted_count
        
    async def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats.total_size = 0
            self.stats.entry_count = 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced cache statistics"""
        with self.lock:
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = (self.stats.hits / total_requests * 100) if total_requests > 0 else 0
            compression_ratio = (self.stats.compressed_size / self.stats.total_size) if self.stats.total_size > 0 else 1.0
            memory_efficiency = 1.0 - compression_ratio
            
            stats = {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "evictions": self.stats.evictions,
                "hit_rate": round(hit_rate, 2),
                "total_size": self.stats.total_size,
                "compressed_size": self.stats.compressed_size,
                "entry_count": self.stats.entry_count,
                "max_size": self.max_size,
                "utilization": round((self.stats.entry_count / self.max_size) * 100, 2),
                "compression_ratio": round(compression_ratio, 3),
                "memory_efficiency": round(memory_efficiency * 100, 2),
                "memory_usage_mb": round(self.stats.compressed_size / (1024 * 1024), 2),
                "max_memory_mb": self.max_memory_mb,
                "eviction_policy": self.eviction_policy.value,
                "compression_level": self.compression_level.value
            }
            
            # Add analytics if enabled
            if self.enable_analytics and self.analytics:
                stats.update({
                    "average_access_time_ms": round(self.analytics.average_access_time * 1000, 2),
                    "cache_efficiency_score": round(self.analytics.cache_efficiency_score, 2),
                    "total_requests": self.analytics.total_requests,
                    "top_keys": self.analytics.top_keys[:10],  # Top 10 most accessed keys
                    "access_patterns_count": len(self.analytics.access_patterns)
                })
            
            return stats
            
    def get_entries_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get cache entries by tag"""
        with self.lock:
            entries = []
            for key, entry in self.cache.items():
                if tag in entry.tags:
                    entries.append({
                        "key": key,
                        "created_at": entry.created_at.isoformat(),
                        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                        "access_count": entry.access_count,
                        "last_accessed": entry.last_accessed.isoformat(),
                        "size_bytes": entry.size_bytes
                    })
            return entries
            
    def cache_decorator(self, ttl: Optional[int] = None, key_prefix: str = "", tags: List[str] = None):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(key_prefix or func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                    
                # Execute function and cache result
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                await self.set(cache_key, result, ttl, tags)
                return result
                
            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(key_prefix or func.__name__, *args, **kwargs)
                
                # Try to get from cache (sync version)
                import asyncio
                loop = asyncio.get_event_loop()
                cached_result = loop.run_until_complete(self.get(cache_key))
                if cached_result is not None:
                    return cached_result
                    
                # Execute function and cache result
                result = func(*args, **kwargs)
                loop.run_until_complete(self.set(cache_key, result, ttl, tags))
                return result
                
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def get_analytics(self) -> Optional[CacheAnalytics]:
        """Get detailed cache analytics"""
        if not self.enable_analytics or not self.analytics:
            return None
        
        with self.lock:
            # Update top keys
            self.analytics.top_keys = sorted(
                self.analytics.access_patterns.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]
            
            # Calculate final metrics
            total_requests = self.stats.hits + self.stats.misses
            self.analytics.hit_rate = (self.stats.hits / total_requests * 100) if total_requests > 0 else 0
            self.analytics.miss_rate = (self.stats.misses / total_requests * 100) if total_requests > 0 else 0
            self.analytics.eviction_rate = (self.stats.evictions / max(self.stats.entry_count, 1)) * 100
            
            return self.analytics
    
    async def benchmark_performance(self, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark cache performance"""
        import random
        import string
        
        def generate_random_data(size: int = 1024) -> str:
            return ''.join(random.choices(string.ascii_letters + string.digits, k=size))
        
        # Benchmark set operations
        set_times = []
        for _ in range(iterations):
            key = f"benchmark_{random.randint(1, 10000)}"
            value = generate_random_data(random.randint(100, 5000))
            
            start_time = time.time()
            await self.set(key, value, ttl=300)
            set_times.append(time.time() - start_time)
        
        # Benchmark get operations
        get_times = []
        for _ in range(iterations):
            key = f"benchmark_{random.randint(1, 10000)}"
            
            start_time = time.time()
            await self.get(key)
            get_times.append(time.time() - start_time)
        
        return {
            "set_operations": {
                "count": len(set_times),
                "average_time_ms": round(statistics.mean(set_times) * 1000, 3),
                "min_time_ms": round(min(set_times) * 1000, 3),
                "max_time_ms": round(max(set_times) * 1000, 3),
                "operations_per_second": round(1 / statistics.mean(set_times), 0)
            },
            "get_operations": {
                "count": len(get_times),
                "average_time_ms": round(statistics.mean(get_times) * 1000, 3),
                "min_time_ms": round(min(get_times) * 1000, 3),
                "max_time_ms": round(max(get_times) * 1000, 3),
                "operations_per_second": round(1 / statistics.mean(get_times), 0)
            }
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get detailed memory usage information"""
        with self.lock:
            return {
                "total_entries": len(self.cache),
                "total_size_bytes": self.stats.total_size,
                "compressed_size_bytes": self.stats.compressed_size,
                "compression_savings_bytes": self.stats.total_size - self.stats.compressed_size,
                "compression_ratio": round(self.stats.compressed_size / max(self.stats.total_size, 1), 3),
                "memory_usage_mb": round(self.stats.compressed_size / (1024 * 1024), 2),
                "max_memory_mb": self.max_memory_mb,
                "memory_utilization_percent": round((self.stats.compressed_size / self.max_memory_bytes) * 100, 2),
                "average_entry_size_bytes": round(self.stats.total_size / max(len(self.cache), 1), 2),
                "average_compressed_size_bytes": round(self.stats.compressed_size / max(len(self.cache), 1), 2)
            }
    
    async def warm_up(self, warm_up_data: Dict[str, Any]) -> Dict[str, Any]:
        """Warm up cache with predefined data"""
        results = {"success": 0, "failed": 0, "total_size": 0}
        
        for key, value in warm_up_data.items():
            try:
                success = await self.set(key, value, ttl=3600)
                if success:
                    results["success"] += 1
                    results["total_size"] += self._calculate_size(value)
                else:
                    results["failed"] += 1
            except Exception as e:
                logger.error(f"Failed to warm up cache with key {key}: {e}")
                results["failed"] += 1
        
        return results
    
    def export_cache_data(self) -> Dict[str, Any]:
        """Export cache data for analysis"""
        with self.lock:
            return {
                "cache_entries": {
                    key: {
                        "created_at": entry.created_at.isoformat(),
                        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                        "access_count": entry.access_count,
                        "last_accessed": entry.last_accessed.isoformat(),
                        "tags": entry.tags,
                        "size_bytes": entry.size_bytes,
                        "compressed_size": entry.compressed_size,
                        "compression_ratio": entry.compression_ratio,
                        "priority": entry.priority,
                        "cost": entry.cost,
                        "metadata": entry.metadata
                    }
                    for key, entry in self.cache.items()
                },
                "statistics": self.get_stats(),
                "analytics": self.get_analytics().__dict__ if self.analytics else None,
                "memory_usage": self.get_memory_usage()
            }
    
    async def recover_from_errors(self) -> Dict[str, Any]:
        """Recover from cache errors and clean up corrupted entries"""
        recovery_stats = {
            "corrupted_entries_removed": 0,
            "memory_freed_bytes": 0,
            "recovery_errors": [],
            "success": True
        }
        
        try:
            with self.lock:
                corrupted_keys = []
                
                # Check each entry for corruption
                for key, entry in list(self.cache.items()):
                    try:
                        # Try to decompress/deserialize the entry
                        if entry.compression_ratio < 1.0:
                            self._decompress_value(entry.value, entry.compression_ratio)
                        else:
                            pickle.loads(entry.value)
                    except (CacheCompressionError, CacheSerializationError, Exception) as e:
                        logger.warning(f"Found corrupted cache entry {key}: {e}")
                        corrupted_keys.append(key)
                        recovery_stats["corrupted_entries_removed"] += 1
                        recovery_stats["memory_freed_bytes"] += entry.compressed_size
                
                # Remove corrupted entries
                for key in corrupted_keys:
                    entry = self.cache.pop(key, None)
                    if entry:
                        self.stats.total_size -= entry.size_bytes
                        self.stats.compressed_size -= entry.compressed_size
                        self.stats.entry_count = len(self.cache)
                
                # Reset compression stats if they seem inconsistent
                if self.stats.compressed_size < 0 or self.stats.total_size < 0:
                    logger.warning("Resetting inconsistent cache statistics")
                    self.stats.total_size = sum(entry.size_bytes for entry in self.cache.values())
                    self.stats.compressed_size = sum(entry.compressed_size for entry in self.cache.values())
                    self.stats.entry_count = len(self.cache)
                
                logger.info(f"Cache recovery completed: removed {recovery_stats['corrupted_entries_removed']} corrupted entries")
                
        except Exception as e:
            logger.error(f"Error during cache recovery: {e}")
            recovery_stats["recovery_errors"].append(str(e))
            recovery_stats["success"] = False
        
        return recovery_stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for the cache system"""
        health_status = {
            "status": "healthy",
            "checks": {},
            "timestamp": time.time(),
            "errors": []
        }
        
        try:
            # Test basic operations
            test_key = "health_check_test"
            test_value = {"test": True, "timestamp": time.time()}
            
            # Test set operation
            try:
                set_success = await self.set(test_key, test_value, ttl=60)
                health_status["checks"]["set_operation"] = set_success
            except Exception as e:
                health_status["checks"]["set_operation"] = False
                health_status["errors"].append(f"Set operation failed: {e}")
            
            # Test get operation
            try:
                retrieved_value = await self.get(test_key)
                health_status["checks"]["get_operation"] = retrieved_value is not None
            except Exception as e:
                health_status["checks"]["get_operation"] = False
                health_status["errors"].append(f"Get operation failed: {e}")
            
            # Test delete operation
            try:
                delete_success = await self.delete(test_key)
                health_status["checks"]["delete_operation"] = delete_success
            except Exception as e:
                health_status["checks"]["delete_operation"] = False
                health_status["errors"].append(f"Delete operation failed: {e}")
            
            # Check memory usage
            memory_info = self.get_memory_usage()
            memory_usage_percent = memory_info["memory_utilization_percent"]
            health_status["checks"]["memory_usage"] = memory_usage_percent < 95  # Healthy if < 95%
            
            if memory_usage_percent >= 95:
                health_status["errors"].append(f"High memory usage: {memory_usage_percent}%")
            
            # Check for corrupted entries
            try:
                recovery_stats = await self.recover_from_errors()
                health_status["checks"]["corruption_check"] = recovery_stats["success"]
                if recovery_stats["corrupted_entries_removed"] > 0:
                    health_status["errors"].append(f"Found {recovery_stats['corrupted_entries_removed']} corrupted entries")
            except Exception as e:
                health_status["checks"]["corruption_check"] = False
                health_status["errors"].append(f"Corruption check failed: {e}")
            
            # Overall health status
            all_checks_passed = all(health_status["checks"].values())
            if not all_checks_passed or health_status["errors"]:
                health_status["status"] = "unhealthy"
            
        except Exception as e:
            health_status["status"] = "critical"
            health_status["errors"].append(f"Health check failed: {e}")
        
        return health_status


# Global cache instance with enhanced features
enhanced_cache = EnhancedCache(
    max_size=2000, 
    default_ttl=3600,
    eviction_policy=EvictionPolicy.HYBRID,
    compression_level=CompressionLevel.DEFAULT,
    enable_analytics=True,
    max_memory_mb=100
)
