"""
Enhanced Caching System with Intelligent Cache Management
Provides multi-level caching with automatic invalidation and optimization
"""

import asyncio
import logging
import json
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry data structure"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    size_bytes: int = 0


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0


class EnhancedCache:
    """Enhanced caching system with intelligent management"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.RLock()
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_interval = 300  # 5 minutes
        
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
                logger.debug(f"LRU cache entry evicted: {key}")
                
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
            
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.expires_at and entry.expires_at <= datetime.now():
                    del self.cache[key]
                    self.stats.misses += 1
                    self.stats.total_size -= entry.size_bytes
                    return None
                    
                # Update access info
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                self.stats.hits += 1
                return entry.value
            else:
                self.stats.misses += 1
                return None
                
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, tags: List[str] = None) -> bool:
        """Set value in cache"""
        try:
            expires_at = None
            if ttl is not None:
                expires_at = datetime.now() + timedelta(seconds=ttl)
            elif self.default_ttl:
                expires_at = datetime.now() + timedelta(seconds=self.default_ttl)
                
            size_bytes = self._calculate_size(value)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                tags=tags or [],
                size_bytes=size_bytes
            )
            
            with self.lock:
                # Remove existing entry if it exists
                if key in self.cache:
                    old_entry = self.cache[key]
                    self.stats.total_size -= old_entry.size_bytes
                    
                self.cache[key] = entry
                self.stats.total_size += size_bytes
                self.stats.entry_count = len(self.cache)
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache entry {key}: {e}")
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
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats.hits + self.stats.misses
            hit_rate = (self.stats.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "evictions": self.stats.evictions,
                "hit_rate": round(hit_rate, 2),
                "total_size": self.stats.total_size,
                "entry_count": self.stats.entry_count,
                "max_size": self.max_size,
                "utilization": round((self.stats.entry_count / self.max_size) * 100, 2)
            }
            
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


# Global cache instance
enhanced_cache = EnhancedCache(max_size=2000, default_ttl=3600)
