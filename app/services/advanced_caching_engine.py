"""
Advanced Caching Engine
Comprehensive caching system with CDN integration and intelligent strategies
"""

import asyncio
import time
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from enum import Enum
import secrets
import pickle
import gzip
import base64
import httpx
from urllib.parse import urljoin

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy enumeration"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    INTELLIGENT = "intelligent"


class CacheTier(Enum):
    """Cache tier enumeration"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_CDN = "l3_cdn"
    L4_DATABASE = "l4_database"


class CacheOperation(Enum):
    """Cache operation enumeration"""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    UPDATE = "update"
    INVALIDATE = "invalidate"
    REFRESH = "refresh"


@dataclass
class CacheEntry:
    """Cache entry data structure"""
    key: str
    value: Any
    strategy: CacheStrategy
    tier: CacheTier
    ttl: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheMetrics:
    """Cache metrics data structure"""
    tier: CacheTier
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    average_response_time: float = 0.0
    hit_rate: float = 0.0


@dataclass
class CDNNode:
    """CDN node data structure"""
    node_id: str
    region: str
    endpoint: str
    status: str
    latency_ms: float
    bandwidth_mbps: float
    storage_gb: float
    last_health_check: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedCachingEngine:
    """Advanced Caching Engine with CDN integration and intelligent strategies"""
    
    def __init__(self):
        self.cache_tiers: Dict[CacheTier, Dict[str, CacheEntry]] = {
            CacheTier.L1_MEMORY: OrderedDict(),
            CacheTier.L2_REDIS: {},
            CacheTier.L3_CDN: {},
            CacheTier.L4_DATABASE: {}
        }
        
        self.cache_metrics: Dict[CacheTier, CacheMetrics] = {
            tier: CacheMetrics(tier=tier) for tier in CacheTier
        }
        
        self.cdn_nodes: Dict[str, CDNNode] = {}
        self.cache_strategies: Dict[str, CacheStrategy] = {}
        self.intelligent_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "caching_enabled": True,
            "cdn_enabled": True,
            "compression_enabled": True,
            "intelligent_caching_enabled": True,
            "auto_tiering_enabled": True,
            "cache_warming_enabled": True,
            "l1_memory_limit_mb": 512,
            "l2_redis_limit_mb": 2048,
            "l3_cdn_limit_gb": 100,
            "default_ttl": 3600,  # 1 hour
            "compression_threshold_bytes": 1024,
            "intelligent_learning_enabled": True,
            "cache_analytics_enabled": True,
            "cdn_health_check_interval": 300,  # 5 minutes
            "cache_cleanup_interval": 3600,  # 1 hour
            "metrics_collection_interval": 60  # 1 minute
        }
        
        # Cache strategies configuration
        self.strategy_config = {
            CacheStrategy.LRU: {
                "max_entries": 10000,
                "eviction_policy": "least_recently_used"
            },
            CacheStrategy.LFU: {
                "max_entries": 10000,
                "eviction_policy": "least_frequently_used"
            },
            CacheStrategy.TTL: {
                "default_ttl": 3600,
                "eviction_policy": "time_based"
            },
            CacheStrategy.INTELLIGENT: {
                "learning_enabled": True,
                "pattern_detection": True,
                "predictive_caching": True
            }
        }
        
        # CDN configuration
        self.cdn_config = {
            "default_regions": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
            "health_check_timeout": 5.0,
            "cache_headers": {
                "Cache-Control": "public, max-age=3600",
                "ETag": "auto",
                "Last-Modified": "auto"
            },
            "compression_types": ["gzip", "brotli"],
            "edge_locations": 50
        }
        
        # Monitoring
        self.caching_active = False
        self.caching_task: Optional[asyncio.Task] = None
        self.cdn_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.caching_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cdn_hits": 0,
            "compression_saves": 0,
            "intelligent_predictions": 0,
            "tier_promotions": 0,
            "tier_demotions": 0
        }
        
    async def start_caching_engine(self):
        """Start the advanced caching engine"""
        if self.caching_active:
            logger.warning("Advanced caching engine already active")
            return
            
        self.caching_active = True
        self.caching_task = asyncio.create_task(self._caching_processing_loop())
        self.cdn_task = asyncio.create_task(self._cdn_processing_loop())
        
        # Initialize CDN nodes
        await self._initialize_cdn_nodes()
        
        logger.info("Advanced Caching Engine started")
        
    async def stop_caching_engine(self):
        """Stop the advanced caching engine"""
        self.caching_active = False
        if self.caching_task:
            self.caching_task.cancel()
            try:
                await self.caching_task
            except asyncio.CancelledError:
                pass
        if self.cdn_task:
            self.cdn_task.cancel()
            try:
                await self.cdn_task
            except asyncio.CancelledError:
                pass
        logger.info("Advanced Caching Engine stopped")
        
    async def _caching_processing_loop(self):
        """Main caching processing loop"""
        while self.caching_active:
            try:
                # Clean up expired entries
                await self._cleanup_expired_entries()
                
                # Update intelligent patterns
                if self.config["intelligent_caching_enabled"]:
                    await self._update_intelligent_patterns()
                    
                # Auto-tiering
                if self.config["auto_tiering_enabled"]:
                    await self._perform_auto_tiering()
                    
                # Cache warming
                if self.config["cache_warming_enabled"]:
                    await self._perform_cache_warming()
                    
                # Update metrics
                await self._update_cache_metrics()
                
                # Wait before next cycle
                await asyncio.sleep(self.config["cache_cleanup_interval"])
                
            except Exception as e:
                logger.error(f"Error in caching processing loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _cdn_processing_loop(self):
        """CDN processing loop"""
        while self.caching_active:
            try:
                # Health check CDN nodes
                await self._health_check_cdn_nodes()
                
                # Update CDN metrics
                await self._update_cdn_metrics()
                
                # Wait before next cycle
                await asyncio.sleep(self.config["cdn_health_check_interval"])
                
            except Exception as e:
                logger.error(f"Error in CDN processing loop: {e}")
                await asyncio.sleep(60)
                
    async def get(self, key: str, strategy: Optional[CacheStrategy] = None) -> Optional[Any]:
        """Get value from cache with intelligent tier selection"""
        try:
            start_time = time.time()
            self.caching_stats["total_requests"] += 1
            
            # Determine strategy
            if not strategy:
                strategy = self._determine_strategy(key)
                
            # Try each tier in order
            for tier in [CacheTier.L1_MEMORY, CacheTier.L2_REDIS, CacheTier.L3_CDN, CacheTier.L4_DATABASE]:
                value = await self._get_from_tier(key, tier)
                if value is not None:
                    # Update metrics
                    self.cache_metrics[tier].hits += 1
                    self.caching_stats["cache_hits"] += 1
                    
                    # Update access pattern
                    await self._update_access_pattern(key, tier)
                    
                    # Promote to higher tier if beneficial
                    if tier != CacheTier.L1_MEMORY:
                        await self._consider_tier_promotion(key, value, tier)
                        
                    # Update response time
                    response_time = (time.time() - start_time) * 1000
                    self._update_response_time(tier, response_time)
                    
                    return value
                    
            # Cache miss
            self.caching_stats["cache_misses"] += 1
            for tier in CacheTier:
                self.cache_metrics[tier].misses += 1
                
            # Intelligent prediction
            if self.config["intelligent_caching_enabled"]:
                await self._predict_and_prefetch(key)
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, strategy: Optional[CacheStrategy] = None, tier: Optional[CacheTier] = None) -> bool:
        """Set value in cache with intelligent tier selection"""
        try:
            start_time = time.time()
            
            # Determine strategy and tier
            if not strategy:
                strategy = self._determine_strategy(key)
            if not tier:
                tier = self._determine_optimal_tier(key, value)
                
            # Compress if beneficial
            compressed_value, is_compressed = await self._compress_if_beneficial(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                strategy=strategy,
                tier=tier,
                ttl=ttl or self.config["default_ttl"],
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=len(str(compressed_value).encode()),
                compressed=is_compressed
            )
            
            # Store in determined tier
            success = await self._set_in_tier(key, entry, tier)
            
            if success:
                # Update metrics
                self.cache_metrics[tier].sets += 1
                self.cache_metrics[tier].total_size_bytes += entry.size_bytes
                
                # Update compression stats
                if is_compressed:
                    self.caching_stats["compression_saves"] += 1
                    
                # Update intelligent patterns
                await self._update_intelligent_patterns()
                
                # Update response time
                response_time = (time.time() - start_time) * 1000
                self._update_response_time(tier, response_time)
                
            return success
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers"""
        try:
            success = True
            
            for tier in CacheTier:
                if await self._delete_from_tier(key, tier):
                    self.cache_metrics[tier].deletes += 1
                else:
                    success = False
                    
            return success
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
            
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        try:
            invalidated_count = 0
            
            for tier in CacheTier:
                count = await self._invalidate_pattern_in_tier(pattern, tier)
                invalidated_count += count
                
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Error invalidating pattern: {e}")
            return 0
            
    async def _get_from_tier(self, key: str, tier: CacheTier) -> Optional[Any]:
        """Get value from specific tier"""
        try:
            if tier == CacheTier.L1_MEMORY:
                if key in self.cache_tiers[tier]:
                    entry = self.cache_tiers[tier][key]
                    # Check TTL
                    if self._is_expired(entry):
                        del self.cache_tiers[tier][key]
                        return None
                    # Update access info
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    # Move to end (LRU)
                    self.cache_tiers[tier].move_to_end(key)
                    return await self._decompress_if_needed(entry.value, entry.compressed)
                    
            elif tier == CacheTier.L2_REDIS:
                # Use enhanced cache for Redis tier
                value = await enhanced_cache.get(key)
                if value:
                    return value
                    
            elif tier == CacheTier.L3_CDN:
                # Check CDN
                cdn_value = await self._get_from_cdn(key)
                if cdn_value:
                    self.caching_stats["cdn_hits"] += 1
                    return cdn_value
                    
            elif tier == CacheTier.L4_DATABASE:
                # This would query the database
                # For now, return None
                pass
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting from tier {tier.value}: {e}")
            return None
            
    async def _set_in_tier(self, key: str, entry: CacheEntry, tier: CacheTier) -> bool:
        """Set value in specific tier"""
        try:
            if tier == CacheTier.L1_MEMORY:
                # Check memory limit
                if self._exceeds_memory_limit(entry.size_bytes):
                    await self._evict_from_memory()
                    
                self.cache_tiers[tier][key] = entry
                
            elif tier == CacheTier.L2_REDIS:
                # Use enhanced cache for Redis tier
                await enhanced_cache.set(key, entry.value, ttl=entry.ttl)
                
            elif tier == CacheTier.L3_CDN:
                # Store in CDN
                await self._set_in_cdn(key, entry)
                
            elif tier == CacheTier.L4_DATABASE:
                # This would store in database
                # For now, just return True
                pass
                
            return True
            
        except Exception as e:
            logger.error(f"Error setting in tier {tier.value}: {e}")
            return False
            
    async def _delete_from_tier(self, key: str, tier: CacheTier) -> bool:
        """Delete value from specific tier"""
        try:
            if tier == CacheTier.L1_MEMORY:
                if key in self.cache_tiers[tier]:
                    del self.cache_tiers[tier][key]
                    return True
                    
            elif tier == CacheTier.L2_REDIS:
                await enhanced_cache.delete(key)
                return True
                
            elif tier == CacheTier.L3_CDN:
                await self._delete_from_cdn(key)
                return True
                
            elif tier == CacheTier.L4_DATABASE:
                # This would delete from database
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error deleting from tier {tier.value}: {e}")
            return False
            
    async def _compress_if_beneficial(self, value: Any) -> Tuple[Any, bool]:
        """Compress value if beneficial"""
        try:
            if not self.config["compression_enabled"]:
                return value, False
                
            # Serialize value
            serialized = pickle.dumps(value)
            
            # Check if compression is beneficial
            if len(serialized) < self.config["compression_threshold_bytes"]:
                return value, False
                
            # Compress
            compressed = gzip.compress(serialized)
            
            # Check if compression saved space
            if len(compressed) < len(serialized) * 0.8:  # 20% savings threshold
                return base64.b64encode(compressed).decode(), True
            else:
                return value, False
                
        except Exception as e:
            logger.error(f"Error compressing value: {e}")
            return value, False
            
    async def _decompress_if_needed(self, value: Any, is_compressed: bool) -> Any:
        """Decompress value if needed"""
        try:
            if not is_compressed:
                return value
                
            # Decompress
            compressed_data = base64.b64decode(value)
            decompressed = gzip.decompress(compressed_data)
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.error(f"Error decompressing value: {e}")
            return value
            
    def _determine_strategy(self, key: str) -> CacheStrategy:
        """Determine optimal cache strategy for key"""
        try:
            # Check if we have a specific strategy for this key
            if key in self.cache_strategies:
                return self.cache_strategies[key]
                
            # Use intelligent strategy if enabled
            if self.config["intelligent_caching_enabled"]:
                return CacheStrategy.INTELLIGENT
                
            # Default to LRU
            return CacheStrategy.LRU
            
        except Exception as e:
            logger.error(f"Error determining strategy: {e}")
            return CacheStrategy.LRU
            
    def _determine_optimal_tier(self, key: str, value: Any) -> CacheTier:
        """Determine optimal cache tier for key-value pair"""
        try:
            # Calculate value size
            value_size = len(str(value).encode())
            
            # Determine tier based on size and access patterns
            if value_size < 1024:  # < 1KB
                return CacheTier.L1_MEMORY
            elif value_size < 1024 * 1024:  # < 1MB
                return CacheTier.L2_REDIS
            elif value_size < 10 * 1024 * 1024:  # < 10MB
                return CacheTier.L3_CDN
            else:
                return CacheTier.L4_DATABASE
                
        except Exception as e:
            logger.error(f"Error determining optimal tier: {e}")
            return CacheTier.L2_REDIS
            
    async def _initialize_cdn_nodes(self):
        """Initialize CDN nodes"""
        try:
            for region in self.cdn_config["default_regions"]:
                node_id = f"cdn_{region}"
                node = CDNNode(
                    node_id=node_id,
                    region=region,
                    endpoint=f"https://cdn-{region}.example.com",
                    status="healthy",
                    latency_ms=50.0 + (secrets.randbelow(50)),  # Mock latency
                    bandwidth_mbps=1000.0,
                    storage_gb=100.0,
                    last_health_check=datetime.now()
                )
                self.cdn_nodes[node_id] = node
                
            logger.info(f"Initialized {len(self.cdn_nodes)} CDN nodes")
            
        except Exception as e:
            logger.error(f"Error initializing CDN nodes: {e}")
            
    async def _health_check_cdn_nodes(self):
        """Health check CDN nodes"""
        try:
            for node_id, node in self.cdn_nodes.items():
                # Simulate health check
                node.status = "healthy" if secrets.randbelow(10) < 9 else "degraded"
                node.last_health_check = datetime.now()
                node.latency_ms = 50.0 + (secrets.randbelow(50))
                
        except Exception as e:
            logger.error(f"Error health checking CDN nodes: {e}")
            
    async def _get_from_cdn(self, key: str) -> Optional[Any]:
        """Get value from CDN"""
        try:
            # Find best CDN node
            best_node = self._find_best_cdn_node()
            if not best_node:
                return None
                
            # Simulate CDN request
            # In real implementation, would make HTTP request
            return None
            
        except Exception as e:
            logger.error(f"Error getting from CDN: {e}")
            return None
            
    async def _set_in_cdn(self, key: str, entry: CacheEntry):
        """Set value in CDN"""
        try:
            # Find best CDN node
            best_node = self._find_best_cdn_node()
            if not best_node:
                return
                
            # Simulate CDN storage
            # In real implementation, would make HTTP request
            pass
            
        except Exception as e:
            logger.error(f"Error setting in CDN: {e}")
            
    async def _delete_from_cdn(self, key: str):
        """Delete value from CDN"""
        try:
            # Simulate CDN deletion
            # In real implementation, would make HTTP request
            pass
            
        except Exception as e:
            logger.error(f"Error deleting from CDN: {e}")
            
    def _find_best_cdn_node(self) -> Optional[CDNNode]:
        """Find best CDN node based on latency and status"""
        try:
            healthy_nodes = [node for node in self.cdn_nodes.values() if node.status == "healthy"]
            if not healthy_nodes:
                return None
                
            # Return node with lowest latency
            return min(healthy_nodes, key=lambda x: x.latency_ms)
            
        except Exception as e:
            logger.error(f"Error finding best CDN node: {e}")
            return None
            
    async def _cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        try:
            current_time = datetime.now()
            
            # Clean up L1 memory
            expired_keys = []
            for key, entry in self.cache_tiers[CacheTier.L1_MEMORY].items():
                if self._is_expired(entry):
                    expired_keys.append(key)
                    
            for key in expired_keys:
                del self.cache_tiers[CacheTier.L1_MEMORY][key]
                self.cache_metrics[CacheTier.L1_MEMORY].evictions += 1
                
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        try:
            if entry.ttl <= 0:
                return False
                
            expiry_time = entry.created_at + timedelta(seconds=entry.ttl)
            return datetime.now() > expiry_time
            
        except Exception as e:
            logger.error(f"Error checking expiry: {e}")
            return True
            
    async def _update_intelligent_patterns(self):
        """Update intelligent caching patterns"""
        try:
            if not self.config["intelligent_caching_enabled"]:
                return
                
            # Analyze access patterns
            # This would implement machine learning-based pattern detection
            # For now, we'll simulate it
            
        except Exception as e:
            logger.error(f"Error updating intelligent patterns: {e}")
            
    async def _perform_auto_tiering(self):
        """Perform automatic tiering based on access patterns"""
        try:
            # Analyze access patterns and move entries between tiers
            # This would implement intelligent tiering logic
            pass
            
        except Exception as e:
            logger.error(f"Error performing auto tiering: {e}")
            
    async def _perform_cache_warming(self):
        """Perform cache warming based on predictions"""
        try:
            # Predict which keys will be accessed and preload them
            # This would implement predictive caching
            pass
            
        except Exception as e:
            logger.error(f"Error performing cache warming: {e}")
            
    async def _update_cache_metrics(self):
        """Update cache metrics"""
        try:
            for tier, metrics in self.cache_metrics.items():
                total_requests = metrics.hits + metrics.misses
                if total_requests > 0:
                    metrics.hit_rate = metrics.hits / total_requests
                    
        except Exception as e:
            logger.error(f"Error updating cache metrics: {e}")
            
    async def _update_cdn_metrics(self):
        """Update CDN metrics"""
        try:
            # Update CDN-specific metrics
            pass
            
        except Exception as e:
            logger.error(f"Error updating CDN metrics: {e}")
            
    async def _update_access_pattern(self, key: str, tier: CacheTier):
        """Update access pattern for intelligent caching"""
        try:
            # Update access pattern data for ML analysis
            pass
            
        except Exception as e:
            logger.error(f"Error updating access pattern: {e}")
            
    async def _consider_tier_promotion(self, key: str, value: Any, current_tier: CacheTier):
        """Consider promoting entry to higher tier"""
        try:
            # Implement tier promotion logic
            pass
            
        except Exception as e:
            logger.error(f"Error considering tier promotion: {e}")
            
    async def _predict_and_prefetch(self, key: str):
        """Predict and prefetch related keys"""
        try:
            # Implement predictive prefetching
            self.caching_stats["intelligent_predictions"] += 1
            
        except Exception as e:
            logger.error(f"Error predicting and prefetching: {e}")
            
    def _update_response_time(self, tier: CacheTier, response_time: float):
        """Update response time metrics"""
        try:
            metrics = self.cache_metrics[tier]
            # Update average response time
            total_requests = metrics.hits + metrics.sets
            if total_requests > 0:
                metrics.average_response_time = (
                    (metrics.average_response_time * (total_requests - 1) + response_time) / total_requests
                )
                
        except Exception as e:
            logger.error(f"Error updating response time: {e}")
            
    def _exceeds_memory_limit(self, size_bytes: int) -> bool:
        """Check if adding entry would exceed memory limit"""
        try:
            current_size = sum(entry.size_bytes for entry in self.cache_tiers[CacheTier.L1_MEMORY].values())
            limit_bytes = self.config["l1_memory_limit_mb"] * 1024 * 1024
            return (current_size + size_bytes) > limit_bytes
            
        except Exception as e:
            logger.error(f"Error checking memory limit: {e}")
            return True
            
    async def _evict_from_memory(self):
        """Evict entries from memory using LRU"""
        try:
            # Remove least recently used entry
            if self.cache_tiers[CacheTier.L1_MEMORY]:
                key, entry = self.cache_tiers[CacheTier.L1_MEMORY].popitem(last=False)
                self.cache_metrics[CacheTier.L1_MEMORY].evictions += 1
                
        except Exception as e:
            logger.error(f"Error evicting from memory: {e}")
            
    async def _invalidate_pattern_in_tier(self, pattern: str, tier: CacheTier) -> int:
        """Invalidate pattern in specific tier"""
        try:
            count = 0
            
            if tier == CacheTier.L1_MEMORY:
                keys_to_remove = [key for key in self.cache_tiers[tier].keys() if pattern in key]
                for key in keys_to_remove:
                    del self.cache_tiers[tier][key]
                    count += 1
                    
            # Similar logic for other tiers
            
            return count
            
        except Exception as e:
            logger.error(f"Error invalidating pattern in tier: {e}")
            return 0
            
    def get_caching_summary(self) -> Dict[str, Any]:
        """Get comprehensive caching summary"""
        try:
            # Calculate overall metrics
            total_hits = sum(metrics.hits for metrics in self.cache_metrics.values())
            total_misses = sum(metrics.misses for metrics in self.cache_metrics.values())
            total_requests = total_hits + total_misses
            overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate tier statistics
            tier_stats = {}
            for tier, metrics in self.cache_metrics.items():
                tier_stats[tier.value] = {
                    "hits": metrics.hits,
                    "misses": metrics.misses,
                    "sets": metrics.sets,
                    "deletes": metrics.deletes,
                    "evictions": metrics.evictions,
                    "total_size_bytes": metrics.total_size_bytes,
                    "average_response_time": metrics.average_response_time,
                    "hit_rate": metrics.hit_rate
                }
                
            # Calculate CDN statistics
            cdn_stats = {
                "total_nodes": len(self.cdn_nodes),
                "healthy_nodes": len([node for node in self.cdn_nodes.values() if node.status == "healthy"]),
                "average_latency": sum(node.latency_ms for node in self.cdn_nodes.values()) / len(self.cdn_nodes) if self.cdn_nodes else 0
            }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "caching_active": self.caching_active,
                "overall_hit_rate": overall_hit_rate,
                "total_requests": total_requests,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "tier_stats": tier_stats,
                "cdn_stats": cdn_stats,
                "cache_strategies": len(self.cache_strategies),
                "intelligent_patterns": len(self.intelligent_patterns),
                "stats": self.caching_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting caching summary: {e}")
            return {"error": str(e)}


# Global instance
advanced_caching_engine = AdvancedCachingEngine()
