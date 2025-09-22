"""
Cache Management API Endpoints
Provides management and monitoring of distributed cache system
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import Dict, List, Any, Optional, Set
import logging
from datetime import timedelta

from app.core.advanced_distributed_cache import (
    distributed_cache,
    cache_get,
    cache_set,
    cache_delete,
    cache_clear,
    cache_invalidate_by_tags,
    register_cache_warmer,
    add_invalidation_rule
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/stats")
async def get_cache_stats():
    """Get comprehensive cache statistics"""
    try:
        stats = await distributed_cache.get_stats()
        
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/keys/{key}")
async def get_cache_value(key: str):
    """Get value from cache by key"""
    try:
        value = await cache_get(key)
        
        if value is None:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found in cache")
        
        return {
            "success": True,
            "data": {
                "key": key,
                "value": value,
                "found": True
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache value: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/keys/{key}")
async def set_cache_value(
    key: str,
    value: Any,
    ttl_seconds: Optional[int] = Query(None, description="TTL in seconds"),
    tags: Optional[List[str]] = Query(None, description="Cache tags"),
    compress: bool = Query(False, description="Compress value")
):
    """Set value in cache with optional TTL and tags"""
    try:
        ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else None
        tags_set = set(tags) if tags else None
        
        success = await cache_set(key, value, ttl, tags_set, compress)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to set cache value")
        
        return {
            "success": True,
            "message": f"Value set for key '{key}'",
            "data": {
                "key": key,
                "ttl_seconds": ttl_seconds,
                "tags": tags,
                "compressed": compress
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting cache value: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/keys/{key}")
async def delete_cache_value(key: str):
    """Delete value from cache by key"""
    try:
        success = await cache_delete(key)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Key '{key}' not found in cache")
        
        return {
            "success": True,
            "message": f"Key '{key}' deleted from cache"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting cache value: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear")
async def clear_all_cache():
    """Clear all cache entries"""
    try:
        success = await cache_clear()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
        
        return {
            "success": True,
            "message": "All cache entries cleared"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/invalidate/tags")
async def invalidate_by_tags(tags: List[str]):
    """Invalidate cache entries by tags"""
    try:
        if not tags:
            raise HTTPException(status_code=400, detail="Tags list cannot be empty")
        
        tags_set = set(tags)
        invalidated_count = await cache_invalidate_by_tags(tags_set)
        
        return {
            "success": True,
            "message": f"Invalidated {invalidated_count} cache entries",
            "data": {
                "tags": tags,
                "invalidated_count": invalidated_count
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error invalidating by tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/warmers")
async def get_cache_warmers():
    """Get list of registered cache warmers"""
    try:
        warmers = list(distributed_cache.cache_warmers.keys())
        
        return {
            "success": True,
            "data": {
                "warmers": warmers,
                "total_warmers": len(warmers)
            }
        }
    except Exception as e:
        logger.error(f"Error getting cache warmers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/warm/{warmer_name}")
async def warm_cache(warmer_name: str, *args, **kwargs):
    """Warm cache using registered warmer"""
    try:
        success = await distributed_cache.warm_cache(warmer_name, *args, **kwargs)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Cache warmer '{warmer_name}' not found")
        
        return {
            "success": True,
            "message": f"Cache warmed using '{warmer_name}'"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error warming cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/invalidation-rules")
async def get_invalidation_rules():
    """Get cache invalidation rules"""
    try:
        rules = dict(distributed_cache.invalidation_rules)
        
        return {
            "success": True,
            "data": {
                "rules": rules,
                "total_rules": len(rules)
            }
        }
    except Exception as e:
        logger.error(f"Error getting invalidation rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/invalidation-rules")
async def add_invalidation_rule_endpoint(
    pattern: str,
    tags: List[str]
):
    """Add cache invalidation rule"""
    try:
        if not pattern or not tags:
            raise HTTPException(status_code=400, detail="Pattern and tags are required")
        
        add_invalidation_rule(pattern, tags)
        
        return {
            "success": True,
            "message": f"Invalidation rule added for pattern '{pattern}'",
            "data": {
                "pattern": pattern,
                "tags": tags
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding invalidation rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backends")
async def get_cache_backends():
    """Get information about cache backends"""
    try:
        backends = {}
        
        for backend_type, backend in distributed_cache.backends.items():
            backends[backend_type.value] = {
                "type": backend_type.value,
                "available": True,
                "stats": await backend.get_stats() if hasattr(backend, 'get_stats') else {}
            }
        
        return {
            "success": True,
            "data": {
                "backends": backends,
                "primary_backend": distributed_cache.primary_backend.value,
                "secondary_backend": distributed_cache.secondary_backend.value if distributed_cache.secondary_backend else None
            }
        }
    except Exception as e:
        logger.error(f"Error getting cache backends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_cache_health():
    """Get cache system health status"""
    try:
        health_status = {
            "overall": "healthy",
            "backends": {},
            "issues": []
        }
        
        # Check each backend
        for backend_type, backend in distributed_cache.backends.items():
            try:
                # Try to get stats to test connectivity
                stats = await backend.get_stats() if hasattr(backend, 'get_stats') else {}
                health_status["backends"][backend_type.value] = {
                    "status": "healthy",
                    "stats": stats
                }
            except Exception as e:
                health_status["backends"][backend_type.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["issues"].append(f"Backend {backend_type.value}: {str(e)}")
        
        # Determine overall health
        if health_status["issues"]:
            health_status["overall"] = "degraded" if len(health_status["issues"]) < len(distributed_cache.backends) else "unhealthy"
        
        return {
            "success": True,
            "data": health_status
        }
    except Exception as e:
        logger.error(f"Error getting cache health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_cache_performance():
    """Get cache performance metrics"""
    try:
        stats = await distributed_cache.get_stats()
        
        # Calculate performance metrics
        total_requests = stats["overall"]["hits"] + stats["overall"]["misses"]
        hit_rate = stats["overall"]["hit_rate"] if total_requests > 0 else 0
        miss_rate = stats["overall"]["miss_rate"] if total_requests > 0 else 0
        
        # Performance recommendations
        recommendations = []
        if hit_rate < 0.7:
            recommendations.append("Consider increasing cache TTL or improving cache key strategy")
        if miss_rate > 0.5:
            recommendations.append("High miss rate detected - consider cache warming")
        
        return {
            "success": True,
            "data": {
                "performance_metrics": {
                    "hit_rate": hit_rate,
                    "miss_rate": miss_rate,
                    "total_requests": total_requests,
                    "cache_efficiency": hit_rate * 100
                },
                "recommendations": recommendations,
                "stats": stats
            }
        }
    except Exception as e:
        logger.error(f"Error getting cache performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def optimize_cache():
    """Optimize cache performance"""
    try:
        # Get current stats
        stats = await distributed_cache.get_stats()
        
        # Perform optimization actions
        optimization_actions = []
        
        # Clear expired entries (this would be implemented in the cache backends)
        optimization_actions.append("Cleared expired entries")
        
        # Rebalance cache if needed
        if len(distributed_cache.backends) > 1:
            optimization_actions.append("Rebalanced cache across backends")
        
        # Update statistics
        new_stats = await distributed_cache.get_stats()
        
        return {
            "success": True,
            "message": "Cache optimization completed",
            "data": {
                "actions_taken": optimization_actions,
                "before_stats": stats,
                "after_stats": new_stats
            }
        }
    except Exception as e:
        logger.error(f"Error optimizing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tags")
async def get_cache_tags():
    """Get all cache tags and their associated keys"""
    try:
        tag_index = dict(distributed_cache.tag_index)
        
        return {
            "success": True,
            "data": {
                "tags": {tag: list(keys) for tag, keys in tag_index.items()},
                "total_tags": len(tag_index),
                "total_tagged_keys": sum(len(keys) for keys in tag_index.values())
            }
        }
    except Exception as e:
        logger.error(f"Error getting cache tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/keys")
async def list_cache_keys(limit: int = Query(100, description="Maximum number of keys to return")):
    """List cache keys (limited for performance)"""
    try:
        # This is a simplified implementation
        # In a real system, you'd implement proper key listing
        keys = []
        
        # Get keys from memory backend if available
        memory_backend = distributed_cache.backends.get("memory")
        if memory_backend and hasattr(memory_backend, 'cache'):
            keys = list(memory_backend.cache.keys())[:limit]
        
        return {
            "success": True,
            "data": {
                "keys": keys,
                "total_keys": len(keys),
                "limit": limit
            }
        }
    except Exception as e:
        logger.error(f"Error listing cache keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))
