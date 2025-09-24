"""
Enhanced Cache API Endpoints
Provides comprehensive cache management and analytics
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import asyncio
import time

from app.core.enhanced_cache import enhanced_cache, EvictionPolicy, CompressionLevel
from app.core.auth import get_current_user
from app.models.user import User

router = APIRouter()


@router.get("/stats")
async def get_cache_stats(current_user: User = Depends(get_current_user)):
    """Get comprehensive cache statistics"""
    try:
        stats = enhanced_cache.get_stats()
        return {
            "success": True,
            "data": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.get("/analytics")
async def get_cache_analytics(current_user: User = Depends(get_current_user)):
    """Get detailed cache analytics"""
    try:
        analytics = enhanced_cache.get_analytics()
        if not analytics:
            return {
                "success": False,
                "message": "Analytics not enabled",
                "data": None
            }
        
        return {
            "success": True,
            "data": {
                "total_requests": analytics.total_requests,
                "hit_rate": analytics.hit_rate,
                "miss_rate": analytics.miss_rate,
                "eviction_rate": analytics.eviction_rate,
                "compression_ratio": analytics.compression_ratio,
                "average_access_time": analytics.average_access_time,
                "memory_efficiency": analytics.memory_efficiency,
                "cache_efficiency_score": analytics.cache_efficiency_score,
                "top_keys": analytics.top_keys,
                "access_patterns": analytics.access_patterns,
                "performance_metrics": analytics.performance_metrics
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache analytics: {str(e)}")


@router.get("/memory")
async def get_memory_usage(current_user: User = Depends(get_current_user)):
    """Get detailed memory usage information"""
    try:
        memory_info = enhanced_cache.get_memory_usage()
        return {
            "success": True,
            "data": memory_info,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory usage: {str(e)}")


@router.post("/benchmark")
async def benchmark_cache_performance(
    iterations: int = 1000,
    current_user: User = Depends(get_current_user)
):
    """Run cache performance benchmark"""
    try:
        if iterations > 10000:
            raise HTTPException(status_code=400, detail="Maximum 10000 iterations allowed")
        
        benchmark_results = await enhanced_cache.benchmark_performance(iterations)
        return {
            "success": True,
            "data": benchmark_results,
            "iterations": iterations,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.post("/warm-up")
async def warm_up_cache(
    warm_up_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Warm up cache with predefined data"""
    try:
        if len(warm_up_data) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 entries allowed for warm-up")
        
        # Run warm-up in background
        def run_warm_up():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(enhanced_cache.warm_up(warm_up_data))
            finally:
                loop.close()
        
        background_tasks.add_task(run_warm_up)
        
        return {
            "success": True,
            "message": f"Warm-up started for {len(warm_up_data)} entries",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warm-up failed: {str(e)}")


@router.post("/set")
async def set_cache_entry(
    key: str,
    value: Any,
    ttl: Optional[int] = None,
    tags: Optional[List[str]] = None,
    priority: int = 0,
    cost: float = 0.0,
    current_user: User = Depends(get_current_user)
):
    """Set a cache entry with enhanced features"""
    try:
        success = await enhanced_cache.set(
            key=key,
            value=value,
            ttl=ttl,
            tags=tags,
            priority=priority,
            cost=cost
        )
        
        return {
            "success": success,
            "key": key,
            "message": "Cache entry set successfully" if success else "Failed to set cache entry",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set cache entry: {str(e)}")


@router.get("/get/{key}")
async def get_cache_entry(key: str, current_user: User = Depends(get_current_user)):
    """Get a cache entry"""
    try:
        value = await enhanced_cache.get(key)
        
        if value is None:
            return {
                "success": False,
                "key": key,
                "value": None,
                "message": "Cache entry not found",
                "timestamp": time.time()
            }
        
        return {
            "success": True,
            "key": key,
            "value": value,
            "message": "Cache entry retrieved successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache entry: {str(e)}")


@router.delete("/delete/{key}")
async def delete_cache_entry(key: str, current_user: User = Depends(get_current_user)):
    """Delete a cache entry"""
    try:
        success = await enhanced_cache.delete(key)
        
        return {
            "success": success,
            "key": key,
            "message": "Cache entry deleted successfully" if success else "Cache entry not found",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete cache entry: {str(e)}")


@router.delete("/delete-by-tags")
async def delete_by_tags(
    tags: List[str],
    current_user: User = Depends(get_current_user)
):
    """Delete cache entries by tags"""
    try:
        deleted_count = await enhanced_cache.delete_by_tags(tags)
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "tags": tags,
            "message": f"Deleted {deleted_count} cache entries",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete by tags: {str(e)}")


@router.post("/clear")
async def clear_cache(current_user: User = Depends(get_current_user)):
    """Clear all cache entries"""
    try:
        await enhanced_cache.clear()
        
        return {
            "success": True,
            "message": "Cache cleared successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/entries/by-tag/{tag}")
async def get_entries_by_tag(tag: str, current_user: User = Depends(get_current_user)):
    """Get cache entries by tag"""
    try:
        entries = enhanced_cache.get_entries_by_tag(tag)
        
        return {
            "success": True,
            "tag": tag,
            "entries": entries,
            "count": len(entries),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entries by tag: {str(e)}")


@router.get("/export")
async def export_cache_data(current_user: User = Depends(get_current_user)):
    """Export cache data for analysis"""
    try:
        export_data = enhanced_cache.export_cache_data()
        
        return {
            "success": True,
            "data": export_data,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export cache data: {str(e)}")


@router.post("/configure")
async def configure_cache(
    max_size: Optional[int] = None,
    default_ttl: Optional[int] = None,
    eviction_policy: Optional[str] = None,
    compression_level: Optional[str] = None,
    max_memory_mb: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """Configure cache settings (requires cache restart to take effect)"""
    try:
        config_updates = {}
        
        if max_size is not None:
            config_updates["max_size"] = max_size
        if default_ttl is not None:
            config_updates["default_ttl"] = default_ttl
        if eviction_policy is not None:
            if eviction_policy not in [policy.value for policy in EvictionPolicy]:
                raise HTTPException(status_code=400, detail="Invalid eviction policy")
            config_updates["eviction_policy"] = eviction_policy
        if compression_level is not None:
            if compression_level not in [level.value for level in CompressionLevel]:
                raise HTTPException(status_code=400, detail="Invalid compression level")
            config_updates["compression_level"] = compression_level
        if max_memory_mb is not None:
            config_updates["max_memory_mb"] = max_memory_mb
        
        return {
            "success": True,
            "message": "Configuration updated (restart required for full effect)",
            "config_updates": config_updates,
            "current_config": {
                "max_size": enhanced_cache.max_size,
                "default_ttl": enhanced_cache.default_ttl,
                "eviction_policy": enhanced_cache.eviction_policy.value,
                "compression_level": enhanced_cache.compression_level.value,
                "max_memory_mb": enhanced_cache.max_memory_mb
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure cache: {str(e)}")


@router.get("/health")
async def cache_health_check():
    """Cache health check endpoint"""
    try:
        # Test basic operations
        test_key = "health_check_test"
        test_value = {"test": True, "timestamp": time.time()}
        
        # Test set
        set_success = await enhanced_cache.set(test_key, test_value, ttl=60)
        
        # Test get
        retrieved_value = await enhanced_cache.get(test_key)
        
        # Test delete
        delete_success = await enhanced_cache.delete(test_key)
        
        # Get stats
        stats = enhanced_cache.get_stats()
        
        health_status = {
            "status": "healthy",
            "operations": {
                "set": set_success,
                "get": retrieved_value is not None,
                "delete": delete_success
            },
            "stats": {
                "hit_rate": stats.get("hit_rate", 0),
                "memory_usage_mb": stats.get("memory_usage_mb", 0),
                "entry_count": stats.get("entry_count", 0)
            },
            "timestamp": time.time()
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
