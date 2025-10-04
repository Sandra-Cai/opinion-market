"""
Advanced Caching API
API endpoints for advanced caching and CDN management
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.services.advanced_caching_engine import advanced_caching_engine, CacheStrategy, CacheTier

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class CacheRequest(BaseModel):
    """Cache request model"""
    key: str
    value: Any
    ttl: Optional[int] = None
    strategy: Optional[str] = None
    tier: Optional[str] = None


class CacheGetRequest(BaseModel):
    """Cache get request model"""
    key: str
    strategy: Optional[str] = None


class CacheInvalidateRequest(BaseModel):
    """Cache invalidate request model"""
    pattern: str


# API Endpoints
@router.get("/status")
async def get_advanced_caching_status():
    """Get advanced caching system status"""
    try:
        summary = advanced_caching_engine.get_caching_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting advanced caching status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache")
async def set_cache_value(cache_request: CacheRequest):
    """Set value in advanced cache"""
    try:
        strategy = CacheStrategy(cache_request.strategy) if cache_request.strategy else None
        tier = CacheTier(cache_request.tier) if cache_request.tier else None
        
        success = await advanced_caching_engine.set(
            key=cache_request.key,
            value=cache_request.value,
            ttl=cache_request.ttl,
            strategy=strategy,
            tier=tier
        )
        
        return JSONResponse(content={
            "message": "Cache value set successfully" if success else "Failed to set cache value",
            "key": cache_request.key,
            "success": success
        })
        
    except Exception as e:
        logger.error(f"Error setting cache value: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/{key}")
async def get_cache_value(key: str, strategy: Optional[str] = None):
    """Get value from advanced cache"""
    try:
        cache_strategy = CacheStrategy(strategy) if strategy else None
        
        value = await advanced_caching_engine.get(key=key, strategy=cache_strategy)
        
        if value is not None:
            return JSONResponse(content={
                "key": key,
                "value": value,
                "found": True
            })
        else:
            return JSONResponse(content={
                "key": key,
                "value": None,
                "found": False
            })
        
    except Exception as e:
        logger.error(f"Error getting cache value: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/{key}")
async def delete_cache_value(key: str):
    """Delete value from advanced cache"""
    try:
        success = await advanced_caching_engine.delete(key)
        
        return JSONResponse(content={
            "message": "Cache value deleted successfully" if success else "Failed to delete cache value",
            "key": key,
            "success": success
        })
        
    except Exception as e:
        logger.error(f"Error deleting cache value: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/invalidate")
async def invalidate_cache_pattern(cache_request: CacheInvalidateRequest):
    """Invalidate cache entries matching pattern"""
    try:
        invalidated_count = await advanced_caching_engine.invalidate_pattern(cache_request.pattern)
        
        return JSONResponse(content={
            "message": f"Invalidated {invalidated_count} cache entries",
            "pattern": cache_request.pattern,
            "invalidated_count": invalidated_count
        })
        
    except Exception as e:
        logger.error(f"Error invalidating cache pattern: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tiers")
async def get_cache_tiers():
    """Get cache tier information"""
    try:
        tier_info = {}
        for tier in CacheTier:
            tier_info[tier.value] = {
                "name": tier.value,
                "description": f"Cache tier {tier.value}",
                "entries": len(advanced_caching_engine.cache_tiers[tier]),
                "metrics": {
                    "hits": advanced_caching_engine.cache_metrics[tier].hits,
                    "misses": advanced_caching_engine.cache_metrics[tier].misses,
                    "sets": advanced_caching_engine.cache_metrics[tier].sets,
                    "deletes": advanced_caching_engine.cache_metrics[tier].deletes,
                    "evictions": advanced_caching_engine.cache_metrics[tier].evictions,
                    "total_size_bytes": advanced_caching_engine.cache_metrics[tier].total_size_bytes,
                    "average_response_time": advanced_caching_engine.cache_metrics[tier].average_response_time,
                    "hit_rate": advanced_caching_engine.cache_metrics[tier].hit_rate
                }
            }
            
        return JSONResponse(content={"tiers": tier_info})
        
    except Exception as e:
        logger.error(f"Error getting cache tiers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def get_cache_strategies():
    """Get cache strategies information"""
    try:
        strategies_info = {}
        for strategy in CacheStrategy:
            strategies_info[strategy.value] = {
                "name": strategy.value,
                "description": f"Cache strategy {strategy.value}",
                "config": advanced_caching_engine.strategy_config.get(strategy, {})
            }
            
        return JSONResponse(content={"strategies": strategies_info})
        
    except Exception as e:
        logger.error(f"Error getting cache strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cdn")
async def get_cdn_nodes():
    """Get CDN nodes information"""
    try:
        cdn_nodes = []
        for node_id, node in advanced_caching_engine.cdn_nodes.items():
            cdn_nodes.append({
                "node_id": node.node_id,
                "region": node.region,
                "endpoint": node.endpoint,
                "status": node.status,
                "latency_ms": node.latency_ms,
                "bandwidth_mbps": node.bandwidth_mbps,
                "storage_gb": node.storage_gb,
                "last_health_check": node.last_health_check.isoformat(),
                "metadata": node.metadata
            })
            
        return JSONResponse(content={"cdn_nodes": cdn_nodes})
        
    except Exception as e:
        logger.error(f"Error getting CDN nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_cache_metrics():
    """Get comprehensive cache metrics"""
    try:
        summary = advanced_caching_engine.get_caching_summary()
        
        return JSONResponse(content={
            "overall_metrics": {
                "hit_rate": summary.get("overall_hit_rate", 0),
                "total_requests": summary.get("total_requests", 0),
                "total_hits": summary.get("total_hits", 0),
                "total_misses": summary.get("total_misses", 0)
            },
            "tier_metrics": summary.get("tier_stats", {}),
            "cdn_metrics": summary.get("cdn_stats", {}),
            "performance_stats": summary.get("stats", {}),
            "timestamp": summary.get("timestamp")
        })
        
    except Exception as e:
        logger.error(f"Error getting cache metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_advanced_caching_dashboard():
    """Get advanced caching dashboard data"""
    try:
        summary = advanced_caching_engine.get_caching_summary()
        
        # Get recent cache operations (simulated)
        recent_operations = [
            {
                "operation": "SET",
                "key": f"cache_key_{i}",
                "tier": "L1_MEMORY",
                "timestamp": datetime.now().isoformat()
            }
            for i in range(5)
        ]
        
        dashboard_data = {
            "summary": summary,
            "recent_operations": recent_operations,
            "performance_trends": {
                "hit_rate_trend": [0.85, 0.87, 0.89, 0.91, 0.88],
                "response_time_trend": [1.2, 1.1, 1.0, 0.9, 1.0],
                "throughput_trend": [1000, 1100, 1200, 1300, 1250]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting advanced caching dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_advanced_caching():
    """Start advanced caching engine"""
    try:
        await advanced_caching_engine.start_caching_engine()
        return JSONResponse(content={"message": "Advanced caching engine started"})
        
    except Exception as e:
        logger.error(f"Error starting advanced caching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_advanced_caching():
    """Stop advanced caching engine"""
    try:
        await advanced_caching_engine.stop_caching_engine()
        return JSONResponse(content={"message": "Advanced caching engine stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping advanced caching: {e}")
        raise HTTPException(status_code=500, detail=str(e))
