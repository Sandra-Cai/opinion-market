"""
Rate Limiting API Endpoints
Provides management and monitoring of rate limiting through API
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, List, Any, Optional
import logging

from app.core.advanced_rate_limiter import (
    advanced_rate_limiter,
    RateLimitScope,
    RateLimit,
    RateLimitAlgorithm
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/status")
async def get_rate_limit_status():
    """Get overall rate limiting status"""
    try:
        all_limiters = advanced_rate_limiter.get_all_limiters()
        
        return {
            "success": True,
            "data": {
                "total_limiters": sum(len(scope_data) for scope_data in all_limiters.values()),
                "scopes": list(all_limiters.keys()),
                "limiters": all_limiters
            }
        }
    except Exception as e:
        logger.error(f"Error getting rate limit status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{scope}")
async def get_scope_status(scope: str, identifier: str):
    """Get rate limit status for a specific scope and identifier"""
    try:
        # Validate scope
        try:
            rate_limit_scope = RateLimitScope(scope)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid scope: {scope}")
        
        status_data = advanced_rate_limiter.get_rate_limit_status(rate_limit_scope, identifier)
        
        return {
            "success": True,
            "data": status_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scope status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/limits/user/{user_id}")
async def set_user_rate_limit(
    user_id: str,
    requests_per_minute: int,
    requests_per_hour: int,
    requests_per_day: int,
    burst_limit: int = 10,
    algorithm: str = "token_bucket"
):
    """Set custom rate limit for a user"""
    try:
        # Validate algorithm
        try:
            rate_limit_algorithm = RateLimitAlgorithm(algorithm)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid algorithm: {algorithm}")
        
        # Validate limits
        if requests_per_minute <= 0 or requests_per_hour <= 0 or requests_per_day <= 0:
            raise HTTPException(status_code=400, detail="Rate limits must be positive")
        
        if burst_limit <= 0:
            raise HTTPException(status_code=400, detail="Burst limit must be positive")
        
        # Create rate limit
        rate_limit = RateLimit(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            requests_per_day=requests_per_day,
            burst_limit=burst_limit,
            algorithm=rate_limit_algorithm,
            scope=RateLimitScope.USER
        )
        
        # Set the limit
        advanced_rate_limiter.set_user_limit(user_id, rate_limit)
        
        return {
            "success": True,
            "message": f"Rate limit set for user {user_id}",
            "data": {
                "user_id": user_id,
                "requests_per_minute": requests_per_minute,
                "requests_per_hour": requests_per_hour,
                "requests_per_day": requests_per_day,
                "burst_limit": burst_limit,
                "algorithm": algorithm
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting user rate limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/limits/endpoint/{endpoint:path}")
async def set_endpoint_rate_limit(
    endpoint: str,
    requests_per_minute: int,
    requests_per_hour: int,
    requests_per_day: int,
    burst_limit: int = 10,
    algorithm: str = "sliding_window"
):
    """Set custom rate limit for an endpoint"""
    try:
        # Validate algorithm
        try:
            rate_limit_algorithm = RateLimitAlgorithm(algorithm)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid algorithm: {algorithm}")
        
        # Validate limits
        if requests_per_minute <= 0 or requests_per_hour <= 0 or requests_per_day <= 0:
            raise HTTPException(status_code=400, detail="Rate limits must be positive")
        
        if burst_limit <= 0:
            raise HTTPException(status_code=400, detail="Burst limit must be positive")
        
        # Create rate limit
        rate_limit = RateLimit(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            requests_per_day=requests_per_day,
            burst_limit=burst_limit,
            algorithm=rate_limit_algorithm,
            scope=RateLimitScope.ENDPOINT
        )
        
        # Set the limit
        advanced_rate_limiter.set_endpoint_limit(endpoint, rate_limit)
        
        return {
            "success": True,
            "message": f"Rate limit set for endpoint {endpoint}",
            "data": {
                "endpoint": endpoint,
                "requests_per_minute": requests_per_minute,
                "requests_per_hour": requests_per_hour,
                "requests_per_day": requests_per_day,
                "burst_limit": burst_limit,
                "algorithm": algorithm
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting endpoint rate limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/limits/api-key/{api_key}")
async def set_api_key_rate_limit(
    api_key: str,
    requests_per_minute: int,
    requests_per_hour: int,
    requests_per_day: int,
    burst_limit: int = 10,
    algorithm: str = "token_bucket"
):
    """Set custom rate limit for an API key"""
    try:
        # Validate algorithm
        try:
            rate_limit_algorithm = RateLimitAlgorithm(algorithm)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid algorithm: {algorithm}")
        
        # Validate limits
        if requests_per_minute <= 0 or requests_per_hour <= 0 or requests_per_day <= 0:
            raise HTTPException(status_code=400, detail="Rate limits must be positive")
        
        if burst_limit <= 0:
            raise HTTPException(status_code=400, detail="Burst limit must be positive")
        
        # Create rate limit
        rate_limit = RateLimit(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            requests_per_day=requests_per_day,
            burst_limit=burst_limit,
            algorithm=rate_limit_algorithm,
            scope=RateLimitScope.API_KEY
        )
        
        # Set the limit
        advanced_rate_limiter.set_api_key_limit(api_key, rate_limit)
        
        return {
            "success": True,
            "message": f"Rate limit set for API key {api_key}",
            "data": {
                "api_key": api_key,
                "requests_per_minute": requests_per_minute,
                "requests_per_hour": requests_per_hour,
                "requests_per_day": requests_per_day,
                "burst_limit": burst_limit,
                "algorithm": algorithm
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting API key rate limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reset/{scope}")
async def reset_rate_limit(scope: str, identifier: str):
    """Reset rate limit for a specific scope and identifier"""
    try:
        # Validate scope
        try:
            rate_limit_scope = RateLimitScope(scope)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid scope: {scope}")
        
        # Reset the rate limit
        advanced_rate_limiter.reset_rate_limit(rate_limit_scope, identifier)
        
        return {
            "success": True,
            "message": f"Rate limit reset for {scope}:{identifier}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting rate limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/algorithms")
async def get_available_algorithms():
    """Get list of available rate limiting algorithms"""
    algorithms = [
        {
            "name": algorithm.value,
            "description": _get_algorithm_description(algorithm)
        }
        for algorithm in RateLimitAlgorithm
    ]
    
    return {
        "success": True,
        "data": {
            "algorithms": algorithms
        }
    }

@router.get("/scopes")
async def get_available_scopes():
    """Get list of available rate limiting scopes"""
    scopes = [
        {
            "name": scope.value,
            "description": _get_scope_description(scope)
        }
        for scope in RateLimitScope
    ]
    
    return {
        "success": True,
        "data": {
            "scopes": scopes
        }
    }

@router.get("/stats")
async def get_rate_limit_stats():
    """Get rate limiting statistics"""
    try:
        all_limiters = advanced_rate_limiter.get_all_limiters()
        
        # Calculate statistics
        total_limiters = sum(len(scope_data) for scope_data in all_limiters.values())
        
        scope_stats = {}
        for scope, identifiers in all_limiters.items():
            scope_stats[scope] = {
                "total_identifiers": len(identifiers),
                "identifiers": list(identifiers.keys())
            }
        
        return {
            "success": True,
            "data": {
                "total_limiters": total_limiters,
                "scope_statistics": scope_stats,
                "timestamp": "2024-01-01T00:00:00Z"  # You'd use actual timestamp
            }
        }
    except Exception as e:
        logger.error(f"Error getting rate limit stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup_old_limiters(max_age_hours: int = 24):
    """Clean up old rate limiters to free memory"""
    try:
        if max_age_hours <= 0:
            raise HTTPException(status_code=400, detail="Max age must be positive")
        
        # Perform cleanup
        advanced_rate_limiter.cleanup_old_limiters(max_age_hours)
        
        return {
            "success": True,
            "message": f"Cleaned up limiters older than {max_age_hours} hours"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up limiters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _get_algorithm_description(algorithm: RateLimitAlgorithm) -> str:
    """Get description for rate limiting algorithm"""
    descriptions = {
        RateLimitAlgorithm.TOKEN_BUCKET: "Token bucket algorithm - allows burst traffic up to bucket capacity",
        RateLimitAlgorithm.SLIDING_WINDOW: "Sliding window algorithm - smooth rate limiting over time window",
        RateLimitAlgorithm.FIXED_WINDOW: "Fixed window algorithm - rate limiting over fixed time periods",
        RateLimitAlgorithm.LEAKY_BUCKET: "Leaky bucket algorithm - smooths out traffic bursts"
    }
    return descriptions.get(algorithm, "Unknown algorithm")

def _get_scope_description(scope: RateLimitScope) -> str:
    """Get description for rate limiting scope"""
    descriptions = {
        RateLimitScope.GLOBAL: "Global rate limiting across all requests",
        RateLimitScope.USER: "Rate limiting per authenticated user",
        RateLimitScope.IP: "Rate limiting per IP address",
        RateLimitScope.ENDPOINT: "Rate limiting per API endpoint",
        RateLimitScope.API_KEY: "Rate limiting per API key"
    }
    return descriptions.get(scope, "Unknown scope")
