"""
Administrative endpoints for system management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import logging

from app.core.config_manager import config_manager
from app.core.database_pool import db_pool_manager
from app.core.metrics import metrics_collector
from app.core.caching import memory_cache
from app.api.v1.endpoints.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/admin/system-status")
async def get_system_status(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get comprehensive system status"""
    try:
        # Get configuration
        config = config_manager.get_config()
        
        # Get database health
        db_health = await db_pool_manager.health_check()
        
        # Get metrics
        metrics = await metrics_collector.get_metrics()
        
        # Get cache stats
        cache_stats = await memory_cache.get_stats()
        
        # Get database connection stats
        db_stats = db_pool_manager.get_connection_stats()
        
        return {
            "timestamp": metrics.get("timestamp"),
            "system": {
                "environment": config.environment.value,
                "debug_mode": config.debug,
                "version": config.api.version
            },
            "database": {
                "status": db_health["status"],
                "connection_pool": db_stats,
                "response_time": db_health.get("response_time", 0)
            },
            "cache": cache_stats,
            "metrics": {
                "counters": metrics.get("counters", {}),
                "gauges": metrics.get("gauges", {}),
                "timers": metrics.get("timers", {})
            },
            "overall_status": "healthy" if db_health["status"] == "healthy" else "degraded"
        }
    
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system status"
        )


@router.get("/admin/configuration")
async def get_configuration(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current system configuration"""
    try:
        config = config_manager.get_config()
        
        # Return safe configuration (exclude sensitive data)
        return {
            "environment": config.environment.value,
            "debug": config.debug,
            "api": {
                "title": config.api.title,
                "version": config.api.version,
                "docs_url": config.api.docs_url
            },
            "database": {
                "host": config.database.host,
                "port": config.database.port,
                "name": config.database.name,
                "pool_size": config.database.pool_size,
                "max_overflow": config.database.max_overflow
            },
            "redis": {
                "host": config.redis.host,
                "port": config.redis.port,
                "db": config.redis.db,
                "max_connections": config.redis.max_connections
            },
            "monitoring": {
                "log_level": config.monitoring.log_level,
                "enable_metrics": config.monitoring.enable_metrics,
                "enable_health_checks": config.monitoring.enable_health_checks
            },
            "cache": {
                "enable_memory_cache": config.cache.enable_memory_cache,
                "memory_cache_size": config.cache.memory_cache_size,
                "memory_cache_ttl": config.cache.memory_cache_ttl
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration"
        )


@router.post("/admin/configuration/validate")
async def validate_configuration(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Validate current configuration"""
    try:
        validation_results = config_manager.validate_config()
        
        return {
            "valid": validation_results["valid"],
            "errors": validation_results["errors"],
            "warnings": validation_results["warnings"],
            "recommendations": [
                "Ensure secret keys are properly configured for production",
                "Set appropriate database connection pool sizes",
                "Configure proper logging levels for production",
                "Enable security features for production deployment"
            ]
        }
    
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate configuration"
        )


@router.post("/admin/cache/clear")
async def clear_cache(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Clear application cache"""
    try:
        await memory_cache.clear()
        
        logger.info(f"Cache cleared by user {current_user.get('user_id', 'unknown')}")
        
        return {
            "message": "Cache cleared successfully",
            "timestamp": metrics_collector.get_metrics().get("timestamp")
        }
    
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.get("/admin/metrics/reset")
async def reset_metrics(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Reset application metrics"""
    try:
        await metrics_collector.reset()
        
        logger.info(f"Metrics reset by user {current_user.get('user_id', 'unknown')}")
        
        return {
            "message": "Metrics reset successfully",
            "timestamp": metrics_collector.get_metrics().get("timestamp")
        }
    
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset metrics"
        )


@router.get("/admin/logs")
async def get_recent_logs(
    lines: int = 100,
    level: str = "INFO",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get recent application logs"""
    try:
        # In a real implementation, you would read from log files
        # For now, return a mock response
        return {
            "logs": [
                {
                    "timestamp": "2024-01-01T12:00:00Z",
                    "level": "INFO",
                    "message": "Application started successfully",
                    "module": "main"
                },
                {
                    "timestamp": "2024-01-01T12:01:00Z",
                    "level": "INFO",
                    "message": "Database connection established",
                    "module": "database"
                },
                {
                    "timestamp": "2024-01-01T12:02:00Z",
                    "level": "INFO",
                    "message": "Cache initialized",
                    "module": "cache"
                }
            ],
            "total_lines": 3,
            "requested_lines": lines,
            "level_filter": level
        }
    
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve logs"
        )


@router.get("/admin/health/detailed")
async def get_detailed_health(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get detailed health information"""
    try:
        import psutil
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Database health
        db_health = await db_pool_manager.health_check()
        
        # Application metrics
        metrics = await metrics_collector.get_metrics()
        
        return {
            "timestamp": metrics.get("timestamp"),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "database": db_health,
            "application": {
                "metrics": metrics,
                "cache_stats": await memory_cache.get_stats()
            },
            "status": "healthy" if cpu_percent < 80 and memory.percent < 80 and db_health["status"] == "healthy" else "warning"
        }
    
    except Exception as e:
        logger.error(f"Failed to get detailed health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve detailed health information"
        )
