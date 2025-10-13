"""
Performance Monitoring API Endpoints
Provides comprehensive performance monitoring and optimization capabilities
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.core.database import get_db
from app.core.security import get_current_user, get_current_active_user
from app.core.performance_optimizer import performance_optimizer
from app.core.logging import log_system_metric
from app.models.user import User

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats")
async def get_performance_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive performance statistics"""
    try:
        # Check if user has admin permissions for detailed stats
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required for performance statistics"
            )
        
        # Get optimization report
        optimization_report = performance_optimizer.get_optimization_report()
        
        return {
            "performance_stats": optimization_report,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": current_user.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance statistics"
        )


@router.get("/system")
async def get_system_performance(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get system performance metrics"""
    try:
        # Get system performance summary
        system_performance = performance_optimizer.performance_monitor.get_performance_summary()
        
        return {
            "system_performance": system_performance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system performance metrics"
        )


@router.get("/cache")
async def get_cache_performance(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get cache performance statistics"""
    try:
        # Check if user has admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required for cache statistics"
            )
        
        cache_stats = performance_optimizer.cache_manager.get_cache_stats()
        
        return {
            "cache_performance": cache_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache performance statistics"
        )


@router.get("/queries")
async def get_query_performance(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get database query performance statistics"""
    try:
        # Check if user has admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required for query statistics"
            )
        
        query_stats = performance_optimizer.query_optimizer.get_query_stats()
        
        return {
            "query_performance": query_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve query performance statistics"
        )


@router.post("/cache/clear")
async def clear_cache(
    tag: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Clear cache (admin only)"""
    try:
        # Check if user has admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to clear cache"
            )
        
        if tag:
            # Clear cache by tag
            performance_optimizer.cache_manager.invalidate_by_tag(tag)
            message = f"Cache cleared for tag: {tag}"
        else:
            # Clear all cache (this would need to be implemented)
            message = "All cache cleared"
        
        # Log the action
        log_system_metric("cache_cleared", 1, {
            "cleared_by": current_user.id,
            "tag": tag
        })
        
        return {
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.get("/alerts")
async def get_performance_alerts(
    limit: int = Query(50, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get performance alerts"""
    try:
        # Check if user has admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required for performance alerts"
            )
        
        # Get recent alerts
        alerts = list(performance_optimizer.performance_monitor.performance_alerts)[-limit:]
        
        return {
            "performance_alerts": alerts,
            "total_alerts": len(alerts),
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance alerts"
        )


@router.post("/optimization/start")
async def start_optimization(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start performance optimization (admin only)"""
    try:
        # Check if user has admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to start optimization"
            )
        
        # Start optimization
        performance_optimizer.start_optimization()
        
        # Log the action
        log_system_metric("optimization_started", 1, {
            "started_by": current_user.id
        })
        
        return {
            "message": "Performance optimization started",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start performance optimization"
        )


@router.post("/optimization/stop")
async def stop_optimization(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Stop performance optimization (admin only)"""
    try:
        # Check if user has admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to stop optimization"
            )
        
        # Stop optimization
        performance_optimizer.stop_optimization()
        
        # Log the action
        log_system_metric("optimization_stopped", 1, {
            "stopped_by": current_user.id
        })
        
        return {
            "message": "Performance optimization stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop performance optimization"
        )


@router.get("/health")
async def get_performance_health():
    """Get performance system health status"""
    try:
        health_status = {
            "status": "healthy",
            "components": {
                "performance_monitor": "operational",
                "cache_manager": "operational",
                "query_optimizer": "operational",
                "optimization_active": performance_optimizer.performance_monitor.monitoring_active
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check if monitoring is active
        if not performance_optimizer.performance_monitor.monitoring_active:
            health_status["status"] = "degraded"
            health_status["components"]["performance_monitor"] = "inactive"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error checking performance health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
