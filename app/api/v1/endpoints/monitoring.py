"""
Monitoring API endpoints for Opinion Market
Provides access to system metrics, performance data, and health status
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from app.core.security import get_current_active_user, require_permissions
from app.core.database import get_db
from app.core.cache import get_cache_stats, cache_health_check
from app.core.database import check_database_health, check_redis_health
from app.monitoring.metrics_collector import metrics_collector
from app.monitoring.performance_monitor import performance_monitor
from app.models.user import User

router = APIRouter()


@router.get("/health")
async def get_system_health():
    """Get comprehensive system health status"""
    try:
        # Get health status from metrics collector
        health_status = metrics_collector.get_health_status()
        
        # Add additional health checks
        db_health = check_database_health()
        redis_health = check_redis_health()
        cache_health = cache_health_check()
        
        health_status.update({
            "database": db_health,
            "redis": redis_health,
            "cache": cache_health,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return health_status
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system health: {str(e)}"
        )


@router.get("/metrics")
async def get_system_metrics(
    hours: int = Query(24, ge=1, le=168),  # 1 hour to 1 week
    metric_names: Optional[List[str]] = Query(None)
):
    """Get system metrics"""
    try:
        if metric_names:
            # Get specific metrics
            metrics_data = {}
            for metric_name in metric_names:
                history = metrics_collector.get_metric_history(metric_name, hours)
                metrics_data[metric_name] = [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "unit": m.unit,
                        "tags": m.tags
                    }
                    for m in history
                ]
        else:
            # Get all metrics summary
            metrics_data = metrics_collector.get_all_metrics_summary(hours)
        
        return {
            "metrics": metrics_data,
            "time_range_hours": hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting metrics: {str(e)}"
        )


@router.get("/metrics/{metric_name}")
async def get_metric_details(
    metric_name: str,
    hours: int = Query(24, ge=1, le=168)
):
    """Get detailed information for a specific metric"""
    try:
        # Get metric history
        history = metrics_collector.get_metric_history(metric_name, hours)
        
        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Metric '{metric_name}' not found"
            )
        
        # Get summary statistics
        summary = metrics_collector.get_metric_summary(metric_name, hours)
        
        return {
            "metric_name": metric_name,
            "summary": summary,
            "history": [
                {
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat(),
                    "unit": m.unit,
                    "tags": m.tags
                }
                for m in history
            ],
            "time_range_hours": hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting metric details: {str(e)}"
        )


@router.get("/performance")
async def get_performance_summary(
    hours: int = Query(24, ge=1, le=168)
):
    """Get performance monitoring summary"""
    try:
        summary = performance_monitor.get_performance_summary(hours)
        return summary
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting performance summary: {str(e)}"
        )


@router.get("/performance/slow-queries")
async def get_slow_queries(
    limit: int = Query(10, ge=1, le=100)
):
    """Get slowest database queries"""
    try:
        slow_queries = performance_monitor.get_slow_queries(limit)
        return {
            "slow_queries": slow_queries,
            "count": len(slow_queries),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting slow queries: {str(e)}"
        )


@router.get("/performance/functions")
async def get_function_profiles(
    sort_by: str = Query("time", regex="^(time|calls)$"),
    limit: int = Query(10, ge=1, le=100)
):
    """Get function performance profiles"""
    try:
        if sort_by == "time":
            functions = performance_monitor.get_top_functions_by_time(limit)
        else:
            functions = performance_monitor.get_top_functions_by_calls(limit)
        
        return {
            "functions": functions,
            "sort_by": sort_by,
            "count": len(functions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting function profiles: {str(e)}"
        )


@router.get("/performance/memory-snapshots")
async def get_memory_snapshots(
    limit: int = Query(5, ge=1, le=20)
):
    """Get memory snapshots for analysis"""
    try:
        snapshots = performance_monitor.get_memory_snapshots(limit)
        return {
            "memory_snapshots": snapshots,
            "count": len(snapshots),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting memory snapshots: {str(e)}"
        )


@router.get("/alerts")
async def get_active_alerts():
    """Get active system alerts"""
    try:
        health_status = metrics_collector.get_health_status()
        return {
            "alerts": health_status.get("alerts", []),
            "status": health_status.get("status", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting alerts: {str(e)}"
        )


@router.get("/dashboard")
async def get_monitoring_dashboard():
    """Get comprehensive monitoring dashboard data"""
    try:
        # Get all monitoring data
        health_status = metrics_collector.get_health_status()
        performance_summary = performance_monitor.get_performance_summary(24)
        cache_stats = get_cache_stats()
        
        # Get key metrics
        key_metrics = {}
        metric_names = [
            "system.cpu.usage",
            "system.memory.usage",
            "system.disk.usage",
            "database.response_time",
            "cache.health",
            "business.users.active",
            "business.markets.active",
            "business.volume.24h"
        ]
        
        for metric_name in metric_names:
            latest = metrics_collector.get_latest_metric(metric_name)
            if latest:
                key_metrics[metric_name] = {
                    "value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp.isoformat()
                }
        
        return {
            "health": health_status,
            "performance": performance_summary,
            "cache": cache_stats,
            "key_metrics": key_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting dashboard data: {str(e)}"
        )


@router.post("/performance/memory-snapshot")
async def take_memory_snapshot(
    reason: str = "manual",
    current_user: User = Depends(get_current_active_user)
):
    """Take a manual memory snapshot"""
    try:
        await performance_monitor._take_memory_snapshot(reason)
        return {
            "message": "Memory snapshot taken successfully",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error taking memory snapshot: {str(e)}"
        )


@router.get("/system-info")
async def get_system_info():
    """Get system information"""
    try:
        import platform
        import psutil
        
        return {
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            },
            "application": {
                "name": "Opinion Market API",
                "version": "2.0.0",
                "environment": "development",  # This would come from settings
                "uptime": performance_monitor.get_performance_summary().get("uptime", 0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting system info: {str(e)}"
        )


@router.get("/logs")
async def get_recent_logs(
    level: Optional[str] = Query(None, regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user)
):
    """Get recent application logs"""
    try:
        # This would typically read from log files or a log aggregation system
        # For now, return a placeholder response
        return {
            "message": "Log retrieval not implemented yet",
            "level": level,
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting logs: {str(e)}"
        )


@router.get("/status")
async def get_service_status():
    """Get status of all services"""
    try:
        # This would check the status of all registered services
        # For now, return basic status information
        return {
            "services": {
                "api": "healthy",
                "database": "healthy",
                "cache": "healthy",
                "metrics_collector": "healthy" if metrics_collector.running else "stopped",
                "performance_monitor": "healthy" if performance_monitor.running else "stopped"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting service status: {str(e)}"
        )