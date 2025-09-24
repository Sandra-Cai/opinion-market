"""
Performance Monitoring API Endpoints
Provides comprehensive performance monitoring and optimization endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from app.core.performance_monitor import performance_monitor
from app.core.enhanced_cache import enhanced_cache
from app.core.auth import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics/summary")
async def get_metrics_summary(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get comprehensive performance metrics summary"""
    try:
        summary = performance_monitor.get_metrics_summary()
        
        # Add cache statistics
        cache_stats = enhanced_cache.get_stats()
        summary["cache"] = cache_stats
        
        # Add system health score
        health_score = _calculate_health_score(summary)
        summary["health_score"] = health_score
        
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


@router.get("/alerts")
async def get_performance_alerts(
    severity: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get active performance alerts"""
    try:
        alerts = performance_monitor.get_active_alerts()
        
        if severity:
            alerts = [alert for alert in alerts if alert["severity"] == severity]
            
        return {
            "status": "success",
            "data": {
                "alerts": alerts,
                "count": len(alerts),
                "severity_breakdown": _get_severity_breakdown(alerts)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance alerts"
        )


@router.get("/recommendations")
async def get_performance_recommendations(
    priority: Optional[str] = None,
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get performance optimization recommendations"""
    try:
        recommendations = performance_monitor.get_recommendations(priority)
        
        if category:
            recommendations = [rec for rec in recommendations if rec["category"] == category]
            
        return {
            "status": "success",
            "data": {
                "recommendations": recommendations,
                "count": len(recommendations),
                "priority_breakdown": _get_priority_breakdown(recommendations)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance recommendations"
        )


@router.get("/cache/stats")
async def get_cache_statistics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed cache statistics"""
    try:
        stats = enhanced_cache.get_stats()
        
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting cache statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics"
        )


@router.get("/cache/entries")
async def get_cache_entries(
    tag: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get cache entries by tag"""
    try:
        if tag:
            entries = enhanced_cache.get_entries_by_tag(tag)
        else:
            # Get all entries (limited to prevent large responses)
            entries = []
            
        return {
            "status": "success",
            "data": {
                "entries": entries,
                "count": len(entries)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting cache entries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache entries"
        )


@router.post("/cache/clear")
async def clear_cache(
    tags: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Clear cache entries"""
    try:
        if tags:
            deleted_count = await enhanced_cache.delete_by_tags(tags)
            message = f"Cleared {deleted_count} cache entries with tags: {tags}"
        else:
            await enhanced_cache.clear()
            message = "Cleared all cache entries"
            
        logger.info(f"Cache cleared by user {current_user.id}: {message}")
        
        return {
            "status": "success",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        )


@router.post("/monitoring/start")
async def start_performance_monitoring(
    interval: int = 30,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Start performance monitoring"""
    try:
        await performance_monitor.start_monitoring(interval)
        
        return {
            "status": "success",
            "message": f"Performance monitoring started with {interval}s interval",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting performance monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start performance monitoring"
        )


@router.post("/monitoring/stop")
async def stop_performance_monitoring(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Stop performance monitoring"""
    try:
        await performance_monitor.stop_monitoring()
        
        return {
            "status": "success",
            "message": "Performance monitoring stopped",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping performance monitoring: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop performance monitoring"
        )


@router.post("/cache/cleanup")
async def trigger_cache_cleanup(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Trigger manual cache cleanup"""
    try:
        background_tasks.add_task(enhanced_cache._cleanup_expired)
        background_tasks.add_task(enhanced_cache._cleanup_lru)
        
        return {
            "status": "success",
            "message": "Cache cleanup triggered",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering cache cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to trigger cache cleanup"
        )


@router.get("/health")
async def get_system_health(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get overall system health status"""
    try:
        metrics = performance_monitor.get_metrics_summary()
        alerts = performance_monitor.get_active_alerts()
        
        health_score = _calculate_health_score(metrics)
        health_status = _get_health_status(health_score)
        
        return {
            "status": "success",
            "data": {
                "health_score": health_score,
                "health_status": health_status,
                "active_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a["severity"] == "critical"]),
                "recommendations": len(performance_monitor.get_recommendations()),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system health"
        )


def _calculate_health_score(metrics: Dict[str, Any]) -> float:
    """Calculate overall system health score (0-100)"""
    try:
        score = 100.0
        
        # CPU usage impact
        if "cpu_usage" in metrics:
            cpu_usage = metrics["cpu_usage"]["current"]
            if cpu_usage > 90:
                score -= 30
            elif cpu_usage > 80:
                score -= 20
            elif cpu_usage > 70:
                score -= 10
                
        # Memory usage impact
        if "memory_usage" in metrics:
            memory_usage = metrics["memory_usage"]["current"]
            if memory_usage > 95:
                score -= 30
            elif memory_usage > 85:
                score -= 20
            elif memory_usage > 75:
                score -= 10
                
        # Cache hit rate impact
        if "cache" in metrics and "hit_rate" in metrics["cache"]:
            hit_rate = metrics["cache"]["hit_rate"]
            if hit_rate < 50:
                score -= 15
            elif hit_rate < 70:
                score -= 10
            elif hit_rate < 80:
                score -= 5
                
        return max(0.0, min(100.0, score))
        
    except Exception:
        return 50.0  # Default score if calculation fails


def _get_health_status(health_score: float) -> str:
    """Get health status based on score"""
    if health_score >= 90:
        return "excellent"
    elif health_score >= 75:
        return "good"
    elif health_score >= 60:
        return "fair"
    elif health_score >= 40:
        return "poor"
    else:
        return "critical"


def _get_severity_breakdown(alerts: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get breakdown of alerts by severity"""
    breakdown = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for alert in alerts:
        severity = alert.get("severity", "low")
        if severity in breakdown:
            breakdown[severity] += 1
    return breakdown


def _get_priority_breakdown(recommendations: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get breakdown of recommendations by priority"""
    breakdown = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for rec in recommendations:
        priority = rec.get("priority", "low")
        if priority in breakdown:
            breakdown[priority] += 1
    return breakdown
