"""
Real-time monitoring dashboard endpoints
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    WebSocket,
    WebSocketDisconnect,
)
from typing import Dict, Any, List
import asyncio
import json
import logging
from datetime import datetime, timedelta

from app.core.metrics import metrics_collector
from app.core.caching import memory_cache
from app.core.database_pool import db_pool_manager
from app.core.health_monitor import health_monitor
from app.core.config_manager import config_manager
from app.api.v1.endpoints.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")
                self.disconnect(connection)


manager = ConnectionManager()


@router.get("/dashboard/overview")
async def get_dashboard_overview(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get comprehensive dashboard overview"""
    try:
        # Get system metrics
        metrics = await metrics_collector.get_metrics()

        # Get health status
        health_status = await health_monitor.get_comprehensive_health()

        # Get database health
        db_health = await db_pool_manager.health_check()

        # Get cache statistics
        cache_stats = await memory_cache.get_stats()

        # Get configuration
        config = config_manager.get_config()

        # Calculate system load
        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "timestamp": datetime.now().isoformat(),
            "overview": {
                "status": "operational",
                "environment": config.environment.value,
                "uptime": "running",
                "version": config.api.version,
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2),
            },
            "application": {
                "health_status": health_status["overall_status"],
                "database_status": db_health["status"],
                "cache_stats": cache_stats,
                "metrics": {
                    "total_requests": metrics.get("counters", {}).get(
                        "http_requests_total", 0
                    ),
                    "error_rate": _calculate_error_rate(metrics),
                    "avg_response_time": _calculate_avg_response_time(metrics),
                },
            },
            "alerts": _get_active_alerts(
                cpu_percent, memory.percent, db_health["status"]
            ),
        }

    except Exception as e:
        logger.error(f"Failed to get dashboard overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard overview",
        )


@router.get("/dashboard/metrics")
async def get_dashboard_metrics(
    time_range: str = "1h", current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detailed metrics for dashboard"""
    try:
        metrics = await metrics_collector.get_metrics()

        # Simulate time series data (in real app, this would come from time series DB)
        time_series_data = _generate_time_series_data(time_range)

        return {
            "timestamp": datetime.now().isoformat(),
            "time_range": time_range,
            "current_metrics": metrics,
            "time_series": time_series_data,
            "summary": {
                "total_requests": metrics.get("counters", {}).get(
                    "http_requests_total", 0
                ),
                "active_connections": len(manager.active_connections),
                "cache_hit_rate": _calculate_cache_hit_rate(cache_stats),
                "error_count": metrics.get("counters", {}).get("http_requests_500", 0),
            },
        }

    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard metrics",
        )


@router.get("/dashboard/alerts")
async def get_dashboard_alerts(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get active alerts and notifications"""
    try:
        # Get current system state
        import psutil

        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        db_health = await db_pool_manager.health_check()

        alerts = []

        # CPU alerts
        if cpu_percent > 90:
            alerts.append(
                {
                    "id": "high_cpu",
                    "severity": "critical",
                    "title": "High CPU Usage",
                    "message": f"CPU usage is {cpu_percent}%",
                    "timestamp": datetime.now().isoformat(),
                    "status": "active",
                }
            )
        elif cpu_percent > 80:
            alerts.append(
                {
                    "id": "medium_cpu",
                    "severity": "warning",
                    "title": "Elevated CPU Usage",
                    "message": f"CPU usage is {cpu_percent}%",
                    "timestamp": datetime.now().isoformat(),
                    "status": "active",
                }
            )

        # Memory alerts
        if memory_percent > 90:
            alerts.append(
                {
                    "id": "high_memory",
                    "severity": "critical",
                    "title": "High Memory Usage",
                    "message": f"Memory usage is {memory_percent}%",
                    "timestamp": datetime.now().isoformat(),
                    "status": "active",
                }
            )

        # Database alerts
        if db_health["status"] != "healthy":
            alerts.append(
                {
                    "id": "database_issue",
                    "severity": "critical",
                    "title": "Database Issue",
                    "message": f"Database status: {db_health['status']}",
                    "timestamp": datetime.now().isoformat(),
                    "status": "active",
                }
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_alerts": len(alerts),
            "active_alerts": len([a for a in alerts if a["status"] == "active"]),
            "alerts": alerts,
        }

    except Exception as e:
        logger.error(f"Failed to get dashboard alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard alerts",
        )


@router.websocket("/dashboard/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring"""
    await manager.connect(websocket)

    try:
        while True:
            # Send real-time data every 5 seconds
            await asyncio.sleep(5)

            # Get current metrics
            metrics = await metrics_collector.get_metrics()
            health_status = await health_monitor.get_comprehensive_health()

            # Get system metrics
            import psutil

            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent

            # Prepare real-time data
            real_time_data = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                },
                "application": {
                    "health_status": health_status["overall_status"],
                    "total_requests": metrics.get("counters", {}).get(
                        "http_requests_total", 0
                    ),
                    "active_connections": len(manager.active_connections),
                },
            }

            # Send to all connected clients
            await manager.broadcast(json.dumps(real_time_data))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/dashboard/performance")
async def get_performance_analysis(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detailed performance analysis"""
    try:
        metrics = await metrics_collector.get_metrics()

        # Analyze performance metrics
        performance_analysis = {
            "timestamp": datetime.now().isoformat(),
            "response_times": _analyze_response_times(metrics),
            "throughput": _analyze_throughput(metrics),
            "error_rates": _analyze_error_rates(metrics),
            "resource_usage": _analyze_resource_usage(),
            "recommendations": _generate_performance_recommendations(metrics),
        }

        return performance_analysis

    except Exception as e:
        logger.error(f"Failed to get performance analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance analysis",
        )


# Helper methods
def _calculate_error_rate(self, metrics: Dict[str, Any]) -> float:
    """Calculate error rate from metrics"""
    counters = metrics.get("counters", {})
    total_requests = counters.get("http_requests_total", 0)
    error_requests = counters.get("http_requests_500", 0)

    if total_requests == 0:
        return 0.0

    return (error_requests / total_requests) * 100


def _calculate_avg_response_time(self, metrics: Dict[str, Any]) -> float:
    """Calculate average response time from metrics"""
    timers = metrics.get("timers", {})
    http_timer = timers.get("http_request_duration", {})

    return http_timer.get("avg", 0.0)


def _get_active_alerts(
    self, cpu_percent: float, memory_percent: float, db_status: str
) -> List[Dict[str, Any]]:
    """Get active alerts based on current system state"""
    alerts = []

    if cpu_percent > 80:
        alerts.append(
            {
                "type": "cpu",
                "severity": "warning" if cpu_percent < 90 else "critical",
                "message": f"High CPU usage: {cpu_percent}%",
            }
        )

    if memory_percent > 80:
        alerts.append(
            {
                "type": "memory",
                "severity": "warning" if memory_percent < 90 else "critical",
                "message": f"High memory usage: {memory_percent}%",
            }
        )

    if db_status != "healthy":
        alerts.append(
            {
                "type": "database",
                "severity": "critical",
                "message": f"Database issue: {db_status}",
            }
        )

    return alerts


def _generate_time_series_data(self, time_range: str) -> Dict[str, List]:
    """Generate mock time series data"""
    # In a real app, this would query a time series database
    return {
        "cpu_usage": [{"timestamp": "2024-01-01T12:00:00Z", "value": 45.2}],
        "memory_usage": [{"timestamp": "2024-01-01T12:00:00Z", "value": 67.8}],
        "request_rate": [{"timestamp": "2024-01-01T12:00:00Z", "value": 150}],
        "response_time": [{"timestamp": "2024-01-01T12:00:00Z", "value": 0.125}],
    }


def _calculate_cache_hit_rate(self, cache_stats: Dict[str, Any]) -> float:
    """Calculate cache hit rate"""
    # Mock calculation - in real app, this would be tracked
    return 85.5


def _analyze_response_times(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze response time metrics"""
    timers = metrics.get("timers", {})
    http_timer = timers.get("http_request_duration", {})

    return {
        "average": http_timer.get("avg", 0.0),
        "p95": http_timer.get("p95", 0.0),
        "p99": http_timer.get("p99", 0.0),
        "max": http_timer.get("max", 0.0),
        "min": http_timer.get("min", 0.0),
    }


def _analyze_throughput(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze throughput metrics"""
    counters = metrics.get("counters", {})
    total_requests = counters.get("http_requests_total", 0)

    return {
        "total_requests": total_requests,
        "requests_per_second": total_requests / 3600,  # Mock calculation
        "peak_throughput": total_requests * 1.5,  # Mock calculation
    }


def _analyze_error_rates(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze error rate metrics"""
    counters = metrics.get("counters", {})
    total_requests = counters.get("http_requests_total", 0)
    error_requests = counters.get("http_requests_500", 0)

    error_rate = (error_requests / total_requests * 100) if total_requests > 0 else 0

    return {
        "error_rate_percent": error_rate,
        "total_errors": error_requests,
        "error_trend": "stable",  # Mock trend
    }


def _analyze_resource_usage(self) -> Dict[str, Any]:
    """Analyze resource usage"""
    import psutil

    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "load_average": (
            psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0
        ),
    }


def _generate_performance_recommendations(
    self, metrics: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate performance recommendations"""
    recommendations = []

    # Check response times
    timers = metrics.get("timers", {})
    http_timer = timers.get("http_request_duration", {})
    avg_response_time = http_timer.get("avg", 0.0)

    if avg_response_time > 0.5:  # 500ms
        recommendations.append(
            {
                "type": "performance",
                "priority": "high",
                "title": "Optimize Response Times",
                "description": f"Average response time is {avg_response_time:.3f}s",
                "action": "Consider implementing caching or query optimization",
            }
        )

    # Check error rates
    counters = metrics.get("counters", {})
    total_requests = counters.get("http_requests_total", 0)
    error_requests = counters.get("http_requests_500", 0)

    if total_requests > 0:
        error_rate = (error_requests / total_requests) * 100
        if error_rate > 5:
            recommendations.append(
                {
                    "type": "reliability",
                    "priority": "high",
                    "title": "High Error Rate",
                    "description": f"Error rate is {error_rate:.1f}%",
                    "action": "Investigate and fix error sources",
                }
            )

    return recommendations


# Helper functions
def _calculate_error_rate(metrics: Dict[str, Any]) -> float:
    """Calculate error rate from metrics"""
    counters = metrics.get("counters", {})
    total_requests = counters.get("http_requests_total", 0)
    error_requests = counters.get("http_requests_500", 0)

    if total_requests == 0:
        return 0.0

    return (error_requests / total_requests) * 100


def _calculate_avg_response_time(metrics: Dict[str, Any]) -> float:
    """Calculate average response time from metrics"""
    timers = metrics.get("timers", {})
    http_timer = timers.get("http_request_duration", {})
    return http_timer.get("avg", 0.0)


def _get_active_alerts(
    cpu_percent: float, memory_percent: float, db_status: str
) -> List[Dict[str, Any]]:
    """Get active system alerts"""
    alerts = []

    if cpu_percent > 80:
        alerts.append(
            {
                "type": "warning",
                "title": "High CPU Usage",
                "message": f"CPU usage is {cpu_percent:.1f}%",
                "timestamp": datetime.now().isoformat(),
            }
        )

    if memory_percent > 85:
        alerts.append(
            {
                "type": "warning",
                "title": "High Memory Usage",
                "message": f"Memory usage is {memory_percent:.1f}%",
                "timestamp": datetime.now().isoformat(),
            }
        )

    if db_status != "healthy":
        alerts.append(
            {
                "type": "error",
                "title": "Database Issue",
                "message": f"Database status: {db_status}",
                "timestamp": datetime.now().isoformat(),
            }
        )

    return alerts


def _generate_time_series_data(time_range: str) -> Dict[str, Any]:
    """Generate time series data for dashboard"""
    # This would typically come from a time series database
    # For now, return mock data
    import random

    points = 24 if time_range == "24h" else 168  # 24h or 7d

    return {
        "cpu_usage": [random.uniform(20, 80) for _ in range(points)],
        "memory_usage": [random.uniform(30, 70) for _ in range(points)],
        "response_times": [random.uniform(0.1, 0.5) for _ in range(points)],
        "request_count": [random.randint(100, 1000) for _ in range(points)],
        "error_rate": [random.uniform(0, 5) for _ in range(points)],
    }


def _calculate_cache_hit_rate(cache_stats: Dict[str, Any]) -> float:
    """Calculate cache hit rate"""
    hits = cache_stats.get("hits", 0)
    misses = cache_stats.get("misses", 0)
    total = hits + misses

    if total == 0:
        return 0.0

    return (hits / total) * 100


def _analyze_response_times(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze response time metrics"""
    timers = metrics.get("timers", {})
    http_timer = timers.get("http_request_duration", {})

    return {
        "avg": http_timer.get("avg", 0.0),
        "min": http_timer.get("min", 0.0),
        "max": http_timer.get("max", 0.0),
        "p95": http_timer.get("p95", 0.0),
        "p99": http_timer.get("p99", 0.0),
    }


def _analyze_throughput(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze throughput metrics"""
    counters = metrics.get("counters", {})

    return {
        "requests_per_second": counters.get("http_requests_total", 0)
        / 3600,  # Approximate
        "total_requests": counters.get("http_requests_total", 0),
        "successful_requests": counters.get("http_requests_200", 0),
        "failed_requests": counters.get("http_requests_500", 0),
    }


def _analyze_error_rates(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze error rate metrics"""
    counters = metrics.get("counters", {})
    total = counters.get("http_requests_total", 0)

    if total == 0:
        return {"overall": 0.0, "by_status": {}}

    return {
        "overall": (counters.get("http_requests_500", 0) / total) * 100,
        "by_status": {
            "4xx": (counters.get("http_requests_400", 0) / total) * 100,
            "5xx": (counters.get("http_requests_500", 0) / total) * 100,
        },
    }


def _analyze_resource_usage() -> Dict[str, Any]:
    """Analyze system resource usage"""
    import psutil

    return {
        "cpu": {"percent": psutil.cpu_percent(interval=1), "count": psutil.cpu_count()},
        "memory": {
            "percent": psutil.virtual_memory().percent,
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        },
        "disk": {
            "percent": psutil.disk_usage("/").percent,
            "free_gb": round(psutil.disk_usage("/").free / (1024**3), 2),
        },
    }
