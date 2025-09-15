"""
Enhanced analytics endpoints with advanced features
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import logging

from app.core.metrics import metrics_collector, TimerContext
from app.core.caching import cache_decorator
from app.core.performance_optimizer import performance_monitor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/analytics/performance")
@performance_monitor
async def get_performance_analytics():
    """Get comprehensive performance analytics"""
    async with TimerContext(metrics_collector, "analytics_performance"):
        # Get all metrics
        metrics = await metrics_collector.get_metrics()

        # Calculate performance insights
        performance_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_performance": "excellent",
            "metrics": metrics,
            "insights": {
                "avg_response_time": 0,
                "request_rate": 0,
                "error_rate": 0,
                "cache_hit_rate": 0,
            },
        }

        # Calculate insights from metrics
        if "timers" in metrics and "http_request_duration" in metrics["timers"]:
            timer_data = metrics["timers"]["http_request_duration"]
            performance_data["insights"]["avg_response_time"] = timer_data.get("avg", 0)

        if "counters" in metrics and "http_requests_total" in metrics["counters"]:
            total_requests = metrics["counters"]["http_requests_total"]
            performance_data["insights"]["request_rate"] = total_requests

        # Determine overall performance rating
        avg_response_time = performance_data["insights"]["avg_response_time"]
        if avg_response_time < 100:  # ms
            performance_data["overall_performance"] = "excellent"
        elif avg_response_time < 500:
            performance_data["overall_performance"] = "good"
        elif avg_response_time < 1000:
            performance_data["overall_performance"] = "acceptable"
        else:
            performance_data["overall_performance"] = "needs_improvement"

        return performance_data


@router.get("/analytics/trends")
@cache_decorator
@performance_monitor
async def get_trend_analytics(
    period: str = Query("7d", description="Analysis period"),
    metric: str = Query("requests", description="Metric to analyze"),
):
    """Get trend analytics for specified metric"""
    async with TimerContext(metrics_collector, "analytics_trends"):
        # Simulate trend analysis
        trends = {
            "period": period,
            "metric": metric,
            "trend": "increasing",
            "change_percent": 15.5,
            "data_points": [
                {"timestamp": "2024-01-01", "value": 100},
                {"timestamp": "2024-01-02", "value": 115},
                {"timestamp": "2024-01-03", "value": 130},
                {"timestamp": "2024-01-04", "value": 145},
                {"timestamp": "2024-01-05", "value": 160},
            ],
            "forecast": {"next_week": 180, "confidence": 0.85},
        }

        return trends


@router.get("/analytics/health-score")
@performance_monitor
async def get_health_score():
    """Calculate overall system health score"""
    async with TimerContext(metrics_collector, "analytics_health_score"):
        # Get system metrics
        import psutil

        # Calculate health score based on multiple factors
        cpu_health = max(0, 100 - psutil.cpu_percent())
        memory_health = max(0, 100 - psutil.virtual_memory().percent)
        disk_health = max(0, 100 - psutil.disk_usage("/").percent)

        # Get application metrics
        metrics = await metrics_collector.get_metrics()

        # Calculate response time health
        response_time_health = 100
        if "timers" in metrics and "http_request_duration" in metrics["timers"]:
            avg_response_time = metrics["timers"]["http_request_duration"].get("avg", 0)
            if avg_response_time > 0:
                response_time_health = max(
                    0, 100 - (avg_response_time * 1000)
                )  # Convert to ms

        # Calculate overall health score
        health_score = (
            cpu_health * 0.25
            + memory_health * 0.25
            + disk_health * 0.25
            + response_time_health * 0.25
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health_score": round(health_score, 2),
            "components": {
                "cpu_health": round(cpu_health, 2),
                "memory_health": round(memory_health, 2),
                "disk_health": round(disk_health, 2),
                "response_time_health": round(response_time_health, 2),
            },
            "status": (
                "excellent"
                if health_score >= 90
                else (
                    "good"
                    if health_score >= 70
                    else "warning" if health_score >= 50 else "critical"
                )
            ),
        }


@router.get("/analytics/recommendations")
@performance_monitor
async def get_optimization_recommendations():
    """Get optimization recommendations based on current metrics"""
    async with TimerContext(metrics_collector, "analytics_recommendations"):
        recommendations = []

        # Get current metrics
        metrics = await metrics_collector.get_metrics()

        # Check response times
        if "timers" in metrics and "http_request_duration" in metrics["timers"]:
            avg_response_time = metrics["timers"]["http_request_duration"].get("avg", 0)
            if avg_response_time > 0.5:  # 500ms
                recommendations.append(
                    {
                        "type": "performance",
                        "priority": "high",
                        "title": "Optimize Response Times",
                        "description": f"Average response time is {avg_response_time:.3f}s, consider caching or query optimization",
                        "impact": "high",
                    }
                )

        # Check memory usage
        import psutil

        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            recommendations.append(
                {
                    "type": "resource",
                    "priority": "medium",
                    "title": "High Memory Usage",
                    "description": f"Memory usage is {memory_percent}%, consider memory optimization",
                    "impact": "medium",
                }
            )

        # Check CPU usage
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > 80:
            recommendations.append(
                {
                    "type": "resource",
                    "priority": "high",
                    "title": "High CPU Usage",
                    "description": f"CPU usage is {cpu_percent}%, consider load balancing or optimization",
                    "impact": "high",
                }
            )

        # Default recommendations if no issues found
        if not recommendations:
            recommendations.append(
                {
                    "type": "general",
                    "priority": "low",
                    "title": "System Running Smoothly",
                    "description": "No immediate optimization needed, continue monitoring",
                    "impact": "low",
                }
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
        }
