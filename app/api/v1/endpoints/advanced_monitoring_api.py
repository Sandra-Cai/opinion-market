"""
Advanced Monitoring API
API endpoints for advanced monitoring and observability
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.services.advanced_monitoring_engine import advanced_monitoring_engine, AlertSeverity, ServiceStatus

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class MetricRequest(BaseModel):
    """Metric request model"""
    name: str
    value: float
    labels: Optional[Dict[str, str]] = None
    description: Optional[str] = None


class AlertRuleRequest(BaseModel):
    """Alert rule request model"""
    name: str
    metric: str
    threshold: float
    severity: str
    description: str


class ServiceHealthRequest(BaseModel):
    """Service health request model"""
    service_name: str
    status: str
    response_time: float
    error_rate: float
    throughput: float


# API Endpoints
@router.get("/status")
async def get_monitoring_status():
    """Get monitoring system status"""
    try:
        summary = advanced_monitoring_engine.get_monitoring_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics():
    """Get all metrics"""
    try:
        metrics = {}
        for name, metric in advanced_monitoring_engine.metrics.items():
            metrics[name] = {
                "value": metric.value,
                "type": metric.metric_type.value,
                "labels": metric.labels,
                "timestamp": metric.timestamp.isoformat(),
                "description": metric.description,
                "unit": metric.unit
            }
            
        return JSONResponse(content={"metrics": metrics})
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics")
async def record_metric(metric_request: MetricRequest):
    """Record a new metric"""
    try:
        from app.services.advanced_monitoring_engine import MetricType
        
        # Record the metric
        await advanced_monitoring_engine._record_metric(
            metric_request.name,
            metric_request.value,
            MetricType.GAUGE,
            metric_request.labels
        )
        
        return JSONResponse(content={"message": "Metric recorded successfully"})
        
    except Exception as e:
        logger.error(f"Error recording metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prometheus")
async def get_prometheus_metrics():
    """Get Prometheus metrics in text format"""
    try:
        metrics_text = advanced_monitoring_engine.get_prometheus_metrics()
        return PlainTextResponse(content=metrics_text)
        
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts():
    """Get all alerts"""
    try:
        alerts = []
        for alert in advanced_monitoring_engine.alerts:
            alerts.append({
                "alert_id": alert.alert_id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "starts_at": alert.starts_at.isoformat(),
                "ends_at": alert.ends_at.isoformat() if alert.ends_at else None,
                "value": alert.value,
                "threshold": alert.threshold
            })
            
        return JSONResponse(content={"alerts": alerts})
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/rules")
async def create_alert_rule(rule_request: AlertRuleRequest):
    """Create a new alert rule"""
    try:
        # Add alert rule
        advanced_monitoring_engine.alert_rules[rule_request.name] = {
            "metric": rule_request.metric,
            "threshold": rule_request.threshold,
            "severity": AlertSeverity(rule_request.severity),
            "description": rule_request.description
        }
        
        return JSONResponse(content={"message": "Alert rule created successfully"})
        
    except Exception as e:
        logger.error(f"Error creating alert rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/health")
async def get_service_health():
    """Get health status of all services"""
    try:
        health_data = {}
        for service_name, health in advanced_monitoring_engine.service_health.items():
            health_data[service_name] = {
                "status": health.status.value,
                "uptime": health.uptime,
                "response_time": health.response_time,
                "error_rate": health.error_rate,
                "throughput": health.throughput,
                "last_check": health.last_check.isoformat(),
                "dependencies": health.dependencies,
                "metadata": health.metadata
            }
            
        return JSONResponse(content={"services": health_data})
        
    except Exception as e:
        logger.error(f"Error getting service health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/health")
async def update_service_health(health_request: ServiceHealthRequest):
    """Update service health status"""
    try:
        from app.services.advanced_monitoring_engine import ServiceHealth
        
        # Create health record
        health = ServiceHealth(
            service_name=health_request.service_name,
            status=ServiceStatus(health_request.status),
            uptime=0,  # Will be calculated
            response_time=health_request.response_time,
            error_rate=health_request.error_rate,
            throughput=health_request.throughput,
            last_check=datetime.now()
        )
        
        # Update health
        advanced_monitoring_engine.service_health[health_request.service_name] = health
        
        return JSONResponse(content={"message": "Service health updated successfully"})
        
    except Exception as e:
        logger.error(f"Error updating service health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_monitoring_dashboard():
    """Get monitoring dashboard data"""
    try:
        # Get comprehensive monitoring data
        summary = advanced_monitoring_engine.get_monitoring_summary()
        
        # Get recent metrics
        recent_metrics = {}
        for name, metric in list(advanced_monitoring_engine.metrics.items())[-10:]:
            recent_metrics[name] = {
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat()
            }
            
        # Get active alerts
        active_alerts = [
            {
                "name": alert.name,
                "severity": alert.severity.value,
                "description": alert.description,
                "value": alert.value,
                "threshold": alert.threshold
            }
            for alert in advanced_monitoring_engine.alerts
            if alert.status == "firing"
        ]
        
        dashboard_data = {
            "summary": summary,
            "recent_metrics": recent_metrics,
            "active_alerts": active_alerts,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_monitoring():
    """Start monitoring engine"""
    try:
        await advanced_monitoring_engine.start_monitoring_engine()
        return JSONResponse(content={"message": "Monitoring engine started"})
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_monitoring():
    """Stop monitoring engine"""
    try:
        await advanced_monitoring_engine.stop_monitoring_engine()
        return JSONResponse(content={"message": "Monitoring engine stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_monitoring_stats():
    """Get monitoring statistics"""
    try:
        stats = advanced_monitoring_engine.monitoring_stats
        return JSONResponse(content={"stats": stats})
        
    except Exception as e:
        logger.error(f"Error getting monitoring stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
