"""
Monitoring and Analytics API Endpoints
Provides comprehensive monitoring, alerting, and analytics capabilities
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import time

from app.core.auth import get_current_user
from app.models.user import User
from app.monitoring.monitoring_manager import monitoring_manager, AlertSeverity, MetricType
from app.analytics.analytics_engine import analytics_engine, AnalyticsType, MetricCategory

router = APIRouter()


@router.get("/health")
async def get_system_health(current_user: User = Depends(get_current_user)):
    """Get overall system health status"""
    try:
        health_status = await monitoring_manager.get_system_health()
        
        return {
            "success": True,
            "data": health_status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@router.get("/metrics")
async def get_metrics(
    metric_name: Optional[str] = Query(None, description="Specific metric name to retrieve"),
    time_range: int = Query(3600, description="Time range in seconds"),
    current_user: User = Depends(get_current_user)
):
    """Get system metrics"""
    try:
        if metric_name:
            metrics = await monitoring_manager.get_metrics(metric_name, time_range)
            return {
                "success": True,
                "data": {
                    "metric_name": metric_name,
                    "metrics": [{
                        "value": m.value,
                        "timestamp": m.timestamp,
                        "labels": m.labels
                    } for m in metrics]
                },
                "timestamp": time.time()
            }
        else:
            # Get all available metrics
            all_metrics = {}
            for metric_name in monitoring_manager.metrics.keys():
                metrics = await monitoring_manager.get_metrics(metric_name, time_range)
                if metrics:
                    all_metrics[metric_name] = [{
                        "value": m.value,
                        "timestamp": m.timestamp,
                        "labels": m.labels
                    } for m in metrics]
            
            return {
                "success": True,
                "data": all_metrics,
                "timestamp": time.time()
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by alert severity"),
    active_only: bool = Query(True, description="Show only active alerts"),
    current_user: User = Depends(get_current_user)
):
    """Get system alerts"""
    try:
        if active_only:
            alerts = await monitoring_manager.get_active_alerts()
        else:
            alerts = await monitoring_manager.get_alert_history()
        
        # Filter by severity if specified
        if severity:
            try:
                severity_enum = AlertSeverity(severity)
                alerts = [a for a in alerts if a.severity == severity_enum]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        return {
            "success": True,
            "data": {
                "alerts": [{
                    "alert_id": a.alert_id,
                    "name": a.name,
                    "severity": a.severity.value,
                    "message": a.message,
                    "metric_name": a.metric_name,
                    "threshold": a.threshold,
                    "current_value": a.current_value,
                    "timestamp": a.timestamp,
                    "resolved": a.resolved,
                    "resolved_at": a.resolved_at
                } for a in alerts],
                "total_count": len(alerts)
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.post("/alerts/resolve/{alert_name}")
async def resolve_alert(
    alert_name: str,
    current_user: User = Depends(get_current_user)
):
    """Manually resolve an alert"""
    try:
        if alert_name not in monitoring_manager.alerts:
            raise HTTPException(status_code=404, detail=f"Alert {alert_name} not found")
        
        alert = monitoring_manager.alerts[alert_name]
        if alert.resolved:
            raise HTTPException(status_code=400, detail=f"Alert {alert_name} is already resolved")
        
        # Resolve the alert
        await monitoring_manager._resolve_alert(alert_name)
        
        return {
            "success": True,
            "message": f"Alert {alert_name} resolved successfully",
            "data": {
                "alert_name": alert_name,
                "resolved_at": time.time()
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")


@router.post("/alerts/rules")
async def add_alert_rule(
    rule_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Add a new alert rule"""
    try:
        rule_name = rule_data.get("name")
        if not rule_name:
            raise HTTPException(status_code=400, detail="Rule name is required")
        
        # Validate rule data
        required_fields = ["metric", "threshold", "severity", "message"]
        for field in required_fields:
            if field not in rule_data:
                raise HTTPException(status_code=400, detail=f"Field {field} is required")
        
        # Validate severity
        try:
            severity = AlertSeverity(rule_data["severity"])
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {rule_data['severity']}")
        
        # Create rule
        rule = {
            "metric": rule_data["metric"],
            "threshold": rule_data["threshold"],
            "severity": severity,
            "message": rule_data["message"]
        }
        
        monitoring_manager.add_alert_rule(rule_name, rule)
        
        return {
            "success": True,
            "message": f"Alert rule {rule_name} added successfully",
            "data": {
                "rule_name": rule_name,
                "rule": rule
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add alert rule: {str(e)}")


@router.get("/analytics")
async def get_analytics(
    analysis_type: Optional[str] = Query(None, description="Filter by analysis type"),
    metric_category: Optional[str] = Query(None, description="Filter by metric category"),
    time_range: int = Query(86400, description="Time range in seconds"),
    current_user: User = Depends(get_current_user)
):
    """Get analytics results"""
    try:
        # Parse analysis type
        analysis_type_enum = None
        if analysis_type:
            try:
                analysis_type_enum = AnalyticsType(analysis_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid analysis type: {analysis_type}")
        
        # Parse metric category
        metric_category_enum = None
        if metric_category:
            try:
                metric_category_enum = MetricCategory(metric_category)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid metric category: {metric_category}")
        
        # Get analytics results
        results = await analytics_engine.get_analytics_results(
            analysis_type_enum, metric_category_enum, time_range
        )
        
        return {
            "success": True,
            "data": {
                "analytics_results": [{
                    "analysis_id": r.analysis_id,
                    "analysis_type": r.analysis_type.value,
                    "metric_category": r.metric_category.value,
                    "result_data": r.result_data,
                    "confidence_score": r.confidence_score,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata
                } for r in results],
                "total_count": len(results)
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.get("/predictions")
async def get_predictions(
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    time_range: int = Query(3600, description="Time range in seconds"),
    current_user: User = Depends(get_current_user)
):
    """Get predictions"""
    try:
        predictions = await analytics_engine.get_predictions(model_name, time_range)
        
        return {
            "success": True,
            "data": {
                "predictions": [{
                    "prediction_id": p.prediction_id,
                    "model_name": p.model_name,
                    "predicted_value": p.predicted_value,
                    "confidence_interval": p.confidence_interval,
                    "features_used": p.features_used,
                    "timestamp": p.timestamp,
                    "actual_value": p.actual_value
                } for p in predictions],
                "total_count": len(predictions)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")


@router.get("/dashboard")
async def get_dashboard_data(current_user: User = Depends(get_current_user)):
    """Get comprehensive dashboard data"""
    try:
        # Get system health
        health_status = await monitoring_manager.get_system_health()
        
        # Get active alerts
        active_alerts = await monitoring_manager.get_active_alerts()
        
        # Get recent analytics
        recent_analytics = await analytics_engine.get_analytics_results(time_range=3600)
        
        # Get recent predictions
        recent_predictions = await analytics_engine.get_predictions(time_range=3600)
        
        # Get monitoring stats
        monitoring_stats = monitoring_manager.get_monitoring_stats()
        
        # Get analytics stats
        analytics_stats = analytics_engine.get_analytics_stats()
        
        return {
            "success": True,
            "data": {
                "system_health": health_status,
                "active_alerts": {
                    "count": len(active_alerts),
                    "alerts": [{
                        "name": a.name,
                        "severity": a.severity.value,
                        "message": a.message,
                        "timestamp": a.timestamp
                    } for a in active_alerts[:5]]  # Show only top 5
                },
                "recent_analytics": {
                    "count": len(recent_analytics),
                    "analytics": [{
                        "analysis_type": r.analysis_type.value,
                        "metric_category": r.metric_category.value,
                        "confidence_score": r.confidence_score,
                        "timestamp": r.timestamp
                    } for r in recent_analytics[:5]]  # Show only top 5
                },
                "recent_predictions": {
                    "count": len(recent_predictions),
                    "predictions": [{
                        "model_name": p.model_name,
                        "predicted_value": p.predicted_value,
                        "confidence_interval": p.confidence_interval,
                        "timestamp": p.timestamp
                    } for p in recent_predictions[:5]]  # Show only top 5
                },
                "monitoring_stats": monitoring_stats,
                "analytics_stats": analytics_stats
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@router.post("/metrics/record")
async def record_custom_metric(
    metric_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Record a custom metric"""
    try:
        metric_name = metric_data.get("name")
        metric_value = metric_data.get("value")
        metric_type = metric_data.get("type", "gauge")
        labels = metric_data.get("labels", {})
        
        if not metric_name or metric_value is None:
            raise HTTPException(status_code=400, detail="Metric name and value are required")
        
        # Validate metric type
        try:
            metric_type_enum = MetricType(metric_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric type: {metric_type}")
        
        # Record the metric
        await monitoring_manager._record_metric(metric_name, metric_value, metric_type_enum, labels)
        
        return {
            "success": True,
            "message": f"Metric {metric_name} recorded successfully",
            "data": {
                "metric_name": metric_name,
                "value": metric_value,
                "type": metric_type,
                "labels": labels,
                "timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record metric: {str(e)}")


@router.get("/stats")
async def get_monitoring_stats(current_user: User = Depends(get_current_user)):
    """Get monitoring system statistics"""
    try:
        monitoring_stats = monitoring_manager.get_monitoring_stats()
        analytics_stats = analytics_engine.get_analytics_stats()
        
        return {
            "success": True,
            "data": {
                "monitoring": monitoring_stats,
                "analytics": analytics_stats
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring stats: {str(e)}")


@router.post("/analytics/trigger")
async def trigger_analytics_analysis(
    analysis_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Trigger a custom analytics analysis"""
    try:
        analysis_type = analysis_data.get("type", "descriptive")
        metric_category = analysis_data.get("category", "user_behavior")
        
        # Validate analysis type
        try:
            analysis_type_enum = AnalyticsType(analysis_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid analysis type: {analysis_type}")
        
        # Validate metric category
        try:
            metric_category_enum = MetricCategory(metric_category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid metric category: {metric_category}")
        
        # Trigger analysis in background
        background_tasks.add_task(
            analytics_engine._perform_analytics
        )
        
        return {
            "success": True,
            "message": "Analytics analysis triggered successfully",
            "data": {
                "analysis_type": analysis_type,
                "metric_category": metric_category,
                "triggered_at": time.time()
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger analytics: {str(e)}")


@router.get("/export")
async def export_monitoring_data(
    data_type: str = Query("all", description="Type of data to export (metrics, alerts, analytics, predictions, all)"),
    format: str = Query("json", description="Export format (json, csv)"),
    time_range: int = Query(86400, description="Time range in seconds"),
    current_user: User = Depends(get_current_user)
):
    """Export monitoring data"""
    try:
        export_data = {}
        
        if data_type in ["metrics", "all"]:
            # Export metrics
            all_metrics = {}
            for metric_name in monitoring_manager.metrics.keys():
                metrics = await monitoring_manager.get_metrics(metric_name, time_range)
                if metrics:
                    all_metrics[metric_name] = [{
                        "value": m.value,
                        "timestamp": m.timestamp,
                        "labels": m.labels
                    } for m in metrics]
            export_data["metrics"] = all_metrics
        
        if data_type in ["alerts", "all"]:
            # Export alerts
            alerts = await monitoring_manager.get_alert_history(time_range)
            export_data["alerts"] = [{
                "alert_id": a.alert_id,
                "name": a.name,
                "severity": a.severity.value,
                "message": a.message,
                "metric_name": a.metric_name,
                "threshold": a.threshold,
                "current_value": a.current_value,
                "timestamp": a.timestamp,
                "resolved": a.resolved,
                "resolved_at": a.resolved_at
            } for a in alerts]
        
        if data_type in ["analytics", "all"]:
            # Export analytics
            analytics_results = await analytics_engine.get_analytics_results(time_range=time_range)
            export_data["analytics"] = [{
                "analysis_id": r.analysis_id,
                "analysis_type": r.analysis_type.value,
                "metric_category": r.metric_category.value,
                "result_data": r.result_data,
                "confidence_score": r.confidence_score,
                "timestamp": r.timestamp,
                "metadata": r.metadata
            } for r in analytics_results]
        
        if data_type in ["predictions", "all"]:
            # Export predictions
            predictions = await analytics_engine.get_predictions(time_range=time_range)
            export_data["predictions"] = [{
                "prediction_id": p.prediction_id,
                "model_name": p.model_name,
                "predicted_value": p.predicted_value,
                "confidence_interval": p.confidence_interval,
                "features_used": p.features_used,
                "timestamp": p.timestamp,
                "actual_value": p.actual_value
            } for p in predictions]
        
        return {
            "success": True,
            "data": export_data,
            "metadata": {
                "export_type": data_type,
                "format": format,
                "time_range": time_range,
                "exported_at": time.time()
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")
