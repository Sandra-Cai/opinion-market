"""
Intelligent Alerting API
API endpoints for managing intelligent alerting system
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime

from app.core.auth import get_current_user
from app.models.user import User
from app.core.intelligent_alerting import (
    intelligent_alerting_system,
    AlertRule,
    Alert,
    AlertSeverity,
    AlertStatus,
    NotificationChannel
)

router = APIRouter()


@router.post("/start-alerting")
async def start_intelligent_alerting(
    current_user: User = Depends(get_current_user)
):
    """Start the intelligent alerting system"""
    try:
        await intelligent_alerting_system.start_alerting()
        return {
            "success": True,
            "message": "Intelligent alerting system started",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start alerting: {str(e)}")


@router.post("/stop-alerting")
async def stop_intelligent_alerting(
    current_user: User = Depends(get_current_user)
):
    """Stop the intelligent alerting system"""
    try:
        await intelligent_alerting_system.stop_alerting()
        return {
            "success": True,
            "message": "Intelligent alerting system stopped",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop alerting: {str(e)}")


@router.get("/summary")
async def get_alerting_summary(
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive alerting system summary"""
    try:
        summary = intelligent_alerting_system.get_alerting_summary()
        
        return {
            "success": True,
            "data": summary,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerting summary: {str(e)}")


@router.get("/alerts")
async def get_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get alerts with optional filtering"""
    try:
        alerts = []
        
        # Get active alerts
        for alert in intelligent_alerting_system.active_alerts.values():
            alerts.append({
                "id": alert.id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "triggered_at": alert.triggered_at.isoformat(),
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "acknowledged_by": alert.acknowledged_by,
                "metadata": alert.metadata
            })
        
        # Get recent resolved alerts from history
        recent_resolved = [
            alert for alert in intelligent_alerting_system.alert_history
            if alert.status == AlertStatus.RESOLVED and
            (datetime.now() - alert.triggered_at).total_seconds() < 86400  # Last 24 hours
        ]
        
        for alert in recent_resolved[-10:]:  # Last 10 resolved
            alerts.append({
                "id": alert.id,
                "rule_id": alert.rule_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "triggered_at": alert.triggered_at.isoformat(),
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "acknowledged_by": alert.acknowledged_by,
                "resolution_notes": alert.resolution_notes,
                "metadata": alert.metadata
            })
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a["status"] == status]
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        # Limit results
        alerts = alerts[-limit:]
        
        return {
            "success": True,
            "data": {
                "alerts": alerts,
                "total_count": len(alerts),
                "active_count": len(intelligent_alerting_system.active_alerts)
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.get("/rules")
async def get_alert_rules(
    enabled_only: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Get alert rules"""
    try:
        rules = []
        
        for rule_id, rule in intelligent_alerting_system.alert_rules.items():
            if enabled_only and not rule.enabled:
                continue
                
            rules.append({
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "metric_name": rule.metric_name,
                "condition": rule.condition,
                "threshold": rule.threshold,
                "severity": rule.severity.value,
                "enabled": rule.enabled,
                "cooldown_period": rule.cooldown_period,
                "notification_channels": rule.notification_channels,
                "tags": rule.tags,
                "created_at": rule.created_at.isoformat()
            })
        
        return {
            "success": True,
            "data": {
                "rules": rules,
                "total_count": len(rules)
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alert rules: {str(e)}")


@router.post("/rules")
async def create_alert_rule(
    rule_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Create a new alert rule"""
    try:
        # Validate required fields
        required_fields = ["id", "name", "description", "metric_name", "condition", "threshold", "severity"]
        for field in required_fields:
            if field not in rule_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Validate severity
        try:
            severity = AlertSeverity(rule_data["severity"])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid severity value")
        
        # Create alert rule
        rule = AlertRule(
            id=rule_data["id"],
            name=rule_data["name"],
            description=rule_data["description"],
            metric_name=rule_data["metric_name"],
            condition=rule_data["condition"],
            threshold=rule_data["threshold"],
            severity=severity,
            enabled=rule_data.get("enabled", True),
            cooldown_period=rule_data.get("cooldown_period", 300),
            notification_channels=rule_data.get("notification_channels", []),
            tags=rule_data.get("tags", [])
        )
        
        intelligent_alerting_system.add_alert_rule(rule)
        
        return {
            "success": True,
            "message": f"Alert rule '{rule.name}' created successfully",
            "data": {
                "id": rule.id,
                "name": rule.name,
                "metric_name": rule.metric_name,
                "severity": rule.severity.value,
                "enabled": rule.enabled
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create alert rule: {str(e)}")


@router.put("/rules/{rule_id}")
async def update_alert_rule(
    rule_id: str,
    rule_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Update an existing alert rule"""
    try:
        if rule_id not in intelligent_alerting_system.alert_rules:
            raise HTTPException(status_code=404, detail=f"Alert rule '{rule_id}' not found")
        
        # Update rule
        intelligent_alerting_system.update_alert_rule(rule_id, **rule_data)
        
        # Get updated rule
        updated_rule = intelligent_alerting_system.alert_rules[rule_id]
        
        return {
            "success": True,
            "message": f"Alert rule '{rule_id}' updated successfully",
            "data": {
                "id": updated_rule.id,
                "name": updated_rule.name,
                "metric_name": updated_rule.metric_name,
                "severity": updated_rule.severity.value,
                "enabled": updated_rule.enabled
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update alert rule: {str(e)}")


@router.delete("/rules/{rule_id}")
async def delete_alert_rule(
    rule_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete an alert rule"""
    try:
        if rule_id not in intelligent_alerting_system.alert_rules:
            raise HTTPException(status_code=404, detail=f"Alert rule '{rule_id}' not found")
        
        rule_name = intelligent_alerting_system.alert_rules[rule_id].name
        intelligent_alerting_system.remove_alert_rule(rule_id)
        
        return {
            "success": True,
            "message": f"Alert rule '{rule_name}' deleted successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete alert rule: {str(e)}")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    acknowledged_by: str,
    current_user: User = Depends(get_current_user)
):
    """Acknowledge an alert"""
    try:
        if alert_id not in intelligent_alerting_system.active_alerts:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")
        
        intelligent_alerting_system.acknowledge_alert(alert_id, acknowledged_by)
        
        return {
            "success": True,
            "message": f"Alert '{alert_id}' acknowledged by {acknowledged_by}",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")


@router.get("/channels")
async def get_notification_channels(
    current_user: User = Depends(get_current_user)
):
    """Get notification channels"""
    try:
        channels = []
        
        for channel_id, channel in intelligent_alerting_system.notification_channels.items():
            channels.append({
                "id": channel.id,
                "name": channel.name,
                "type": channel.type,
                "enabled": channel.enabled,
                "severity_filter": [s.value for s in channel.severity_filter],
                "config": {k: v for k, v in channel.config.items() if k not in ["password", "token"]}  # Hide sensitive data
            })
        
        return {
            "success": True,
            "data": {
                "channels": channels,
                "total_count": len(channels)
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get notification channels: {str(e)}")


@router.post("/channels")
async def create_notification_channel(
    channel_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Create a new notification channel"""
    try:
        # Validate required fields
        required_fields = ["id", "name", "type", "config"]
        for field in required_fields:
            if field not in channel_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Validate severity filter
        severity_filter = []
        for severity_str in channel_data.get("severity_filter", []):
            try:
                severity_filter.append(AlertSeverity(severity_str))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity_str}")
        
        # Create notification channel
        channel = NotificationChannel(
            id=channel_data["id"],
            name=channel_data["name"],
            type=channel_data["type"],
            config=channel_data["config"],
            enabled=channel_data.get("enabled", True),
            severity_filter=severity_filter
        )
        
        intelligent_alerting_system.notification_channels[channel.id] = channel
        
        return {
            "success": True,
            "message": f"Notification channel '{channel.name}' created successfully",
            "data": {
                "id": channel.id,
                "name": channel.name,
                "type": channel.type,
                "enabled": channel.enabled
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create notification channel: {str(e)}")


@router.get("/stats")
async def get_alerting_stats(
    current_user: User = Depends(get_current_user)
):
    """Get alerting system statistics"""
    try:
        stats = intelligent_alerting_system.alerting_stats
        
        return {
            "success": True,
            "data": {
                "alerts_triggered": stats["alerts_triggered"],
                "alerts_resolved": stats["alerts_resolved"],
                "false_positives": stats["false_positives"],
                "notifications_sent": stats["notifications_sent"],
                "average_resolution_time": stats["average_resolution_time"],
                "active_alerts": len(intelligent_alerting_system.active_alerts),
                "total_rules": len(intelligent_alerting_system.alert_rules),
                "enabled_rules": len([r for r in intelligent_alerting_system.alert_rules.values() if r.enabled])
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerting stats: {str(e)}")


@router.post("/test-alert")
async def test_alert(
    rule_id: str,
    current_user: User = Depends(get_current_user)
):
    """Test an alert rule by triggering a test alert"""
    try:
        if rule_id not in intelligent_alerting_system.alert_rules:
            raise HTTPException(status_code=404, detail=f"Alert rule '{rule_id}' not found")
        
        rule = intelligent_alerting_system.alert_rules[rule_id]
        
        # Create a test alert
        test_value = rule.threshold + 10  # Exceed threshold by 10
        await intelligent_alerting_system._create_alert(rule, test_value)
        
        return {
            "success": True,
            "message": f"Test alert triggered for rule '{rule.name}'",
            "data": {
                "rule_id": rule_id,
                "rule_name": rule.name,
                "test_value": test_value,
                "threshold": rule.threshold
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test alert: {str(e)}")


@router.get("/health")
async def get_alerting_health(
    current_user: User = Depends(get_current_user)
):
    """Get alerting system health status"""
    try:
        summary = intelligent_alerting_system.get_alerting_summary()
        
        health_status = {
            "alerting_active": summary.get("alerting_active", False),
            "last_evaluation": summary.get("last_evaluation"),
            "active_alerts": summary.get("active_alerts", {}).get("total", 0),
            "enabled_rules": summary.get("enabled_rules", 0),
            "notification_channels": summary.get("notification_channels", 0),
            "status": "healthy" if summary.get("alerting_active", False) else "inactive"
        }
        
        return {
            "success": True,
            "data": health_status,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")
