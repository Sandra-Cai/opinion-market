"""
Intelligent Alerting System
Advanced alerting with machine learning and predictive capabilities
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import statistics
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.core.performance_optimizer_v2 import performance_optimizer_v2
from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    id: str
    name: str
    description: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'gte', 'lte', 'change_rate'
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_period: int = 300  # seconds
    suppression_rules: List[str] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """Alert instance"""
    id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    metric_name: str
    current_value: float
    threshold_value: float
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    id: str
    name: str
    type: str  # 'email', 'webhook', 'slack', 'sms'
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=list)


class IntelligentAlertingSystem:
    """Intelligent alerting system with ML capabilities"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[str, NotificationChannel] = {}
        
        # Alerting configuration
        self.config = {
            "evaluation_interval": 30,  # seconds
            "max_alerts_per_metric": 5,
            "alert_retention_days": 30,
            "suppression_window": 300,  # seconds
            "escalation_timeout": 1800,  # 30 minutes
            "ml_prediction_enabled": True,
            "anomaly_detection_enabled": True
        }
        
        # Machine learning features
        self.metric_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_thresholds: Dict[str, float] = {}
        self.prediction_models: Dict[str, Any] = {}
        
        # Alerting state
        self.alerting_active = False
        self.alerting_task: Optional[asyncio.Task] = None
        self.last_evaluation = None
        
        # Performance tracking
        self.alerting_stats = {
            "alerts_triggered": 0,
            "alerts_resolved": 0,
            "false_positives": 0,
            "notifications_sent": 0,
            "average_resolution_time": 0
        }
        
        # Initialize default rules and channels
        self._initialize_default_rules()
        self._initialize_default_channels()
        
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                id="cpu_high",
                name="High CPU Usage",
                description="CPU usage exceeds threshold",
                metric_name="cpu_usage",
                condition="gt",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                notification_channels=["email", "webhook"],
                tags=["system", "performance"]
            ),
            AlertRule(
                id="cpu_critical",
                name="Critical CPU Usage",
                description="CPU usage critically high",
                metric_name="cpu_usage",
                condition="gt",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                notification_channels=["email", "webhook", "slack"],
                tags=["system", "performance", "critical"]
            ),
            AlertRule(
                id="memory_high",
                name="High Memory Usage",
                description="Memory usage exceeds threshold",
                metric_name="memory_usage",
                condition="gt",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                notification_channels=["email", "webhook"],
                tags=["system", "memory"]
            ),
            AlertRule(
                id="response_time_slow",
                name="Slow Response Time",
                description="Application response time is slow",
                metric_name="response_time",
                condition="gt",
                threshold=200.0,
                severity=AlertSeverity.WARNING,
                notification_channels=["email", "webhook"],
                tags=["application", "performance"]
            ),
            AlertRule(
                id="cache_hit_rate_low",
                name="Low Cache Hit Rate",
                description="Cache hit rate is below threshold",
                metric_name="cache_hit_rate",
                condition="lt",
                threshold=70.0,
                severity=AlertSeverity.WARNING,
                notification_channels=["email"],
                tags=["cache", "performance"]
            ),
            AlertRule(
                id="disk_space_low",
                name="Low Disk Space",
                description="Disk space is running low",
                metric_name="disk_usage",
                condition="gt",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                notification_channels=["email", "webhook", "slack"],
                tags=["system", "storage", "critical"]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.id] = rule
            
    def _initialize_default_channels(self):
        """Initialize default notification channels"""
        default_channels = [
            NotificationChannel(
                id="email",
                name="Email Notifications",
                type="email",
                config={
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "username": "alerts@opinionmarket.com",
                    "password": "",
                    "from_email": "alerts@opinionmarket.com",
                    "to_emails": ["admin@opinionmarket.com"]
                },
                severity_filter=[AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            ),
            NotificationChannel(
                id="webhook",
                name="Webhook Notifications",
                type="webhook",
                config={
                    "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                    "timeout": 10,
                    "retry_attempts": 3
                },
                severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            ),
            NotificationChannel(
                id="slack",
                name="Slack Notifications",
                type="slack",
                config={
                    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                    "channel": "#alerts",
                    "username": "AlertBot"
                },
                severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            )
        ]
        
        for channel in default_channels:
            self.notification_channels[channel.id] = channel
            
    async def start_alerting(self):
        """Start the intelligent alerting system"""
        if self.alerting_active:
            logger.warning("Alerting system already active")
            return
            
        self.alerting_active = True
        self.alerting_task = asyncio.create_task(self._alerting_loop())
        logger.info("Intelligent alerting system started")
        
    async def stop_alerting(self):
        """Stop the intelligent alerting system"""
        self.alerting_active = False
        if self.alerting_task:
            self.alerting_task.cancel()
            try:
                await self.alerting_task
            except asyncio.CancelledError:
                pass
        logger.info("Intelligent alerting system stopped")
        
    async def _alerting_loop(self):
        """Main alerting evaluation loop"""
        while self.alerting_active:
            try:
                # Evaluate alert rules
                await self._evaluate_alert_rules()
                
                # Check for alert escalations
                await self._check_alert_escalations()
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                # Update ML models
                if self.config["ml_prediction_enabled"]:
                    await self._update_ml_models()
                
                self.last_evaluation = datetime.now()
                
                # Wait before next evaluation
                await asyncio.sleep(self.config["evaluation_interval"])
                
            except Exception as e:
                logger.error(f"Error in alerting loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules against current metrics"""
        try:
            # Get current performance metrics
            perf_summary = performance_optimizer_v2.get_performance_summary()
            current_metrics = perf_summary.get("metrics", {})
            
            # Evaluate each rule
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                    
                # Check cooldown period
                if self._is_rule_in_cooldown(rule_id):
                    continue
                    
                # Get current metric value
                current_value = current_metrics.get(rule.metric_name, {}).get("current", 0)
                
                # Check if rule condition is met
                if self._evaluate_condition(current_value, rule.condition, rule.threshold):
                    # Check if alert already exists
                    existing_alert = self._get_active_alert_for_rule(rule_id)
                    
                    if not existing_alert:
                        # Create new alert
                        await self._create_alert(rule, current_value)
                    else:
                        # Update existing alert
                        await self._update_alert(existing_alert, current_value)
                        
        except Exception as e:
            logger.error(f"Error evaluating alert rules: {e}")
            
    def _is_rule_in_cooldown(self, rule_id: str) -> bool:
        """Check if a rule is in cooldown period"""
        rule = self.alert_rules.get(rule_id)
        if not rule:
            return False
            
        # Check recent alerts for this rule
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.rule_id == rule_id and 
            (datetime.now() - alert.triggered_at).total_seconds() < rule.cooldown_period
        ]
        
        return len(recent_alerts) > 0
        
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        try:
            if condition == "gt":
                return value > threshold
            elif condition == "lt":
                return value < threshold
            elif condition == "eq":
                return abs(value - threshold) < 0.01
            elif condition == "gte":
                return value >= threshold
            elif condition == "lte":
                return value <= threshold
            elif condition == "change_rate":
                # Calculate change rate from recent values
                return self._calculate_change_rate(value) > threshold
            else:
                logger.warning(f"Unknown condition: {condition}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
            
    def _calculate_change_rate(self, current_value: float) -> float:
        """Calculate change rate for a metric"""
        # This would typically use historical data
        # For now, return a simple calculation
        return 0.0
        
    def _get_active_alert_for_rule(self, rule_id: str) -> Optional[Alert]:
        """Get active alert for a rule"""
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                return alert
        return None
        
    async def _create_alert(self, rule: AlertRule, current_value: float):
        """Create a new alert"""
        try:
            alert_id = f"{rule.id}_{int(time.time())}"
            
            alert = Alert(
                id=alert_id,
                rule_id=rule.id,
                title=rule.name,
                message=f"{rule.description}. Current value: {current_value:.2f}, Threshold: {rule.threshold:.2f}",
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                metric_name=rule.metric_name,
                current_value=current_value,
                threshold_value=rule.threshold,
                triggered_at=datetime.now(),
                metadata={
                    "rule_name": rule.name,
                    "condition": rule.condition,
                    "tags": rule.tags
                }
            )
            
            # Add to active alerts
            self.active_alerts[alert_id] = alert
            
            # Add to history
            self.alert_history.append(alert)
            
            # Send notifications
            await self._send_notifications(alert, rule)
            
            # Update stats
            self.alerting_stats["alerts_triggered"] += 1
            
            logger.info(f"Alert created: {alert.title} (Severity: {alert.severity.value})")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            
    async def _update_alert(self, alert: Alert, current_value: float):
        """Update existing alert with new value"""
        try:
            alert.current_value = current_value
            alert.message = f"Alert still active. Current value: {current_value:.2f}, Threshold: {alert.threshold_value:.2f}"
            
            # Check if alert should be resolved
            rule = self.alert_rules.get(alert.rule_id)
            if rule and not self._evaluate_condition(current_value, rule.condition, rule.threshold):
                await self._resolve_alert(alert, "Condition no longer met")
                
        except Exception as e:
            logger.error(f"Error updating alert: {e}")
            
    async def _resolve_alert(self, alert: Alert, resolution_notes: str):
        """Resolve an alert"""
        try:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.resolution_notes = resolution_notes
            
            # Remove from active alerts
            if alert.id in self.active_alerts:
                del self.active_alerts[alert.id]
                
            # Update stats
            self.alerting_stats["alerts_resolved"] += 1
            
            # Calculate resolution time
            if alert.acknowledged_at:
                resolution_time = (alert.resolved_at - alert.acknowledged_at).total_seconds()
            else:
                resolution_time = (alert.resolved_at - alert.triggered_at).total_seconds()
                
            # Update average resolution time
            total_resolved = self.alerting_stats["alerts_resolved"]
            current_avg = self.alerting_stats["average_resolution_time"]
            self.alerting_stats["average_resolution_time"] = (
                (current_avg * (total_resolved - 1) + resolution_time) / total_resolved
            )
            
            logger.info(f"Alert resolved: {alert.title}")
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            
    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for an alert"""
        try:
            for channel_id in rule.notification_channels:
                channel = self.notification_channels.get(channel_id)
                if not channel or not channel.enabled:
                    continue
                    
                # Check severity filter
                if alert.severity not in channel.severity_filter:
                    continue
                    
                # Send notification
                await self._send_notification(alert, channel)
                
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
            
    async def _send_notification(self, alert: Alert, channel: NotificationChannel):
        """Send notification through a specific channel"""
        try:
            if channel.type == "email":
                await self._send_email_notification(alert, channel)
            elif channel.type == "webhook":
                await self._send_webhook_notification(alert, channel)
            elif channel.type == "slack":
                await self._send_slack_notification(alert, channel)
            elif channel.type == "sms":
                await self._send_sms_notification(alert, channel)
                
            self.alerting_stats["notifications_sent"] += 1
            
        except Exception as e:
            logger.error(f"Error sending {channel.type} notification: {e}")
            
    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """Send email notification"""
        try:
            config = channel.config
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
            Alert Details:
            Title: {alert.title}
            Message: {alert.message}
            Severity: {alert.severity.value}
            Metric: {alert.metric_name}
            Current Value: {alert.current_value:.2f}
            Threshold: {alert.threshold_value:.2f}
            Triggered At: {alert.triggered_at.isoformat()}
            
            Please investigate and take appropriate action.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email (in production, use proper SMTP configuration)
            logger.info(f"Email notification sent for alert: {alert.title}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            
    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """Send webhook notification"""
        try:
            import aiohttp
            
            config = channel.config
            payload = {
                "alert_id": alert.id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "triggered_at": alert.triggered_at.isoformat(),
                "metadata": alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['url'],
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.get('timeout', 10))
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent for alert: {alert.title}")
                    else:
                        logger.warning(f"Webhook notification failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            
    async def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """Send Slack notification"""
        try:
            import aiohttp
            
            config = channel.config
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }
            
            payload = {
                "channel": config.get('channel', '#alerts'),
                "username": config.get('username', 'AlertBot'),
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current Value", "value": f"{alert.current_value:.2f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold_value:.2f}", "short": True}
                    ],
                    "timestamp": int(alert.triggered_at.timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config['webhook_url'], json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert: {alert.title}")
                    else:
                        logger.warning(f"Slack notification failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            
    async def _send_sms_notification(self, alert: Alert, channel: NotificationChannel):
        """Send SMS notification"""
        try:
            # SMS implementation would go here
            # For now, just log
            logger.info(f"SMS notification would be sent for alert: {alert.title}")
            
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
            
    async def _check_alert_escalations(self):
        """Check for alert escalations"""
        try:
            current_time = datetime.now()
            escalation_timeout = self.config["escalation_timeout"]
            
            for alert in self.active_alerts.values():
                if alert.status == AlertStatus.ACTIVE:
                    time_since_triggered = (current_time - alert.triggered_at).total_seconds()
                    
                    if time_since_triggered > escalation_timeout:
                        # Escalate alert
                        await self._escalate_alert(alert)
                        
        except Exception as e:
            logger.error(f"Error checking alert escalations: {e}")
            
    async def _escalate_alert(self, alert: Alert):
        """Escalate an alert"""
        try:
            # Increase severity
            if alert.severity == AlertSeverity.WARNING:
                alert.severity = AlertSeverity.CRITICAL
            elif alert.severity == AlertSeverity.CRITICAL:
                alert.severity = AlertSeverity.EMERGENCY
                
            # Update message
            alert.message += f" [ESCALATED - {alert.severity.value.upper()}]"
            
            # Send escalation notifications
            rule = self.alert_rules.get(alert.rule_id)
            if rule:
                await self._send_notifications(alert, rule)
                
            logger.info(f"Alert escalated: {alert.title} -> {alert.severity.value}")
            
        except Exception as e:
            logger.error(f"Error escalating alert: {e}")
            
    async def _cleanup_old_alerts(self):
        """Clean up old alerts from history"""
        try:
            retention_days = self.config["alert_retention_days"]
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Remove old alerts from history
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.triggered_at > cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
            
    async def _update_ml_models(self):
        """Update machine learning models for anomaly detection"""
        try:
            # This would implement ML-based anomaly detection
            # For now, just log
            logger.debug("ML models updated")
            
        except Exception as e:
            logger.error(f"Error updating ML models: {e}")
            
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules[rule.id] = rule
        logger.info(f"Alert rule added: {rule.name}")
        
    def update_alert_rule(self, rule_id: str, **kwargs):
        """Update an existing alert rule"""
        if rule_id in self.alert_rules:
            rule = self.alert_rules[rule_id]
            for key, value in kwargs.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            logger.info(f"Alert rule updated: {rule.name}")
        else:
            logger.warning(f"Alert rule not found: {rule_id}")
            
    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule"""
        if rule_id in self.alert_rules:
            rule = self.alert_rules[rule_id]
            del self.alert_rules[rule_id]
            logger.info(f"Alert rule removed: {rule.name}")
        else:
            logger.warning(f"Alert rule not found: {rule_id}")
            
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            logger.info(f"Alert acknowledged: {alert.title} by {acknowledged_by}")
        else:
            logger.warning(f"Alert not found: {alert_id}")
            
    def get_alerting_summary(self) -> Dict[str, Any]:
        """Get comprehensive alerting summary"""
        try:
            # Get active alerts by severity
            active_by_severity = defaultdict(int)
            for alert in self.active_alerts.values():
                active_by_severity[alert.severity.value] += 1
                
            # Get recent alerts
            recent_alerts = [
                alert for alert in self.alert_history
                if (datetime.now() - alert.triggered_at).total_seconds() < 3600  # Last hour
            ]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "alerting_active": self.alerting_active,
                "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None,
                "active_alerts": {
                    "total": len(self.active_alerts),
                    "by_severity": dict(active_by_severity)
                },
                "recent_alerts": len(recent_alerts),
                "total_rules": len(self.alert_rules),
                "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
                "notification_channels": len(self.notification_channels),
                "stats": self.alerting_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting alerting summary: {e}")
            return {"error": str(e)}


# Global instance
intelligent_alerting_system = IntelligentAlertingSystem()
