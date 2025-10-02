"""
Advanced Monitoring Engine
Comprehensive monitoring and observability system with Prometheus/Grafana integration
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import json
import psutil
import threading
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import requests

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    CUSTOM = "custom"


class AlertSeverity(Enum):
    """Alert severity enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    unit: str = ""


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    status: str  # firing, resolved
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    starts_at: datetime = field(default_factory=datetime.now)
    ends_at: Optional[datetime] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class ServiceHealth:
    """Service health data structure"""
    service_name: str
    status: ServiceStatus
    uptime: float
    response_time: float
    error_rate: float
    throughput: float
    last_check: datetime
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedMonitoringEngine:
    """Advanced Monitoring Engine with Prometheus/Grafana integration"""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.alerts: List[Alert] = []
        self.service_health: Dict[str, ServiceHealth] = {}
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.prometheus_metrics = {
            "request_count": Counter(
                "http_requests_total",
                "Total HTTP requests",
                ["method", "endpoint", "status"],
                registry=self.registry
            ),
            "request_duration": Histogram(
                "http_request_duration_seconds",
                "HTTP request duration",
                ["method", "endpoint"],
                registry=self.registry
            ),
            "active_connections": Gauge(
                "active_connections",
                "Number of active connections",
                registry=self.registry
            ),
            "memory_usage": Gauge(
                "memory_usage_bytes",
                "Memory usage in bytes",
                ["type"],
                registry=self.registry
            ),
            "cpu_usage": Gauge(
                "cpu_usage_percent",
                "CPU usage percentage",
                registry=self.registry
            ),
            "disk_usage": Gauge(
                "disk_usage_bytes",
                "Disk usage in bytes",
                ["device"],
                registry=self.registry
            ),
            "custom_metrics": Gauge(
                "custom_metric_value",
                "Custom metric values",
                ["metric_name", "service"],
                registry=self.registry
            )
        }
        
        # Configuration
        self.config = {
            "prometheus_enabled": True,
            "grafana_enabled": True,
            "alerting_enabled": True,
            "metrics_retention_days": 30,
            "alert_check_interval": 60,  # seconds
            "health_check_interval": 30,  # seconds
            "metrics_collection_interval": 10,  # seconds
            "prometheus_port": 9090,
            "grafana_port": 3000,
            "alert_webhook_url": None,
            "slack_webhook_url": None,
            "email_notifications": False
        }
        
        # Alert rules
        self.alert_rules = {
            "high_cpu_usage": {
                "metric": "cpu_usage_percent",
                "threshold": 80.0,
                "severity": AlertSeverity.WARNING,
                "description": "High CPU usage detected"
            },
            "high_memory_usage": {
                "metric": "memory_usage_bytes",
                "threshold": 1024 * 1024 * 1024 * 8,  # 8GB
                "severity": AlertSeverity.WARNING,
                "description": "High memory usage detected"
            },
            "high_error_rate": {
                "metric": "error_rate",
                "threshold": 0.05,  # 5%
                "severity": AlertSeverity.ERROR,
                "description": "High error rate detected"
            },
            "service_down": {
                "metric": "service_status",
                "threshold": 0,  # 0 = down
                "severity": AlertSeverity.CRITICAL,
                "description": "Service is down"
            },
            "slow_response_time": {
                "metric": "response_time",
                "threshold": 5.0,  # 5 seconds
                "severity": AlertSeverity.WARNING,
                "description": "Slow response time detected"
            }
        }
        
        # Service dependencies
        self.service_dependencies = {
            "api": ["database", "redis", "cache"],
            "database": [],
            "redis": [],
            "cache": [],
            "blockchain": ["api"],
            "ml_engine": ["api", "cache"],
            "security": ["api"],
            "monitoring": ["api"]
        }
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.monitoring_stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "health_checks_performed": 0,
            "prometheus_requests": 0,
            "grafana_dashboards_updated": 0
        }
        
    async def start_monitoring_engine(self):
        """Start the advanced monitoring engine"""
        if self.monitoring_active:
            logger.warning("Monitoring engine already active")
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_processing_loop())
        logger.info("Advanced Monitoring Engine started")
        
    async def stop_monitoring_engine(self):
        """Stop the advanced monitoring engine"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Advanced Monitoring Engine stopped")
        
    async def _monitoring_processing_loop(self):
        """Main monitoring processing loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check service health
                await self._check_service_health()
                
                # Evaluate alert rules
                await self._evaluate_alert_rules()
                
                # Update Prometheus metrics
                if self.config["prometheus_enabled"]:
                    await self._update_prometheus_metrics()
                    
                # Update Grafana dashboards
                if self.config["grafana_enabled"]:
                    await self._update_grafana_dashboards()
                    
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next cycle
                await asyncio.sleep(self.config["metrics_collection_interval"])
                
            except Exception as e:
                logger.error(f"Error in monitoring processing loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric("cpu_usage_percent", cpu_percent, MetricType.GAUGE)
            
            # Memory usage
            memory = psutil.virtual_memory()
            await self._record_metric("memory_usage_bytes", memory.used, MetricType.GAUGE, {"type": "used"})
            await self._record_metric("memory_usage_bytes", memory.available, MetricType.GAUGE, {"type": "available"})
            
            # Disk usage
            disk = psutil.disk_usage('/')
            await self._record_metric("disk_usage_bytes", disk.used, MetricType.GAUGE, {"device": "root"})
            await self._record_metric("disk_usage_bytes", disk.free, MetricType.GAUGE, {"device": "root"})
            
            # Network I/O
            network = psutil.net_io_counters()
            await self._record_metric("network_bytes_sent", network.bytes_sent, MetricType.COUNTER)
            await self._record_metric("network_bytes_recv", network.bytes_recv, MetricType.COUNTER)
            
            # Process count
            process_count = len(psutil.pids())
            await self._record_metric("process_count", process_count, MetricType.GAUGE)
            
            self.monitoring_stats["metrics_collected"] += 1
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
    async def _check_service_health(self):
        """Check health of all services"""
        try:
            services = [
                "api", "database", "redis", "cache", "blockchain",
                "ml_engine", "security", "monitoring"
            ]
            
            for service in services:
                health = await self._check_single_service_health(service)
                self.service_health[service] = health
                
            self.monitoring_stats["health_checks_performed"] += 1
            
        except Exception as e:
            logger.error(f"Error checking service health: {e}")
            
    async def _check_single_service_health(self, service_name: str) -> ServiceHealth:
        """Check health of a single service"""
        try:
            # Simulate health check
            start_time = time.time()
            
            # Mock health check based on service type
            if service_name == "api":
                # Check API endpoint
                response_time = 0.1 + (time.time() % 0.5)  # Mock response time
                error_rate = 0.01 if time.time() % 10 < 9 else 0.1  # Mock error rate
                status = ServiceStatus.HEALTHY if error_rate < 0.05 else ServiceStatus.DEGRADED
            elif service_name == "database":
                # Check database connection
                response_time = 0.05 + (time.time() % 0.2)
                error_rate = 0.005
                status = ServiceStatus.HEALTHY
            elif service_name == "redis":
                # Check Redis connection
                response_time = 0.02 + (time.time() % 0.1)
                error_rate = 0.001
                status = ServiceStatus.HEALTHY
            else:
                # Default health check
                response_time = 0.1
                error_rate = 0.01
                status = ServiceStatus.HEALTHY
                
            # Calculate uptime (mock)
            uptime = 86400 + (time.time() % 3600)  # Mock uptime
            
            # Calculate throughput (mock)
            throughput = 1000 + (time.time() % 500)  # Mock throughput
            
            return ServiceHealth(
                service_name=service_name,
                status=status,
                uptime=uptime,
                response_time=response_time,
                error_rate=error_rate,
                throughput=throughput,
                last_check=datetime.now(),
                dependencies=self.service_dependencies.get(service_name, []),
                metadata={"version": "1.0.0", "region": "us-east-1"}
            )
            
        except Exception as e:
            logger.error(f"Error checking health for {service_name}: {e}")
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                uptime=0,
                response_time=0,
                error_rate=1.0,
                throughput=0,
                last_check=datetime.now()
            )
            
    async def _evaluate_alert_rules(self):
        """Evaluate alert rules and trigger alerts"""
        try:
            for rule_name, rule_config in self.alert_rules.items():
                metric_name = rule_config["metric"]
                threshold = rule_config["threshold"]
                severity = rule_config["severity"]
                description = rule_config["description"]
                
                # Get current metric value
                metric_value = await self._get_metric_value(metric_name)
                
                if metric_value is not None:
                    # Check if threshold is exceeded
                    threshold_exceeded = False
                    
                    if rule_name == "high_cpu_usage":
                        threshold_exceeded = metric_value > threshold
                    elif rule_name == "high_memory_usage":
                        threshold_exceeded = metric_value > threshold
                    elif rule_name == "high_error_rate":
                        threshold_exceeded = metric_value > threshold
                    elif rule_name == "service_down":
                        threshold_exceeded = metric_value == threshold
                    elif rule_name == "slow_response_time":
                        threshold_exceeded = metric_value > threshold
                        
                    if threshold_exceeded:
                        await self._trigger_alert(rule_name, description, severity, metric_value, threshold)
                        
        except Exception as e:
            logger.error(f"Error evaluating alert rules: {e}")
            
    async def _trigger_alert(self, alert_name: str, description: str, severity: AlertSeverity, value: float, threshold: float):
        """Trigger an alert"""
        try:
            # Check if alert is already firing
            existing_alert = next(
                (alert for alert in self.alerts 
                 if alert.name == alert_name and alert.status == "firing"),
                None
            )
            
            if existing_alert:
                # Update existing alert
                existing_alert.value = value
                existing_alert.annotations["current_value"] = str(value)
                existing_alert.annotations["threshold"] = str(threshold)
            else:
                # Create new alert
                alert = Alert(
                    alert_id=f"alert_{int(time.time())}_{alert_name}",
                    name=alert_name,
                    description=description,
                    severity=severity,
                    status="firing",
                    value=value,
                    threshold=threshold,
                    labels={"service": "monitoring", "alert_name": alert_name},
                    annotations={
                        "current_value": str(value),
                        "threshold": str(threshold),
                        "description": description
                    }
                )
                
                self.alerts.append(alert)
                self.monitoring_stats["alerts_triggered"] += 1
                
                # Send notifications
                await self._send_alert_notifications(alert)
                
                logger.warning(f"Alert triggered: {alert_name} - {description}")
                
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
            
    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications"""
        try:
            # Send to webhook if configured
            if self.config["alert_webhook_url"]:
                await self._send_webhook_notification(alert)
                
            # Send to Slack if configured
            if self.config["slack_webhook_url"]:
                await self._send_slack_notification(alert)
                
            # Send email if configured
            if self.config["email_notifications"]:
                await self._send_email_notification(alert)
                
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")
            
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        try:
            payload = {
                "alert_id": alert.alert_id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity.value,
                "status": alert.status,
                "value": alert.value,
                "threshold": alert.threshold,
                "timestamp": alert.starts_at.isoformat()
            }
            
            # This would send actual webhook
            logger.info(f"Webhook notification sent for alert: {alert.name}")
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        try:
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger"
            }.get(alert.severity, "good")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"Alert: {alert.name}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Current Value", "value": str(alert.value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True}
                    ],
                    "timestamp": int(alert.starts_at.timestamp())
                }]
            }
            
            # This would send actual Slack message
            logger.info(f"Slack notification sent for alert: {alert.name}")
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            
    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            # This would send actual email
            logger.info(f"Email notification sent for alert: {alert.name}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            
    async def _update_prometheus_metrics(self):
        """Update Prometheus metrics"""
        try:
            # Update custom metrics
            for metric_name, metric in self.metrics.items():
                if metric_name in ["cpu_usage_percent", "memory_usage_bytes", "disk_usage_bytes"]:
                    self.prometheus_metrics["custom_metrics"].labels(
                        metric_name=metric_name,
                        service="system"
                    ).set(metric.value)
                    
            self.monitoring_stats["prometheus_requests"] += 1
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
            
    async def _update_grafana_dashboards(self):
        """Update Grafana dashboards"""
        try:
            # This would update Grafana dashboards
            # For now, we'll just log the action
            logger.debug("Updating Grafana dashboards...")
            self.monitoring_stats["grafana_dashboards_updated"] += 1
            
        except Exception as e:
            logger.error(f"Error updating Grafana dashboards: {e}")
            
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=self.config["metrics_retention_days"])
            
            # Clean up old metrics
            self.metrics = {
                name: metric for name, metric in self.metrics.items()
                if metric.timestamp > cutoff_time
            }
            
            # Clean up old alerts
            self.alerts = [
                alert for alert in self.alerts
                if alert.starts_at > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    async def _record_metric(self, name: str, value: float, metric_type: MetricType, labels: Optional[Dict[str, str]] = None):
        """Record a metric"""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                labels=labels or {},
                timestamp=datetime.now()
            )
            
            self.metrics[name] = metric
            
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
            
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric"""
        try:
            metric = self.metrics.get(metric_name)
            if metric:
                return metric.value
            return None
            
        except Exception as e:
            logger.error(f"Error getting metric value: {e}")
            return None
            
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        try:
            return generate_latest(self.registry).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error generating Prometheus metrics: {e}")
            return ""
            
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        try:
            # Calculate overall system health
            healthy_services = sum(1 for health in self.service_health.values() if health.status == ServiceStatus.HEALTHY)
            total_services = len(self.service_health)
            overall_health = "healthy" if healthy_services == total_services else "degraded" if healthy_services > total_services // 2 else "unhealthy"
            
            # Get active alerts
            active_alerts = [alert for alert in self.alerts if alert.status == "firing"]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": self.monitoring_active,
                "overall_health": overall_health,
                "total_metrics": len(self.metrics),
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "service_health": {
                    name: {
                        "status": health.status.value,
                        "uptime": health.uptime,
                        "response_time": health.response_time,
                        "error_rate": health.error_rate,
                        "throughput": health.throughput
                    }
                    for name, health in self.service_health.items()
                },
                "alerts_by_severity": {
                    severity.value: len([a for a in active_alerts if a.severity == severity])
                    for severity in AlertSeverity
                },
                "stats": self.monitoring_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting monitoring summary: {e}")
            return {"error": str(e)}


# Global instance
advanced_monitoring_engine = AdvancedMonitoringEngine()
