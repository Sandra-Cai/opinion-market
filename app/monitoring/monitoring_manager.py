"""
Advanced Monitoring Manager
Comprehensive system monitoring with real-time metrics and alerting
"""

import asyncio
import logging
import time
import json
import psutil
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque

from app.core.enhanced_cache import enhanced_cache
from app.core.database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str]
    timestamp: float
    description: Optional[str] = None


@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None


@dataclass
class SystemMetrics:
    """System metrics data structure"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    load_average: List[float]


class MonitoringManager:
    """Advanced monitoring and metrics collection system"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.metric_collectors: List[Callable] = []
        self.alert_handlers: List[Callable] = []
        
        # Configuration
        self.collection_interval = 30  # seconds
        self.retention_period = 7 * 24 * 3600  # 7 days
        self.alert_cooldown = 300  # 5 minutes
        
        # Statistics
        self.stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "alerts_resolved": 0,
            "collection_errors": 0
        }
        
        # Start monitoring tasks
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._alert_evaluation_loop())
        asyncio.create_task(self._cleanup_loop())
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        logger.info("Starting advanced monitoring system")
        
        # Register default metric collectors
        self._register_default_collectors()
        
        # Register default alert rules
        self._register_default_alert_rules()
        
        logger.info("Advanced monitoring system started")
    
    def _register_default_collectors(self):
        """Register default metric collectors"""
        self.metric_collectors.extend([
            self._collect_system_metrics,
            self._collect_application_metrics,
            self._collect_database_metrics,
            self._collect_cache_metrics,
            self._collect_network_metrics
        ])
    
    def _register_default_alert_rules(self):
        """Register default alert rules"""
        self.alert_rules = {
            "high_cpu_usage": {
                "metric": "system.cpu_percent",
                "threshold": 80.0,
                "severity": AlertSeverity.WARNING,
                "message": "High CPU usage detected"
            },
            "high_memory_usage": {
                "metric": "system.memory_percent",
                "threshold": 85.0,
                "severity": AlertSeverity.WARNING,
                "message": "High memory usage detected"
            },
            "low_disk_space": {
                "metric": "system.disk_usage_percent",
                "threshold": 90.0,
                "severity": AlertSeverity.ERROR,
                "message": "Low disk space detected"
            },
            "high_error_rate": {
                "metric": "application.error_rate",
                "threshold": 5.0,
                "severity": AlertSeverity.ERROR,
                "message": "High error rate detected"
            },
            "slow_response_time": {
                "metric": "application.avg_response_time",
                "threshold": 2.0,
                "severity": AlertSeverity.WARNING,
                "message": "Slow response time detected"
            },
            "database_connection_issues": {
                "metric": "database.connection_errors",
                "threshold": 10.0,
                "severity": AlertSeverity.ERROR,
                "message": "Database connection issues detected"
            },
            "cache_hit_rate_low": {
                "metric": "cache.hit_rate",
                "threshold": 70.0,
                "severity": AlertSeverity.WARNING,
                "message": "Low cache hit rate detected"
            }
        }
    
    async def _metrics_collection_loop(self):
        """Background task to collect metrics"""
        while True:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                self.stats["collection_errors"] += 1
                await asyncio.sleep(60)
    
    async def _collect_all_metrics(self):
        """Collect all registered metrics"""
        for collector in self.metric_collectors:
            try:
                await collector()
            except Exception as e:
                logger.error(f"Error in metric collector {collector.__name__}: {e}")
                self.stats["collection_errors"] += 1
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric("system.cpu_percent", cpu_percent, MetricType.GAUGE, {
                "host": os.uname().nodename
            })
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self._record_metric("system.memory_percent", memory.percent, MetricType.GAUGE, {
                "host": os.uname().nodename
            })
            await self._record_metric("system.memory_used_mb", memory.used / 1024 / 1024, MetricType.GAUGE, {
                "host": os.uname().nodename
            })
            await self._record_metric("system.memory_available_mb", memory.available / 1024 / 1024, MetricType.GAUGE, {
                "host": os.uname().nodename
            })
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            await self._record_metric("system.disk_usage_percent", (disk.used / disk.total) * 100, MetricType.GAUGE, {
                "host": os.uname().nodename,
                "mount": "/"
            })
            await self._record_metric("system.disk_free_gb", disk.free / 1024 / 1024 / 1024, MetricType.GAUGE, {
                "host": os.uname().nodename,
                "mount": "/"
            })
            
            # Network metrics
            network = psutil.net_io_counters()
            await self._record_metric("system.network_bytes_sent", network.bytes_sent, MetricType.COUNTER, {
                "host": os.uname().nodename
            })
            await self._record_metric("system.network_bytes_recv", network.bytes_recv, MetricType.COUNTER, {
                "host": os.uname().nodename
            })
            
            # Load average
            load_avg = os.getloadavg()
            await self._record_metric("system.load_average_1m", load_avg[0], MetricType.GAUGE, {
                "host": os.uname().nodename
            })
            await self._record_metric("system.load_average_5m", load_avg[1], MetricType.GAUGE, {
                "host": os.uname().nodename
            })
            await self._record_metric("system.load_average_15m", load_avg[2], MetricType.GAUGE, {
                "host": os.uname().nodename
            })
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application-level metrics"""
        try:
            # Get application metrics from cache
            app_metrics = await enhanced_cache.get("application_metrics")
            if app_metrics:
                await self._record_metric("application.request_count", app_metrics.get("request_count", 0), MetricType.COUNTER)
                await self._record_metric("application.error_count", app_metrics.get("error_count", 0), MetricType.COUNTER)
                await self._record_metric("application.avg_response_time", app_metrics.get("avg_response_time", 0), MetricType.GAUGE)
                
                # Calculate error rate
                request_count = app_metrics.get("request_count", 0)
                error_count = app_metrics.get("error_count", 0)
                error_rate = (error_count / max(request_count, 1)) * 100
                await self._record_metric("application.error_rate", error_rate, MetricType.GAUGE)
            
            # Active connections
            process = psutil.Process()
            connections = len(process.connections())
            await self._record_metric("application.active_connections", connections, MetricType.GAUGE)
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    async def _collect_database_metrics(self):
        """Collect database metrics"""
        try:
            with engine.connect() as conn:
                # Connection count
                result = conn.execute(text("SELECT count(*) FROM pg_stat_activity"))
                connection_count = result.fetchone()[0]
                await self._record_metric("database.connections", connection_count, MetricType.GAUGE)
                
                # Database size
                result = conn.execute(text("SELECT pg_database_size(current_database())"))
                db_size = result.fetchone()[0]
                await self._record_metric("database.size_bytes", db_size, MetricType.GAUGE)
                
                # Query performance (simplified)
                start_time = time.time()
                conn.execute(text("SELECT 1"))
                query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                await self._record_metric("database.query_time_ms", query_time, MetricType.HISTOGRAM)
                
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")
            await self._record_metric("database.connection_errors", 1, MetricType.COUNTER)
    
    async def _collect_cache_metrics(self):
        """Collect cache metrics"""
        try:
            cache_stats = enhanced_cache.get_stats()
            
            await self._record_metric("cache.hits", cache_stats["hits"], MetricType.COUNTER)
            await self._record_metric("cache.misses", cache_stats["misses"], MetricType.COUNTER)
            await self._record_metric("cache.entry_count", cache_stats["entry_count"], MetricType.GAUGE)
            await self._record_metric("cache.total_size_bytes", cache_stats["total_size_bytes"], MetricType.GAUGE)
            
            # Calculate hit rate
            total_requests = cache_stats["hits"] + cache_stats["misses"]
            hit_rate = (cache_stats["hits"] / max(total_requests, 1)) * 100
            await self._record_metric("cache.hit_rate", hit_rate, MetricType.GAUGE)
            
        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}")
    
    async def _collect_network_metrics(self):
        """Collect network metrics"""
        try:
            # HTTP request metrics (simplified)
            http_metrics = await enhanced_cache.get("http_metrics")
            if http_metrics:
                await self._record_metric("network.http_requests", http_metrics.get("requests", 0), MetricType.COUNTER)
                await self._record_metric("network.http_errors", http_metrics.get("errors", 0), MetricType.COUNTER)
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
    
    async def _record_metric(self, name: str, value: float, metric_type: MetricType, 
                           labels: Dict[str, str] = None):
        """Record a metric"""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                labels=labels or {},
                timestamp=time.time()
            )
            
            self.metrics[name].append(metric)
            self.stats["metrics_collected"] += 1
            
            # Store in cache for quick access
            await enhanced_cache.set(
                f"metric_{name}",
                asdict(metric),
                ttl=3600,
                tags=["metrics", name]
            )
            
        except Exception as e:
            logger.error(f"Error recording metric {name}: {e}")
    
    async def _alert_evaluation_loop(self):
        """Background task to evaluate alert rules"""
        while True:
            try:
                await self._evaluate_alerts()
                await asyncio.sleep(60)  # Check alerts every minute
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_alerts(self):
        """Evaluate all alert rules"""
        for alert_name, rule in self.alert_rules.items():
            try:
                metric_name = rule["metric"]
                threshold = rule["threshold"]
                severity = rule["severity"]
                message = rule["message"]
                
                # Get latest metric value
                latest_metric = await self._get_latest_metric(metric_name)
                if not latest_metric:
                    continue
                
                current_value = latest_metric.value
                
                # Check if alert should be triggered
                should_alert = False
                if metric_name in ["system.cpu_percent", "system.memory_percent", "system.disk_usage_percent", "application.error_rate", "application.avg_response_time", "database.connection_errors"]:
                    should_alert = current_value > threshold
                elif metric_name in ["cache.hit_rate"]:
                    should_alert = current_value < threshold
                
                if should_alert:
                    await self._trigger_alert(alert_name, severity, message, metric_name, threshold, current_value)
                else:
                    await self._resolve_alert(alert_name)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert {alert_name}: {e}")
    
    async def _get_latest_metric(self, metric_name: str) -> Optional[Metric]:
        """Get the latest metric value"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
    
    async def _trigger_alert(self, alert_name: str, severity: AlertSeverity, message: str,
                           metric_name: str, threshold: float, current_value: float):
        """Trigger an alert"""
        try:
            # Check if alert is already active
            if alert_name in self.alerts and not self.alerts[alert_name].resolved:
                return
            
            # Check cooldown
            if alert_name in self.alerts:
                last_alert = self.alerts[alert_name]
                if time.time() - last_alert.timestamp < self.alert_cooldown:
                    return
            
            alert = Alert(
                alert_id=f"{alert_name}_{int(time.time())}",
                name=alert_name,
                severity=severity,
                message=message,
                metric_name=metric_name,
                threshold=threshold,
                current_value=current_value,
                timestamp=time.time()
            )
            
            self.alerts[alert_name] = alert
            self.stats["alerts_triggered"] += 1
            
            # Store alert in cache
            await enhanced_cache.set(
                f"alert_{alert_name}",
                asdict(alert),
                ttl=86400,  # 24 hours
                tags=["alerts", alert_name, severity.value]
            )
            
            # Call alert handlers
            for handler in self.alert_handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
            
            logger.warning(f"Alert triggered: {alert_name} - {message} (Current: {current_value}, Threshold: {threshold})")
            
        except Exception as e:
            logger.error(f"Error triggering alert {alert_name}: {e}")
    
    async def _resolve_alert(self, alert_name: str):
        """Resolve an alert"""
        try:
            if alert_name in self.alerts and not self.alerts[alert_name].resolved:
                alert = self.alerts[alert_name]
                alert.resolved = True
                alert.resolved_at = time.time()
                
                self.stats["alerts_resolved"] += 1
                
                # Update alert in cache
                await enhanced_cache.set(
                    f"alert_{alert_name}",
                    asdict(alert),
                    ttl=86400,
                    tags=["alerts", alert_name, "resolved"]
                )
                
                logger.info(f"Alert resolved: {alert_name}")
                
        except Exception as e:
            logger.error(f"Error resolving alert {alert_name}: {e}")
    
    async def _cleanup_loop(self):
        """Background task to cleanup old metrics and alerts"""
        while True:
            try:
                current_time = time.time()
                cutoff_time = current_time - self.retention_period
                
                # Cleanup old metrics
                for metric_name, metric_deque in self.metrics.items():
                    while metric_deque and metric_deque[0].timestamp < cutoff_time:
                        metric_deque.popleft()
                
                # Cleanup old resolved alerts
                old_alerts = [
                    name for name, alert in self.alerts.items()
                    if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
                ]
                for alert_name in old_alerts:
                    del self.alerts[alert_name]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    def register_metric_collector(self, collector: Callable):
        """Register a custom metric collector"""
        self.metric_collectors.append(collector)
        logger.info(f"Registered metric collector: {collector.__name__}")
    
    def register_alert_handler(self, handler: Callable):
        """Register a custom alert handler"""
        self.alert_handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__name__}")
    
    def add_alert_rule(self, name: str, rule: Dict[str, Any]):
        """Add a custom alert rule"""
        self.alert_rules[name] = rule
        logger.info(f"Added alert rule: {name}")
    
    async def get_metrics(self, metric_name: str, time_range: int = 3600) -> List[Metric]:
        """Get metrics for a specific time range"""
        try:
            if metric_name not in self.metrics:
                return []
            
            cutoff_time = time.time() - time_range
            return [
                metric for metric in self.metrics[metric_name]
                if metric.timestamp >= cutoff_time
            ]
        except Exception as e:
            logger.error(f"Error getting metrics for {metric_name}: {e}")
            return []
    
    async def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    async def get_alert_history(self, time_range: int = 86400) -> List[Alert]:
        """Get alert history for a specific time range"""
        try:
            cutoff_time = time.time() - time_range
            return [
                alert for alert in self.alerts.values()
                if alert.timestamp >= cutoff_time
            ]
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
            return []
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics"""
        return {
            "metrics_collected": self.stats["metrics_collected"],
            "alerts_triggered": self.stats["alerts_triggered"],
            "alerts_resolved": self.stats["alerts_resolved"],
            "collection_errors": self.stats["collection_errors"],
            "active_alerts": len([a for a in self.alerts.values() if not a.resolved]),
            "total_metrics": len(self.metrics),
            "alert_rules": len(self.alert_rules),
            "metric_collectors": len(self.metric_collectors),
            "alert_handlers": len(self.alert_handlers)
        }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            # Get latest system metrics
            cpu_metric = await self._get_latest_metric("system.cpu_percent")
            memory_metric = await self._get_latest_metric("system.memory_percent")
            disk_metric = await self._get_latest_metric("system.disk_usage_percent")
            
            # Determine health status
            health_status = "healthy"
            issues = []
            
            if cpu_metric and cpu_metric.value > 80:
                health_status = "degraded"
                issues.append("High CPU usage")
            
            if memory_metric and memory_metric.value > 85:
                health_status = "degraded"
                issues.append("High memory usage")
            
            if disk_metric and disk_metric.value > 90:
                health_status = "critical"
                issues.append("Low disk space")
            
            # Check active alerts
            active_alerts = await self.get_active_alerts()
            critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
            
            if critical_alerts:
                health_status = "critical"
                issues.extend([a.message for a in critical_alerts])
            
            return {
                "status": health_status,
                "issues": issues,
                "active_alerts": len(active_alerts),
                "critical_alerts": len(critical_alerts),
                "timestamp": time.time(),
                "metrics": {
                    "cpu_percent": cpu_metric.value if cpu_metric else 0,
                    "memory_percent": memory_metric.value if memory_metric else 0,
                    "disk_usage_percent": disk_metric.value if disk_metric else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "status": "unknown",
                "issues": ["Unable to determine system health"],
                "timestamp": time.time()
            }


# Global monitoring manager instance
monitoring_manager = MonitoringManager()
