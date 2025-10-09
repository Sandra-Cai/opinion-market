"""
Comprehensive metrics collection system for Opinion Market
Provides real-time monitoring, alerting, and performance tracking
"""

import asyncio
import time
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json

from app.core.logging import log_system_metric
from app.core.config import settings
from app.core.database import get_db_session, check_database_health, check_redis_health
from app.core.cache import cache_health_check, get_cache_stats


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    """Alert configuration"""
    name: str
    condition: Callable[[float], bool]
    severity: str = "warning"  # info, warning, error, critical
    message: str = ""
    cooldown: int = 300  # seconds
    last_triggered: Optional[datetime] = None


class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Alert] = {}
        self.collectors: Dict[str, Callable] = {}
        self.running = False
        self.collection_interval = 30  # seconds
        self._lock = threading.Lock()
        
        # Initialize collectors
        self._setup_collectors()
        self._setup_alerts()
    
    def _setup_collectors(self):
        """Setup metric collectors"""
        self.collectors = {
            "system": self._collect_system_metrics,
            "application": self._collect_application_metrics,
            "database": self._collect_database_metrics,
            "cache": self._collect_cache_metrics,
            "business": self._collect_business_metrics,
        }
    
    def _setup_alerts(self):
        """Setup alert conditions"""
        self.alerts = {
            "high_cpu": Alert(
                name="high_cpu",
                condition=lambda x: x > 80,
                severity="warning",
                message="High CPU usage detected",
                cooldown=300
            ),
            "high_memory": Alert(
                name="high_memory",
                condition=lambda x: x > 85,
                severity="warning",
                message="High memory usage detected",
                cooldown=300
            ),
            "low_disk_space": Alert(
                name="low_disk_space",
                condition=lambda x: x > 90,
                severity="error",
                message="Low disk space detected",
                cooldown=600
            ),
            "database_slow": Alert(
                name="database_slow",
                condition=lambda x: x > 1.0,
                severity="warning",
                message="Slow database response time",
                cooldown=300
            ),
            "cache_unhealthy": Alert(
                name="cache_unhealthy",
                condition=lambda x: x == 0,  # 0 means unhealthy
                severity="error",
                message="Cache system is unhealthy",
                cooldown=300
            ),
            "high_error_rate": Alert(
                name="high_error_rate",
                condition=lambda x: x > 5,  # 5% error rate
                severity="critical",
                message="High error rate detected",
                cooldown=60
            ),
        }
    
    async def start(self):
        """Start metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting metrics collection")
        
        # Start collection loop
        asyncio.create_task(self._collection_loop())
    
    async def stop(self):
        """Stop metrics collection"""
        self.running = False
        self.logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                await self._collect_all_metrics()
                await self._check_alerts()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_all_metrics(self):
        """Collect all metrics"""
        for collector_name, collector_func in self.collectors.items():
            try:
                metrics = await collector_func()
                for metric in metrics:
                    self._add_metric(metric)
            except Exception as e:
                self.logger.error(f"Error collecting {collector_name} metrics: {e}")
    
    def _add_metric(self, metric: Metric):
        """Add metric to collection"""
        with self._lock:
            self.metrics[metric.name].append(metric)
    
    async def _collect_system_metrics(self) -> List[Metric]:
        """Collect system-level metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(Metric("system.cpu.usage", cpu_percent, now, unit="percent"))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(Metric("system.memory.usage", memory.percent, now, unit="percent"))
            metrics.append(Metric("system.memory.available", memory.available, now, unit="bytes"))
            metrics.append(Metric("system.memory.total", memory.total, now, unit="bytes"))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.append(Metric("system.disk.usage", (disk.used / disk.total) * 100, now, unit="percent"))
            metrics.append(Metric("system.disk.available", disk.free, now, unit="bytes"))
            metrics.append(Metric("system.disk.total", disk.total, now, unit="bytes"))
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics.append(Metric("system.network.bytes_sent", network.bytes_sent, now, unit="bytes"))
            metrics.append(Metric("system.network.bytes_recv", network.bytes_recv, now, unit="bytes"))
            
            # Process metrics
            process = psutil.Process()
            metrics.append(Metric("system.process.cpu_percent", process.cpu_percent(), now, unit="percent"))
            metrics.append(Metric("system.process.memory_percent", process.memory_percent(), now, unit="percent"))
            metrics.append(Metric("system.process.memory_rss", process.memory_info().rss, now, unit="bytes"))
            metrics.append(Metric("system.process.memory_vms", process.memory_info().vms, now, unit="bytes"))
            
            # Load average (Unix only)
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
                metrics.append(Metric("system.load.1min", load_avg[0], now))
                metrics.append(Metric("system.load.5min", load_avg[1], now))
                metrics.append(Metric("system.load.15min", load_avg[2], now))
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    async def _collect_application_metrics(self) -> List[Metric]:
        """Collect application-level metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # Application uptime
            if hasattr(self, 'start_time'):
                uptime = (now - self.start_time).total_seconds()
                metrics.append(Metric("application.uptime", uptime, now, unit="seconds"))
            
            # Request metrics (would be collected from middleware)
            metrics.append(Metric("application.requests.total", 0, now))  # Placeholder
            metrics.append(Metric("application.requests.rate", 0, now, unit="requests_per_second"))
            metrics.append(Metric("application.response_time.avg", 0, now, unit="seconds"))
            metrics.append(Metric("application.response_time.p95", 0, now, unit="seconds"))
            metrics.append(Metric("application.response_time.p99", 0, now, unit="seconds"))
            
            # Error metrics
            metrics.append(Metric("application.errors.total", 0, now))
            metrics.append(Metric("application.errors.rate", 0, now, unit="errors_per_second"))
            
            # Active connections
            metrics.append(Metric("application.connections.active", 0, now))
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
        
        return metrics
    
    async def _collect_database_metrics(self) -> List[Metric]:
        """Collect database metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # Database health
            db_health = check_database_health()
            metrics.append(Metric("database.health", 1 if db_health["status"] == "healthy" else 0, now))
            metrics.append(Metric("database.response_time", db_health.get("response_time", 0), now, unit="seconds"))
            
            # Connection pool metrics
            if "connection_pool" in db_health:
                pool = db_health["connection_pool"]
                metrics.append(Metric("database.pool.size", pool.get("size", 0), now))
                metrics.append(Metric("database.pool.checked_in", pool.get("checked_in", 0), now))
                metrics.append(Metric("database.pool.checked_out", pool.get("checked_out", 0), now))
                metrics.append(Metric("database.pool.overflow", pool.get("overflow", 0), now))
                metrics.append(Metric("database.pool.invalid", pool.get("invalid", 0), now))
            
            # Query metrics (would be collected from database middleware)
            metrics.append(Metric("database.queries.total", 0, now))
            metrics.append(Metric("database.queries.rate", 0, now, unit="queries_per_second"))
            metrics.append(Metric("database.queries.slow", 0, now))
            
        except Exception as e:
            self.logger.error(f"Error collecting database metrics: {e}")
        
        return metrics
    
    async def _collect_cache_metrics(self) -> List[Metric]:
        """Collect cache metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            # Cache health
            cache_health = cache_health_check()
            metrics.append(Metric("cache.health", 1 if cache_health["status"] == "healthy" else 0, now))
            
            # Cache statistics
            cache_stats = get_cache_stats()
            if "l1_cache" in cache_stats:
                l1 = cache_stats["l1_cache"]
                metrics.append(Metric("cache.l1.hits", l1.get("hits", 0), now))
                metrics.append(Metric("cache.l1.misses", l1.get("misses", 0), now))
                metrics.append(Metric("cache.l1.hit_rate", l1.get("hit_rate", 0), now, unit="percent"))
                metrics.append(Metric("cache.l1.size", l1.get("size", 0), now))
            
            if "l2_cache" in cache_stats:
                l2 = cache_stats["l2_cache"]
                metrics.append(Metric("cache.l2.hits", l2.get("hits", 0), now))
                metrics.append(Metric("cache.l2.misses", l2.get("misses", 0), now))
                metrics.append(Metric("cache.l2.hit_rate", l2.get("hit_rate", 0), now, unit="percent"))
            
            # Redis metrics
            redis_health = check_redis_health()
            if redis_health["status"] == "healthy" and "info" in redis_health:
                info = redis_health["info"]
                metrics.append(Metric("cache.redis.connected_clients", info.get("connected_clients", 0), now))
                metrics.append(Metric("cache.redis.total_commands", info.get("total_commands_processed", 0), now))
            
        except Exception as e:
            self.logger.error(f"Error collecting cache metrics: {e}")
        
        return metrics
    
    async def _collect_business_metrics(self) -> List[Metric]:
        """Collect business-level metrics"""
        metrics = []
        now = datetime.utcnow()
        
        try:
            with get_db_session() as db:
                # User metrics
                from app.models.user import User
                total_users = db.query(User).count()
                active_users = db.query(User).filter(User.is_active == True).count()
                metrics.append(Metric("business.users.total", total_users, now))
                metrics.append(Metric("business.users.active", active_users, now))
                
                # Market metrics
                from app.models.market import Market, MarketStatus
                total_markets = db.query(Market).count()
                active_markets = db.query(Market).filter(Market.status == MarketStatus.OPEN).count()
                metrics.append(Metric("business.markets.total", total_markets, now))
                metrics.append(Metric("business.markets.active", active_markets, now))
                
                # Trading metrics
                from app.models.trade import Trade
                from sqlalchemy import func
                total_trades = db.query(Trade).count()
                total_volume = db.query(func.sum(Trade.total_value)).scalar() or 0
                metrics.append(Metric("business.trades.total", total_trades, now))
                metrics.append(Metric("business.volume.total", total_volume, now))
                
                # 24h metrics
                since_24h = now - timedelta(hours=24)
                trades_24h = db.query(Trade).filter(Trade.created_at >= since_24h).count()
                volume_24h = db.query(func.sum(Trade.total_value)).filter(Trade.created_at >= since_24h).scalar() or 0
                metrics.append(Metric("business.trades.24h", trades_24h, now))
                metrics.append(Metric("business.volume.24h", volume_24h, now))
                
        except Exception as e:
            self.logger.error(f"Error collecting business metrics: {e}")
        
        return metrics
    
    async def _check_alerts(self):
        """Check alert conditions"""
        for alert_name, alert in self.alerts.items():
            try:
                # Get latest metric value
                metric_name = self._get_metric_name_for_alert(alert_name)
                if metric_name in self.metrics and self.metrics[metric_name]:
                    latest_metric = self.metrics[metric_name][-1]
                    
                    # Check if alert should trigger
                    if alert.condition(latest_metric.value):
                        # Check cooldown
                        if (alert.last_triggered is None or 
                            (datetime.utcnow() - alert.last_triggered).total_seconds() > alert.cooldown):
                            
                            # Trigger alert
                            await self._trigger_alert(alert, latest_metric)
                            alert.last_triggered = datetime.utcnow()
                            
            except Exception as e:
                self.logger.error(f"Error checking alert {alert_name}: {e}")
    
    def _get_metric_name_for_alert(self, alert_name: str) -> str:
        """Get metric name for alert"""
        alert_mapping = {
            "high_cpu": "system.cpu.usage",
            "high_memory": "system.memory.usage",
            "low_disk_space": "system.disk.usage",
            "database_slow": "database.response_time",
            "cache_unhealthy": "cache.health",
            "high_error_rate": "application.errors.rate",
        }
        return alert_mapping.get(alert_name, "")
    
    async def _trigger_alert(self, alert: Alert, metric: Metric):
        """Trigger an alert"""
        alert_data = {
            "alert_name": alert.name,
            "severity": alert.severity,
            "message": alert.message,
            "metric_name": metric.name,
            "metric_value": metric.value,
            "metric_unit": metric.unit,
            "timestamp": metric.timestamp.isoformat(),
        }
        
        # Log the alert
        log_system_metric("alert_triggered", 1, alert_data)
        
        # Send to external monitoring system (e.g., Slack, email, etc.)
        await self._send_alert_notification(alert_data)
    
    async def _send_alert_notification(self, alert_data: Dict[str, Any]):
        """Send alert notification to external systems"""
        # This would integrate with external alerting systems
        # For now, just log the alert
        self.logger.warning(f"ALERT: {alert_data['message']}", **alert_data)
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Metric]:
        """Get metric history for a specific metric"""
        if metric_name not in self.metrics:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [m for m in self.metrics[metric_name] if m.timestamp >= cutoff_time]
    
    def get_latest_metric(self, metric_name: str) -> Optional[Metric]:
        """Get latest metric value"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1]
    
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get metric summary statistics"""
        history = self.get_metric_history(metric_name, hours)
        if not history:
            return {}
        
        values = [m.value for m in history]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "first": values[0],
            "unit": history[0].unit if history else "",
        }
    
    def get_all_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        for metric_name in self.metrics.keys():
            summary[metric_name] = self.get_metric_summary(metric_name, hours)
        return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
            "alerts": []
        }
        
        # Check critical metrics
        critical_metrics = {
            "system.cpu.usage": 80,
            "system.memory.usage": 85,
            "system.disk.usage": 90,
            "database.health": 0.5,
            "cache.health": 0.5,
        }
        
        unhealthy_count = 0
        for metric_name, threshold in critical_metrics.items():
            latest = self.get_latest_metric(metric_name)
            if latest:
                health_status["metrics"][metric_name] = {
                    "value": latest.value,
                    "threshold": threshold,
                    "healthy": latest.value < threshold
                }
                if not health_status["metrics"][metric_name]["healthy"]:
                    unhealthy_count += 1
        
        # Check active alerts
        for alert_name, alert in self.alerts.items():
            if alert.last_triggered:
                time_since_triggered = (datetime.utcnow() - alert.last_triggered).total_seconds()
                if time_since_triggered < alert.cooldown:
                    health_status["alerts"].append({
                        "name": alert_name,
                        "severity": alert.severity,
                        "message": alert.message,
                        "triggered_at": alert.last_triggered.isoformat()
                    })
        
        # Determine overall status
        if unhealthy_count > 2:
            health_status["status"] = "critical"
        elif unhealthy_count > 0 or health_status["alerts"]:
            health_status["status"] = "warning"
        
        return health_status


# Global metrics collector instance
metrics_collector = MetricsCollector()
