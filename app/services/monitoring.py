import asyncio
import json
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from sqlalchemy.orm import Session
import redis as redis_sync
import redis.asyncio as redis
import aiohttp

from app.core.database import SessionLocal
from app.models.market import Market, MarketStatus
from app.models.trade import Trade
from app.models.user import User
from app.services.notification_service import get_notification_service

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics"""

    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    database_connections: int
    redis_connections: int


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""

    timestamp: datetime
    total_users: int
    active_markets: int
    total_trades_24h: int
    total_volume_24h: float
    average_response_time: float
    error_rate: float
    api_requests_per_minute: float


@dataclass
class Alert:
    """System alert"""

    alert_id: str
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime
    data: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class SystemMonitor:
    """Comprehensive system monitoring and alerting"""

    def __init__(self):
        self.redis_client: Optional[redis_sync.Redis] = None
        self.notification_service = get_notification_service()
        self.metrics_history: List[SystemMetrics] = []
        self.application_metrics_history: List[ApplicationMetrics] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_thresholds = {
            "cpu_usage": 80.0,  # 80% CPU usage
            "memory_usage": 85.0,  # 85% memory usage
            "disk_usage": 90.0,  # 90% disk usage
            "error_rate": 5.0,  # 5% error rate
            "response_time": 2000.0,  # 2 seconds average response time
            "database_connections": 100,  # 100 concurrent DB connections
            "redis_connections": 50,  # 50 concurrent Redis connections
        }
        self.metrics_collection_interval = 60  # seconds
        self.alert_check_interval = 30  # seconds
        self.health_check_interval = 300  # seconds

    async def initialize(self, redis_url: str):
        """Initialize the monitoring system"""
        self.redis_client = redis.from_url(redis_url)
        await self.redis_client.ping()
        logger.info("System monitoring initialized")

    async def start_monitoring(self):
        """Start all monitoring tasks"""
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._alert_monitoring_loop())
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._performance_analysis_loop())
        logger.info("System monitoring started")

    async def _metrics_collection_loop(self):
        """Collect system and application metrics"""
        while True:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                self.metrics_history.append(system_metrics)

                # Collect application metrics
                app_metrics = await self._collect_application_metrics()
                self.application_metrics_history.append(app_metrics)

                # Keep only last 24 hours of metrics
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history if m.timestamp > cutoff_time
                ]
                self.application_metrics_history = [
                    m
                    for m in self.application_metrics_history
                    if m.timestamp > cutoff_time
                ]

                # Store metrics in Redis for external monitoring
                if self.redis_client:
                    await self._store_metrics_in_redis(system_metrics, app_metrics)

                await asyncio.sleep(self.metrics_collection_interval)

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage = (disk.used / disk.total) * 100

            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv,
            }

            # Connection counts (approximate)
            active_connections = len(psutil.net_connections())
            database_connections = await self._get_database_connection_count()
            redis_connections = await self._get_redis_connection_count()

            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=active_connections,
                database_connections=database_connections,
                redis_connections=redis_connections,
            )

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                active_connections=0,
                database_connections=0,
                redis_connections=0,
            )

    async def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        try:
            db = SessionLocal()

            # User metrics
            total_users = db.query(User).count()

            # Market metrics
            active_markets = (
                db.query(Market).filter(Market.status == MarketStatus.ACTIVE).count()
            )

            # Trade metrics (last 24h)
            yesterday = datetime.utcnow() - timedelta(days=1)
            trades_24h = db.query(Trade).filter(Trade.created_at >= yesterday).all()

            total_trades_24h = len(trades_24h)
            total_volume_24h = sum(trade.total_value for trade in trades_24h)

            # Performance metrics (from Redis)
            average_response_time = await self._get_average_response_time()
            error_rate = await self._get_error_rate()
            api_requests_per_minute = await self._get_api_requests_per_minute()

            db.close()

            return ApplicationMetrics(
                timestamp=datetime.utcnow(),
                total_users=total_users,
                active_markets=active_markets,
                total_trades_24h=total_trades_24h,
                total_volume_24h=total_volume_24h,
                average_response_time=average_response_time,
                error_rate=error_rate,
                api_requests_per_minute=api_requests_per_minute,
            )

        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return ApplicationMetrics(
                timestamp=datetime.utcnow(),
                total_users=0,
                active_markets=0,
                total_trades_24h=0,
                total_volume_24h=0.0,
                average_response_time=0.0,
                error_rate=0.0,
                api_requests_per_minute=0.0,
            )

    async def _alert_monitoring_loop(self):
        """Monitor metrics and generate alerts"""
        while True:
            try:
                if self.metrics_history:
                    latest_system_metrics = self.metrics_history[-1]
                    await self._check_system_alerts(latest_system_metrics)

                if self.application_metrics_history:
                    latest_app_metrics = self.application_metrics_history[-1]
                    await self._check_application_alerts(latest_app_metrics)

                await asyncio.sleep(self.alert_check_interval)

            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system-level alerts"""
        # CPU usage alert
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            await self._create_alert(
                alert_type="high_cpu_usage",
                severity="high" if metrics.cpu_usage > 90 else "medium",
                message=f"High CPU usage detected: {metrics.cpu_usage:.1f}%",
                data={"cpu_usage": metrics.cpu_usage},
            )

        # Memory usage alert
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            await self._create_alert(
                alert_type="high_memory_usage",
                severity="high" if metrics.memory_usage > 95 else "medium",
                message=f"High memory usage detected: {metrics.memory_usage:.1f}%",
                data={"memory_usage": metrics.memory_usage},
            )

        # Disk usage alert
        if metrics.disk_usage > self.alert_thresholds["disk_usage"]:
            await self._create_alert(
                alert_type="high_disk_usage",
                severity="critical" if metrics.disk_usage > 95 else "high",
                message=f"High disk usage detected: {metrics.disk_usage:.1f}%",
                data={"disk_usage": metrics.disk_usage},
            )

        # Database connections alert
        if metrics.database_connections > self.alert_thresholds["database_connections"]:
            await self._create_alert(
                alert_type="high_db_connections",
                severity="high",
                message=f"High database connections: {metrics.database_connections}",
                data={"database_connections": metrics.database_connections},
            )

        # Redis connections alert
        if metrics.redis_connections > self.alert_thresholds["redis_connections"]:
            await self._create_alert(
                alert_type="high_redis_connections",
                severity="medium",
                message=f"High Redis connections: {metrics.redis_connections}",
                data={"redis_connections": metrics.redis_connections},
            )

    async def _check_application_alerts(self, metrics: ApplicationMetrics):
        """Check for application-level alerts"""
        # Error rate alert
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            await self._create_alert(
                alert_type="high_error_rate",
                severity="high" if metrics.error_rate > 10 else "medium",
                message=f"High error rate detected: {metrics.error_rate:.1f}%",
                data={"error_rate": metrics.error_rate},
            )

        # Response time alert
        if metrics.average_response_time > self.alert_thresholds["response_time"]:
            await self._create_alert(
                alert_type="slow_response_time",
                severity="medium",
                message=f"Slow response time: {metrics.average_response_time:.0f}ms",
                data={"response_time": metrics.average_response_time},
            )

        # Low activity alert
        if metrics.total_trades_24h < 10:
            await self._create_alert(
                alert_type="low_trading_activity",
                severity="low",
                message=f"Low trading activity: {metrics.total_trades_24h} trades in 24h",
                data={"total_trades_24h": metrics.total_trades_24h},
            )

    async def _create_alert(
        self, alert_type: str, severity: str, message: str, data: Dict[str, Any]
    ):
        """Create and manage an alert"""
        alert_id = f"{alert_type}_{int(time.time())}"

        # Check if similar alert already exists
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.alert_type == alert_type and not alert.resolved:
                existing_alert = alert
                break

        if existing_alert:
            # Update existing alert
            existing_alert.message = message
            existing_alert.data.update(data)
            existing_alert.timestamp = datetime.utcnow()
        else:
            # Create new alert
            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                timestamp=datetime.utcnow(),
                data=data,
            )
            self.active_alerts[alert_id] = alert

            # Send notification for new alerts
            await self._send_alert_notification(alert)

            # Log alert
            logger.warning(f"Alert created: {message}")

    async def _health_check_loop(self):
        """Perform periodic health checks"""
        while True:
            try:
                # Database health check
                await self._check_database_health()

                # Redis health check
                await self._check_redis_health()

                # External service health checks
                await self._check_external_services()

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)

    async def _performance_analysis_loop(self):
        """Analyze performance trends and generate reports"""
        while True:
            try:
                if len(self.application_metrics_history) > 10:
                    await self._analyze_performance_trends()

                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                logger.error(f"Error in performance analysis loop: {e}")
                await asyncio.sleep(3600)

    async def _get_database_connection_count(self) -> int:
        """Get current database connection count"""
        try:
            # This is a simplified approach - in production you'd query the database directly
            return 10  # Placeholder
        except Exception as e:
            logger.error(f"Error getting database connection count: {e}")
            return 0

    async def _get_redis_connection_count(self) -> int:
        """Get current Redis connection count"""
        try:
            if self.redis_client:
                info = await self.redis_client.info("clients")
                return info.get("connected_clients", 0)
            return 0
        except Exception as e:
            logger.error(f"Error getting Redis connection count: {e}")
            return 0

    async def _get_average_response_time(self) -> float:
        """Get average API response time"""
        try:
            if self.redis_client:
                # This would be tracked in Redis by middleware
                return 150.0  # Placeholder
            return 0.0
        except Exception as e:
            logger.error(f"Error getting average response time: {e}")
            return 0.0

    async def _get_error_rate(self) -> float:
        """Get current error rate"""
        try:
            if self.redis_client:
                # This would be tracked in Redis by middleware
                return 1.5  # Placeholder 1.5%
            return 0.0
        except Exception as e:
            logger.error(f"Error getting error rate: {e}")
            return 0.0

    async def _get_api_requests_per_minute(self) -> float:
        """Get API requests per minute"""
        try:
            if self.redis_client:
                # This would be tracked in Redis by middleware
                return 120.0  # Placeholder
            return 0.0
        except Exception as e:
            logger.error(f"Error getting API requests per minute: {e}")
            return 0.0

    async def _store_metrics_in_redis(
        self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics
    ):
        """Store metrics in Redis for external monitoring"""
        try:
            metrics_data = {
                "system": {
                    "timestamp": system_metrics.timestamp.isoformat(),
                    "cpu_usage": system_metrics.cpu_usage,
                    "memory_usage": system_metrics.memory_usage,
                    "disk_usage": system_metrics.disk_usage,
                    "active_connections": system_metrics.active_connections,
                    "database_connections": system_metrics.database_connections,
                    "redis_connections": system_metrics.redis_connections,
                },
                "application": {
                    "timestamp": app_metrics.timestamp.isoformat(),
                    "total_users": app_metrics.total_users,
                    "active_markets": app_metrics.active_markets,
                    "total_trades_24h": app_metrics.total_trades_24h,
                    "total_volume_24h": app_metrics.total_volume_24h,
                    "average_response_time": app_metrics.average_response_time,
                    "error_rate": app_metrics.error_rate,
                    "api_requests_per_minute": app_metrics.api_requests_per_minute,
                },
            }

            await self.redis_client.set(
                "system_metrics", json.dumps(metrics_data), ex=3600  # Expire in 1 hour
            )

        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")

    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification to administrators"""
        try:
            # Send to notification service
            await self.notification_service.create_notification(
                user_id=1,  # Admin user ID
                notification_type="system_alert",
                title=f"System Alert: {alert.alert_type}",
                message=alert.message,
                priority=alert.severity,
                metadata=alert.data,
            )

        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")

    async def _check_database_health(self):
        """Check database health"""
        try:
            db = SessionLocal()
            # Simple query to check database connectivity
            db.execute("SELECT 1")
            db.close()
        except Exception as e:
            await self._create_alert(
                alert_type="database_health_check_failed",
                severity="critical",
                message=f"Database health check failed: {str(e)}",
                data={"error": str(e)},
            )

    async def _check_redis_health(self):
        """Check Redis health"""
        try:
            if self.redis_client:
                await self.redis_client.ping()
        except Exception as e:
            await self._create_alert(
                alert_type="redis_health_check_failed",
                severity="critical",
                message=f"Redis health check failed: {str(e)}",
                data={"error": str(e)},
            )

    async def _check_external_services(self):
        """Check external service health"""
        try:
            # Check if the application is responding
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/health") as response:
                    if response.status != 200:
                        await self._create_alert(
                            alert_type="application_health_check_failed",
                            severity="critical",
                            message=f"Application health check failed: {response.status}",
                            data={"status_code": response.status},
                        )
        except Exception as e:
            await self._create_alert(
                alert_type="external_service_check_failed",
                severity="high",
                message=f"External service check failed: {str(e)}",
                data={"error": str(e)},
            )

    async def _analyze_performance_trends(self):
        """Analyze performance trends and generate insights"""
        try:
            if len(self.application_metrics_history) < 10:
                return

            # Calculate trends
            recent_metrics = self.application_metrics_history[-10:]

            # Response time trend
            response_times = [m.average_response_time for m in recent_metrics]
            response_time_trend = (response_times[-1] - response_times[0]) / len(
                response_times
            )

            # Error rate trend
            error_rates = [m.error_rate for m in recent_metrics]
            error_rate_trend = (error_rates[-1] - error_rates[0]) / len(error_rates)

            # Generate insights
            if response_time_trend > 50:  # Increasing by more than 50ms per interval
                await self._create_alert(
                    alert_type="response_time_increasing",
                    severity="medium",
                    message=f"Response time is increasing: +{response_time_trend:.1f}ms per interval",
                    data={"trend": response_time_trend},
                )

            if (
                error_rate_trend > 1
            ):  # Error rate increasing by more than 1% per interval
                await self._create_alert(
                    alert_type="error_rate_increasing",
                    severity="high",
                    message=f"Error rate is increasing: +{error_rate_trend:.1f}% per interval",
                    data={"trend": error_rate_trend},
                )

        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")

    async def get_system_metrics(self, hours: int = 24) -> List[SystemMetrics]:
        """Get system metrics for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]

    async def get_application_metrics(
        self, hours: int = 24
    ) -> List[ApplicationMetrics]:
        """Get application metrics for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            m for m in self.application_metrics_history if m.timestamp > cutoff_time
        ]

    async def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get active alerts with optional severity filtering"""
        alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return alerts

    async def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            logger.info(f"Alert resolved: {alert.message}")


# Global system monitor instance
system_monitor = SystemMonitor()


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance"""
    return system_monitor
