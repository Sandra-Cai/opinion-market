"""
Performance Monitoring and Optimization Service
Tracks system performance, identifies bottlenecks, and provides optimization recommendations
"""

import asyncio
import logging
import time
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point"""

    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]


@dataclass
class PerformanceAlert:
    """Performance alert"""

    alert_id: str
    alert_type: str  # 'threshold_exceeded', 'trend_anomaly', 'resource_exhaustion'
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    resolved: bool = False


@dataclass
class PerformanceBottleneck:
    """Performance bottleneck identification"""

    bottleneck_id: str
    component: str  # 'database', 'cache', 'api', 'memory', 'cpu'
    metric_name: str
    current_value: float
    baseline_value: float
    impact_score: float  # 0-1, how much it impacts performance
    description: str
    recommendations: List[str]
    detected_at: datetime


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""

    recommendation_id: str
    category: str  # 'database', 'caching', 'code', 'infrastructure'
    priority: str  # 'low', 'medium', 'high', 'critical'
    title: str
    description: str
    expected_improvement: float  # percentage improvement expected
    implementation_effort: str  # 'low', 'medium', 'high'
    estimated_cost: str  # 'free', 'low', 'medium', 'high'
    created_at: datetime


class PerformanceMonitoringService:
    """Comprehensive performance monitoring and optimization service"""

    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, PerformanceAlert] = {}
        self.bottlenecks: Dict[str, PerformanceBottleneck] = {}
        self.recommendations: Dict[str, OptimizationRecommendation] = {}

        # Performance thresholds
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "api_response_time": 1000.0,  # ms
            "database_query_time": 500.0,  # ms
            "cache_hit_rate": 70.0,  # percentage
            "error_rate": 5.0,  # percentage
            "active_connections": 1000,
            "queue_depth": 100,
        }

        # Baseline values for comparison
        self.baselines = {}

    async def initialize(self):
        """Initialize the performance monitoring service"""
        logger.info("Initializing Performance Monitoring Service")

        # Start monitoring tasks
        asyncio.create_task(self._monitor_system_metrics())
        asyncio.create_task(self._monitor_application_metrics())
        asyncio.create_task(self._monitor_database_metrics())
        asyncio.create_task(self._monitor_cache_metrics())
        asyncio.create_task(self._detect_bottlenecks())
        asyncio.create_task(self._generate_recommendations())

        # Initialize baselines
        await self._initialize_baselines()

        logger.info("Performance Monitoring Service initialized successfully")

    async def _initialize_baselines(self):
        """Initialize baseline performance values"""
        try:
            # Collect baseline metrics over 5 minutes
            baseline_metrics = []
            for _ in range(10):
                metrics = await self._collect_current_metrics()
                baseline_metrics.append(metrics)
                await asyncio.sleep(30)

            # Calculate baseline averages
            for metric_name in baseline_metrics[0].keys():
                values = [m[metric_name] for m in baseline_metrics if metric_name in m]
                if values:
                    self.baselines[metric_name] = np.mean(values)

            logger.info(f"Initialized baselines for {len(self.baselines)} metrics")

        except Exception as e:
            logger.error(f"Error initializing baselines: {e}")

    async def _monitor_system_metrics(self):
        """Monitor system-level performance metrics"""
        while True:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()

                # Memory metrics
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()

                # Disk metrics
                disk = psutil.disk_usage("/")
                disk_io = psutil.disk_io_counters()

                # Network metrics
                network = psutil.net_io_counters()

                # Record metrics
                await self._record_metric("cpu_usage", cpu_percent, "%")
                await self._record_metric("cpu_count", cpu_count, "cores")
                if cpu_freq:
                    await self._record_metric("cpu_frequency", cpu_freq.current, "MHz")

                await self._record_metric("memory_usage", memory.percent, "%")
                await self._record_metric(
                    "memory_available", memory.available / (1024**3), "GB"
                )
                await self._record_metric("swap_usage", swap.percent, "%")

                await self._record_metric("disk_usage", disk.percent, "%")
                await self._record_metric("disk_free", disk.free / (1024**3), "GB")
                if disk_io:
                    await self._record_metric(
                        "disk_read_bytes", disk_io.read_bytes, "bytes"
                    )
                    await self._record_metric(
                        "disk_write_bytes", disk_io.write_bytes, "bytes"
                    )

                await self._record_metric(
                    "network_bytes_sent", network.bytes_sent, "bytes"
                )
                await self._record_metric(
                    "network_bytes_recv", network.bytes_recv, "bytes"
                )

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(60)

    async def _monitor_application_metrics(self):
        """Monitor application-level performance metrics"""
        while True:
            try:
                # API response times (simulated)
                api_response_time = await self._measure_api_response_time()
                await self._record_metric("api_response_time", api_response_time, "ms")

                # Error rates (simulated)
                error_rate = await self._calculate_error_rate()
                await self._record_metric("error_rate", error_rate, "%")

                # Active connections (simulated)
                active_connections = await self._get_active_connections()
                await self._record_metric(
                    "active_connections", active_connections, "connections"
                )

                # Queue depth (simulated)
                queue_depth = await self._get_queue_depth()
                await self._record_metric("queue_depth", queue_depth, "items")

                # Memory usage of current process
                process = psutil.Process()
                process_memory = process.memory_info()
                await self._record_metric(
                    "process_memory", process_memory.rss / (1024**2), "MB"
                )

                # Garbage collection stats
                gc_stats = gc.get_stats()
                if gc_stats:
                    total_collections = sum(stat["collections"] for stat in gc_stats)
                    await self._record_metric(
                        "gc_collections", total_collections, "count"
                    )

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error monitoring application metrics: {e}")
                await asyncio.sleep(120)

    async def _monitor_database_metrics(self):
        """Monitor database performance metrics"""
        while True:
            try:
                # Database connection pool stats
                pool_stats = await self._get_database_pool_stats()
                await self._record_metric(
                    "db_connections_active", pool_stats.get("active", 0), "connections"
                )
                await self._record_metric(
                    "db_connections_idle", pool_stats.get("idle", 0), "connections"
                )

                # Query performance (simulated)
                query_time = await self._measure_database_query_time()
                await self._record_metric("database_query_time", query_time, "ms")

                # Database size (simulated)
                db_size = await self._get_database_size()
                await self._record_metric("database_size", db_size, "MB")

                await asyncio.sleep(120)  # Update every 2 minutes

            except Exception as e:
                logger.error(f"Error monitoring database metrics: {e}")
                await asyncio.sleep(300)

    async def _monitor_cache_metrics(self):
        """Monitor cache performance metrics"""
        while True:
            try:
                # Redis cache stats
                cache_stats = await self._get_cache_stats()
                await self._record_metric(
                    "cache_hit_rate", cache_stats.get("hit_rate", 0), "%"
                )
                await self._record_metric(
                    "cache_memory_usage", cache_stats.get("memory_usage", 0), "MB"
                )
                await self._record_metric(
                    "cache_keys", cache_stats.get("keys", 0), "count"
                )

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error monitoring cache metrics: {e}")
                await asyncio.sleep(120)

    async def _detect_bottlenecks(self):
        """Detect performance bottlenecks"""
        while True:
            try:
                current_metrics = await self._collect_current_metrics()

                for metric_name, current_value in current_metrics.items():
                    if metric_name in self.baselines:
                        baseline = self.baselines[metric_name]

                        # Calculate deviation from baseline
                        if baseline > 0:
                            deviation = abs(current_value - baseline) / baseline

                            # Detect significant deviations
                            if deviation > 0.5:  # 50% deviation
                                bottleneck = await self._create_bottleneck(
                                    metric_name, current_value, baseline, deviation
                                )
                                if bottleneck:
                                    self.bottlenecks[bottleneck.bottleneck_id] = (
                                        bottleneck
                                    )

                # Clean up old bottlenecks
                await self._cleanup_old_bottlenecks()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error detecting bottlenecks: {e}")
                await asyncio.sleep(600)

    async def _generate_recommendations(self):
        """Generate performance optimization recommendations"""
        while True:
            try:
                # Analyze current performance state
                performance_analysis = await self._analyze_performance()

                # Generate recommendations based on analysis
                new_recommendations = await self._create_recommendations(
                    performance_analysis
                )

                for recommendation in new_recommendations:
                    self.recommendations[recommendation.recommendation_id] = (
                        recommendation
                    )

                # Clean up old recommendations
                await self._cleanup_old_recommendations()

                await asyncio.sleep(1800)  # Generate every 30 minutes

            except Exception as e:
                logger.error(f"Error generating recommendations: {e}")
                await asyncio.sleep(3600)

    async def _record_metric(
        self, metric_name: str, value: float, unit: str, tags: Dict[str, str] = None
    ):
        """Record a performance metric"""
        try:
            metric = PerformanceMetric(
                metric_name=metric_name,
                value=value,
                unit=unit,
                timestamp=datetime.utcnow(),
                tags=tags or {},
            )

            # Store in history
            self.metrics_history[metric_name].append(metric)

            # Check thresholds and create alerts
            await self._check_thresholds(metric)

            # Cache metric
            await self._cache_metric(metric)

        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")

    async def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds and create alerts"""
        try:
            threshold = self.thresholds.get(metric.metric_name)
            if threshold is None:
                return

            # Check if threshold is exceeded
            if metric.value > threshold:
                alert_id = (
                    f"{metric.metric_name}_{metric.timestamp.strftime('%Y%m%d_%H%M%S')}"
                )

                # Determine severity
                if metric.value > threshold * 1.5:
                    severity = "critical"
                elif metric.value > threshold * 1.2:
                    severity = "high"
                elif metric.value > threshold * 1.1:
                    severity = "medium"
                else:
                    severity = "low"

                alert = PerformanceAlert(
                    alert_id=alert_id,
                    alert_type="threshold_exceeded",
                    severity=severity,
                    metric_name=metric.metric_name,
                    current_value=metric.value,
                    threshold_value=threshold,
                    message=f"{metric.metric_name} exceeded threshold: {metric.value} {metric.unit} > {threshold} {metric.unit}",
                    timestamp=metric.timestamp,
                )

                self.alerts[alert_id] = alert
                await self._cache_alert(alert)

                logger.warning(f"Performance alert: {alert.message}")

        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")

    async def _create_bottleneck(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        deviation: float,
    ) -> Optional[PerformanceBottleneck]:
        """Create a performance bottleneck"""
        try:
            bottleneck_id = f"bottleneck_{metric_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # Determine component based on metric name
            component_map = {
                "cpu_usage": "cpu",
                "memory_usage": "memory",
                "disk_usage": "disk",
                "api_response_time": "api",
                "database_query_time": "database",
                "cache_hit_rate": "cache",
            }
            component = component_map.get(metric_name, "unknown")

            # Calculate impact score based on deviation and metric importance
            importance_weights = {
                "cpu_usage": 0.9,
                "memory_usage": 0.8,
                "api_response_time": 0.7,
                "database_query_time": 0.7,
                "cache_hit_rate": 0.6,
                "disk_usage": 0.5,
            }
            importance = importance_weights.get(metric_name, 0.5)
            impact_score = min(deviation * importance, 1.0)

            # Generate description and recommendations
            description = f"{metric_name} is {deviation:.1%} above baseline ({current_value:.2f} vs {baseline_value:.2f})"
            recommendations = await self._get_bottleneck_recommendations(
                metric_name, component, current_value
            )

            return PerformanceBottleneck(
                bottleneck_id=bottleneck_id,
                component=component,
                metric_name=metric_name,
                current_value=current_value,
                baseline_value=baseline_value,
                impact_score=impact_score,
                description=description,
                recommendations=recommendations,
                detected_at=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error creating bottleneck: {e}")
            return None

    async def _get_bottleneck_recommendations(
        self, metric_name: str, component: str, current_value: float
    ) -> List[str]:
        """Get recommendations for a specific bottleneck"""
        recommendations = []

        try:
            if component == "cpu":
                if current_value > 90:
                    recommendations.extend(
                        [
                            "Consider scaling horizontally by adding more instances",
                            "Optimize CPU-intensive operations",
                            "Implement caching for expensive computations",
                        ]
                    )
                elif current_value > 80:
                    recommendations.extend(
                        [
                            "Monitor CPU usage trends",
                            "Consider optimizing database queries",
                            "Review background job scheduling",
                        ]
                    )

            elif component == "memory":
                if current_value > 90:
                    recommendations.extend(
                        [
                            "Increase available memory",
                            "Implement memory pooling",
                            "Review memory leaks in application code",
                        ]
                    )
                elif current_value > 80:
                    recommendations.extend(
                        [
                            "Monitor memory usage patterns",
                            "Consider implementing object pooling",
                            "Review caching strategies",
                        ]
                    )

            elif component == "database":
                recommendations.extend(
                    [
                        "Optimize slow queries",
                        "Add database indexes",
                        "Consider read replicas for read-heavy workloads",
                        "Implement query result caching",
                    ]
                )

            elif component == "cache":
                if current_value < 50:
                    recommendations.extend(
                        [
                            "Increase cache size",
                            "Implement cache warming strategies",
                            "Review cache eviction policies",
                            "Add more cache layers",
                        ]
                    )

            elif component == "api":
                recommendations.extend(
                    [
                        "Implement API response caching",
                        "Optimize database queries",
                        "Consider async processing for heavy operations",
                        "Add API rate limiting",
                    ]
                )

            if not recommendations:
                recommendations.append("Monitor the metric and investigate root cause")

            return recommendations

        except Exception as e:
            logger.error(f"Error getting bottleneck recommendations: {e}")
            return ["Investigate performance issue"]

    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance state"""
        try:
            analysis = {
                "current_metrics": {},
                "trends": {},
                "bottlenecks": {},
                "alerts": {},
                "overall_health": "unknown",
            }

            # Get current metrics
            current_metrics = await self._collect_current_metrics()
            analysis["current_metrics"] = current_metrics

            # Calculate trends
            for metric_name in current_metrics.keys():
                trend = await self._calculate_trend(metric_name)
                if trend:
                    analysis["trends"][metric_name] = trend

            # Get active bottlenecks
            active_bottlenecks = [
                b
                for b in self.bottlenecks.values()
                if (datetime.utcnow() - b.detected_at).total_seconds() < 3600
            ]
            analysis["bottlenecks"] = {
                b.bottleneck_id: {
                    "component": b.component,
                    "impact_score": b.impact_score,
                    "description": b.description,
                }
                for b in active_bottlenecks
            }

            # Get active alerts
            active_alerts = [a for a in self.alerts.values() if not a.resolved]
            analysis["alerts"] = {
                a.alert_id: {
                    "severity": a.severity,
                    "metric_name": a.metric_name,
                    "message": a.message,
                }
                for a in active_alerts
            }

            # Calculate overall health
            analysis["overall_health"] = self._calculate_overall_health(analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {}

    async def _create_recommendations(
        self, performance_analysis: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Create optimization recommendations based on performance analysis"""
        recommendations = []

        try:
            current_metrics = performance_analysis.get("current_metrics", {})
            bottlenecks = performance_analysis.get("bottlenecks", {})
            alerts = performance_analysis.get("alerts", {})

            # High CPU usage recommendations
            if current_metrics.get("cpu_usage", 0) > 80:
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"cpu_optimization_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        category="infrastructure",
                        priority=(
                            "high" if current_metrics["cpu_usage"] > 90 else "medium"
                        ),
                        title="Optimize CPU Usage",
                        description="High CPU usage detected. Consider scaling or optimization.",
                        expected_improvement=20.0,
                        implementation_effort="medium",
                        estimated_cost="medium",
                        created_at=datetime.utcnow(),
                    )
                )

            # High memory usage recommendations
            if current_metrics.get("memory_usage", 0) > 85:
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"memory_optimization_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        category="infrastructure",
                        priority=(
                            "high" if current_metrics["memory_usage"] > 95 else "medium"
                        ),
                        title="Optimize Memory Usage",
                        description="High memory usage detected. Consider memory optimization.",
                        expected_improvement=15.0,
                        implementation_effort="medium",
                        estimated_cost="low",
                        created_at=datetime.utcnow(),
                    )
                )

            # Low cache hit rate recommendations
            if current_metrics.get("cache_hit_rate", 100) < 70:
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"cache_optimization_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        category="caching",
                        priority="medium",
                        title="Improve Cache Hit Rate",
                        description="Low cache hit rate detected. Consider cache optimization.",
                        expected_improvement=25.0,
                        implementation_effort="low",
                        estimated_cost="low",
                        created_at=datetime.utcnow(),
                    )
                )

            # High API response time recommendations
            if current_metrics.get("api_response_time", 0) > 1000:
                recommendations.append(
                    OptimizationRecommendation(
                        recommendation_id=f"api_optimization_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        category="code",
                        priority="high",
                        title="Optimize API Response Time",
                        description="High API response time detected. Consider code optimization.",
                        expected_improvement=30.0,
                        implementation_effort="high",
                        estimated_cost="free",
                        created_at=datetime.utcnow(),
                    )
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error creating recommendations: {e}")
            return []

    def _calculate_overall_health(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall system health"""
        try:
            health_score = 100

            # Deduct points for alerts
            alerts = analysis.get("alerts", {})
            for alert in alerts.values():
                if alert["severity"] == "critical":
                    health_score -= 30
                elif alert["severity"] == "high":
                    health_score -= 20
                elif alert["severity"] == "medium":
                    health_score -= 10
                else:
                    health_score -= 5

            # Deduct points for bottlenecks
            bottlenecks = analysis.get("bottlenecks", {})
            for bottleneck in bottlenecks.values():
                health_score -= int(bottleneck["impact_score"] * 20)

            # Deduct points for poor metrics
            current_metrics = analysis.get("current_metrics", {})
            if current_metrics.get("cpu_usage", 0) > 90:
                health_score -= 20
            elif current_metrics.get("cpu_usage", 0) > 80:
                health_score -= 10

            if current_metrics.get("memory_usage", 0) > 90:
                health_score -= 20
            elif current_metrics.get("memory_usage", 0) > 80:
                health_score -= 10

            if current_metrics.get("cache_hit_rate", 100) < 50:
                health_score -= 15
            elif current_metrics.get("cache_hit_rate", 100) < 70:
                health_score -= 5

            # Determine health level
            if health_score >= 90:
                return "excellent"
            elif health_score >= 75:
                return "good"
            elif health_score >= 60:
                return "fair"
            elif health_score >= 40:
                return "poor"
            else:
                return "critical"

        except Exception as e:
            logger.error(f"Error calculating overall health: {e}")
            return "unknown"

    async def _calculate_trend(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Calculate trend for a metric"""
        try:
            history = list(self.metrics_history[metric_name])
            if len(history) < 10:
                return None

            # Get recent values
            recent_values = [m.value for m in history[-10:]]

            # Calculate trend
            if len(recent_values) >= 2:
                trend_direction = (
                    "increasing"
                    if recent_values[-1] > recent_values[0]
                    else "decreasing"
                )
                trend_magnitude = (
                    abs(recent_values[-1] - recent_values[0]) / recent_values[0]
                    if recent_values[0] > 0
                    else 0
                )

                return {
                    "direction": trend_direction,
                    "magnitude": trend_magnitude,
                    "recent_values": recent_values[-5:],
                }

            return None

        except Exception as e:
            logger.error(f"Error calculating trend for {metric_name}: {e}")
            return None

    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics"""
        try:
            metrics = {}

            # Get latest values from history
            for metric_name, history in self.metrics_history.items():
                if history:
                    metrics[metric_name] = history[-1].value

            return metrics

        except Exception as e:
            logger.error(f"Error collecting current metrics: {e}")
            return {}

    # Simulated metric collection methods
    async def _measure_api_response_time(self) -> float:
        """Measure API response time (simulated)"""
        return 150.0 + np.random.normal(0, 50)  # Simulated response time

    async def _calculate_error_rate(self) -> float:
        """Calculate error rate (simulated)"""
        return 2.0 + np.random.normal(0, 1)  # Simulated error rate

    async def _get_active_connections(self) -> int:
        """Get active connections (simulated)"""
        return 50 + int(np.random.normal(0, 10))  # Simulated connection count

    async def _get_queue_depth(self) -> int:
        """Get queue depth (simulated)"""
        return 10 + int(np.random.normal(0, 5))  # Simulated queue depth

    async def _get_database_pool_stats(self) -> Dict[str, int]:
        """Get database pool stats (simulated)"""
        return {
            "active": 5 + int(np.random.normal(0, 2)),
            "idle": 10 + int(np.random.normal(0, 3)),
        }

    async def _measure_database_query_time(self) -> float:
        """Measure database query time (simulated)"""
        return 200.0 + np.random.normal(0, 100)  # Simulated query time

    async def _get_database_size(self) -> float:
        """Get database size (simulated)"""
        return 500.0 + np.random.normal(0, 50)  # Simulated database size

    async def _get_cache_stats(self) -> Dict[str, float]:
        """Get cache stats (simulated)"""
        return {
            "hit_rate": 85.0 + np.random.normal(0, 10),
            "memory_usage": 100.0 + np.random.normal(0, 20),
            "keys": 1000 + int(np.random.normal(0, 100)),
        }

    # Caching methods
    async def _cache_metric(self, metric: PerformanceMetric):
        """Cache metric in Redis"""
        try:
            cache_key = f"metric:{metric.metric_name}:{metric.timestamp.strftime('%Y%m%d_%H%M%S')}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "value": metric.value,
                        "unit": metric.unit,
                        "timestamp": metric.timestamp.isoformat(),
                        "tags": metric.tags,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching metric: {e}")

    async def _cache_alert(self, alert: PerformanceAlert):
        """Cache alert in Redis"""
        try:
            cache_key = f"alert:{alert.alert_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching alert: {e}")

    # Cleanup methods
    async def _cleanup_old_bottlenecks(self):
        """Clean up old bottlenecks"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            old_bottlenecks = [
                bid
                for bid, bottleneck in self.bottlenecks.items()
                if bottleneck.detected_at < cutoff_time
            ]
            for bid in old_bottlenecks:
                del self.bottlenecks[bid]
        except Exception as e:
            logger.error(f"Error cleaning up old bottlenecks: {e}")

    async def _cleanup_old_recommendations(self):
        """Clean up old recommendations"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            old_recommendations = [
                rid
                for rid, recommendation in self.recommendations.items()
                if recommendation.created_at < cutoff_time
            ]
            for rid in old_recommendations:
                del self.recommendations[rid]
        except Exception as e:
            logger.error(f"Error cleaning up old recommendations: {e}")


# Factory function
async def get_performance_monitoring_service(
    redis_client: redis.Redis, db_session: Session
) -> PerformanceMonitoringService:
    """Get performance monitoring service instance"""
    service = PerformanceMonitoringService(redis_client, db_session)
    await service.initialize()
    return service
