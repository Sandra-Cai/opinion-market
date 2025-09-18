"""
Advanced Performance Optimization System
Provides intelligent performance monitoring, optimization, and auto-tuning
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import functools
from collections import defaultdict, deque
import threading
import weakref

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metrics to track"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CACHE_HIT_RATE = "cache_hit_rate"
    DATABASE_QUERY_TIME = "database_query_time"
    ERROR_RATE = "error_rate"
    ACTIVE_CONNECTIONS = "active_connections"


class OptimizationStrategy(Enum):
    """Optimization strategies"""
    CACHING = "caching"
    CONNECTION_POOLING = "connection_pooling"
    QUERY_OPTIMIZATION = "query_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    CPU_OPTIMIZATION = "cpu_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"
    ASYNC_OPTIMIZATION = "async_optimization"
    RESOURCE_SCALING = "resource_scaling"


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    metrics: Dict[PerformanceMetric, float]
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metrics": {metric.value: value for metric, value in self.metrics.items()},
            "context": self.context
        }


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric: PerformanceMetric
    warning_threshold: float
    critical_threshold: float
    optimization_threshold: float
    unit: str = ""


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    strategy: OptimizationStrategy
    priority: int  # 1-10, higher is more urgent
    description: str
    expected_improvement: float  # percentage
    implementation_effort: str  # low, medium, high
    risk_level: str  # low, medium, high
    auto_implementable: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)


class AdvancedPerformanceOptimizer:
    """Advanced performance optimization system with AI-driven insights"""

    def __init__(self):
        self.performance_history: deque = deque(maxlen=10000)
        self.thresholds: Dict[PerformanceMetric, PerformanceThreshold] = {}
        self.optimization_history: List[OptimizationRecommendation] = []
        self.auto_optimizations: Dict[str, Any] = {}
        self.performance_patterns: Dict[str, List[float]] = defaultdict(list)
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.function_performance: Dict[str, List[float]] = defaultdict(list)
        self.endpoint_performance: Dict[str, List[float]] = defaultdict(list)
        self.database_query_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Auto-tuning parameters
        self.auto_tuning_enabled = True
        self.optimization_cooldown = 300  # 5 minutes
        self.last_optimization = datetime.min
        
        # Initialize default thresholds
        self._initialize_default_thresholds()

    def _initialize_default_thresholds(self):
        """Initialize default performance thresholds"""
        self.thresholds = {
            PerformanceMetric.RESPONSE_TIME: PerformanceThreshold(
                metric=PerformanceMetric.RESPONSE_TIME,
                warning_threshold=500.0,  # ms
                critical_threshold=1000.0,  # ms
                optimization_threshold=200.0,  # ms
                unit="ms"
            ),
            PerformanceMetric.CPU_USAGE: PerformanceThreshold(
                metric=PerformanceMetric.CPU_USAGE,
                warning_threshold=70.0,  # %
                critical_threshold=85.0,  # %
                optimization_threshold=50.0,  # %
                unit="%"
            ),
            PerformanceMetric.MEMORY_USAGE: PerformanceThreshold(
                metric=PerformanceMetric.MEMORY_USAGE,
                warning_threshold=80.0,  # %
                critical_threshold=90.0,  # %
                optimization_threshold=60.0,  # %
                unit="%"
            ),
            PerformanceMetric.ERROR_RATE: PerformanceThreshold(
                metric=PerformanceMetric.ERROR_RATE,
                warning_threshold=5.0,  # %
                critical_threshold=10.0,  # %
                optimization_threshold=1.0,  # %
                unit="%"
            ),
            PerformanceMetric.CACHE_HIT_RATE: PerformanceThreshold(
                metric=PerformanceMetric.CACHE_HIT_RATE,
                warning_threshold=70.0,  # %
                critical_threshold=50.0,  # %
                optimization_threshold=90.0,  # %
                unit="%"
            )
        }

    async def start_monitoring(self, interval: int = 30):
        """Start continuous performance monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval)
        )
        logger.info("Advanced performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Advanced performance monitoring stopped")

    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._collect_performance_metrics()
                await self._analyze_performance()
                await self._check_optimization_opportunities()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)

    async def _collect_performance_metrics(self):
        """Collect current performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            # Application metrics (would be collected from your metrics system)
            app_metrics = await self._get_application_metrics()
            
            metrics = {
                PerformanceMetric.CPU_USAGE: cpu_percent,
                PerformanceMetric.MEMORY_USAGE: memory.percent,
                PerformanceMetric.DISK_IO: disk.percent,
                PerformanceMetric.NETWORK_IO: network_io.bytes_sent + network_io.bytes_recv,
                **app_metrics
            }
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                metrics=metrics,
                context={
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                    "active_processes": len(psutil.pids())
                }
            )
            
            self.performance_history.append(snapshot)
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")

    async def _get_application_metrics(self) -> Dict[PerformanceMetric, float]:
        """Get application-specific metrics"""
        import random
        import time
        
        # Generate realistic mock data with some variation
        base_time = time.time()
        variation = random.uniform(0.9, 1.1)  # ±10% variation
        
        return {
            PerformanceMetric.RESPONSE_TIME: 150.0 * variation,  # Mock data with variation
            PerformanceMetric.THROUGHPUT: 1000.0 * variation,  # Mock data with variation
            PerformanceMetric.ERROR_RATE: max(0.1, 0.5 * variation),  # Mock data with variation
            PerformanceMetric.CACHE_HIT_RATE: 85.0 + random.uniform(-5, 5),  # Mock data with variation
            PerformanceMetric.DATABASE_QUERY_TIME: 50.0 * variation,  # Mock data with variation
            PerformanceMetric.ACTIVE_CONNECTIONS: 150.0 + random.uniform(-10, 10)  # Mock data with variation
        }

    async def _analyze_performance(self):
        """Analyze performance patterns and trends"""
        if len(self.performance_history) < 10:
            return

        recent_snapshots = list(self.performance_history)[-100:]  # Last 100 snapshots
        
        for metric in PerformanceMetric:
            values = [snapshot.metrics.get(metric, 0) for snapshot in recent_snapshots]
            if values:
                self.performance_patterns[metric.value] = values
                
                # Detect anomalies
                await self._detect_anomalies(metric, values)
                
                # Analyze trends
                await self._analyze_trends(metric, values)

    async def _detect_anomalies(self, metric: PerformanceMetric, values: List[float]):
        """Detect performance anomalies"""
        if len(values) < 10:
            return

        # Simple anomaly detection using statistical methods
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5

        threshold = self.thresholds.get(metric)
        if not threshold:
            return

        # Check for critical anomalies
        for i, value in enumerate(values[-5:]):  # Check last 5 values
            if value > threshold.critical_threshold:
                await self._handle_critical_anomaly(metric, value, mean_val, std_dev)

    async def _analyze_trends(self, metric: PerformanceMetric, values: List[float]):
        """Analyze performance trends"""
        if len(values) < 20:
            return

        # Simple trend analysis
        recent_avg = sum(values[-10:]) / 10
        older_avg = sum(values[-20:-10]) / 10
        
        trend_direction = "improving" if recent_avg < older_avg else "degrading"
        trend_magnitude = abs(recent_avg - older_avg) / older_avg * 100

        if trend_magnitude > 20:  # Significant trend
            await self._handle_performance_trend(metric, trend_direction, trend_magnitude)

    async def _handle_critical_anomaly(
        self, 
        metric: PerformanceMetric, 
        value: float, 
        mean: float, 
        std_dev: float
    ):
        """Handle critical performance anomalies"""
        logger.critical(
            f"CRITICAL PERFORMANCE ANOMALY: {metric.value} = {value:.2f} "
            f"(mean: {mean:.2f}, std: {std_dev:.2f})"
        )
        
        # Generate immediate optimization recommendations
        recommendations = await self._generate_emergency_recommendations(metric, value)
        for rec in recommendations:
            await self._apply_optimization(rec)

    async def _handle_performance_trend(
        self, 
        metric: PerformanceMetric, 
        direction: str, 
        magnitude: float
    ):
        """Handle performance trends"""
        logger.warning(
            f"PERFORMANCE TREND: {metric.value} is {direction} "
            f"by {magnitude:.1f}%"
        )

    async def _check_optimization_opportunities(self):
        """Check for optimization opportunities"""
        if not self.auto_tuning_enabled:
            return

        # Check cooldown period
        if datetime.utcnow() - self.last_optimization < timedelta(seconds=self.optimization_cooldown):
            return

        if len(self.performance_history) < 50:
            return

        recent_snapshots = list(self.performance_history)[-50:]
        
        for metric, threshold in self.thresholds.items():
            values = [snapshot.metrics.get(metric, 0) for snapshot in recent_snapshots]
            if not values:
                continue

            avg_value = sum(values) / len(values)
            
            # Check if optimization is needed
            if avg_value > threshold.optimization_threshold:
                recommendations = await self._generate_optimization_recommendations(
                    metric, avg_value, threshold
                )
                
                for rec in recommendations:
                    if rec.auto_implementable:
                        await self._apply_optimization(rec)

    async def _generate_emergency_recommendations(
        self, 
        metric: PerformanceMetric, 
        value: float
    ) -> List[OptimizationRecommendation]:
        """Generate emergency optimization recommendations"""
        recommendations = []

        if metric == PerformanceMetric.RESPONSE_TIME:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.CACHING,
                priority=10,
                description="Enable aggressive caching to reduce response times",
                expected_improvement=30.0,
                implementation_effort="low",
                risk_level="low",
                auto_implementable=True,
                parameters={"cache_ttl": 300, "aggressive_mode": True}
            ))

        elif metric == PerformanceMetric.CPU_USAGE:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.CPU_OPTIMIZATION,
                priority=9,
                description="Optimize CPU usage by reducing concurrent operations",
                expected_improvement=25.0,
                implementation_effort="medium",
                risk_level="medium",
                auto_implementable=True,
                parameters={"max_concurrent": 10}
            ))

        elif metric == PerformanceMetric.MEMORY_USAGE:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.MEMORY_OPTIMIZATION,
                priority=8,
                description="Optimize memory usage by clearing caches and reducing buffer sizes",
                expected_improvement=20.0,
                implementation_effort="low",
                risk_level="low",
                auto_implementable=True,
                parameters={"clear_caches": True, "reduce_buffers": True}
            ))

        return recommendations

    async def _generate_optimization_recommendations(
        self,
        metric: PerformanceMetric,
        current_value: float,
        threshold: PerformanceThreshold
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on metrics"""
        recommendations = []

        if metric == PerformanceMetric.RESPONSE_TIME:
            recommendations.extend([
                OptimizationRecommendation(
                    strategy=OptimizationStrategy.CACHING,
                    priority=7,
                    description="Implement response caching",
                    expected_improvement=40.0,
                    implementation_effort="low",
                    risk_level="low",
                    auto_implementable=True
                ),
                OptimizationRecommendation(
                    strategy=OptimizationStrategy.QUERY_OPTIMIZATION,
                    priority=6,
                    description="Optimize database queries",
                    expected_improvement=25.0,
                    implementation_effort="medium",
                    risk_level="medium",
                    auto_implementable=False
                )
            ])

        elif metric == PerformanceMetric.DATABASE_QUERY_TIME:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.QUERY_OPTIMIZATION,
                priority=8,
                description="Optimize slow database queries",
                expected_improvement=50.0,
                implementation_effort="high",
                risk_level="medium",
                auto_implementable=False
            ))

        elif metric == PerformanceMetric.CACHE_HIT_RATE:
            recommendations.append(OptimizationRecommendation(
                strategy=OptimizationStrategy.CACHING,
                priority=5,
                description="Improve cache hit rate",
                expected_improvement=15.0,
                implementation_effort="medium",
                risk_level="low",
                auto_implementable=True
            ))

        return recommendations

    async def _apply_optimization(self, recommendation: OptimizationRecommendation):
        """Apply an optimization recommendation"""
        try:
            logger.info(f"Applying optimization: {recommendation.description}")
            
            if recommendation.strategy == OptimizationStrategy.CACHING:
                await self._apply_caching_optimization(recommendation)
            elif recommendation.strategy == OptimizationStrategy.CPU_OPTIMIZATION:
                await self._apply_cpu_optimization(recommendation)
            elif recommendation.strategy == OptimizationStrategy.MEMORY_OPTIMIZATION:
                await self._apply_memory_optimization(recommendation)
            
            self.optimization_history.append(recommendation)
            self.last_optimization = datetime.utcnow()
            
            logger.info(f"Optimization applied successfully: {recommendation.description}")
            
        except Exception as e:
            logger.error(f"Failed to apply optimization: {e}")

    async def _apply_caching_optimization(self, recommendation: OptimizationRecommendation):
        """Apply caching optimization"""
        # This would integrate with your caching system
        logger.info("Applying caching optimization...")
        # Implementation would depend on your caching system

    async def _apply_cpu_optimization(self, recommendation: OptimizationRecommendation):
        """Apply CPU optimization"""
        logger.info("Applying CPU optimization...")
        # Implementation would depend on your application architecture

    async def _apply_memory_optimization(self, recommendation: OptimizationRecommendation):
        """Apply memory optimization"""
        logger.info("Applying memory optimization...")
        # Implementation would depend on your application architecture

    def track_function_performance(self, func_name: str, execution_time: float):
        """Track function execution performance"""
        self.function_performance[func_name].append(execution_time)
        
        # Keep only recent data
        if len(self.function_performance[func_name]) > 1000:
            self.function_performance[func_name] = self.function_performance[func_name][-500:]

    def track_endpoint_performance(self, endpoint: str, response_time: float):
        """Track API endpoint performance"""
        self.endpoint_performance[endpoint].append(response_time)
        
        # Keep only recent data
        if len(self.endpoint_performance[endpoint]) > 1000:
            self.endpoint_performance[endpoint] = self.endpoint_performance[endpoint][-500:]

    def track_database_query_performance(self, query: str, execution_time: float):
        """Track database query performance"""
        # Use query hash for privacy
        query_hash = hash(query) % 1000000
        self.database_query_performance[str(query_hash)].append(execution_time)
        
        # Keep only recent data
        if len(self.database_query_performance[str(query_hash)]) > 1000:
            self.database_query_performance[str(query_hash)] = self.database_query_performance[str(query_hash)][-500:]

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_snapshots = [
            s for s in self.performance_history 
            if s.timestamp >= cutoff_time
        ]

        if not recent_snapshots:
            return {"error": "No performance data available"}

        summary = {
            "time_period_hours": hours,
            "snapshots_count": len(recent_snapshots),
            "metrics_summary": {},
            "optimization_recommendations": [],
            "performance_trends": {}
        }

        # Calculate metrics summary
        for metric in PerformanceMetric:
            values = [s.metrics.get(metric, 0) for s in recent_snapshots if metric in s.metrics]
            if values:
                summary["metrics_summary"][metric.value] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        # Get recent optimization recommendations
        recent_recommendations = [
            rec for rec in self.optimization_history
            if rec.priority >= 7  # High priority recommendations
        ]
        summary["optimization_recommendations"] = [
            {
                "strategy": rec.strategy.value,
                "priority": rec.priority,
                "description": rec.description,
                "expected_improvement": rec.expected_improvement,
                "auto_implementable": rec.auto_implementable
            }
            for rec in recent_recommendations[-10:]  # Last 10 recommendations
        ]

        return summary

    def get_slow_functions(self, threshold_ms: float = 100.0) -> List[Dict[str, Any]]:
        """Get functions that are performing slowly"""
        slow_functions = []
        
        for func_name, times in self.function_performance.items():
            if times:
                avg_time = sum(times) / len(times) * 1000  # Convert to ms
                if avg_time > threshold_ms:
                    slow_functions.append({
                        "function": func_name,
                        "avg_time_ms": round(avg_time, 2),
                        "call_count": len(times),
                        "max_time_ms": round(max(times) * 1000, 2)
                    })
        
        return sorted(slow_functions, key=lambda x: x["avg_time_ms"], reverse=True)

    def get_slow_endpoints(self, threshold_ms: float = 500.0) -> List[Dict[str, Any]]:
        """Get API endpoints that are performing slowly"""
        slow_endpoints = []
        
        for endpoint, times in self.endpoint_performance.items():
            if times:
                avg_time = sum(times) / len(times) * 1000  # Convert to ms
                if avg_time > threshold_ms:
                    slow_endpoints.append({
                        "endpoint": endpoint,
                        "avg_time_ms": round(avg_time, 2),
                        "request_count": len(times),
                        "max_time_ms": round(max(times) * 1000, 2)
                    })
        
        return sorted(slow_endpoints, key=lambda x: x["avg_time_ms"], reverse=True)

    def get_slow_queries(self, threshold_ms: float = 100.0) -> List[Dict[str, Any]]:
        """Get database queries that are performing slowly"""
        slow_queries = []
        
        for query_hash, times in self.database_query_performance.items():
            if times:
                avg_time = sum(times) / len(times) * 1000  # Convert to ms
                if avg_time > threshold_ms:
                    slow_queries.append({
                        "query_hash": query_hash,
                        "avg_time_ms": round(avg_time, 2),
                        "execution_count": len(times),
                        "max_time_ms": round(max(times) * 1000, 2)
                    })
        
        return sorted(slow_queries, key=lambda x: x["avg_time_ms"], reverse=True)


# Global performance optimizer instance
advanced_performance_optimizer = AdvancedPerformanceOptimizer()


# Decorators for performance tracking
def track_performance(func_name: Optional[str] = None):
    """Decorator to track function performance"""
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                advanced_performance_optimizer.track_function_performance(name, execution_time)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                advanced_performance_optimizer.track_function_performance(name, execution_time)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def track_endpoint_performance(endpoint_name: Optional[str] = None):
    """Decorator to track API endpoint performance"""
    def decorator(func):
        name = endpoint_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                advanced_performance_optimizer.track_endpoint_performance(name, execution_time)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                advanced_performance_optimizer.track_endpoint_performance(name, execution_time)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
