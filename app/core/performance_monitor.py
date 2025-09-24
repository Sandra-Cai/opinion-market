"""
Enhanced Performance Monitoring System
Provides comprehensive performance tracking and optimization recommendations
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    metric_name: str
    threshold: float
    current_value: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    resolved: bool = False


@dataclass
class PerformanceRecommendation:
    """Performance optimization recommendation"""
    category: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    title: str
    description: str
    impact: str
    effort: str  # 'low', 'medium', 'high'
    implementation: str
    created_at: datetime


class PerformanceMonitor:
    """Enhanced performance monitoring system"""
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[PerformanceAlert] = []
        self.recommendations: List[PerformanceRecommendation] = []
        self.thresholds: Dict[str, Dict[str, float]] = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "api_response_time": {"warning": 1000.0, "critical": 5000.0},
            "database_query_time": {"warning": 500.0, "critical": 2000.0},
            "cache_hit_rate": {"warning": 70.0, "critical": 50.0},
            "error_rate": {"warning": 5.0, "critical": 10.0},
        }
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.lock = threading.Lock()
        
    async def start_monitoring(self, interval: int = 30):
        """Start performance monitoring"""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("Performance monitoring started")
        
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
        
    async def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await self._check_thresholds()
                await self._generate_recommendations()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
                
    async def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric("cpu_usage", cpu_percent, "percent")
            
            # Memory usage
            memory = psutil.virtual_memory()
            await self._record_metric("memory_usage", memory.percent, "percent")
            await self._record_metric("memory_available", memory.available / (1024**3), "GB")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self._record_metric("disk_usage", disk_percent, "percent")
            await self._record_metric("disk_free", disk.free / (1024**3), "GB")
            
            # Network I/O
            net_io = psutil.net_io_counters()
            await self._record_metric("network_bytes_sent", net_io.bytes_sent / (1024**2), "MB")
            await self._record_metric("network_bytes_recv", net_io.bytes_recv / (1024**2), "MB")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
    async def _collect_application_metrics(self):
        """Collect application-level performance metrics"""
        try:
            # Process-specific metrics
            process = psutil.Process()
            await self._record_metric("process_cpu", process.cpu_percent(), "percent")
            await self._record_metric("process_memory", process.memory_info().rss / (1024**2), "MB")
            await self._record_metric("process_threads", process.num_threads(), "count")
            await self._record_metric("process_fds", process.num_fds() if hasattr(process, 'num_fds') else 0, "count")
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            
    async def _record_metric(self, name: str, value: float, unit: str, tags: Dict[str, str] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        with self.lock:
            self.metrics_history[name].append(metric)
            
    async def _check_thresholds(self):
        """Check metrics against thresholds and generate alerts"""
        current_time = datetime.now()
        
        for metric_name, thresholds in self.thresholds.items():
            if metric_name not in self.metrics_history:
                continue
                
            recent_metrics = list(self.metrics_history[metric_name])[-5:]  # Last 5 measurements
            if not recent_metrics:
                continue
                
            avg_value = sum(m.value for m in recent_metrics) / len(recent_metrics)
            
            # Check warning threshold
            if avg_value >= thresholds.get("warning", float('inf')):
                severity = "high" if avg_value >= thresholds.get("critical", float('inf')) else "medium"
                
                # Check if alert already exists and is recent
                existing_alert = next(
                    (a for a in self.alerts 
                     if a.metric_name == metric_name and 
                     not a.resolved and 
                     (current_time - a.timestamp).seconds < 300),  # 5 minutes
                    None
                )
                
                if not existing_alert:
                    alert = PerformanceAlert(
                        metric_name=metric_name,
                        threshold=thresholds["warning"],
                        current_value=avg_value,
                        severity=severity,
                        message=f"{metric_name} is {avg_value:.2f}, exceeding threshold of {thresholds['warning']}",
                        timestamp=current_time
                    )
                    self.alerts.append(alert)
                    logger.warning(f"Performance alert: {alert.message}")
                    
    async def _generate_recommendations(self):
        """Generate performance optimization recommendations"""
        current_time = datetime.now()
        
        # Check for high CPU usage
        if "cpu_usage" in self.metrics_history:
            recent_cpu = list(self.metrics_history["cpu_usage"])[-10:]
            if recent_cpu:
                avg_cpu = sum(m.value for m in recent_cpu) / len(recent_cpu)
                if avg_cpu > 80:
                    await self._add_recommendation(
                        "system",
                        "high",
                        "High CPU Usage Detected",
                        f"Average CPU usage is {avg_cpu:.1f}%. Consider optimizing CPU-intensive operations.",
                        "Reduced system responsiveness and potential timeouts",
                        "medium",
                        "Profile application code, optimize database queries, implement caching"
                    )
                    
        # Check for high memory usage
        if "memory_usage" in self.metrics_history:
            recent_memory = list(self.metrics_history["memory_usage"])[-10:]
            if recent_memory:
                avg_memory = sum(m.value for m in recent_memory) / len(recent_memory)
                if avg_memory > 85:
                    await self._add_recommendation(
                        "system",
                        "high",
                        "High Memory Usage Detected",
                        f"Average memory usage is {avg_memory:.1f}%. Consider memory optimization.",
                        "Potential out-of-memory errors and system instability",
                        "high",
                        "Review memory leaks, optimize data structures, implement memory pooling"
                    )
                    
        # Check for slow API responses
        if "api_response_time" in self.metrics_history:
            recent_response = list(self.metrics_history["api_response_time"])[-10:]
            if recent_response:
                avg_response = sum(m.value for m in recent_response) / len(recent_response)
                if avg_response > 1000:
                    await self._add_recommendation(
                        "api",
                        "medium",
                        "Slow API Response Times",
                        f"Average API response time is {avg_response:.0f}ms. Consider API optimization.",
                        "Poor user experience and potential timeouts",
                        "medium",
                        "Optimize database queries, implement caching, review API logic"
                    )
                    
    async def _add_recommendation(self, category: str, priority: str, title: str, 
                                description: str, impact: str, effort: str, implementation: str):
        """Add a performance recommendation"""
        recommendation = PerformanceRecommendation(
            category=category,
            priority=priority,
            title=title,
            description=description,
            impact=impact,
            effort=effort,
            implementation=implementation,
            created_at=datetime.now()
        )
        
        # Check if similar recommendation already exists
        existing = next(
            (r for r in self.recommendations 
             if r.title == title and (datetime.now() - r.created_at).days < 1),
            None
        )
        
        if not existing:
            self.recommendations.append(recommendation)
            logger.info(f"Performance recommendation added: {title}")
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        summary = {}
        
        for metric_name, metrics in self.metrics_history.items():
            if not metrics:
                continue
                
            recent_metrics = list(metrics)[-10:]  # Last 10 measurements
            values = [m.value for m in recent_metrics]
            
            summary[metric_name] = {
                "current": values[-1] if values else 0,
                "average": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "unit": recent_metrics[-1].unit if recent_metrics else "unknown",
                "timestamp": recent_metrics[-1].timestamp.isoformat() if recent_metrics else None
            }
            
        return summary
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active performance alerts"""
        active_alerts = [a for a in self.alerts if not a.resolved]
        return [
            {
                "metric_name": alert.metric_name,
                "threshold": alert.threshold,
                "current_value": alert.current_value,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in active_alerts
        ]
        
    def get_recommendations(self, priority: str = None) -> List[Dict[str, Any]]:
        """Get performance recommendations"""
        recommendations = self.recommendations
        if priority:
            recommendations = [r for r in recommendations if r.priority == priority]
            
        return [
            {
                "category": rec.category,
                "priority": rec.priority,
                "title": rec.title,
                "description": rec.description,
                "impact": rec.impact,
                "effort": rec.effort,
                "implementation": rec.implementation,
                "created_at": rec.created_at.isoformat()
            }
            for rec in recommendations
        ]
        
    @asynccontextmanager
    async def measure_execution_time(self, operation_name: str):
        """Context manager to measure execution time of operations"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            await self._record_metric(f"execution_time_{operation_name}", execution_time, "ms")
            
    def record_custom_metric(self, name: str, value: float, unit: str = "count", tags: Dict[str, str] = None):
        """Record a custom performance metric"""
        asyncio.create_task(self._record_metric(name, value, unit, tags))


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
