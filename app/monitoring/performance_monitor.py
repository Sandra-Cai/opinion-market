"""
Performance monitoring system for Opinion Market
Provides detailed performance tracking, profiling, and optimization insights
"""

import asyncio
import time
import functools
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import gc
import tracemalloc
from contextlib import asynccontextmanager, contextmanager

from app.core.logging import log_system_metric
from app.core.config import settings


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionProfile:
    """Function profiling data"""
    function_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_called: Optional[datetime] = None
    errors: int = 0


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.slow_queries: deque = deque(maxlen=100)
        self.memory_snapshots: deque = deque(maxlen=50)
        self.running = False
        self._lock = threading.Lock()
        self._request_counter = 0
        
        # Performance thresholds
        self.thresholds = {
            "slow_request": 1.0,  # seconds
            "slow_query": 0.5,    # seconds
            "high_memory": 0.8,   # 80% of available memory
            "high_cpu": 0.8,      # 80% CPU usage
        }
        
        # Start memory tracking
        if settings.DEBUG:
            tracemalloc.start()
    
    async def start(self):
        """Start performance monitoring"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting performance monitoring")
        
        # Start monitoring tasks
        asyncio.create_task(self._memory_monitoring_loop())
        asyncio.create_task(self._gc_monitoring_loop())
        asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """Stop performance monitoring"""
        self.running = False
        self.logger.info("Stopped performance monitoring")
    
    async def _memory_monitoring_loop(self):
        """Monitor memory usage"""
        while self.running:
            try:
                await self._collect_memory_metrics()
                await asyncio.sleep(30)  # Every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _gc_monitoring_loop(self):
        """Monitor garbage collection"""
        while self.running:
            try:
                await self._collect_gc_metrics()
                await asyncio.sleep(60)  # Every minute
            except Exception as e:
                self.logger.error(f"Error in GC monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Cleanup old data"""
        while self.running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Every hour
            except Exception as e:
                self.logger.error(f"Error in cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_memory_metrics(self):
        """Collect memory-related metrics"""
        now = datetime.utcnow()
        
        try:
            # System memory
            memory = psutil.virtual_memory()
            self._add_metric(PerformanceMetric(
                "memory.system.usage", memory.percent, now, unit="percent"
            ))
            self._add_metric(PerformanceMetric(
                "memory.system.available", memory.available, now, unit="bytes"
            ))
            
            # Process memory
            process = psutil.Process()
            memory_info = process.memory_info()
            self._add_metric(PerformanceMetric(
                "memory.process.rss", memory_info.rss, now, unit="bytes"
            ))
            self._add_metric(PerformanceMetric(
                "memory.process.vms", memory_info.vms, now, unit="bytes"
            ))
            self._add_metric(PerformanceMetric(
                "memory.process.percent", process.memory_percent(), now, unit="percent"
            ))
            
            # Python memory (if tracemalloc is enabled)
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                self._add_metric(PerformanceMetric(
                    "memory.python.current", current, now, unit="bytes"
                ))
                self._add_metric(PerformanceMetric(
                    "memory.python.peak", peak, now, unit="bytes"
                ))
            
            # Memory pressure check
            if memory.percent > self.thresholds["high_memory"] * 100:
                await self._handle_memory_pressure(memory.percent)
            
        except Exception as e:
            self.logger.error(f"Error collecting memory metrics: {e}")
    
    async def _collect_gc_metrics(self):
        """Collect garbage collection metrics"""
        now = datetime.utcnow()
        
        try:
            # GC statistics
            gc_stats = gc.get_stats()
            for i, stats in enumerate(gc_stats):
                self._add_metric(PerformanceMetric(
                    f"gc.generation_{i}.collections", stats["collections"], now
                ))
                self._add_metric(PerformanceMetric(
                    f"gc.generation_{i}.collected", stats["collected"], now
                ))
            
            # GC counts
            gc_counts = gc.get_count()
            for i, count in enumerate(gc_counts):
                self._add_metric(PerformanceMetric(
                    f"gc.counts.generation_{i}", count, now
                ))
            
            # Object counts
            self._add_metric(PerformanceMetric(
                "gc.objects.total", len(gc.get_objects()), now
            ))
            
        except Exception as e:
            self.logger.error(f"Error collecting GC metrics: {e}")
    
    async def _cleanup_old_data(self):
        """Cleanup old performance data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        with self._lock:
            # Clean up old active requests
            old_requests = [
                req_id for req_id, req_data in self.active_requests.items()
                if req_data.get("start_time", datetime.utcnow()) < cutoff_time
            ]
            for req_id in old_requests:
                del self.active_requests[req_id]
            
            # Clean up old memory snapshots
            while self.memory_snapshots and self.memory_snapshots[0]["timestamp"] < cutoff_time:
                self.memory_snapshots.popleft()
    
    async def _handle_memory_pressure(self, memory_percent: float):
        """Handle high memory usage"""
        self.logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
        
        # Force garbage collection
        collected = gc.collect()
        self.logger.info(f"Garbage collection freed {collected} objects")
        
        # Take memory snapshot
        await self._take_memory_snapshot("memory_pressure")
        
        # Log memory pressure event
        log_system_metric("memory_pressure", memory_percent, {
            "threshold": self.thresholds["high_memory"] * 100,
            "gc_collected": collected
        })
    
    async def _take_memory_snapshot(self, reason: str = "scheduled"):
        """Take a memory snapshot for analysis"""
        if not tracemalloc.is_tracing():
            return
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            snapshot_data = {
                "timestamp": datetime.utcnow(),
                "reason": reason,
                "top_allocations": [
                    {
                        "filename": stat.traceback.format()[0] if stat.traceback else "unknown",
                        "size": stat.size,
                        "count": stat.count
                    }
                    for stat in top_stats[:10]
                ],
                "total_size": sum(stat.size for stat in top_stats),
                "total_count": sum(stat.count for stat in top_stats)
            }
            
            with self._lock:
                self.memory_snapshots.append(snapshot_data)
            
        except Exception as e:
            self.logger.error(f"Error taking memory snapshot: {e}")
    
    def _add_metric(self, metric: PerformanceMetric):
        """Add performance metric"""
        with self._lock:
            self.metrics[metric.name].append(metric)
    
    def start_request_tracking(self, request_id: str, method: str, path: str) -> str:
        """Start tracking a request"""
        if not request_id:
            self._request_counter += 1
            request_id = f"req_{self._request_counter}_{int(time.time())}"
        
        with self._lock:
            self.active_requests[request_id] = {
                "method": method,
                "path": path,
                "start_time": datetime.utcnow(),
                "start_memory": psutil.Process().memory_info().rss,
                "queries": [],
                "functions": []
            }
        
        return request_id
    
    def end_request_tracking(self, request_id: str, status_code: int = 200, error: Optional[str] = None):
        """End tracking a request"""
        if request_id not in self.active_requests:
            return
        
        with self._lock:
            request_data = self.active_requests[request_id]
            end_time = datetime.utcnow()
            duration = (end_time - request_data["start_time"]).total_seconds()
            end_memory = psutil.Process().memory_info().rss
            memory_delta = end_memory - request_data["start_memory"]
            
            # Record request metrics
            self._add_metric(PerformanceMetric(
                "request.duration", duration, end_time,
                tags={"method": request_data["method"], "path": request_data["path"], "status": str(status_code)},
                unit="seconds"
            ))
            
            self._add_metric(PerformanceMetric(
                "request.memory_delta", memory_delta, end_time,
                tags={"method": request_data["method"], "path": request_data["path"]},
                unit="bytes"
            ))
            
            # Check for slow requests
            if duration > self.thresholds["slow_request"]:
                self.logger.warning(f"Slow request detected: {request_data['path']} took {duration:.2f}s")
                log_system_metric("slow_request", duration, {
                    "method": request_data["method"],
                    "path": request_data["path"],
                    "status_code": status_code,
                    "queries": len(request_data["queries"]),
                    "memory_delta": memory_delta
                })
            
            # Remove from active requests
            del self.active_requests[request_id]
    
    def track_query(self, request_id: str, query: str, duration: float):
        """Track database query performance"""
        if request_id in self.active_requests:
            with self._lock:
                self.active_requests[request_id]["queries"].append({
                    "query": query,
                    "duration": duration,
                    "timestamp": datetime.utcnow()
                })
        
        # Record query metric
        self._add_metric(PerformanceMetric(
            "query.duration", duration, datetime.utcnow(),
            tags={"query_type": self._get_query_type(query)},
            unit="seconds"
        ))
        
        # Check for slow queries
        if duration > self.thresholds["slow_query"]:
            with self._lock:
                self.slow_queries.append({
                    "query": query,
                    "duration": duration,
                    "timestamp": datetime.utcnow()
                })
            
            self.logger.warning(f"Slow query detected: {duration:.2f}s - {query[:100]}...")
            log_system_metric("slow_query", duration, {"query": query[:200]})
    
    def _get_query_type(self, query: str) -> str:
        """Determine query type from SQL"""
        query_upper = query.upper().strip()
        if query_upper.startswith("SELECT"):
            return "select"
        elif query_upper.startswith("INSERT"):
            return "insert"
        elif query_upper.startswith("UPDATE"):
            return "update"
        elif query_upper.startswith("DELETE"):
            return "delete"
        else:
            return "other"
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function performance"""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                self._update_function_profile(function_name, duration, False)
                return result
            except Exception as e:
                duration = time.time() - start_time
                self._update_function_profile(function_name, duration, True)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                self._update_function_profile(function_name, duration, False)
                return result
            except Exception as e:
                duration = time.time() - start_time
                self._update_function_profile(function_name, duration, True)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    def _update_function_profile(self, function_name: str, duration: float, is_error: bool):
        """Update function profiling data"""
        with self._lock:
            if function_name not in self.function_profiles:
                self.function_profiles[function_name] = FunctionProfile(function_name)
            
            profile = self.function_profiles[function_name]
            profile.call_count += 1
            profile.total_time += duration
            profile.min_time = min(profile.min_time, duration)
            profile.max_time = max(profile.max_time, duration)
            profile.avg_time = profile.total_time / profile.call_count
            profile.last_called = datetime.utcnow()
            
            if is_error:
                profile.errors += 1
    
    @contextmanager
    def time_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss
            memory_delta = end_memory - start_memory
            
            self._add_metric(PerformanceMetric(
                f"operation.{operation_name}.duration", duration, datetime.utcnow(),
                tags=tags or {}, unit="seconds"
            ))
            
            self._add_metric(PerformanceMetric(
                f"operation.{operation_name}.memory_delta", memory_delta, datetime.utcnow(),
                tags=tags or {}, unit="bytes"
            ))
    
    @asynccontextmanager
    async def time_async_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Async context manager for timing operations"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss
            memory_delta = end_memory - start_memory
            
            self._add_metric(PerformanceMetric(
                f"operation.{operation_name}.duration", duration, datetime.utcnow(),
                tags=tags or {}, unit="seconds"
            ))
            
            self._add_metric(PerformanceMetric(
                f"operation.{operation_name}.memory_delta", memory_delta, datetime.utcnow(),
                tags=tags or {}, unit="bytes"
            ))
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "time_range_hours": hours,
            "active_requests": len(self.active_requests),
            "slow_queries_count": len(self.slow_queries),
            "memory_snapshots_count": len(self.memory_snapshots),
            "function_profiles": {},
            "metrics_summary": {},
            "recommendations": []
        }
        
        # Function profiles
        for func_name, profile in self.function_profiles.items():
            if profile.last_called and profile.last_called >= cutoff_time:
                summary["function_profiles"][func_name] = {
                    "call_count": profile.call_count,
                    "total_time": profile.total_time,
                    "avg_time": profile.avg_time,
                    "min_time": profile.min_time,
                    "max_time": profile.max_time,
                    "errors": profile.errors,
                    "error_rate": (profile.errors / profile.call_count * 100) if profile.call_count > 0 else 0
                }
        
        # Metrics summary
        for metric_name, metrics in self.metrics.items():
            recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                summary["metrics_summary"][metric_name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[-1],
                    "unit": recent_metrics[0].unit
                }
        
        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(summary)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check for slow functions
        for func_name, profile in summary["function_profiles"].items():
            if profile["avg_time"] > 0.1:  # 100ms
                recommendations.append(f"Consider optimizing {func_name} (avg: {profile['avg_time']:.3f}s)")
            
            if profile["error_rate"] > 5:  # 5% error rate
                recommendations.append(f"High error rate in {func_name} ({profile['error_rate']:.1f}%)")
        
        # Check for memory issues
        memory_metrics = summary["metrics_summary"].get("memory.process.percent", {})
        if memory_metrics.get("latest", 0) > 80:
            recommendations.append("High memory usage detected - consider memory optimization")
        
        # Check for slow queries
        if summary["slow_queries_count"] > 10:
            recommendations.append("Multiple slow queries detected - consider database optimization")
        
        # Check for high request volume
        request_metrics = summary["metrics_summary"].get("request.duration", {})
        if request_metrics.get("avg", 0) > 0.5:  # 500ms average
            recommendations.append("High average request duration - consider performance optimization")
        
        return recommendations
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest queries"""
        with self._lock:
            return list(self.slow_queries)[-limit:]
    
    def get_memory_snapshots(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent memory snapshots"""
        with self._lock:
            return list(self.memory_snapshots)[-limit:]
    
    def get_top_functions_by_time(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get functions that consume the most time"""
        sorted_functions = sorted(
            self.function_profiles.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )
        
        return [
            {
                "function": name,
                "total_time": profile.total_time,
                "call_count": profile.call_count,
                "avg_time": profile.avg_time,
                "errors": profile.errors
            }
            for name, profile in sorted_functions[:limit]
        ]
    
    def get_top_functions_by_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently called functions"""
        sorted_functions = sorted(
            self.function_profiles.items(),
            key=lambda x: x[1].call_count,
            reverse=True
        )
        
        return [
            {
                "function": name,
                "call_count": profile.call_count,
                "total_time": profile.total_time,
                "avg_time": profile.avg_time,
                "errors": profile.errors
            }
            for name, profile in sorted_functions[:limit]
        ]


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
