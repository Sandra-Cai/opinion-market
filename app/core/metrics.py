"""
Advanced metrics collection and monitoring
"""

import time
import asyncio
from typing import Dict, Any, List
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Comprehensive metrics collection system"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric"""
        async with self.lock:
            self.counters[name] += value

    async def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric value"""
        async with self.lock:
            self.gauges[name] = value

    async def record_histogram(self, name: str, value: float) -> None:
        """Record a histogram value"""
        async with self.lock:
            self.histograms[name].append(value)

    async def record_timer(self, name: str, duration: float) -> None:
        """Record a timer duration"""
        async with self.lock:
            self.timers[name].append(duration)
            # Keep only recent measurements
            if len(self.timers[name]) > self.max_history:
                self.timers[name] = self.timers[name][-self.max_history :]

    async def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        async with self.lock:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {},
                "timers": {},
            }

            # Calculate histogram statistics
            for name, values in self.histograms.items():
                if values:
                    values_list = list(values)
                    metrics["histograms"][name] = {
                        "count": len(values_list),
                        "min": min(values_list),
                        "max": max(values_list),
                        "avg": sum(values_list) / len(values_list),
                        "p50": self._percentile(values_list, 50),
                        "p95": self._percentile(values_list, 95),
                        "p99": self._percentile(values_list, 99),
                    }

            # Calculate timer statistics
            for name, durations in self.timers.items():
                if durations:
                    metrics["timers"][name] = {
                        "count": len(durations),
                        "min": min(durations),
                        "max": max(durations),
                        "avg": sum(durations) / len(durations),
                        "p50": self._percentile(durations, 50),
                        "p95": self._percentile(durations, 95),
                        "p99": self._percentile(durations, 99),
                    }

            return metrics

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

    async def reset(self) -> None:
        """Reset all metrics"""
        async with self.lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()


class TimerContext:
    """Context manager for timing operations"""

    def __init__(self, metrics_collector: MetricsCollector, timer_name: str):
        self.metrics_collector = metrics_collector
        self.timer_name = timer_name
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            await self.metrics_collector.record_timer(self.timer_name, duration)


class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector

    async def __call__(self, request, call_next):
        # Record request start
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time

        await self.metrics.increment_counter("http_requests_total")
        await self.metrics.increment_counter(f"http_requests_{response.status_code}")
        await self.metrics.record_timer("http_request_duration", duration)
        await self.metrics.record_histogram(
            "response_size", len(response.body) if hasattr(response, "body") else 0
        )

        return response


# Global metrics collector
metrics_collector = MetricsCollector()
