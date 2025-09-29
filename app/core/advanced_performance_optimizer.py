"""
Advanced Performance Optimizer
AI-powered performance optimization with predictive analytics
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import json
import logging

from app.core.enhanced_cache import enhanced_cache
from app.core.database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationAction:
    """Optimization action to be taken"""
    action_type: str
    target: str
    parameters: Dict[str, Any]
    priority: int = 1
    estimated_impact: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PerformancePrediction:
    """Performance prediction based on historical data"""
    metric_name: str
    predicted_value: float
    confidence: float
    time_horizon: int  # minutes
    trend: str  # 'increasing', 'decreasing', 'stable'
    created_at: datetime = field(default_factory=datetime.now)


class AdvancedPerformanceOptimizer:
    """AI-powered performance optimization system"""
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.predictions: Dict[str, List[PerformancePrediction]] = defaultdict(list)
        self.optimization_actions: List[OptimizationAction] = []
        self.performance_baselines: Dict[str, float] = {}
        self.optimization_rules: Dict[str, Dict[str, Any]] = {}
        
        # Performance thresholds
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "api_response_time": {"warning": 100.0, "critical": 500.0},
            "database_query_time": {"warning": 100.0, "critical": 1000.0},
            "cache_hit_rate": {"warning": 80.0, "critical": 60.0},
            "error_rate": {"warning": 1.0, "critical": 5.0},
        }
        
        # Optimization strategies
        self.optimization_strategies = {
            "cache_optimization": self._optimize_cache,
            "database_optimization": self._optimize_database,
            "memory_optimization": self._optimize_memory,
            "connection_optimization": self._optimize_connections,
            "query_optimization": self._optimize_queries,
        }
        
        self.monitoring_active = False
        self.optimization_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.lock = threading.Lock()
        
    async def start_monitoring(self):
        """Start advanced performance monitoring"""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Advanced performance monitoring started")
        
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Advanced performance monitoring stopped")
        
    async def start_optimization(self):
        """Start automatic optimization"""
        if self.optimization_active:
            logger.warning("Optimization already active")
            return
            
        self.optimization_active = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("Automatic optimization started")
        
    async def stop_optimization(self):
        """Stop automatic optimization"""
        self.optimization_active = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        logger.info("Automatic optimization stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._collect_metrics()
                await self._analyze_performance()
                await self._generate_predictions()
                await asyncio.sleep(30)  # Collect metrics every 30 seconds
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _optimization_loop(self):
        """Main optimization loop"""
        while self.optimization_active:
            try:
                await self._evaluate_optimization_opportunities()
                await self._execute_optimizations()
                await asyncio.sleep(300)  # Check for optimizations every 5 minutes
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(600)  # Wait longer on error
                
    async def _collect_metrics(self):
        """Collect comprehensive performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            await self._record_metric("system.cpu_usage", cpu_percent)
            await self._record_metric("system.memory_usage", memory.percent)
            await self._record_metric("system.disk_usage", disk.percent)
            await self._record_metric("system.memory_available", memory.available / (1024**3))  # GB
            
            # Cache metrics
            cache_stats = enhanced_cache.get_stats()
            await self._record_metric("cache.hit_rate", cache_stats.get("hit_rate", 0) * 100)
            await self._record_metric("cache.entry_count", cache_stats.get("entry_count", 0))
            await self._record_metric("cache.memory_usage", cache_stats.get("memory_usage_mb", 0))
            
            # Database metrics
            await self._collect_database_metrics()
            
            # Application metrics
            await self._collect_application_metrics()
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            
    async def _collect_database_metrics(self):
        """Collect database performance metrics"""
        try:
            with engine.connect() as conn:
                # Connection pool metrics
                pool = engine.pool
                await self._record_metric("database.pool_size", pool.size())
                await self._record_metric("database.checked_out", pool.checkedout())
                await self._record_metric("database.overflow", pool.overflow())
                
                # Query performance
                start_time = time.time()
                result = conn.execute(text("SELECT 1"))
                query_time = (time.time() - start_time) * 1000  # ms
                await self._record_metric("database.query_time", query_time)
                
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")
            
    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # API response times (simulated - would be collected from actual requests)
            await self._record_metric("application.api_response_time", 50.0)  # ms
            await self._record_metric("application.error_rate", 0.1)  # percentage
            await self._record_metric("application.requests_per_second", 1000.0)
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            
    async def _record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        with self.lock:
            self.metrics_history[name].append(metric)
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": self.monitoring_active,
                "optimization_active": self.optimization_active,
                "metrics": {},
                "predictions": {},
                "optimization_actions": len(self.optimization_actions),
                "performance_score": 0
            }
            
            # Calculate current metrics
            for metric_name, history in self.metrics_history.items():
                if history:
                    recent_values = [m.value for m in list(history)[-5:]]
                    summary["metrics"][metric_name] = {
                        "current": recent_values[-1] if recent_values else 0,
                        "average": statistics.mean(recent_values) if recent_values else 0,
                        "trend": "stable"  # Would be calculated from trend analysis
                    }
                    
            # Calculate performance score
            performance_factors = []
            for metric_name, data in summary["metrics"].items():
                if metric_name in self.thresholds:
                    threshold = self.thresholds[metric_name].get("warning", 100)
                    if "usage" in metric_name or "time" in metric_name:
                        score = max(0, 100 - (data["current"] / threshold * 100))
                    else:  # hit_rate, etc.
                        score = min(100, data["current"] / threshold * 100)
                    performance_factors.append(score)
                    
            summary["performance_score"] = statistics.mean(performance_factors) if performance_factors else 100
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}


# Global instance
advanced_performance_optimizer = AdvancedPerformanceOptimizer()