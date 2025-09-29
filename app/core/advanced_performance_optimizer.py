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


# Global instance
advanced_performance_optimizer = AdvancedPerformanceOptimizer()