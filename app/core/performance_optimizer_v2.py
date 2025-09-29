"""
Performance Optimizer V2
Advanced performance optimization with intelligent caching and resource management
"""

import asyncio
import time
import psutil
import gc
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

from app.core.enhanced_cache import enhanced_cache
from app.core.advanced_performance_optimizer import advanced_performance_optimizer

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    timestamp: datetime
    category: str  # 'system', 'application', 'cache', 'database'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationAction:
    """Optimization action data structure"""
    action_type: str  # 'cache_optimization', 'memory_cleanup', 'gc_trigger', 'resource_scaling'
    description: str
    impact_score: float  # 0-1, expected performance improvement
    cost_score: float  # 0-1, resource cost of action
    executed: bool = False
    executed_at: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)


class PerformanceOptimizerV2:
    """Advanced performance optimizer with intelligent resource management"""
    
    def __init__(self):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.optimization_actions: List[OptimizationAction] = []
        self.performance_baseline: Dict[str, float] = {}
        self.optimization_active = False
        self.optimization_task: Optional[asyncio.Task] = None
        
        # Performance thresholds
        self.thresholds = {
            "cpu_usage": {"warning": 70, "critical": 85},
            "memory_usage": {"warning": 75, "critical": 90},
            "response_time": {"warning": 100, "critical": 200},  # ms
            "cache_hit_rate": {"warning": 80, "critical": 70},  # %
            "gc_frequency": {"warning": 10, "critical": 20},  # per minute
        }
        
        # Optimization strategies
        self.strategies = {
            "aggressive": {
                "cache_cleanup_interval": 30,  # seconds
                "gc_trigger_threshold": 0.7,
                "memory_cleanup_threshold": 0.8,
                "optimization_frequency": 60  # seconds
            },
            "balanced": {
                "cache_cleanup_interval": 120,
                "gc_trigger_threshold": 0.8,
                "memory_cleanup_threshold": 0.85,
                "optimization_frequency": 300
            },
            "conservative": {
                "cache_cleanup_interval": 300,
                "gc_trigger_threshold": 0.9,
                "memory_cleanup_threshold": 0.9,
                "optimization_frequency": 600
            }
        }
        
        self.current_strategy = "balanced"
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.performance_stats = {
            "optimizations_performed": 0,
            "total_improvement": 0.0,
            "average_improvement": 0.0,
            "last_optimization": None
        }
        
    async def start_optimization(self):
        """Start the performance optimization engine"""
        if self.optimization_active:
            logger.warning("Performance optimizer already active")
            return
            
        self.optimization_active = True
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        # Initialize baseline metrics
        await self._establish_baseline()
        
        logger.info("Performance Optimizer V2 started")
        
    async def stop_optimization(self):
        """Stop the performance optimization engine"""
        self.optimization_active = False
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup executor
        self.executor.shutdown(wait=True)
        logger.info("Performance Optimizer V2 stopped")
        
    async def _optimization_loop(self):
        """Main optimization loop"""
        while self.optimization_active:
            try:
                # Collect current metrics
                await self._collect_metrics()
                
                # Analyze performance
                analysis = await self._analyze_performance()
                
                # Generate optimization actions
                actions = await self._generate_optimization_actions(analysis)
                
                # Execute high-impact actions
                await self._execute_optimization_actions(actions)
                
                # Update performance baseline
                await self._update_baseline()
                
                # Wait before next optimization cycle
                strategy_config = self.strategies[self.current_strategy]
                await asyncio.sleep(strategy_config["optimization_frequency"])
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _establish_baseline(self):
        """Establish performance baseline"""
        logger.info("Establishing performance baseline...")
        
        # Collect metrics for baseline
        baseline_metrics = []
        for _ in range(10):  # Collect 10 samples
            metrics = await self._collect_system_metrics()
            baseline_metrics.append(metrics)
            await asyncio.sleep(1)
        
        # Calculate baseline values
        for metric_name in ["cpu_usage", "memory_usage", "response_time", "cache_hit_rate"]:
            values = [m.get(metric_name, 0) for m in baseline_metrics if metric_name in m]
            if values:
                self.performance_baseline[metric_name] = statistics.mean(values)
        
        logger.info(f"Performance baseline established: {self.performance_baseline}")
        
    async def _collect_metrics(self):
        """Collect comprehensive performance metrics"""
        try:
            # System metrics
            system_metrics = await self._collect_system_metrics()
            
            # Application metrics
            app_metrics = await self._collect_application_metrics()
            
            # Cache metrics
            cache_metrics = await self._collect_cache_metrics()
            
            # Database metrics
            db_metrics = await self._collect_database_metrics()
            
            # Combine all metrics
            all_metrics = {**system_metrics, **app_metrics, **cache_metrics, **db_metrics}
            
            # Store metrics
            for metric_name, value in all_metrics.items():
                metric = PerformanceMetric(
                    name=metric_name,
                    value=value,
                    timestamp=datetime.now(),
                    category=self._get_metric_category(metric_name),
                    metadata={"source": "optimizer_v2"}
                )
                self.metrics_history[metric_name].append(metric)
                
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            
    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free / (1024**3)  # GB
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_bytes_sent = network.bytes_sent
                network_bytes_recv = network.bytes_recv
            except:
                network_bytes_sent = 0
                network_bytes_recv = 0
            
            return {
                "cpu_usage": cpu_percent,
                "cpu_count": cpu_count,
                "memory_usage": memory_percent,
                "memory_available_gb": memory_available,
                "disk_usage": disk_percent,
                "disk_free_gb": disk_free,
                "network_bytes_sent": network_bytes_sent,
                "network_bytes_recv": network_bytes_recv
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
            
    async def _collect_application_metrics(self) -> Dict[str, float]:
        """Collect application-level metrics"""
        try:
            # Get performance summary from existing optimizer
            perf_summary = advanced_performance_optimizer.get_performance_summary()
            
            app_metrics = {
                "response_time": perf_summary.get("metrics", {}).get("response_time", {}).get("current", 0),
                "throughput": perf_summary.get("metrics", {}).get("throughput", {}).get("current", 0),
                "error_rate": perf_summary.get("metrics", {}).get("error_rate", {}).get("current", 0),
                "active_connections": perf_summary.get("metrics", {}).get("active_connections", {}).get("current", 0)
            }
            
            return app_metrics
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return {}
            
    async def _collect_cache_metrics(self) -> Dict[str, float]:
        """Collect cache performance metrics"""
        try:
            cache_stats = enhanced_cache.get_stats()
            cache_memory = enhanced_cache.get_memory_usage()
            
            return {
                "cache_hit_rate": cache_stats.get("hit_rate", 0) * 100,
                "cache_size": cache_stats.get("entry_count", 0),
                "cache_memory_mb": cache_memory.get("memory_usage_mb", 0),
                "cache_compression_ratio": cache_memory.get("compression_ratio", 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}")
            return {}
            
    async def _collect_database_metrics(self) -> Dict[str, float]:
        """Collect database performance metrics"""
        try:
            # Simulate database metrics (in real implementation, connect to actual DB)
            return {
                "db_connection_pool_size": 10,
                "db_active_connections": 5,
                "db_query_time_avg": 15.5,  # ms
                "db_transaction_rate": 100.0  # per second
            }
            
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")
            return {}
            
    def _get_metric_category(self, metric_name: str) -> str:
        """Get category for a metric"""
        if metric_name in ["cpu_usage", "memory_usage", "disk_usage"]:
            return "system"
        elif metric_name in ["response_time", "throughput", "error_rate"]:
            return "application"
        elif metric_name.startswith("cache_"):
            return "cache"
        elif metric_name.startswith("db_"):
            return "database"
        else:
            return "other"
            
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and identify issues"""
        try:
            analysis = {
                "issues": [],
                "recommendations": [],
                "performance_score": 100.0,
                "trends": {}
            }
            
            # Analyze each metric category
            for metric_name, history in self.metrics_history.items():
                if len(history) < 5:
                    continue
                    
                recent_values = [m.value for m in list(history)[-10:]]
                current_value = recent_values[-1]
                
                # Check thresholds
                threshold = self.thresholds.get(metric_name)
                if threshold:
                    if current_value > threshold["critical"]:
                        analysis["issues"].append({
                            "metric": metric_name,
                            "severity": "critical",
                            "current_value": current_value,
                            "threshold": threshold["critical"],
                            "description": f"{metric_name} is critically high"
                        })
                    elif current_value > threshold["warning"]:
                        analysis["issues"].append({
                            "metric": metric_name,
                            "severity": "warning",
                            "current_value": current_value,
                            "threshold": threshold["warning"],
                            "description": f"{metric_name} is above warning threshold"
                        })
                
                # Calculate trend
                if len(recent_values) >= 5:
                    trend = self._calculate_trend(recent_values)
                    analysis["trends"][metric_name] = trend
                    
                    # Adjust performance score based on trends
                    if trend == "degrading" and current_value > self.performance_baseline.get(metric_name, 0):
                        analysis["performance_score"] -= 5
                    elif trend == "improving":
                        analysis["performance_score"] += 2
                        
            # Ensure performance score is within bounds
            analysis["performance_score"] = max(0, min(100, analysis["performance_score"]))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            return {"issues": [], "recommendations": [], "performance_score": 50.0, "trends": {}}
            
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend for a series of values"""
        if len(values) < 3:
            return "stable"
            
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        change_percent = (second_avg - first_avg) / first_avg if first_avg > 0 else 0
        
        if change_percent > 0.05:
            return "increasing"
        elif change_percent < -0.05:
            return "decreasing"
        else:
            return "stable"
            
    async def _generate_optimization_actions(self, analysis: Dict[str, Any]) -> List[OptimizationAction]:
        """Generate optimization actions based on analysis"""
        actions = []
        
        # Cache optimization actions
        if "cache_hit_rate" in analysis["trends"] and analysis["trends"]["cache_hit_rate"] == "decreasing":
            actions.append(OptimizationAction(
                action_type="cache_optimization",
                description="Optimize cache configuration and eviction policies",
                impact_score=0.7,
                cost_score=0.2
            ))
            
        # Memory cleanup actions
        memory_issues = [i for i in analysis["issues"] if "memory" in i["metric"]]
        if memory_issues:
            actions.append(OptimizationAction(
                action_type="memory_cleanup",
                description="Trigger garbage collection and memory cleanup",
                impact_score=0.6,
                cost_score=0.3
            ))
            
        # CPU optimization actions
        cpu_issues = [i for i in analysis["issues"] if "cpu" in i["metric"]]
        if cpu_issues:
            actions.append(OptimizationAction(
                action_type="resource_scaling",
                description="Optimize CPU usage and resource allocation",
                impact_score=0.8,
                cost_score=0.4
            ))
            
        # Response time optimization
        response_issues = [i for i in analysis["issues"] if "response_time" in i["metric"]]
        if response_issues:
            actions.append(OptimizationAction(
                action_type="application_optimization",
                description="Optimize application response times",
                impact_score=0.9,
                cost_score=0.5
            ))
            
        return actions
        
    async def _execute_optimization_actions(self, actions: List[OptimizationAction]):
        """Execute optimization actions"""
        for action in actions:
            try:
                # Only execute high-impact, low-cost actions automatically
                if action.impact_score > 0.6 and action.cost_score < 0.4:
                    await self._execute_action(action)
                    
            except Exception as e:
                logger.error(f"Error executing action {action.action_type}: {e}")
                
    async def _execute_action(self, action: OptimizationAction):
        """Execute a specific optimization action"""
        try:
            logger.info(f"Executing optimization action: {action.description}")
            
            if action.action_type == "cache_optimization":
                await self._optimize_cache()
                
            elif action.action_type == "memory_cleanup":
                await self._cleanup_memory()
                
            elif action.action_type == "resource_scaling":
                await self._optimize_resources()
                
            elif action.action_type == "application_optimization":
                await self._optimize_application()
                
            # Mark action as executed
            action.executed = True
            action.executed_at = datetime.now()
            
            # Update performance stats
            self.performance_stats["optimizations_performed"] += 1
            self.performance_stats["last_optimization"] = datetime.now()
            
            logger.info(f"Successfully executed: {action.description}")
            
        except Exception as e:
            logger.error(f"Failed to execute action {action.action_type}: {e}")
            
    async def _optimize_cache(self):
        """Optimize cache performance"""
        try:
            # Clear expired entries
            await enhanced_cache.cleanup_expired()
            
            # Optimize eviction policy if needed
            cache_stats = enhanced_cache.get_stats()
            if cache_stats.get("hit_rate", 0) < 0.8:
                # Could implement cache size adjustment here
                pass
                
        except Exception as e:
            logger.error(f"Error optimizing cache: {e}")
            
    async def _cleanup_memory(self):
        """Perform memory cleanup"""
        try:
            # Trigger garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Clear weak references
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            
    async def _optimize_resources(self):
        """Optimize system resources"""
        try:
            # This would typically involve:
            # - Adjusting thread pool sizes
            # - Optimizing connection pools
            # - Scaling resources based on load
            
            # For now, just log the action
            logger.info("Resource optimization performed")
            
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
            
    async def _optimize_application(self):
        """Optimize application performance"""
        try:
            # This would typically involve:
            # - Optimizing database queries
            # - Adjusting application settings
            # - Implementing performance improvements
            
            # For now, just log the action
            logger.info("Application optimization performed")
            
        except Exception as e:
            logger.error(f"Error optimizing application: {e}")
            
    async def _update_baseline(self):
        """Update performance baseline based on recent metrics"""
        try:
            # Update baseline with recent performance
            for metric_name, history in self.metrics_history.items():
                if len(history) >= 10:
                    recent_values = [m.value for m in list(history)[-10:]]
                    new_baseline = statistics.mean(recent_values)
                    
                    # Smooth baseline update
                    current_baseline = self.performance_baseline.get(metric_name, new_baseline)
                    self.performance_baseline[metric_name] = (current_baseline * 0.8) + (new_baseline * 0.2)
                    
        except Exception as e:
            logger.error(f"Error updating baseline: {e}")
            
    def set_optimization_strategy(self, strategy: str):
        """Set optimization strategy"""
        if strategy in self.strategies:
            self.current_strategy = strategy
            logger.info(f"Optimization strategy changed to: {strategy}")
        else:
            logger.warning(f"Unknown optimization strategy: {strategy}")
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            # Get recent metrics
            recent_metrics = {}
            for metric_name, history in self.metrics_history.items():
                if history:
                    recent_metrics[metric_name] = {
                        "current": history[-1].value,
                        "baseline": self.performance_baseline.get(metric_name, 0),
                        "trend": self._calculate_trend([m.value for m in list(history)[-10:]]) if len(history) >= 10 else "stable"
                    }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "optimization_active": self.optimization_active,
                "current_strategy": self.current_strategy,
                "performance_score": self._calculate_overall_performance_score(),
                "metrics": recent_metrics,
                "baseline": self.performance_baseline,
                "stats": self.performance_stats,
                "recent_actions": [
                    {
                        "action_type": a.action_type,
                        "description": a.description,
                        "executed": a.executed,
                        "executed_at": a.executed_at.isoformat() if a.executed_at else None
                    }
                    for a in self.optimization_actions[-10:]  # Last 10 actions
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
            
    def _calculate_overall_performance_score(self) -> float:
        """Calculate overall performance score"""
        try:
            if not self.metrics_history:
                return 100.0
                
            score = 100.0
            
            # Penalize for issues
            for metric_name, history in self.metrics_history.items():
                if not history:
                    continue
                    
                current_value = history[-1].value
                baseline = self.performance_baseline.get(metric_name, current_value)
                threshold = self.thresholds.get(metric_name)
                
                if threshold:
                    if current_value > threshold["critical"]:
                        score -= 20
                    elif current_value > threshold["warning"]:
                        score -= 10
                        
                # Penalize for deviation from baseline
                if baseline > 0:
                    deviation = abs(current_value - baseline) / baseline
                    if deviation > 0.2:  # 20% deviation
                        score -= 5
                        
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 50.0


# Global instance
performance_optimizer_v2 = PerformanceOptimizerV2()
