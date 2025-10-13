"""
Advanced Performance Optimizer for Opinion Market
Provides intelligent caching, query optimization, and performance monitoring
"""

import time
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from functools import wraps, lru_cache
from collections import defaultdict, deque
import logging
import psutil
import threading
from contextlib import asynccontextmanager

from app.core.database import get_redis_client
from app.core.logging import log_system_metric
from app.core.config import settings

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Database query optimization and caching"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.query_cache = {}
        self.slow_queries = deque(maxlen=1000)
        self.query_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "avg_time": 0})
        
    def cache_query_result(self, query_hash: str, result: Any, ttl: int = 300):
        """Cache query result with TTL"""
        try:
            if self.redis_client:
                cache_key = f"query_cache:{query_hash}"
                self.redis_client.setex(cache_key, ttl, json.dumps(result, default=str))
            else:
                # Fallback to in-memory cache
                self.query_cache[query_hash] = {
                    "result": result,
                    "expires_at": time.time() + ttl
                }
        except Exception as e:
            logger.warning(f"Failed to cache query result: {e}")
    
    def get_cached_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result"""
        try:
            if self.redis_client:
                cache_key = f"query_cache:{query_hash}"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            else:
                # Check in-memory cache
                if query_hash in self.query_cache:
                    cache_entry = self.query_cache[query_hash]
                    if time.time() < cache_entry["expires_at"]:
                        return cache_entry["result"]
                    else:
                        del self.query_cache[query_hash]
        except Exception as e:
            logger.warning(f"Failed to get cached result: {e}")
        
        return None
    
    def generate_query_hash(self, query: str, params: Dict[str, Any]) -> str:
        """Generate hash for query caching"""
        query_data = {
            "query": query,
            "params": params
        }
        query_string = json.dumps(query_data, sort_keys=True)
        return hashlib.md5(query_string.encode()).hexdigest()
    
    def record_slow_query(self, query: str, execution_time: float, params: Dict[str, Any]):
        """Record slow query for optimization"""
        slow_query_data = {
            "query": query[:200],  # Truncate for storage
            "execution_time": execution_time,
            "params": params,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.slow_queries.append(slow_query_data)
        
        # Log slow query
        log_system_metric("slow_query", execution_time, {
            "query_hash": self.generate_query_hash(query, params),
            "execution_time": execution_time
        })
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        return {
            "total_queries": sum(stats["count"] for stats in self.query_stats.values()),
            "slow_queries_count": len(self.slow_queries),
            "average_query_time": sum(stats["avg_time"] for stats in self.query_stats.values()) / max(len(self.query_stats), 1),
            "query_breakdown": dict(self.query_stats),
            "recent_slow_queries": list(self.slow_queries)[-10:]  # Last 10 slow queries
        }


class CacheManager:
    """Intelligent caching system with automatic invalidation"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
        self.cache_dependencies = defaultdict(set)  # Track cache dependencies
        self.cache_tags = defaultdict(set)  # Group caches by tags
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    self.cache_stats["hits"] += 1
                    return json.loads(cached_data)
                else:
                    self.cache_stats["misses"] += 1
                    return None
            else:
                # Fallback to in-memory cache
                return self._get_from_memory(key)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300, tags: List[str] = None):
        """Set value in cache with TTL and tags"""
        try:
            if self.redis_client:
                self.redis_client.setex(key, ttl, json.dumps(value, default=str))
            else:
                self._set_in_memory(key, value, ttl)
            
            self.cache_stats["sets"] += 1
            
            # Track tags for invalidation
            if tags:
                for tag in tags:
                    self.cache_tags[tag].add(key)
                    
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete value from cache"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                self._delete_from_memory(key)
            
            self.cache_stats["deletes"] += 1
            
            # Remove from tags
            for tag, keys in self.cache_tags.items():
                keys.discard(key)
                
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
    
    def invalidate_by_tag(self, tag: str):
        """Invalidate all caches with a specific tag"""
        try:
            if tag in self.cache_tags:
                keys_to_delete = list(self.cache_tags[tag])
                if self.redis_client:
                    if keys_to_delete:
                        self.redis_client.delete(*keys_to_delete)
                else:
                    for key in keys_to_delete:
                        self._delete_from_memory(key)
                
                # Clear the tag
                self.cache_tags[tag].clear()
                
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get from in-memory cache"""
        # This would be implemented with a proper in-memory cache
        # For now, just return None
        self.cache_stats["misses"] += 1
        return None
    
    def _set_in_memory(self, key: str, value: Any, ttl: int):
        """Set in in-memory cache"""
        # This would be implemented with a proper in-memory cache
        pass
    
    def _delete_from_memory(self, key: str):
        """Delete from in-memory cache"""
        # This would be implemented with a proper in-memory cache
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_tags_count": len(self.cache_tags),
            "total_tagged_keys": sum(len(keys) for keys in self.cache_tags.values())
        }


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.performance_alerts = deque(maxlen=100)
        self.system_stats = {}
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._check_performance_thresholds()
                time.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available,
                "disk_percent": disk.percent,
                "disk_free": disk.free,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv
            }
            
            # Store metrics
            self.metrics["system"].append(metrics)
            
            # Keep only last 1000 entries
            if len(self.metrics["system"]) > 1000:
                self.metrics["system"] = self.metrics["system"][-1000:]
            
            # Log system metric
            log_system_metric("system_performance", cpu_percent, {
                "memory_percent": memory.percent,
                "disk_percent": disk.percent
            })
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _check_performance_thresholds(self):
        """Check performance thresholds and generate alerts"""
        try:
            if not self.metrics["system"]:
                return
            
            latest_metrics = self.metrics["system"][-1]
            
            # Check CPU threshold
            if latest_metrics["cpu_percent"] > 80:
                self._create_alert("high_cpu", f"CPU usage: {latest_metrics['cpu_percent']:.1f}%")
            
            # Check memory threshold
            if latest_metrics["memory_percent"] > 85:
                self._create_alert("high_memory", f"Memory usage: {latest_metrics['memory_percent']:.1f}%")
            
            # Check disk threshold
            if latest_metrics["disk_percent"] > 90:
                self._create_alert("high_disk", f"Disk usage: {latest_metrics['disk_percent']:.1f}%")
                
        except Exception as e:
            logger.error(f"Error checking performance thresholds: {e}")
    
    def _create_alert(self, alert_type: str, message: str):
        """Create performance alert"""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "warning"
        }
        
        self.performance_alerts.append(alert)
        
        # Log the alert
        log_system_metric("performance_alert", 1, {
            "alert_type": alert_type,
            "message": message
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics["system"]:
            return {"error": "No performance data available"}
        
        latest_metrics = self.metrics["system"][-1]
        
        # Calculate averages over last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.metrics["system"]
            if datetime.fromisoformat(m["timestamp"]) > one_hour_ago
        ]
        
        if recent_metrics:
            avg_cpu = sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m["memory_percent"] for m in recent_metrics) / len(recent_metrics)
            avg_disk = sum(m["disk_percent"] for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = avg_memory = avg_disk = 0
        
        return {
            "current": latest_metrics,
            "averages_last_hour": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "disk_percent": avg_disk
            },
            "alerts": list(self.performance_alerts)[-10:],  # Last 10 alerts
            "monitoring_active": self.monitoring_active
        }


class PerformanceOptimizer:
    """Main performance optimization orchestrator"""
    
    def __init__(self):
        self.query_optimizer = QueryOptimizer()
        self.cache_manager = CacheManager()
        self.performance_monitor = PerformanceMonitor()
        self.optimization_rules = []
        
    def start_optimization(self):
        """Start all optimization services"""
        self.performance_monitor.start_monitoring()
        logger.info("Performance optimization started")
    
    def stop_optimization(self):
        """Stop all optimization services"""
        self.performance_monitor.stop_monitoring()
        logger.info("Performance optimization stopped")
    
    def optimize_query(self, query: str, params: Dict[str, Any], ttl: int = 300):
        """Optimize database query with caching"""
        query_hash = self.query_optimizer.generate_query_hash(query, params)
        
        # Try to get from cache first
        cached_result = self.query_optimizer.get_cached_result(query_hash)
        if cached_result is not None:
            return cached_result
        
        # Execute query and cache result
        start_time = time.time()
        # This would execute the actual query
        # result = execute_query(query, params)
        result = None  # Placeholder
        
        execution_time = time.time() - start_time
        
        # Record slow queries
        if execution_time > 1.0:  # Queries taking more than 1 second
            self.query_optimizer.record_slow_query(query, execution_time, params)
        
        # Cache the result
        if result is not None:
            self.query_optimizer.cache_query_result(query_hash, result, ttl)
        
        return result
    
    def smart_cache(self, key: str, value: Any, ttl: int = 300, tags: List[str] = None):
        """Smart caching with automatic optimization"""
        self.cache_manager.set(key, value, ttl, tags)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            "query_optimization": self.query_optimizer.get_query_stats(),
            "cache_performance": self.cache_manager.get_cache_stats(),
            "system_performance": self.performance_monitor.get_performance_summary(),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()


def cached_query(ttl: int = 300):
    """Decorator for caching database queries"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"query:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = performance_optimizer.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            performance_optimizer.cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def performance_monitor(func_name: str = None):
    """Decorator for monitoring function performance"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func_name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log performance metric
                log_system_metric("function_performance", execution_time, {
                    "function": function_name,
                    "status": "success"
                })
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log error metric
                log_system_metric("function_error", execution_time, {
                    "function": function_name,
                    "error": str(e),
                    "status": "error"
                })
                
                raise
        return wrapper
    return decorator


@asynccontextmanager
async def performance_context(operation_name: str):
    """Context manager for performance monitoring"""
    start_time = time.time()
    try:
        yield
        execution_time = time.time() - start_time
        log_system_metric("operation_performance", execution_time, {
            "operation": operation_name,
            "status": "success"
        })
    except Exception as e:
        execution_time = time.time() - start_time
        log_system_metric("operation_error", execution_time, {
            "operation": operation_name,
            "error": str(e),
            "status": "error"
        })
        raise


# Export functions and classes
__all__ = [
    "PerformanceOptimizer",
    "QueryOptimizer", 
    "CacheManager",
    "PerformanceMonitor",
    "performance_optimizer",
    "cached_query",
    "performance_monitor",
    "performance_context"
]