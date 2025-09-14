"""
Advanced features and utilities for enterprise-grade functionality
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import uuid

logger = logging.getLogger(__name__)


class FeatureFlag(Enum):
    """Feature flags for controlling functionality"""
    AI_OPTIMIZATION = "ai_optimization"
    ADVANCED_ANALYTICS = "advanced_analytics"
    REAL_TIME_MONITORING = "real_time_monitoring"
    AUTO_SCALING = "auto_scaling"
    CIRCUIT_BREAKER = "circuit_breaker"
    RATE_LIMITING = "rate_limiting"
    CACHING = "caching"
    SECURITY_SCANNING = "security_scanning"


@dataclass
class FeatureConfig:
    """Configuration for feature flags"""
    enabled: bool = True
    rollout_percentage: float = 100.0
    user_groups: List[str] = None
    environments: List[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.user_groups is None:
            self.user_groups = ["all"]
        if self.environments is None:
            self.environments = ["development", "staging", "production"]


class FeatureManager:
    """Advanced feature flag management system"""
    
    def __init__(self):
        self.features: Dict[FeatureFlag, FeatureConfig] = {}
        self.user_context: Dict[str, Any] = {}
        self._load_default_features()
    
    def _load_default_features(self):
        """Load default feature configurations"""
        self.features = {
            FeatureFlag.AI_OPTIMIZATION: FeatureConfig(enabled=True, rollout_percentage=100.0),
            FeatureFlag.ADVANCED_ANALYTICS: FeatureConfig(enabled=True, rollout_percentage=100.0),
            FeatureFlag.REAL_TIME_MONITORING: FeatureConfig(enabled=True, rollout_percentage=100.0),
            FeatureFlag.AUTO_SCALING: FeatureConfig(enabled=True, rollout_percentage=100.0),
            FeatureFlag.CIRCUIT_BREAKER: FeatureConfig(enabled=True, rollout_percentage=100.0),
            FeatureFlag.RATE_LIMITING: FeatureConfig(enabled=True, rollout_percentage=100.0),
            FeatureFlag.CACHING: FeatureConfig(enabled=True, rollout_percentage=100.0),
            FeatureFlag.SECURITY_SCANNING: FeatureConfig(enabled=True, rollout_percentage=100.0),
        }
    
    def is_feature_enabled(self, feature: FeatureFlag, user_id: Optional[str] = None) -> bool:
        """Check if a feature is enabled for a user"""
        if feature not in self.features:
            return False
        
        config = self.features[feature]
        
        # Check if feature is globally disabled
        if not config.enabled:
            return False
        
        # Check date range
        now = datetime.now()
        if config.start_date and now < config.start_date:
            return False
        if config.end_date and now > config.end_date:
            return False
        
        # Check rollout percentage
        if user_id:
            user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            user_percentage = (user_hash % 100) + 1
            if user_percentage > config.rollout_percentage:
                return False
        
        return True
    
    def set_user_context(self, user_id: str, context: Dict[str, Any]):
        """Set user context for feature evaluation"""
        self.user_context[user_id] = context
    
    def get_enabled_features(self, user_id: Optional[str] = None) -> List[FeatureFlag]:
        """Get list of enabled features for a user"""
        return [
            feature for feature in FeatureFlag
            if self.is_feature_enabled(feature, user_id)
        ]


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class EventBus:
    """Event bus for decoupled communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: str, handler: callable):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: callable):
        """Unsubscribe from an event type"""
        if event_type in self.subscribers:
            if handler in self.subscribers[event_type]:
                self.subscribers[event_type].remove(handler)
    
    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event"""
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Notify subscribers
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history"""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        
        return events[-limit:]


class DistributedLock:
    """Distributed lock implementation"""
    
    def __init__(self, redis_client, key: str, timeout: int = 30):
        self.redis = redis_client
        self.key = f"lock:{key}"
        self.timeout = timeout
        self.identifier = str(uuid.uuid4())
    
    async def acquire(self) -> bool:
        """Acquire the lock"""
        try:
            result = await self.redis.set(
                self.key, 
                self.identifier, 
                nx=True, 
                ex=self.timeout
            )
            return result is not None
        except Exception as e:
            logger.error(f"Failed to acquire lock {self.key}: {e}")
            return False
    
    async def release(self) -> bool:
        """Release the lock"""
        try:
            # Use Lua script to ensure atomic release
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            result = await self.redis.eval(lua_script, 1, self.key, self.identifier)
            return result == 1
        except Exception as e:
            logger.error(f"Failed to release lock {self.key}: {e}")
            return False
    
    async def extend(self, additional_time: int) -> bool:
        """Extend lock timeout"""
        try:
            lua_script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("expire", KEYS[1], ARGV[2])
            else
                return 0
            end
            """
            result = await self.redis.eval(
                lua_script, 
                1, 
                self.key, 
                self.identifier, 
                str(self.timeout + additional_time)
            )
            return result == 1
        except Exception as e:
            logger.error(f"Failed to extend lock {self.key}: {e}")
            return False


class AdvancedCache:
    """Advanced caching with TTL, compression, and invalidation"""
    
    def __init__(self, redis_client, compression_enabled: bool = True):
        self.redis = redis_client
        self.compression_enabled = compression_enabled
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.redis.get(key)
            if value is None:
                self.cache_stats["misses"] += 1
                return None
            
            # Decompress if needed
            if self.compression_enabled and value.startswith(b"compressed:"):
                import gzip
                compressed_data = value[11:]  # Remove "compressed:" prefix
                decompressed = gzip.decompress(compressed_data)
                self.cache_stats["hits"] += 1
                return json.loads(decompressed.decode())
            
            self.cache_stats["hits"] += 1
            return json.loads(value.decode())
        
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.cache_stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache"""
        try:
            serialized = json.dumps(value, default=str)
            
            if self.compression_enabled and len(serialized) > 1024:  # Compress large values
                import gzip
                compressed = gzip.compress(serialized.encode())
                final_value = b"compressed:" + compressed
            else:
                final_value = serialized.encode()
            
            result = await self.redis.set(key, final_value, ex=ttl)
            self.cache_stats["sets"] += 1
            return result
        
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            result = await self.redis.delete(key)
            self.cache_stats["deletes"] += 1
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                result = await self.redis.delete(*keys)
                self.cache_stats["deletes"] += result
                return result
            return 0
        except Exception as e:
            logger.error(f"Cache pattern invalidation error for {pattern}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }


class AdvancedMetrics:
    """Advanced metrics collection with aggregation and alerting"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.alerts: Dict[str, Dict[str, Any]] = {}
        self.aggregations: Dict[str, Dict[str, float]] = {}
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Keep only last 1000 values
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
        
        # Check alerts
        self._check_alerts(name, value)
        
        # Update aggregations
        self._update_aggregations(name)
    
    def set_alert(self, metric_name: str, threshold: float, operator: str = "gt"):
        """Set an alert for a metric"""
        self.alerts[metric_name] = {
            "threshold": threshold,
            "operator": operator,
            "triggered": False,
            "last_triggered": None
        }
    
    def _check_alerts(self, metric_name: str, value: float):
        """Check if any alerts should be triggered"""
        if metric_name not in self.alerts:
            return
        
        alert = self.alerts[metric_name]
        threshold = alert["threshold"]
        operator = alert["operator"]
        
        triggered = False
        if operator == "gt" and value > threshold:
            triggered = True
        elif operator == "lt" and value < threshold:
            triggered = True
        elif operator == "eq" and value == threshold:
            triggered = True
        
        if triggered and not alert["triggered"]:
            alert["triggered"] = True
            alert["last_triggered"] = datetime.now().isoformat()
            logger.warning(f"Alert triggered for {metric_name}: {value} {operator} {threshold}")
    
    def _update_aggregations(self, metric_name: str):
        """Update metric aggregations"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return
        
        values = self.metrics[metric_name]
        self.aggregations[metric_name] = {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p50": self._percentile(values, 50),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99)
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific metric"""
        if metric_name not in self.aggregations:
            return None
        
        return {
            "name": metric_name,
            "aggregations": self.aggregations[metric_name],
            "alert": self.alerts.get(metric_name),
            "recent_values": self.metrics[metric_name][-10:]  # Last 10 values
        }


# Global instances
feature_manager = FeatureManager()
event_bus = EventBus()
advanced_metrics = AdvancedMetrics()
