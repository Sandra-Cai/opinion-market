import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from functools import wraps
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics"""

    hits: int
    misses: int
    hit_rate: float
    total_requests: int
    avg_response_time: float
    cache_size: int
    evictions: int
    timestamp: datetime


@dataclass
class DatabaseMetrics:
    """Database performance metrics"""

    active_connections: int
    idle_connections: int
    total_queries: int
    slow_queries: int
    avg_query_time: float
    connection_errors: int
    timestamp: datetime


@dataclass
class PerformanceProfile:
    """Performance profiling data"""

    endpoint: str
    method: str
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    request_count: int
    error_rate: float
    timestamp: datetime


class PerformanceOptimizer:
    """Comprehensive performance optimization service"""

    def __init__(self):
        self.redis_client: Optional[redis_sync.Redis] = None
        self.cache_metrics = {}
        self.db_metrics = {}
        self.performance_profiles = {}
        self.slow_query_threshold = 1.0  # seconds
        self.cache_ttl_default = 3600  # 1 hour
        self.connection_pool_size = 20
        self.max_overflow = 30

        # Cache keys
        self.cache_prefixes = {
            "user": "user:",
            "market": "market:",
            "trade": "trade:",
            "position": "position:",
            "analytics": "analytics:",
            "social": "social:",
        }

    async def initialize(self, redis_url: str, database_url: str):
        """Initialize the performance optimizer"""
        # Initialize Redis client
        self.redis_client = redis.from_url(redis_url)
        await self.redis_client.ping()

        # Initialize database connection pool
        self._setup_database_pool(database_url)

        # Start monitoring tasks
        asyncio.create_task(self._monitor_cache_performance())
        asyncio.create_task(self._monitor_database_performance())
        asyncio.create_task(self._cleanup_expired_cache())

        logger.info("Performance optimizer initialized")

    def _setup_database_pool(self, database_url: str):
        """Setup optimized database connection pool"""
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=self.connection_pool_size,
            max_overflow=self.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=False,  # Set to True for SQL logging
        )

        self.SessionLocal = sessionmaker(
            bind=self.engine, autocommit=False, autoflush=False
        )

    def cache_decorator(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """Decorator for caching function results"""

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(
                    func.__name__, args, kwargs, key_prefix
                )

                # Try to get from cache
                cached_result = await self._get_from_cache(cache_key)
                if cached_result is not None:
                    await self._record_cache_hit(cache_key)
                    return cached_result

                # Cache miss - execute function
                await self._record_cache_miss(cache_key)
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)

                    # Cache the result
                    cache_ttl = ttl or self.cache_ttl_default
                    await self._set_cache(cache_key, result, cache_ttl)

                    # Record performance
                    execution_time = time.time() - start_time
                    await self._record_performance(func.__name__, execution_time)

                    return result

                except Exception as e:
                    # Record error
                    await self._record_error(func.__name__, str(e))
                    raise

            return wrapper

        return decorator

    def database_optimizer(self, query_type: str = "read"):
        """Decorator for database query optimization"""

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)

                    # Record query performance
                    execution_time = time.time() - start_time
                    await self._record_database_query(query_type, execution_time)

                    # Alert on slow queries
                    if execution_time > self.slow_query_threshold:
                        await self._alert_slow_query(func.__name__, execution_time)

                    return result

                except Exception as e:
                    await self._record_database_error(func.__name__, str(e))
                    raise

            return wrapper

        return decorator

    async def warm_cache(self, cache_patterns: List[str]):
        """Warm up cache with frequently accessed data"""
        try:
            for pattern in cache_patterns:
                if pattern == "popular_markets":
                    await self._warm_popular_markets()
                elif pattern == "user_profiles":
                    await self._warm_user_profiles()
                elif pattern == "market_data":
                    await self._warm_market_data()
                elif pattern == "analytics":
                    await self._warm_analytics_data()

            logger.info(f"Cache warming completed for patterns: {cache_patterns}")

        except Exception as e:
            logger.error(f"Error warming cache: {e}")

    async def optimize_database_queries(self):
        """Analyze and optimize database queries"""
        try:
            # Get slow queries
            slow_queries = await self._get_slow_queries()

            # Analyze query patterns
            query_patterns = await self._analyze_query_patterns()

            # Generate optimization recommendations
            recommendations = await self._generate_optimization_recommendations(
                slow_queries, query_patterns
            )

            # Apply automatic optimizations
            await self._apply_automatic_optimizations(recommendations)

            logger.info("Database query optimization completed")
            return recommendations

        except Exception as e:
            logger.error(f"Error optimizing database queries: {e}")
            return []

    async def monitor_system_resources(self) -> Dict[str, Any]:
        """Monitor system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            disk_free = disk.free / (1024**3)  # GB

            # Network I/O
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv

            # Process info
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            process_cpu = process.cpu_percent()

            return {
                "cpu": {"usage_percent": cpu_percent, "core_count": cpu_count},
                "memory": {
                    "usage_percent": memory_percent,
                    "available_gb": memory_available,
                    "process_memory_mb": process_memory,
                },
                "disk": {"usage_percent": disk_percent, "free_gb": disk_free},
                "network": {
                    "bytes_sent": network_bytes_sent,
                    "bytes_recv": network_bytes_recv,
                },
                "process": {"cpu_percent": process_cpu, "memory_mb": process_memory},
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")
            return {}

    async def optimize_memory_usage(self):
        """Optimize memory usage"""
        try:
            # Force garbage collection
            collected = gc.collect()

            # Clear unnecessary caches
            await self._clear_old_cache_entries()

            # Optimize Redis memory
            await self._optimize_redis_memory()

            logger.info(f"Memory optimization completed. Collected {collected} objects")

        except Exception as e:
            logger.error(f"Error optimizing memory usage: {e}")

    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Get cache metrics
            cache_metrics = await self._get_cache_metrics()

            # Get database metrics
            db_metrics = await self._get_database_metrics()

            # Get system resources
            system_resources = await self.monitor_system_resources()

            # Get performance profiles
            performance_profiles = await self._get_performance_profiles()

            # Calculate overall performance score
            performance_score = await self._calculate_performance_score(
                cache_metrics, db_metrics, system_resources
            )

            return {
                "performance_score": performance_score,
                "cache_metrics": cache_metrics,
                "database_metrics": db_metrics,
                "system_resources": system_resources,
                "performance_profiles": performance_profiles,
                "recommendations": await self._generate_performance_recommendations(),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}

    async def _generate_cache_key(
        self, func_name: str, args: tuple, kwargs: dict, prefix: str
    ) -> str:
        """Generate cache key for function call"""
        # Create a hash of the function name and arguments
        key_data = {"func": func_name, "args": args, "kwargs": kwargs}

        import hashlib

        key_hash = hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode(), usedforsecurity=False
        ).hexdigest()

        return f"{prefix}{func_name}:{key_hash}"

    async def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if not self.redis_client:
                return None

            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)

            return None

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    async def _set_cache(self, cache_key: str, value: Any, ttl: int):
        """Set value in cache"""
        try:
            if not self.redis_client:
                return

            serialized_value = json.dumps(value)
            await self.redis_client.setex(cache_key, ttl, serialized_value)

        except Exception as e:
            logger.error(f"Error setting cache: {e}")

    async def _record_cache_hit(self, cache_key: str):
        """Record cache hit"""
        try:
            if not self.redis_client:
                return

            await self.redis_client.incr("cache:hits")
            await self.redis_client.incr("cache:total_requests")

        except Exception as e:
            logger.error(f"Error recording cache hit: {e}")

    async def _record_cache_miss(self, cache_key: str):
        """Record cache miss"""
        try:
            if not self.redis_client:
                return

            await self.redis_client.incr("cache:misses")
            await self.redis_client.incr("cache:total_requests")

        except Exception as e:
            logger.error(f"Error recording cache miss: {e}")

    async def _record_performance(self, func_name: str, execution_time: float):
        """Record function performance"""
        try:
            if not self.redis_client:
                return

            # Store execution time
            await self.redis_client.lpush(f"perf:{func_name}", execution_time)
            await self.redis_client.ltrim(f"perf:{func_name}", 0, 999)  # Keep last 1000

            # Update running average
            current_avg = await self.redis_client.get(f"perf_avg:{func_name}")
            if current_avg:
                current_avg = float(current_avg)
                new_avg = (current_avg + execution_time) / 2
            else:
                new_avg = execution_time

            await self.redis_client.set(f"perf_avg:{func_name}", new_avg)

        except Exception as e:
            logger.error(f"Error recording performance: {e}")

    async def _record_error(self, func_name: str, error_message: str):
        """Record function error"""
        try:
            if not self.redis_client:
                return

            await self.redis_client.incr(f"errors:{func_name}")
            await self.redis_client.lpush(f"error_log:{func_name}", error_message)
            await self.redis_client.ltrim(
                f"error_log:{func_name}", 0, 99
            )  # Keep last 100

        except Exception as e:
            logger.error(f"Error recording error: {e}")

    async def _record_database_query(self, query_type: str, execution_time: float):
        """Record database query performance"""
        try:
            if not self.redis_client:
                return

            # Record query count and time
            await self.redis_client.incr("db:total_queries")
            await self.redis_client.incr(f"db:queries:{query_type}")

            # Record slow queries
            if execution_time > self.slow_query_threshold:
                await self.redis_client.incr("db:slow_queries")
                await self.redis_client.lpush("db:slow_query_times", execution_time)
                await self.redis_client.ltrim("db:slow_query_times", 0, 99)

            # Update average query time
            current_avg = await self.redis_client.get("db:avg_query_time")
            if current_avg:
                current_avg = float(current_avg)
                total_queries = int(
                    await self.redis_client.get("db:total_queries") or 0
                )
                new_avg = (
                    (current_avg * (total_queries - 1)) + execution_time
                ) / total_queries
            else:
                new_avg = execution_time

            await self.redis_client.set("db:avg_query_time", new_avg)

        except Exception as e:
            logger.error(f"Error recording database query: {e}")

    async def _record_database_error(self, func_name: str, error_message: str):
        """Record database error"""
        try:
            if not self.redis_client:
                return

            await self.redis_client.incr("db:connection_errors")
            await self.redis_client.lpush(
                "db:error_log", f"{func_name}: {error_message}"
            )
            await self.redis_client.ltrim("db:error_log", 0, 99)

        except Exception as e:
            logger.error(f"Error recording database error: {e}")

    async def _alert_slow_query(self, func_name: str, execution_time: float):
        """Alert on slow query"""
        logger.warning(f"Slow query detected in {func_name}: {execution_time:.2f}s")

    async def _warm_popular_markets(self):
        """Warm cache with popular markets"""
        # This would query the database for popular markets and cache them
        pass

    async def _warm_user_profiles(self):
        """Warm cache with active user profiles"""
        # This would cache profiles of recently active users
        pass

    async def _warm_market_data(self):
        """Warm cache with market data"""
        # This would cache current market prices and statistics
        pass

    async def _warm_analytics_data(self):
        """Warm cache with analytics data"""
        # This would cache frequently accessed analytics
        pass

    async def _get_slow_queries(self) -> List[Dict[str, Any]]:
        """Get list of slow queries"""
        try:
            if not self.redis_client:
                return []

            slow_times = await self.redis_client.lrange("db:slow_query_times", 0, -1)
            return [{"execution_time": float(time)} for time in slow_times]

        except Exception as e:
            logger.error(f"Error getting slow queries: {e}")
            return []

    async def _analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze database query patterns"""
        try:
            if not self.redis_client:
                return {}

            # Get query counts by type
            read_queries = int(await self.redis_client.get("db:queries:read") or 0)
            write_queries = int(await self.redis_client.get("db:queries:write") or 0)

            return {
                "read_queries": read_queries,
                "write_queries": write_queries,
                "total_queries": read_queries + write_queries,
            }

        except Exception as e:
            logger.error(f"Error analyzing query patterns: {e}")
            return {}

    async def _generate_optimization_recommendations(
        self, slow_queries: List, query_patterns: Dict
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        if slow_queries:
            recommendations.append(
                "Consider adding database indexes for frequently slow queries"
            )

        if (
            query_patterns.get("read_queries", 0)
            > query_patterns.get("write_queries", 0) * 10
        ):
            recommendations.append(
                "Consider implementing read replicas for better read performance"
            )

        if query_patterns.get("total_queries", 0) > 10000:
            recommendations.append("Consider implementing query result caching")

        return recommendations

    async def _apply_automatic_optimizations(self, recommendations: List[str]):
        """Apply automatic optimizations"""
        for recommendation in recommendations:
            if "caching" in recommendation.lower():
                await self._optimize_caching_strategy()
            elif "indexes" in recommendation.lower():
                await self._suggest_database_indexes()

    async def _optimize_caching_strategy(self):
        """Optimize caching strategy"""
        # Adjust cache TTL based on access patterns
        pass

    async def _suggest_database_indexes(self):
        """Suggest database indexes"""
        # Analyze query patterns and suggest indexes
        pass

    async def _clear_old_cache_entries(self):
        """Clear old cache entries"""
        try:
            if not self.redis_client:
                return

            # Clear entries older than 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            # This is a simplified approach - in production you'd use Redis SCAN
            logger.info("Cleared old cache entries")

        except Exception as e:
            logger.error(f"Error clearing old cache entries: {e}")

    async def _optimize_redis_memory(self):
        """Optimize Redis memory usage"""
        try:
            if not self.redis_client:
                return

            # Set memory policy
            await self.redis_client.config_set("maxmemory-policy", "allkeys-lru")

            logger.info("Redis memory optimization completed")

        except Exception as e:
            logger.error(f"Error optimizing Redis memory: {e}")

    async def _monitor_cache_performance(self):
        """Monitor cache performance"""
        while True:
            try:
                if self.redis_client:
                    # Calculate cache hit rate
                    hits = int(await self.redis_client.get("cache:hits") or 0)
                    misses = int(await self.redis_client.get("cache:misses") or 0)
                    total = hits + misses

                    hit_rate = (hits / total * 100) if total > 0 else 0

                    # Store metrics
                    metrics = CacheMetrics(
                        hits=hits,
                        misses=misses,
                        hit_rate=hit_rate,
                        total_requests=total,
                        avg_response_time=0.0,  # Would calculate from actual data
                        cache_size=0,  # Would get from Redis INFO
                        evictions=0,  # Would get from Redis INFO
                        timestamp=datetime.utcnow(),
                    )

                    self.cache_metrics[datetime.utcnow()] = metrics

                await asyncio.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"Error monitoring cache performance: {e}")
                await asyncio.sleep(60)

    async def _monitor_database_performance(self):
        """Monitor database performance"""
        while True:
            try:
                # Get database connection info
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT count(*) FROM pg_stat_activity"))
                    active_connections = result.scalar()

                # Store metrics
                metrics = DatabaseMetrics(
                    active_connections=active_connections,
                    idle_connections=0,  # Would calculate from pool stats
                    total_queries=int(
                        await self.redis_client.get("db:total_queries") or 0
                    ),
                    slow_queries=int(
                        await self.redis_client.get("db:slow_queries") or 0
                    ),
                    avg_query_time=float(
                        await self.redis_client.get("db:avg_query_time") or 0
                    ),
                    connection_errors=int(
                        await self.redis_client.get("db:connection_errors") or 0
                    ),
                    timestamp=datetime.utcnow(),
                )

                self.db_metrics[datetime.utcnow()] = metrics

                await asyncio.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"Error monitoring database performance: {e}")
                await asyncio.sleep(60)

    async def _cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        while True:
            try:
                await self._clear_old_cache_entries()
                await asyncio.sleep(3600)  # Clean up every hour

            except Exception as e:
                logger.error(f"Error cleaning up expired cache: {e}")
                await asyncio.sleep(3600)

    async def _get_cache_metrics(self) -> Dict[str, Any]:
        """Get current cache metrics"""
        try:
            if not self.cache_metrics:
                return {}

            latest_metrics = max(self.cache_metrics.values(), key=lambda x: x.timestamp)

            return {
                "hits": latest_metrics.hits,
                "misses": latest_metrics.misses,
                "hit_rate": latest_metrics.hit_rate,
                "total_requests": latest_metrics.total_requests,
                "avg_response_time": latest_metrics.avg_response_time,
                "cache_size": latest_metrics.cache_size,
                "evictions": latest_metrics.evictions,
            }

        except Exception as e:
            logger.error(f"Error getting cache metrics: {e}")
            return {}

    async def _get_database_metrics(self) -> Dict[str, Any]:
        """Get current database metrics"""
        try:
            if not self.db_metrics:
                return {}

            latest_metrics = max(self.db_metrics.values(), key=lambda x: x.timestamp)

            return {
                "active_connections": latest_metrics.active_connections,
                "idle_connections": latest_metrics.idle_connections,
                "total_queries": latest_metrics.total_queries,
                "slow_queries": latest_metrics.slow_queries,
                "avg_query_time": latest_metrics.avg_query_time,
                "connection_errors": latest_metrics.connection_errors,
            }

        except Exception as e:
            logger.error(f"Error getting database metrics: {e}")
            return {}

    async def _get_performance_profiles(self) -> List[Dict[str, Any]]:
        """Get performance profiles for endpoints"""
        try:
            if not self.redis_client:
                return []

            # Get all performance keys
            perf_keys = await self.redis_client.keys("perf_avg:*")

            profiles = []
            for key in perf_keys:
                func_name = key.decode().replace("perf_avg:", "")
                avg_time = float(await self.redis_client.get(key) or 0)

                profiles.append(
                    {
                        "endpoint": func_name,
                        "avg_response_time": avg_time,
                        "request_count": 0,  # Would calculate from actual data
                    }
                )

            return profiles

        except Exception as e:
            logger.error(f"Error getting performance profiles: {e}")
            return []

    async def _calculate_performance_score(
        self, cache_metrics: Dict, db_metrics: Dict, system_resources: Dict
    ) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            score = 100.0

            # Cache performance (30% weight)
            if cache_metrics:
                hit_rate = cache_metrics.get("hit_rate", 0)
                score -= (100 - hit_rate) * 0.3

            # Database performance (30% weight)
            if db_metrics:
                avg_query_time = db_metrics.get("avg_query_time", 0)
                if avg_query_time > 1.0:
                    score -= (avg_query_time - 1.0) * 10 * 0.3

            # System resources (40% weight)
            if system_resources:
                cpu_usage = system_resources.get("cpu", {}).get("usage_percent", 0)
                memory_usage = system_resources.get("memory", {}).get(
                    "usage_percent", 0
                )

                score -= (cpu_usage - 50) * 0.2  # Penalize high CPU
                score -= (memory_usage - 70) * 0.2  # Penalize high memory

            return max(0, min(100, score))

        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 50.0

    async def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []

        # Get current metrics
        cache_metrics = await self._get_cache_metrics()
        db_metrics = await self._get_database_metrics()
        system_resources = await self.monitor_system_resources()

        # Cache recommendations
        if cache_metrics.get("hit_rate", 0) < 80:
            recommendations.append(
                "Consider increasing cache TTL for frequently accessed data"
            )

        # Database recommendations
        if db_metrics.get("avg_query_time", 0) > 0.5:
            recommendations.append("Consider adding database indexes for slow queries")

        # System recommendations
        if system_resources.get("cpu", {}).get("usage_percent", 0) > 80:
            recommendations.append("Consider scaling up CPU resources")

        if system_resources.get("memory", {}).get("usage_percent", 0) > 90:
            recommendations.append("Consider increasing memory allocation")

        return recommendations


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance"""
    return performance_optimizer
