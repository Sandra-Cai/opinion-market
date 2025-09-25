"""
Database Performance Optimization
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

from app.core.database import engine
from sqlalchemy import text, inspect
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


@dataclass
class QueryPerformance:
    query: str
    execution_time: float
    timestamp: float
    parameters: Dict[str, Any]


@dataclass
class DatabaseMetrics:
    connection_count: int
    active_queries: int
    slow_queries: int
    cache_hit_ratio: float
    index_usage: Dict[str, float]


class DatabaseOptimizer:
    """Database performance optimization system"""
    
    def __init__(self):
        self.query_performance = deque(maxlen=1000)
        self.slow_queries = deque(maxlen=100)
        self.database_metrics = {}
        self.optimization_recommendations = []
        
        # Performance thresholds
        self.thresholds = {
            "slow_query_time": 1.0,  # 1 second
            "connection_limit": 80,  # 80% of max connections
            "cache_hit_ratio_min": 0.95  # 95% cache hit ratio
        }
    
    async def start_monitoring(self):
        """Start database monitoring"""
        logger.info("Starting database performance monitoring")
        asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await self._collect_metrics()
                await self._analyze_performance()
                await asyncio.sleep(60)  # Monitor every minute
            except Exception as e:
                logger.error(f"Error in database monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _collect_metrics(self):
        """Collect database metrics"""
        try:
            with engine.connect() as conn:
                # Get connection pool stats
                pool = engine.pool
                connection_count = pool.checkedout()
                max_connections = pool.size()
                
                # Get database stats
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as active_connections,
                        (SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active') as active_queries
                """))
                stats = result.fetchone()
                
                # Get cache hit ratio
                cache_result = conn.execute(text("""
                    SELECT 
                        round(100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read)), 2) as cache_hit_ratio
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """))
                cache_stats = cache_result.fetchone()
                
                self.database_metrics = DatabaseMetrics(
                    connection_count=connection_count,
                    active_queries=stats[1] if stats else 0,
                    slow_queries=len(self.slow_queries),
                    cache_hit_ratio=cache_stats[0] / 100.0 if cache_stats and cache_stats[0] else 0.0,
                    index_usage={}
                )
                
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")
    
    async def _analyze_performance(self):
        """Analyze database performance"""
        try:
            recommendations = []
            
            # Check connection usage
            if self.database_metrics.connection_count > self.thresholds["connection_limit"]:
                recommendations.append({
                    "type": "connection_pool",
                    "severity": "high",
                    "message": f"High connection usage: {self.database_metrics.connection_count}",
                    "suggestion": "Consider increasing connection pool size or optimizing queries"
                })
            
            # Check cache hit ratio
            if self.database_metrics.cache_hit_ratio < self.thresholds["cache_hit_ratio_min"]:
                recommendations.append({
                    "type": "cache_performance",
                    "severity": "medium",
                    "message": f"Low cache hit ratio: {self.database_metrics.cache_hit_ratio:.2%}",
                    "suggestion": "Consider increasing shared_buffers or optimizing queries"
                })
            
            # Check for slow queries
            if len(self.slow_queries) > 10:
                recommendations.append({
                    "type": "slow_queries",
                    "severity": "high",
                    "message": f"Multiple slow queries detected: {len(self.slow_queries)}",
                    "suggestion": "Review and optimize slow queries"
                })
            
            self.optimization_recommendations = recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing database performance: {e}")
    
    async def log_query_performance(self, query: str, execution_time: float, parameters: Dict[str, Any] = None):
        """Log query performance"""
        try:
            performance = QueryPerformance(
                query=query[:200],  # Truncate long queries
                execution_time=execution_time,
                timestamp=time.time(),
                parameters=parameters or {}
            )
            
            self.query_performance.append(performance)
            
            # Check if it's a slow query
            if execution_time > self.thresholds["slow_query_time"]:
                self.slow_queries.append(performance)
                logger.warning(f"Slow query detected: {execution_time:.2f}s - {query[:100]}")
            
        except Exception as e:
            logger.error(f"Error logging query performance: {e}")
    
    async def get_slow_queries(self, limit: int = 10) -> List[QueryPerformance]:
        """Get slow queries"""
        return list(self.slow_queries)[-limit:]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        return {
            "metrics": self.database_metrics.__dict__,
            "recommendations": self.optimization_recommendations,
            "slow_queries_count": len(self.slow_queries),
            "total_queries_logged": len(self.query_performance)
        }
    
    async def optimize_queries(self) -> Dict[str, Any]:
        """Provide query optimization suggestions"""
        try:
            suggestions = []
            
            # Analyze recent slow queries
            recent_slow = list(self.slow_queries)[-20:]
            
            for query_perf in recent_slow:
                query = query_perf.query.lower()
                
                # Check for common optimization opportunities
                if "select *" in query:
                    suggestions.append({
                        "query": query_perf.query[:100],
                        "issue": "SELECT * usage",
                        "suggestion": "Use specific column names instead of SELECT *",
                        "impact": "high"
                    })
                
                if "order by" in query and "limit" not in query:
                    suggestions.append({
                        "query": query_perf.query[:100],
                        "issue": "ORDER BY without LIMIT",
                        "suggestion": "Add LIMIT clause or create appropriate index",
                        "impact": "medium"
                    })
                
                if "like '%" in query:
                    suggestions.append({
                        "query": query_perf.query[:100],
                        "issue": "Leading wildcard in LIKE",
                        "suggestion": "Avoid leading wildcards or use full-text search",
                        "impact": "high"
                    })
            
            return {
                "optimization_suggestions": suggestions,
                "total_suggestions": len(suggestions)
            }
            
        except Exception as e:
            logger.error(f"Error generating optimization suggestions: {e}")
            return {"optimization_suggestions": [], "total_suggestions": 0}
    
    async def get_index_recommendations(self) -> List[Dict[str, Any]]:
        """Get index optimization recommendations"""
        try:
            recommendations = []
            
            with engine.connect() as conn:
                # Get table statistics
                result = conn.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins + n_tup_upd + n_tup_del as total_changes,
                        n_live_tup as live_tuples
                    FROM pg_stat_user_tables 
                    ORDER BY total_changes DESC 
                    LIMIT 10
                """))
                
                for row in result:
                    if row[2] > 1000:  # High activity tables
                        recommendations.append({
                            "table": f"{row[0]}.{row[1]}",
                            "activity": row[2],
                            "tuples": row[3],
                            "suggestion": "Consider adding indexes on frequently queried columns"
                        })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting index recommendations: {e}")
            return []


# Global database optimizer
database_optimizer = DatabaseOptimizer()
