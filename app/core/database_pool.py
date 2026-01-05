"""
Advanced database connection pooling and optimization
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.engine import Engine
import time

logger = logging.getLogger(__name__)


class DatabasePoolManager:
    """
    Advanced database connection pool management.
    
    Manages database connection pooling with monitoring, health checks,
    and optimization features.
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False,
    ) -> None:
        """
        Initialize the database pool manager.
        
        Args:
            database_url: Database connection URL
            pool_size: Number of connections to maintain in the pool
            max_overflow: Maximum number of connections to create beyond pool_size
            pool_timeout: Timeout in seconds for getting a connection from the pool
            pool_recycle: Time in seconds before recycling a connection
            echo: Enable SQL query logging
        """
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.echo = echo

        self.engine: Optional[Engine] = None
        self.async_engine = None
        self.async_session_factory = None
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "pool_overflow": 0,
            "pool_timeouts": 0,
        }

    def create_engine(self) -> Engine:
        """Create optimized database engine with connection pooling"""
        if self.engine is None:
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                echo=self.echo,
                pool_pre_ping=True,  # Verify connections before use
                connect_args={
                    "options": "-c timezone=utc",
                    "application_name": "opinion_market_api",
                },
            )

            # Add connection event listeners
            self._add_connection_listeners()

            logger.info(
                f"Database engine created with pool_size={self.pool_size}, max_overflow={self.max_overflow}"
            )

        return self.engine

    def create_async_engine(self) -> Any:
        """
        Create async database engine with connection pooling.
        
        Returns:
            AsyncEngine instance for async database operations
        """
        if self.async_engine is None:
            self.async_engine = create_async_engine(
                self.database_url.replace("postgresql://", "postgresql+asyncpg://"),
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                echo=self.echo,
                pool_pre_ping=True,
            )

            self.async_session_factory = async_sessionmaker(
                self.async_engine, class_=AsyncSession, expire_on_commit=False
            )

            logger.info("Async database engine created")

        return self.async_engine

    def _add_connection_listeners(self):
        """Add connection event listeners for monitoring"""

        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            """Track new connections"""
            self.connection_stats["total_connections"] += 1
            self.connection_stats["active_connections"] += 1
            logger.debug("New database connection established")

        @event.listens_for(self.engine, "close")
        def receive_close(dbapi_connection, connection_record):
            """Track connection closures"""
            self.connection_stats["active_connections"] = max(
                0, self.connection_stats["active_connections"] - 1
            )
            logger.debug("Database connection closed")

        @event.listens_for(self.engine, "overflow")
        def receive_overflow(dbapi_connection, connection_record):
            """Track pool overflow"""
            self.connection_stats["pool_overflow"] += 1
            logger.warning("Database pool overflow occurred")

        @event.listens_for(self.engine, "timeout")
        def receive_timeout(dbapi_connection, connection_record):
            """Track pool timeouts"""
            self.connection_stats["pool_timeouts"] += 1
            logger.warning("Database pool timeout occurred")

    @asynccontextmanager
    async def get_async_session(self):
        """Get async database session with proper cleanup"""
        if not self.async_session_factory:
            self.create_async_engine()

        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection pool statistics"""
        if self.engine:
            pool = self.engine.pool
            return {
                **self.connection_stats,
                "pool_size": pool.size(),
                "checked_in_connections": pool.checkedin(),
                "checked_out_connections": pool.checkedout(),
                "overflow_connections": pool.overflow(),
                "pool_invalidated": pool.invalidated(),
            }
        return self.connection_stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            if not self.async_engine:
                self.create_async_engine()

            async with self.async_engine.begin() as conn:
                result = await conn.execute("SELECT 1")
                row = result.fetchone()

                return {
                    "status": "healthy",
                    "response_time": 0,  # Could measure actual response time
                    "connection_pool": self.get_connection_stats(),
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_pool": self.get_connection_stats(),
            }

    async def close_all_connections(self):
        """Close all database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("All database connections closed")

        if self.async_engine:
            await self.async_engine.dispose()
            logger.info("All async database connections closed")


class QueryOptimizer:
    """Database query optimization utilities"""

    @staticmethod
    def optimize_query(query: str) -> str:
        """Basic query optimization"""
        # Remove unnecessary whitespace
        optimized = " ".join(query.split())

        # Add common optimizations
        if "SELECT" in optimized.upper():
            # Ensure proper indexing hints
            if "WHERE" in optimized.upper() and "ORDER BY" in optimized.upper():
                # Suggest index optimization
                logger.debug(
                    "Query could benefit from composite index on WHERE and ORDER BY columns"
                )

        return optimized

    @staticmethod
    def explain_query(query: str) -> str:
        """Generate EXPLAIN query for analysis"""
        return f"EXPLAIN ANALYZE {query}"

    @staticmethod
    def get_query_metrics(query: str) -> Dict[str, Any]:
        """Get query performance metrics"""
        return {
            "query_length": len(query),
            "has_joins": "JOIN" in query.upper(),
            "has_subqueries": (
                "SELECT" in query.upper().split("FROM")[1:]
                if "FROM" in query.upper()
                else False
            ),
            "has_aggregations": any(
                func in query.upper() for func in ["COUNT", "SUM", "AVG", "MIN", "MAX"]
            ),
            "complexity_score": len(query.split()) * 0.1,
        }


# Global database pool manager
db_pool_manager = DatabasePoolManager(
    database_url="postgresql://user:password@localhost/opinion_market",
    pool_size=20,
    max_overflow=30,
)
