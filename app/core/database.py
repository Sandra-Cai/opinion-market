from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy.engine import Engine
from app.core.config import settings
import logging
import os
import time
from contextlib import contextmanager
from typing import Generator, Optional
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# Use enhanced configuration if available, fallback to settings
def get_database_url():
    try:
        from app.core.enhanced_config import enhanced_config_manager
        return enhanced_config_manager.get("database.url", settings.DATABASE_URL)
    except ImportError:
        logger.warning("Enhanced config not available, using default settings")
        return settings.DATABASE_URL

# Create engine with optimized configuration
database_url = get_database_url()
database_config = settings.database_config

# Choose pool class based on database type
if "sqlite" in database_url:
    poolclass = NullPool  # SQLite doesn't support connection pooling
else:
    poolclass = QueuePool

engine = create_engine(
    database_url,
    poolclass=poolclass,
    **database_config,
    echo=settings.DEBUG,  # Enable SQL logging in debug mode
    connect_args={
        "options": "-c timezone=utc",
        "application_name": "opinion_market_api",
    } if "postgresql" in database_url else {}
)

# Add connection event listeners for monitoring
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance"""
    if "sqlite" in database_url:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log connection checkout for monitoring"""
    logger.debug("Connection checked out from pool")

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log connection checkin for monitoring"""
    logger.debug("Connection checked in to pool")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis connection for caching and session management
redis_client: Optional[redis.Redis] = None

def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client with proper configuration"""
    global redis_client
    if redis_client is None and settings.ENABLE_CACHING:
        try:
            redis_config = settings.redis_config
            redis_client = redis.from_url(
                settings.REDIS_URL,
                **redis_config,
                decode_responses=True
            )
            # Test connection
            redis_client.ping()
            logger.info("Redis connection established")
        except RedisError as e:
            logger.warning(f"Redis connection failed: {e}")
            redis_client = None
    return redis_client

def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session with proper cleanup and monitoring"""
    db = SessionLocal()
    start_time = time.time()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()
        duration = time.time() - start_time
        if duration > 1.0:  # Log slow queries
            logger.warning(f"Slow database session: {duration:.2f}s")

@contextmanager
def get_db_session():
    """Context manager for database sessions with automatic cleanup"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        db.close()

def init_database():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

def check_database_health() -> dict:
    """Check database health and return status"""
    health_status = {
        "status": "healthy",
        "database": "unknown",
        "connection_pool": "unknown",
        "response_time": 0.0
    }
    
    try:
        start_time = time.time()
        with get_db_session() as db:
            # Simple query to test connection
            db.execute("SELECT 1")
            health_status["response_time"] = time.time() - start_time
            health_status["status"] = "healthy"
            health_status["database"] = "connected"
            
            # Check connection pool status
            pool = engine.pool
            health_status["connection_pool"] = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
            
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        logger.error(f"Database health check failed: {e}")
    
    return health_status

def check_redis_health() -> dict:
    """Check Redis health and return status"""
    health_status = {
        "status": "unknown",
        "response_time": 0.0
    }
    
    redis_client = get_redis_client()
    if redis_client is None:
        health_status["status"] = "disabled"
        return health_status
    
    try:
        start_time = time.time()
        redis_client.ping()
        health_status["response_time"] = time.time() - start_time
        health_status["status"] = "healthy"
        
        # Get Redis info
        info = redis_client.info()
        health_status["info"] = {
            "version": info.get("redis_version"),
            "used_memory": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
            "total_commands_processed": info.get("total_commands_processed")
        }
        
    except RedisError as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        logger.error(f"Redis health check failed: {e}")
    
    return health_status

# Initialize Redis connection on module load
if settings.ENABLE_CACHING:
    get_redis_client()
