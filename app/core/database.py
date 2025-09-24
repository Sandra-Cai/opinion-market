from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool
from app.core.config import settings
import logging
import os

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
engine = create_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False,  # Set to True for SQL debugging
    connect_args={
        "options": "-c timezone=utc",
        "application_name": "opinion_market_api",
    } if "postgresql" in database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency to get database session with proper cleanup"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()
