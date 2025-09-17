from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
import os

# Use enhanced configuration if available, fallback to settings
def get_database_url():
    try:
        from app.core.enhanced_config import enhanced_config_manager
        return enhanced_config_manager.get("database.url", settings.DATABASE_URL)
    except:
        return settings.DATABASE_URL

# Create engine with dynamic configuration
database_url = get_database_url()
engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
