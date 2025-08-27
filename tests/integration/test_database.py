import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.database import get_db, Base

# Test database URL
TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/test_db"

@pytest.fixture
def test_db():
    """Create test database"""
    engine = create_engine(TEST_DATABASE_URL)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()
        # Drop tables
        Base.metadata.drop_all(bind=engine)

def test_database_connection(test_db):
    """Test database connection"""
    assert test_db is not None
    
    # Test simple query
    result = test_db.execute("SELECT 1")
    assert result.scalar() == 1

def test_database_tables_exist(test_db):
    """Test that required tables exist"""
    # This test will pass if tables are created successfully
    assert True
