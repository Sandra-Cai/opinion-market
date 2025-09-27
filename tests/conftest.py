"""
Pytest Configuration and Fixtures
Advanced testing configuration for Opinion Market platform
"""

import pytest
import asyncio
import os
import tempfile
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import get_db, Base
from app.core.enhanced_cache import enhanced_cache
from app.core.security_manager import security_manager
from app.services.service_registry import service_registry
from app.services.inter_service_communication import inter_service_comm
from app.events.event_store import event_store
from app.events.event_bus import event_bus

# Test configuration
TEST_DATABASE_URL = "sqlite:///./test.db"
TEST_REDIS_URL = "redis://localhost:6379/1"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
def test_session_factory(test_engine):
    """Create test session factory."""
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def test_db(test_session_factory):
    """Create test database session."""
    session = test_session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def test_client(test_db):
    """Create test client with database override."""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def test_cache():
    """Create test cache instance."""
    # Create a temporary cache for testing
    test_cache = enhanced_cache
    await test_cache.clear()
    yield test_cache
    await test_cache.clear()


@pytest.fixture(scope="function")
async def test_security_manager():
    """Create test security manager."""
    # Reset security manager state
    security_manager.security_events.clear()
    security_manager.blocked_ips.clear()
    security_manager.rate_limits.clear()
    yield security_manager


@pytest.fixture(scope="function")
async def test_service_registry():
    """Create test service registry."""
    # Reset service registry
    service_registry.services.clear()
    yield service_registry


@pytest.fixture(scope="function")
async def test_inter_service_comm():
    """Create test inter-service communication."""
    # Initialize test communication
    await inter_service_comm.initialize()
    yield inter_service_comm
    await inter_service_comm.cleanup()


@pytest.fixture(scope="function")
async def test_event_store():
    """Create test event store."""
    # Clear event store
    event_store.events_cache.clear()
    event_store.snapshots_cache.clear()
    yield event_store


@pytest.fixture(scope="function")
async def test_event_bus():
    """Create test event bus."""
    # Reset event bus
    event_bus.subscriptions.clear()
    event_bus.published_events = 0
    event_bus.failed_events = 0
    yield event_bus


@pytest.fixture(scope="function")
def test_user_data():
    """Create test user data."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }


@pytest.fixture(scope="function")
def test_market_data():
    """Create test market data."""
    return {
        "title": "Test Market",
        "description": "A test prediction market",
        "category": "test",
        "market_type": "binary",
        "outcomes": ["Yes", "No"],
        "end_date": "2024-12-31T23:59:59Z"
    }


@pytest.fixture(scope="function")
def test_trade_data():
    """Create test trade data."""
    return {
        "market_id": 1,
        "outcome": "Yes",
        "amount": 100.0,
        "order_type": "market",
        "side": "buy"
    }


@pytest.fixture(scope="function")
def mock_external_services():
    """Mock external services for testing."""
    class MockExternalServices:
        def __init__(self):
            self.mock_responses = {}
        
        def mock_http_response(self, url: str, response_data: dict, status_code: int = 200):
            self.mock_responses[url] = {
                "data": response_data,
                "status_code": status_code
            }
        
        def get_mock_response(self, url: str):
            return self.mock_responses.get(url, {"data": {}, "status_code": 404})
    
    return MockExternalServices()


@pytest.fixture(scope="function")
def test_data_factory():
    """Create test data factory."""
    class TestDataFactory:
        @staticmethod
        def create_user(**kwargs):
            default_user = {
                "username": "testuser",
                "email": "test@example.com",
                "password": "testpassword123",
                "full_name": "Test User"
            }
            default_user.update(kwargs)
            return default_user
        
        @staticmethod
        def create_market(**kwargs):
            default_market = {
                "title": "Test Market",
                "description": "A test prediction market",
                "category": "test",
                "market_type": "binary",
                "outcomes": ["Yes", "No"],
                "end_date": "2024-12-31T23:59:59Z"
            }
            default_market.update(kwargs)
            return default_market
        
        @staticmethod
        def create_trade(**kwargs):
            default_trade = {
                "market_id": 1,
                "outcome": "Yes",
                "amount": 100.0,
                "order_type": "market",
                "side": "buy"
            }
            default_trade.update(kwargs)
            return default_trade
        
        @staticmethod
        def create_event(**kwargs):
            default_event = {
                "event_type": "test_event",
                "aggregate_id": "test_aggregate",
                "aggregate_type": "test_type",
                "event_data": {"test": "data"},
                "metadata": {"source": "test"}
            }
            default_event.update(kwargs)
            return default_event
    
    return TestDataFactory()


# Performance testing fixtures
@pytest.fixture(scope="function")
def performance_test_config():
    """Configuration for performance tests."""
    return {
        "concurrent_users": 100,
        "test_duration": 60,  # seconds
        "ramp_up_time": 10,   # seconds
        "target_response_time": 100,  # milliseconds
        "max_error_rate": 0.01  # 1%
    }


@pytest.fixture(scope="function")
def load_test_data():
    """Generate load test data."""
    return {
        "users": [f"user_{i}" for i in range(1000)],
        "markets": [f"market_{i}" for i in range(100)],
        "trades": [f"trade_{i}" for i in range(10000)]
    }


# Integration testing fixtures
@pytest.fixture(scope="function")
async def integration_test_setup():
    """Setup for integration tests."""
    # Initialize all services
    await enhanced_cache.initialize()
    await security_manager.start_security_monitoring()
    await service_registry.start_health_monitoring()
    await inter_service_comm.initialize()
    
    yield {
        "cache": enhanced_cache,
        "security": security_manager,
        "registry": service_registry,
        "communication": inter_service_comm
    }
    
    # Cleanup
    await enhanced_cache.cleanup()
    await security_manager.stop_security_monitoring()
    await inter_service_comm.cleanup()


# Mock fixtures for external dependencies
@pytest.fixture(scope="function")
def mock_database():
    """Mock database for testing."""
    class MockDatabase:
        def __init__(self):
            self.data = {}
        
        def get(self, key):
            return self.data.get(key)
        
        def set(self, key, value):
            self.data[key] = value
        
        def delete(self, key):
            self.data.pop(key, None)
        
        def clear(self):
            self.data.clear()
    
    return MockDatabase()


@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis for testing."""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        async def get(self, key):
            return self.data.get(key)
        
        async def set(self, key, value, ex=None):
            self.data[key] = value
        
        async def delete(self, key):
            self.data.pop(key, None)
        
        async def clear(self):
            self.data.clear()
    
    return MockRedis()


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker for tests that take longer
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add external marker for tests requiring external services
        if "external" in item.name or "api" in item.name:
            item.add_marker(pytest.mark.external)