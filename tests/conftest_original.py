import pytest
import asyncio
import json
import tempfile
import os
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import redis.asyncio as redis
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.config import settings
from app.core.database import get_db, Base
from app.services.performance_optimization import PerformanceOptimizer
from app.services.enterprise_security import EnterpriseSecurityService
from app.services.market_data_feed import MarketDataFeed
from app.services.machine_learning import MachineLearningService
from app.services.blockchain_integration import BlockchainIntegration
from app.services.social_features import SocialFeaturesService
from app.services.advanced_orders import AdvancedOrderManager
from app.services.system_monitor import SystemMonitor

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test.db"


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
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def test_db(test_engine):
    """Create test database tables."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session(test_engine, test_db):
    """Create a new database session for a test."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session) -> TestClient:
    """Create a test client with database session."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
async def redis_client():
    """Create a test Redis client."""
    redis_client = redis.from_url(
        "redis://localhost:6379/1"
    )  # Use database 1 for testing
    await redis_client.ping()
    yield redis_client
    await redis_client.flushdb()  # Clean up after tests
    await redis_client.close()


@pytest.fixture
async def performance_optimizer(redis_client):
    """Create a test performance optimizer."""
    optimizer = PerformanceOptimizer()
    await optimizer.initialize("redis://localhost:6379/1", TEST_DATABASE_URL)
    yield optimizer


@pytest.fixture
async def enterprise_security(redis_client):
    """Create a test enterprise security service."""
    security = EnterpriseSecurityService()
    await security.initialize("redis://localhost:6379/1")
    yield security


@pytest.fixture
async def market_data_feed(redis_client):
    """Create a test market data feed."""
    feed = MarketDataFeed()
    await feed.initialize("redis://localhost:6379/1")
    yield feed


@pytest.fixture
async def ml_service(redis_client):
    """Create a test machine learning service."""
    ml_service = MachineLearningService()
    await ml_service.initialize("redis://localhost:6379/1")
    yield ml_service


@pytest.fixture
async def blockchain_integration(redis_client):
    """Create a test blockchain integration service."""
    blockchain = BlockchainIntegration()
    await blockchain.initialize("redis://localhost:6379/1")
    yield blockchain


@pytest.fixture
async def social_features(redis_client):
    """Create a test social features service."""
    social = SocialFeaturesService()
    await social.initialize("redis://localhost:6379/1")
    yield social


@pytest.fixture
async def advanced_orders(redis_client):
    """Create a test advanced order manager."""
    orders = AdvancedOrderManager()
    await orders.initialize("redis://localhost:6379/1")
    yield orders


@pytest.fixture
async def system_monitor(redis_client):
    """Create a test system monitor."""
    monitor = SystemMonitor()
    await monitor.initialize("redis://localhost:6379/1")
    yield monitor


@pytest.fixture
def sample_user_data() -> Dict[str, Any]:
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "display_name": "Test User",
        "bio": "Test user bio",
    }


@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """Sample market data for testing."""
    return {
        "title": "Will Bitcoin reach $100,000 by end of 2024?",
        "description": "Test prediction market",
        "outcome_a": "Yes",
        "outcome_b": "No",
        "category": "cryptocurrency",
        "end_date": "2024-12-31T23:59:59Z",
        "initial_liquidity": 1000.0,
    }


@pytest.fixture
def sample_trade_data() -> Dict[str, Any]:
    """Sample trade data for testing."""
    return {
        "market_id": 1,
        "outcome": "outcome_a",
        "shares": 10.0,
        "order_type": "market",
    }


@pytest.fixture
def sample_social_post_data() -> Dict[str, Any]:
    """Sample social post data for testing."""
    return {
        "content": "Test social post content",
        "post_type": "general",
        "tags": ["test", "social"],
    }


@pytest.fixture
def sample_advanced_order_data() -> Dict[str, Any]:
    """Sample advanced order data for testing."""
    return {
        "market_id": 1,
        "order_type": "stop_loss",
        "trigger_price": 0.6,
        "shares": 5.0,
        "outcome": "outcome_a",
    }


@pytest.fixture
def mock_external_services():
    """Mock external services for testing."""
    with pytest.MonkeyPatch().context() as m:
        # Mock external API calls
        m.setattr("app.services.blockchain_integration.requests.get", AsyncMock())
        m.setattr("app.services.market_data_feed.requests.get", AsyncMock())
        m.setattr("app.services.machine_learning.requests.get", AsyncMock())
        yield m


@pytest.fixture
def test_files():
    """Create temporary test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = {
            "config": os.path.join(temp_dir, "test_config.json"),
            "data": os.path.join(temp_dir, "test_data.csv"),
            "model": os.path.join(temp_dir, "test_model.pkl"),
        }

        # Write test data
        with open(test_files["config"], "w") as f:
            json.dump({"test": "config"}, f)

        with open(test_files["data"], "w") as f:
            f.write("id,name,value\n1,test1,100\n2,test2,200\n")

        yield test_files


@pytest.fixture
def auth_headers(client, sample_user_data):
    """Get authentication headers for testing."""
    # Register user
    response = client.post("/v1/auth/register", json=sample_user_data)
    assert response.status_code == 201

    # Login to get token
    login_data = {
        "username": sample_user_data["username"],
        "password": sample_user_data["password"],
    }
    response = client.post("/v1/auth/login", data=login_data)
    assert response.status_code == 200

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(client, sample_user_data):
    """Get admin authentication headers for testing."""
    # Create admin user
    admin_data = sample_user_data.copy()
    admin_data["username"] = "admin"
    admin_data["email"] = "admin@example.com"
    admin_data["is_admin"] = True

    # Register admin user
    response = client.post("/v1/auth/register", json=admin_data)
    assert response.status_code == 201

    # Login to get token
    login_data = {
        "username": admin_data["username"],
        "password": admin_data["password"],
    }
    response = client.post("/v1/auth/login", data=login_data)
    assert response.status_code == 200

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_market_id(client, auth_headers, sample_market_data):
    """Create a sample market and return its ID."""
    response = client.post("/v1/markets", json=sample_market_data, headers=auth_headers)
    assert response.status_code == 201
    return response.json()["id"]


@pytest.fixture
def sample_user_id(client, auth_headers):
    """Get the current user ID from the authenticated session."""
    response = client.get("/v1/users/me", headers=auth_headers)
    assert response.status_code == 200
    return response.json()["id"]


@pytest.fixture
def websocket_client():
    """Create a WebSocket test client."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        with client.websocket_connect("/v1/ws/market-data") as websocket:
            yield websocket


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.incr.return_value = 1
    mock_redis.lpush.return_value = 1
    mock_redis.ltrim.return_value = True
    mock_redis.sismember.return_value = False
    mock_redis.flushdb.return_value = True
    mock_redis.close.return_value = None
    return mock_redis


@pytest.fixture
def mock_database():
    """Mock database session for testing."""
    mock_session = AsyncMock()
    mock_session.commit.return_value = None
    mock_session.rollback.return_value = None
    mock_session.close.return_value = None
    return mock_session


@pytest.fixture
def mock_external_api():
    """Mock external API responses."""
    mock_responses = {
        "market_data": {"price": 0.65, "volume": 1000.0, "liquidity": 5000.0},
        "ml_prediction": {
            "predicted_price": 0.68,
            "confidence": 0.85,
            "horizon": "24h",
        },
        "blockchain_tx": {
            "tx_hash": "0x1234567890abcdef",
            "status": "confirmed",
            "block_number": 12345,
        },
    }
    return mock_responses


@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        "database_url": TEST_DATABASE_URL,
        "redis_url": "redis://localhost:6379/1",
        "secret_key": "test-secret-key",
        "algorithm": "HS256",
        "access_token_expire_minutes": 30,
        "encryption_key": "test-encryption-key",
        "environment": "test",
    }


@pytest.fixture
def performance_test_data():
    """Sample data for performance testing."""
    return {
        "users": [
            {
                "username": f"user{i}",
                "email": f"user{i}@example.com",
                "password": "password123",
            }
            for i in range(100)
        ],
        "markets": [
            {
                "title": f"Test Market {i}",
                "description": f"Test market description {i}",
                "outcome_a": "Yes",
                "outcome_b": "No",
                "category": "test",
                "end_date": "2024-12-31T23:59:59Z",
                "initial_liquidity": 1000.0,
            }
            for i in range(50)
        ],
        "trades": [
            {
                "market_id": i % 10 + 1,
                "outcome": "outcome_a" if i % 2 == 0 else "outcome_b",
                "shares": 10.0 + (i * 0.1),
                "order_type": "market",
            }
            for i in range(200)
        ],
    }


@pytest.fixture
def security_test_data():
    """Sample data for security testing."""
    return {
        "malicious_requests": [
            {"sql_injection": "'; DROP TABLE users; --"},
            {"xss": "<script>alert('xss')</script>"},
            {"path_traversal": "../../../etc/passwd"},
            {"rate_limit": {"requests": 1000, "timeframe": "1s"}},
        ],
        "valid_requests": [
            {"normal_request": "valid data"},
            {"safe_input": "safe user input"},
            {"legitimate_trade": {"market_id": 1, "shares": 10.0}},
        ],
    }


@pytest.fixture
def load_test_scenarios():
    """Load testing scenarios."""
    return {
        "light_load": {"users": 10, "requests_per_user": 10, "duration": 60},
        "medium_load": {"users": 50, "requests_per_user": 20, "duration": 120},
        "heavy_load": {"users": 100, "requests_per_user": 50, "duration": 300},
    }


@pytest.fixture
def integration_test_data():
    """Data for integration testing."""
    return {
        "user_workflow": [
            {
                "action": "register",
                "data": {"username": "testuser", "email": "test@example.com"},
            },
            {
                "action": "login",
                "data": {"username": "testuser", "password": "password123"},
            },
            {
                "action": "create_market",
                "data": {"title": "Test Market", "outcome_a": "Yes", "outcome_b": "No"},
            },
            {
                "action": "place_trade",
                "data": {"market_id": 1, "shares": 10.0, "outcome": "outcome_a"},
            },
            {
                "action": "create_post",
                "data": {"content": "Test post", "post_type": "general"},
            },
        ],
        "admin_workflow": [
            {"action": "login", "data": {"username": "admin", "password": "admin123"}},
            {"action": "verify_market", "data": {"market_id": 1, "status": "verified"}},
            {
                "action": "resolve_dispute",
                "data": {"dispute_id": 1, "resolution": "resolved"},
            },
            {"action": "view_analytics", "data": {"timeframe": "24h"}},
        ],
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Mark slow tests
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if "performance" in item.keywords:
            item.add_marker(pytest.mark.performance)

        # Mark security tests
        if "security" in item.keywords:
            item.add_marker(pytest.mark.security)


# Test utilities
class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def create_test_user(client, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a test user and return user data."""
        response = client.post("/v1/auth/register", json=user_data)
        assert response.status_code == 201
        return response.json()

    @staticmethod
    def login_user(client, username: str, password: str) -> str:
        """Login user and return access token."""
        login_data = {"username": username, "password": password}
        response = client.post("/v1/auth/login", data=login_data)
        assert response.status_code == 200
        return response.json()["access_token"]

    @staticmethod
    def create_test_market(
        client, headers: Dict[str, str], market_data: Dict[str, Any]
    ) -> int:
        """Create a test market and return market ID."""
        response = client.post("/v1/markets", json=market_data, headers=headers)
        assert response.status_code == 201
        return response.json()["id"]

    @staticmethod
    def place_test_trade(
        client, headers: Dict[str, str], trade_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Place a test trade and return trade data."""
        response = client.post("/v1/trades", json=trade_data, headers=headers)
        assert response.status_code == 201
        return response.json()

    @staticmethod
    def create_test_post(
        client, headers: Dict[str, str], post_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a test social post and return post data."""
        response = client.post("/v1/social/posts", json=post_data, headers=headers)
        assert response.status_code == 201
        return response.json()


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils
