"""
Pytest configuration and fixtures for Opinion Market tests
"""

import asyncio
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import tempfile
import os
from datetime import datetime, timedelta
from typing import Generator, AsyncGenerator

from app.main import app
from app.core.database import get_db, Base
from app.core.config import settings
from app.models.user import User
from app.models.market import Market, MarketCategory, MarketStatus
from app.models.trade import Trade, TradeType, TradeOutcome, TradeStatus
from app.core.security import security_manager


# Test database configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def db_session() -> Generator:
    """Create a fresh database session for each test."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        # Drop tables
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session) -> Generator:
    """Create a test client with database dependency override."""
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
def test_user(db_session) -> User:
    """Create a test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=security_manager.hash_password("testpassword"),
        full_name="Test User",
        is_active=True,
        is_verified=True,
        preferences={},
        notification_settings={}
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_market(db_session, test_user) -> Market:
    """Create a test market."""
    market = Market(
        title="Test Market",
        description="A test market for testing purposes",
        question="Will this test pass?",
        category=MarketCategory.POLITICS,
        outcome_a="Yes",
        outcome_b="No",
        creator_id=test_user.id,
        closes_at=datetime.utcnow() + timedelta(days=7),
        status=MarketStatus.OPEN,
        price_a=0.5,
        price_b=0.5,
        volume_total=0.0,
        volume_24h=0.0,
        trending_score=0.0,
        sentiment_score=0.0,
        liquidity_score=0.0,
        risk_score=0.0
    )
    db_session.add(market)
    db_session.commit()
    db_session.refresh(market)
    return market


@pytest.fixture
def test_trade(db_session, test_user, test_market) -> Trade:
    """Create a test trade."""
    trade = Trade(
        trade_type=TradeType.BUY,
        outcome=TradeOutcome.OUTCOME_A,
        amount=10.0,
        price_a=0.5,
        price_b=0.5,
        price_per_share=0.5,
        total_value=5.0,
        fee=0.1,
        market_id=test_market.id,
        user_id=test_user.id,
        status=TradeStatus.COMPLETED,
        trade_hash="test_trade_hash"
    )
    db_session.add(trade)
    db_session.commit()
    db_session.refresh(trade)
    return trade


@pytest.fixture
def auth_headers(test_user) -> dict:
    """Create authentication headers for test user."""
    access_token = security_manager.create_access_token(
        data={"sub": test_user.id, "username": test_user.username}
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"test content")
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        def get(self, key):
            return self.data.get(key)
        
        def set(self, key, value, ex=None):
            self.data[key] = value
            return True
        
        def delete(self, key):
            if key in self.data:
                del self.data[key]
                return 1
            return 0
        
        def exists(self, key):
            return 1 if key in self.data else 0
        
        def incr(self, key):
            if key not in self.data:
                self.data[key] = 0
            self.data[key] += 1
            return self.data[key]
        
        def expire(self, key, time):
            return True
        
        def hset(self, name, mapping=None, **kwargs):
            if name not in self.data:
                self.data[name] = {}
            if mapping:
                self.data[name].update(mapping)
            if kwargs:
                self.data[name].update(kwargs)
            return len(kwargs) + (len(mapping) if mapping else 0)
        
        def hget(self, name, key):
            return self.data.get(name, {}).get(key)
        
        def hgetall(self, name):
            return self.data.get(name, {})
        
        def lpush(self, name, *values):
            if name not in self.data:
                self.data[name] = []
            self.data[name] = list(values) + self.data[name]
            return len(self.data[name])
        
        def ping(self):
            return True
        
        def info(self):
            return {
                "redis_version": "6.0.0",
                "used_memory_human": "1M",
                "connected_clients": 1,
                "total_commands_processed": 100
            }
    
    return MockRedis()


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing."""
    class MockWebSocket:
        def __init__(self):
            self.client_state = "connected"
            self.sent_messages = []
        
        async def accept(self):
            pass
        
        async def receive_text(self):
            return '{"type": "ping", "data": {}}'
        
        async def send_text(self, data):
            self.sent_messages.append(data)
        
        async def close(self):
            self.client_state = "disconnected"
    
    return MockWebSocket()


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "title": "Sample Market",
        "description": "A sample market for testing",
        "question": "Will this sample market work?",
        "category": "politics",
        "outcome_a": "Yes",
        "outcome_b": "No",
        "closes_at": (datetime.utcnow() + timedelta(days=7)).isoformat(),
        "resolution_criteria": "Based on official results",
        "initial_liquidity": 1000.0,
        "trading_fee": 0.02
    }


@pytest.fixture
def sample_trade_data():
    """Sample trade data for testing."""
    return {
        "trade_type": "buy",
        "outcome": "outcome_a",
        "amount": 10.0,
        "market_id": 1
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "username": "sampleuser",
        "email": "sample@example.com",
        "password": "SamplePassword123!",
        "full_name": "Sample User",
        "bio": "A sample user for testing"
    }


@pytest.fixture
def multiple_test_users(db_session) -> list:
    """Create multiple test users."""
    users = []
    for i in range(5):
        user = User(
            username=f"testuser{i}",
            email=f"test{i}@example.com",
            hashed_password=security_manager.hash_password("testpassword"),
            full_name=f"Test User {i}",
            is_active=True,
            is_verified=True,
            preferences={},
            notification_settings={}
        )
        db_session.add(user)
        users.append(user)
    
    db_session.commit()
    for user in users:
        db_session.refresh(user)
    
    return users


@pytest.fixture
def multiple_test_markets(db_session, multiple_test_users) -> list:
    """Create multiple test markets."""
    markets = []
    categories = [MarketCategory.POLITICS, MarketCategory.SPORTS, MarketCategory.ECONOMICS]
    
    for i, user in enumerate(multiple_test_users):
        market = Market(
            title=f"Test Market {i}",
            description=f"Test market {i} description",
            question=f"Will test market {i} work?",
            category=categories[i % len(categories)],
            outcome_a="Yes",
            outcome_b="No",
            creator_id=user.id,
            closes_at=datetime.utcnow() + timedelta(days=7),
            status=MarketStatus.OPEN,
            price_a=0.5,
            price_b=0.5,
            volume_total=0.0,
            volume_24h=0.0,
            trending_score=0.0,
            sentiment_score=0.0,
            liquidity_score=0.0,
            risk_score=0.0
        )
        db_session.add(market)
        markets.append(market)
    
    db_session.commit()
    for market in markets:
        db_session.refresh(market)
    
    return markets


@pytest.fixture
def multiple_test_trades(db_session, multiple_test_users, multiple_test_markets) -> list:
    """Create multiple test trades."""
    trades = []
    
    for i, (user, market) in enumerate(zip(multiple_test_users, multiple_test_markets)):
        trade = Trade(
            trade_type=TradeType.BUY if i % 2 == 0 else TradeType.SELL,
            outcome=TradeOutcome.OUTCOME_A if i % 2 == 0 else TradeOutcome.OUTCOME_B,
            amount=10.0 + i,
            price_a=0.5,
            price_b=0.5,
            price_per_share=0.5,
            total_value=(10.0 + i) * 0.5,
            fee=0.1,
            market_id=market.id,
            user_id=user.id,
            status=TradeStatus.COMPLETED,
            trade_hash=f"test_trade_hash_{i}"
        )
        db_session.add(trade)
        trades.append(trade)
    
    db_session.commit()
    for trade in trades:
        db_session.refresh(trade)
    
    return trades


# Async fixtures for async tests
@pytest_asyncio.fixture
async def async_client():
    """Create an async test client."""
    from httpx import AsyncClient
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def async_db_session():
    """Create an async database session."""
    # This would be used with async database drivers
    # For now, we'll use the sync session
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


# Performance testing fixtures
@pytest.fixture
def performance_test_data():
    """Generate large datasets for performance testing."""
    return {
        "users": 1000,
        "markets": 500,
        "trades": 10000
    }


# Integration test fixtures
@pytest.fixture
def integration_test_setup(db_session):
    """Setup for integration tests."""
    # Create test data for integration tests
    users = []
    markets = []
    trades = []
    
    # Create users
    for i in range(10):
        user = User(
            username=f"integration_user_{i}",
            email=f"integration{i}@example.com",
            hashed_password=security_manager.hash_password("testpassword"),
            full_name=f"Integration User {i}",
            is_active=True,
            is_verified=True,
            preferences={},
            notification_settings={}
        )
        db_session.add(user)
        users.append(user)
    
    db_session.commit()
    for user in users:
        db_session.refresh(user)
    
    # Create markets
    for i in range(5):
        market = Market(
            title=f"Integration Market {i}",
            description=f"Integration test market {i}",
            question=f"Will integration test {i} pass?",
            category=MarketCategory.POLITICS,
            outcome_a="Yes",
            outcome_b="No",
            creator_id=users[i].id,
            closes_at=datetime.utcnow() + timedelta(days=7),
            status=MarketStatus.OPEN,
            price_a=0.5,
            price_b=0.5,
            volume_total=0.0,
            volume_24h=0.0,
            trending_score=0.0,
            sentiment_score=0.0,
            liquidity_score=0.0,
            risk_score=0.0
        )
        db_session.add(market)
        markets.append(market)
    
    db_session.commit()
    for market in markets:
        db_session.refresh(market)
    
    # Create trades
    for i in range(20):
        trade = Trade(
            trade_type=TradeType.BUY if i % 2 == 0 else TradeType.SELL,
            outcome=TradeOutcome.OUTCOME_A if i % 2 == 0 else TradeOutcome.OUTCOME_B,
            amount=10.0 + i,
            price_a=0.5,
            price_b=0.5,
            price_per_share=0.5,
            total_value=(10.0 + i) * 0.5,
            fee=0.1,
            market_id=markets[i % len(markets)].id,
            user_id=users[i % len(users)].id,
            status=TradeStatus.COMPLETED,
            trade_hash=f"integration_trade_hash_{i}"
        )
        db_session.add(trade)
        trades.append(trade)
    
    db_session.commit()
    for trade in trades:
        db_session.refresh(trade)
    
    return {
        "users": users,
        "markets": markets,
        "trades": trades
    }