"""
Comprehensive API Test Suite
Tests all major API endpoints and functionality
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import get_db, Base
from app.models import user, market, trade, vote, position, order

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)

# Test data
TEST_USER = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPassword123!",
    "full_name": "Test User"
}

TEST_MARKET = {
    "title": "Test Market: Will it work?",
    "description": "A test market for API testing",
    "outcome_a": "Yes, it will work",
    "outcome_b": "No, it won't work",
    "end_date": "2024-12-31T23:59:59Z",
    "category": "test",
    "tags": ["test", "api", "comprehensive"]
}

TEST_TRADE = {
    "market_id": 1,
    "trade_type": "buy",
    "outcome": "outcome_a",
    "amount": 100.0,
    "price_per_share": 0.5
}

class TestAuthentication:
    """Test authentication endpoints"""
    
    def test_register_user(self):
        """Test user registration"""
        response = client.post("/api/v1/auth/register", json=TEST_USER)
        assert response.status_code == 201
        data = response.json()
        assert "user_id" in data
        assert "access_token" in data
        assert data["username"] == TEST_USER["username"]
        assert data["email"] == TEST_USER["email"]
    
    def test_login_user(self):
        """Test user login"""
        # First register a user
        client.post("/api/v1/auth/register", json=TEST_USER)
        
        # Then login
        login_data = {
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "user" in data
        assert data["user"]["username"] == TEST_USER["username"]
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        login_data = {
            "username": "nonexistent",
            "password": "wrongpassword"
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401
    
    def test_get_current_user(self):
        """Test getting current user info"""
        # Register and login
        client.post("/api/v1/auth/register", json=TEST_USER)
        login_response = client.post("/api/v1/auth/login", json={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        })
        token = login_response.json()["access_token"]
        
        # Get current user
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == TEST_USER["username"]


class TestMarkets:
    """Test market endpoints"""
    
    def test_list_markets(self):
        """Test listing all markets"""
        response = client.get("/api/v1/markets/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_create_market(self):
        """Test creating a new market"""
        # Register and login
        client.post("/api/v1/auth/register", json=TEST_USER)
        login_response = client.post("/api/v1/auth/login", json={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        })
        token = login_response.json()["access_token"]
        
        # Create market
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post("/api/v1/markets/", json=TEST_MARKET, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert "market_id" in data
        assert data["title"] == TEST_MARKET["title"]
        assert data["status"] == "open"
    
    def test_get_market(self):
        """Test getting a specific market"""
        # Create a market first
        client.post("/api/v1/auth/register", json=TEST_USER)
        login_response = client.post("/api/v1/auth/login", json={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        })
        token = login_response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        create_response = client.post("/api/v1/markets/", json=TEST_MARKET, headers=headers)
        market_id = create_response.json()["market_id"]
        
        # Get the market
        response = client.get(f"/api/v1/markets/{market_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["market_id"] == market_id
        assert data["title"] == TEST_MARKET["title"]
    
    def test_get_nonexistent_market(self):
        """Test getting a nonexistent market"""
        response = client.get("/api/v1/markets/999")
        assert response.status_code == 404


class TestTrading:
    """Test trading endpoints"""
    
    def test_execute_trade(self):
        """Test executing a trade"""
        # Setup: Register user and create market
        client.post("/api/v1/auth/register", json=TEST_USER)
        login_response = client.post("/api/v1/auth/login", json={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        })
        token = login_response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        create_response = client.post("/api/v1/markets/", json=TEST_MARKET, headers=headers)
        market_id = create_response.json()["market_id"]
        
        # Execute trade
        trade_data = TEST_TRADE.copy()
        trade_data["market_id"] = market_id
        response = client.post("/api/v1/trades/", json=trade_data, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert "trade_id" in data
        assert data["market_id"] == market_id
        assert data["trade_type"] == trade_data["trade_type"]
        assert data["status"] == "executed"
    
    def test_list_trades(self):
        """Test listing trades"""
        # Setup: Register user and create market
        client.post("/api/v1/auth/register", json=TEST_USER)
        login_response = client.post("/api/v1/auth/login", json={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        })
        token = login_response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        create_response = client.post("/api/v1/markets/", json=TEST_MARKET, headers=headers)
        market_id = create_response.json()["market_id"]
        
        # Execute a trade
        trade_data = TEST_TRADE.copy()
        trade_data["market_id"] = market_id
        client.post("/api/v1/trades/", json=trade_data, headers=headers)
        
        # List trades
        response = client.get("/api/v1/trades/", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
    
    def test_create_order(self):
        """Test creating an order"""
        # Setup: Register user and create market
        client.post("/api/v1/auth/register", json=TEST_USER)
        login_response = client.post("/api/v1/auth/login", json={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        })
        token = login_response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        create_response = client.post("/api/v1/markets/", json=TEST_MARKET, headers=headers)
        market_id = create_response.json()["market_id"]
        
        # Create order
        order_data = {
            "market_id": market_id,
            "order_type": "limit",
            "side": "buy",
            "outcome": "outcome_a",
            "amount": 50.0,
            "limit_price": 0.4
        }
        response = client.post("/api/v1/orders/", json=order_data, headers=headers)
        assert response.status_code == 201
        data = response.json()
        assert "order_id" in data
        assert data["market_id"] == market_id
        assert data["order_type"] == "limit"
        assert data["status"] == "pending"


class TestAnalytics:
    """Test analytics endpoints"""
    
    def test_market_analytics(self):
        """Test market analytics endpoint"""
        # Setup: Create a market
        client.post("/api/v1/auth/register", json=TEST_USER)
        login_response = client.post("/api/v1/auth/login", json={
            "username": TEST_USER["username"],
            "password": TEST_USER["password"]
        })
        token = login_response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {token}"}
        create_response = client.post("/api/v1/markets/", json=TEST_MARKET, headers=headers)
        market_id = create_response.json()["market_id"]
        
        # Get analytics
        response = client.get(f"/api/v1/analytics/market/{market_id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "market_id" in data
        assert data["market_id"] == market_id
    
    def test_global_analytics(self):
        """Test global analytics endpoint"""
        response = client.get("/api/v1/analytics/global")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestHealth:
    """Test health and monitoring endpoints"""
    
    def test_basic_health_check(self):
        """Test basic health check"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "timestamp" in data
    
    def test_detailed_health_check(self):
        """Test detailed health check"""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "system" in data
    
    def test_readiness_check(self):
        """Test readiness check"""
        response = client.get("/api/v1/health/readiness")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "checks" in data
    
    def test_liveness_check(self):
        """Test liveness check"""
        response = client.get("/api/v1/health/liveness")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] == True
    
    def test_health_metrics(self):
        """Test health metrics"""
        response = client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "system_metrics" in data
        assert "application_metrics" in data


class TestDocumentation:
    """Test API documentation endpoints"""
    
    def test_api_overview(self):
        """Test API overview endpoint"""
        response = client.get("/api/v1/docs/overview")
        assert response.status_code == 200
        data = response.json()
        assert "api_name" in data
        assert "version" in data
        assert "endpoints" in data
        assert "features" in data
    
    def test_api_examples(self):
        """Test API examples endpoint"""
        response = client.get("/api/v1/docs/examples")
        assert response.status_code == 200
        data = response.json()
        assert "examples" in data
        assert "authentication" in data["examples"]
        assert "markets" in data["examples"]
        assert "trading" in data["examples"]
    
    def test_status_codes(self):
        """Test status codes documentation"""
        response = client.get("/api/v1/docs/status-codes")
        assert response.status_code == 200
        data = response.json()
        assert "status_codes" in data
        assert "success" in data["status_codes"]
        assert "client_errors" in data["status_codes"]
        assert "server_errors" in data["status_codes"]
    
    def test_authentication_guide(self):
        """Test authentication guide"""
        response = client.get("/api/v1/docs/authentication")
        assert response.status_code == 200
        data = response.json()
        assert "authentication" in data
        assert "overview" in data["authentication"]
        assert "flow" in data["authentication"]
        assert "endpoints" in data["authentication"]
    
    def test_rate_limits(self):
        """Test rate limits documentation"""
        response = client.get("/api/v1/docs/rate-limits")
        assert response.status_code == 200
        data = response.json()
        assert "rate_limits" in data
        assert "overview" in data["rate_limits"]
        assert "limits" in data["rate_limits"]
    
    def test_interactive_docs(self):
        """Test interactive documentation"""
        response = client.get("/api/v1/docs/interactive")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Opinion Market API" in response.text
    
    def test_error_handling_guide(self):
        """Test error handling guide"""
        response = client.get("/api/v1/docs/errors")
        assert response.status_code == 200
        data = response.json()
        assert "error_handling" in data
        assert "overview" in data["error_handling"]
        assert "common_errors" in data["error_handling"]


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_unauthorized_access(self):
        """Test accessing protected endpoint without authentication"""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401
    
    def test_invalid_token(self):
        """Test accessing protected endpoint with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 401
    
    def test_malformed_json(self):
        """Test sending malformed JSON"""
        response = client.post("/api/v1/auth/register", 
                             data="invalid json",
                             headers={"Content-Type": "application/json"})
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test missing required fields"""
        incomplete_user = {"username": "test"}
        response = client.post("/api/v1/auth/register", json=incomplete_user)
        assert response.status_code == 422


# Setup and teardown
@pytest.fixture(scope="module")
def setup_database():
    """Setup test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
