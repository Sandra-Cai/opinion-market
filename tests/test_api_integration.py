"""
API Integration Test Suite
Tests complete user workflows and API integrations
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import get_db, Base
from app.models import user, market, trade, vote, position, order

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_integration.db"
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

class TestCompleteUserWorkflow:
    """Test complete user workflow from registration to trading"""
    
    def setup_method(self):
        """Setup test data for each test"""
        self.test_user = {
            "username": f"testuser_{pytest.current_test}",
            "email": f"test_{pytest.current_test}@example.com",
            "password": "TestPassword123!",
            "full_name": "Test User"
        }
        self.auth_token = None
        self.market_id = None
        self.trade_id = None
    
    def test_complete_user_workflow(self):
        """Test complete user workflow: register -> login -> create market -> trade -> vote"""
        print("\n🧪 Testing Complete User Workflow...")
        
        # Step 1: Register user
        print("   Step 1: Registering user...")
        response = client.post("/api/v1/auth/register", json=self.test_user)
        assert response.status_code == 201
        user_data = response.json()
        assert "user_id" in user_data
        assert "access_token" in user_data
        self.auth_token = user_data["access_token"]
        print(f"      ✅ User registered with ID: {user_data['user_id']}")
        
        # Step 2: Login user
        print("   Step 2: Logging in user...")
        login_data = {
            "username": self.test_user["username"],
            "password": self.test_user["password"]
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        login_response = response.json()
        assert "access_token" in login_response
        self.auth_token = login_response["access_token"]
        print("      ✅ User logged in successfully")
        
        # Step 3: Get current user info
        print("   Step 3: Getting current user info...")
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        user_info = response.json()
        assert user_info["username"] == self.test_user["username"]
        print(f"      ✅ User info retrieved: {user_info['username']}")
        
        # Step 4: Create a market
        print("   Step 4: Creating a market...")
        market_data = {
            "title": f"Integration Test Market: Will it work?",
            "description": "A market created for integration testing",
            "outcome_a": "Yes, it will work",
            "outcome_b": "No, it won't work",
            "end_date": "2024-12-31T23:59:59Z",
            "category": "test",
            "tags": ["integration", "test", "workflow"]
        }
        response = client.post("/api/v1/markets/", json=market_data, headers=headers)
        assert response.status_code == 201
        market_response = response.json()
        assert "market_id" in market_response
        self.market_id = market_response["market_id"]
        print(f"      ✅ Market created with ID: {self.market_id}")
        
        # Step 5: Get market details
        print("   Step 5: Getting market details...")
        response = client.get(f"/api/v1/markets/{self.market_id}")
        assert response.status_code == 200
        market_details = response.json()
        assert market_details["market_id"] == self.market_id
        assert market_details["title"] == market_data["title"]
        print(f"      ✅ Market details retrieved: {market_details['title']}")
        
        # Step 6: Execute a trade
        print("   Step 6: Executing a trade...")
        trade_data = {
            "market_id": self.market_id,
            "trade_type": "buy",
            "outcome": "outcome_a",
            "amount": 100.0,
            "price_per_share": 0.5
        }
        response = client.post("/api/v1/trades/", json=trade_data, headers=headers)
        assert response.status_code == 201
        trade_response = response.json()
        assert "trade_id" in trade_response
        self.trade_id = trade_response["trade_id"]
        print(f"      ✅ Trade executed with ID: {self.trade_id}")
        
        # Step 7: Create an order
        print("   Step 7: Creating an order...")
        order_data = {
            "market_id": self.market_id,
            "order_type": "limit",
            "side": "buy",
            "outcome": "outcome_b",
            "amount": 50.0,
            "limit_price": 0.4
        }
        response = client.post("/api/v1/orders/", json=order_data, headers=headers)
        assert response.status_code == 201
        order_response = response.json()
        assert "order_id" in order_response
        print(f"      ✅ Order created with ID: {order_response['order_id']}")
        
        # Step 8: Get user's trades
        print("   Step 8: Getting user's trades...")
        response = client.get("/api/v1/trades/", headers=headers)
        assert response.status_code == 200
        trades = response.json()
        assert isinstance(trades, list)
        assert len(trades) > 0
        print(f"      ✅ Found {len(trades)} trades for user")
        
        # Step 9: Get user's orders
        print("   Step 9: Getting user's orders...")
        response = client.get("/api/v1/orders/", headers=headers)
        assert response.status_code == 200
        orders = response.json()
        assert isinstance(orders, list)
        print(f"      ✅ Found {len(orders)} orders for user")
        
        # Step 10: Get market analytics
        print("   Step 10: Getting market analytics...")
        response = client.get(f"/api/v1/analytics/market/{self.market_id}", headers=headers)
        assert response.status_code == 200
        analytics = response.json()
        assert "market_id" in analytics
        print("      ✅ Market analytics retrieved")
        
        print("   🎉 Complete user workflow test passed!")

class TestMarketLifecycle:
    """Test market lifecycle from creation to resolution"""
    
    def test_market_lifecycle(self):
        """Test market lifecycle: create -> trade -> close -> resolve"""
        print("\n🧪 Testing Market Lifecycle...")
        
        # Setup: Register user and get token
        user_data = {
            "username": f"marketuser_{pytest.current_test}",
            "email": f"market_{pytest.current_test}@example.com",
            "password": "TestPassword123!",
            "full_name": "Market Test User"
        }
        
        # Register user
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201
        auth_token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Step 1: Create market
        print("   Step 1: Creating market...")
        market_data = {
            "title": "Market Lifecycle Test: Will this test pass?",
            "description": "Testing market lifecycle",
            "outcome_a": "Yes, it will pass",
            "outcome_b": "No, it will fail",
            "end_date": "2024-12-31T23:59:59Z",
            "category": "test",
            "tags": ["lifecycle", "test"]
        }
        response = client.post("/api/v1/markets/", json=market_data, headers=headers)
        assert response.status_code == 201
        market_id = response.json()["market_id"]
        print(f"      ✅ Market created with ID: {market_id}")
        
        # Step 2: Add multiple trades
        print("   Step 2: Adding multiple trades...")
        trades_data = [
            {"market_id": market_id, "trade_type": "buy", "outcome": "outcome_a", "amount": 100.0, "price_per_share": 0.5},
            {"market_id": market_id, "trade_type": "buy", "outcome": "outcome_b", "amount": 50.0, "price_per_share": 0.5},
            {"market_id": market_id, "trade_type": "sell", "outcome": "outcome_a", "amount": 25.0, "price_per_share": 0.6}
        ]
        
        for i, trade_data in enumerate(trades_data):
            response = client.post("/api/v1/trades/", json=trade_data, headers=headers)
            assert response.status_code == 201
            print(f"      ✅ Trade {i+1} executed")
        
        # Step 3: Get market with trades
        print("   Step 3: Getting market with trades...")
        response = client.get(f"/api/v1/markets/{market_id}")
        assert response.status_code == 200
        market = response.json()
        print(f"      ✅ Market retrieved: {market['title']}")
        
        # Step 4: Get market analytics
        print("   Step 4: Getting market analytics...")
        response = client.get(f"/api/v1/analytics/market/{market_id}", headers=headers)
        assert response.status_code == 200
        analytics = response.json()
        print("      ✅ Market analytics retrieved")
        
        print("   🎉 Market lifecycle test passed!")

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_authentication_errors(self):
        """Test authentication error handling"""
        print("\n🧪 Testing Authentication Error Handling...")
        
        # Test accessing protected endpoint without token
        print("   Testing access without token...")
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401
        print("      ✅ Correctly rejected without token")
        
        # Test accessing protected endpoint with invalid token
        print("   Testing access with invalid token...")
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 401
        print("      ✅ Correctly rejected with invalid token")
        
        # Test login with invalid credentials
        print("   Testing login with invalid credentials...")
        login_data = {
            "username": "nonexistent",
            "password": "wrongpassword"
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 401
        print("      ✅ Correctly rejected invalid credentials")
        
        print("   🎉 Authentication error handling test passed!")
    
    def test_validation_errors(self):
        """Test input validation error handling"""
        print("\n🧪 Testing Input Validation Error Handling...")
        
        # Test invalid user registration
        print("   Testing invalid user registration...")
        invalid_user = {
            "username": "test",  # Missing required fields
            "email": "invalid-email"  # Invalid email format
        }
        response = client.post("/api/v1/auth/register", json=invalid_user)
        assert response.status_code == 422
        print("      ✅ Correctly rejected invalid user data")
        
        # Test invalid market creation
        print("   Testing invalid market creation...")
        # First register a user
        user_data = {
            "username": f"validationuser_{pytest.current_test}",
            "email": f"validation_{pytest.current_test}@example.com",
            "password": "TestPassword123!",
            "full_name": "Validation Test User"
        }
        response = client.post("/api/v1/auth/register", json=user_data)
        auth_token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Try to create market with invalid data
        invalid_market = {
            "title": "",  # Empty title
            "description": "Test market"
            # Missing required fields
        }
        response = client.post("/api/v1/markets/", json=invalid_market, headers=headers)
        assert response.status_code == 422
        print("      ✅ Correctly rejected invalid market data")
        
        print("   🎉 Input validation error handling test passed!")
    
    def test_not_found_errors(self):
        """Test not found error handling"""
        print("\n🧪 Testing Not Found Error Handling...")
        
        # Test accessing nonexistent market
        print("   Testing access to nonexistent market...")
        response = client.get("/api/v1/markets/99999")
        assert response.status_code == 404
        print("      ✅ Correctly returned 404 for nonexistent market")
        
        # Test accessing nonexistent user
        print("   Testing access to nonexistent user...")
        response = client.get("/api/v1/users/99999")
        assert response.status_code == 404
        print("      ✅ Correctly returned 404 for nonexistent user")
        
        print("   🎉 Not found error handling test passed!")

class TestPerformance:
    """Test API performance and load handling"""
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        print("\n🧪 Testing Concurrent Request Handling...")
        
        import threading
        import time
        
        # Register a user
        user_data = {
            "username": f"perfuser_{pytest.current_test}",
            "email": f"perf_{pytest.current_test}@example.com",
            "password": "TestPassword123!",
            "full_name": "Performance Test User"
        }
        response = client.post("/api/v1/auth/register", json=user_data)
        auth_token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Create a market
        market_data = {
            "title": "Performance Test Market",
            "description": "Testing concurrent requests",
            "outcome_a": "Yes",
            "outcome_b": "No",
            "end_date": "2024-12-31T23:59:59Z",
            "category": "test",
            "tags": ["performance", "concurrent"]
        }
        response = client.post("/api/v1/markets/", json=market_data, headers=headers)
        market_id = response.json()["market_id"]
        
        # Test concurrent health checks
        print("   Testing concurrent health checks...")
        results = []
        
        def make_health_request():
            response = client.get("/api/v1/health")
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_health_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Check results
        assert all(status == 200 for status in results)
        print(f"      ✅ All {len(results)} concurrent requests succeeded")
        print(f"      ✅ Total time: {end_time - start_time:.2f} seconds")
        
        print("   🎉 Concurrent request handling test passed!")

class TestDataConsistency:
    """Test data consistency across API operations"""
    
    def test_data_consistency(self):
        """Test that data remains consistent across operations"""
        print("\n🧪 Testing Data Consistency...")
        
        # Register user
        user_data = {
            "username": f"consistencyuser_{pytest.current_test}",
            "email": f"consistency_{pytest.current_test}@example.com",
            "password": "TestPassword123!",
            "full_name": "Consistency Test User"
        }
        response = client.post("/api/v1/auth/register", json=user_data)
        auth_token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Create market
        market_data = {
            "title": "Data Consistency Test Market",
            "description": "Testing data consistency",
            "outcome_a": "Consistent",
            "outcome_b": "Inconsistent",
            "end_date": "2024-12-31T23:59:59Z",
            "category": "test",
            "tags": ["consistency", "test"]
        }
        response = client.post("/api/v1/markets/", json=market_data, headers=headers)
        market_id = response.json()["market_id"]
        
        # Execute trade
        trade_data = {
            "market_id": market_id,
            "trade_type": "buy",
            "outcome": "outcome_a",
            "amount": 100.0,
            "price_per_share": 0.5
        }
        response = client.post("/api/v1/trades/", json=trade_data, headers=headers)
        trade_id = response.json()["trade_id"]
        
        # Verify data consistency
        print("   Verifying data consistency...")
        
        # Check market still exists
        response = client.get(f"/api/v1/markets/{market_id}")
        assert response.status_code == 200
        market = response.json()
        assert market["market_id"] == market_id
        print("      ✅ Market data consistent")
        
        # Check trade exists
        response = client.get("/api/v1/trades/", headers=headers)
        assert response.status_code == 200
        trades = response.json()
        assert any(trade["trade_id"] == trade_id for trade in trades)
        print("      ✅ Trade data consistent")
        
        # Check user data
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200
        user_info = response.json()
        assert user_info["username"] == user_data["username"]
        print("      ✅ User data consistent")
        
        print("   🎉 Data consistency test passed!")

# Setup and teardown
@pytest.fixture(scope="module")
def setup_database():
    """Setup test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

