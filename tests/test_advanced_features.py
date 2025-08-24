import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import json

from app.main import app
from app.core.database import Base, get_db
from app.models.user import User
from app.models.market import Market, MarketStatus
from app.models.trade import Trade
from app.models.position import Position
from app.services.ai_analytics import get_ai_analytics_service
from app.services.rewards_system import get_rewards_system
from app.services.mobile_api import get_mobile_api_service
from app.services.advanced_orders import get_advanced_order_manager

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

@pytest.fixture(scope="function")
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_user():
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }

@pytest.fixture
def test_market():
    return {
        "title": "Test Market",
        "description": "A test market for testing",
        "category": "politics",
        "question": "Will the test pass?",
        "outcome_a": "Yes",
        "outcome_b": "No",
        "closes_at": (datetime.utcnow() + timedelta(days=7)).isoformat(),
        "total_liquidity": 1000.0
    }

class TestAIAnalytics:
    """Test AI Analytics features"""
    
    def test_market_prediction(self, setup_database, test_user, test_market):
        """Test AI market prediction"""
        # Create user and market
        user_response = client.post("/api/v1/auth/register", json=test_user)
        assert user_response.status_code == 200
        
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        market_response = client.post("/api/v1/markets/", json=test_market, headers=headers)
        assert market_response.status_code == 200
        market_id = market_response.json()["id"]
        
        # Test AI prediction
        prediction_response = client.get(f"/api/v1/ai-analytics/market/{market_id}/prediction")
        assert prediction_response.status_code == 200
        
        prediction_data = prediction_response.json()
        assert "market_id" in prediction_data
        assert "prediction" in prediction_data
        assert "confidence" in prediction_data
        assert "technical_indicators" in prediction_data
    
    def test_user_insights(self, setup_database, test_user):
        """Test AI user insights"""
        # Create user
        user_response = client.post("/api/v1/auth/register", json=test_user)
        assert user_response.status_code == 200
        
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test user insights
        insights_response = client.get("/api/v1/ai-analytics/user/insights", headers=headers)
        assert insights_response.status_code == 200
        
        insights_data = insights_response.json()
        assert "performance_metrics" in insights_data
        assert "trading_patterns" in insights_data
        assert "recommendations" in insights_data
    
    def test_market_sentiment(self, setup_database, test_user, test_market):
        """Test market sentiment analysis"""
        # Create user and market
        user_response = client.post("/api/v1/auth/register", json=test_user)
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        market_response = client.post("/api/v1/markets/", json=test_market, headers=headers)
        market_id = market_response.json()["id"]
        
        # Test sentiment analysis
        sentiment_response = client.get(f"/api/v1/ai-analytics/market/{market_id}/sentiment")
        assert sentiment_response.status_code == 200
        
        sentiment_data = sentiment_response.json()
        assert "market_id" in sentiment_data
        assert "sentiment_score" in sentiment_data
        assert "sentiment_label" in sentiment_data

class TestRewardsSystem:
    """Test Rewards and Gamification features"""
    
    def test_daily_login_reward(self, setup_database, test_user):
        """Test daily login reward system"""
        # Create user
        user_response = client.post("/api/v1/auth/register", json=test_user)
        assert user_response.status_code == 200
        
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test daily login reward
        reward_response = client.post("/api/v1/rewards/daily-login", headers=headers)
        assert reward_response.status_code == 200
        
        reward_data = reward_response.json()
        assert "reward_type" in reward_data
        assert "tokens_awarded" in reward_data
        assert "xp_gained" in reward_data
    
    def test_achievements(self, setup_database, test_user):
        """Test achievements system"""
        # Create user
        user_response = client.post("/api/v1/auth/register", json=test_user)
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test achievements
        achievements_response = client.get("/api/v1/rewards/achievements", headers=headers)
        assert achievements_response.status_code == 200
        
        achievements_data = achievements_response.json()
        assert "achievements" in achievements_data
        assert "total_unlocked" in achievements_data
        assert "total_achievements" in achievements_data
    
    def test_rewards_stats(self, setup_database, test_user):
        """Test rewards statistics"""
        # Create user
        user_response = client.post("/api/v1/auth/register", json=test_user)
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test rewards stats
        stats_response = client.get("/api/v1/rewards/stats", headers=headers)
        assert stats_response.status_code == 200
        
        stats_data = stats_response.json()
        assert "total_tokens_earned" in stats_data
        assert "total_xp_earned" in stats_data
        assert "achievements_unlocked" in stats_data

class TestMobileAPI:
    """Test Mobile API features"""
    
    def test_mobile_dashboard(self, setup_database, test_user):
        """Test mobile dashboard"""
        # Create user
        user_response = client.post("/api/v1/auth/register", json=test_user)
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test mobile dashboard
        dashboard_response = client.get("/api/v1/mobile/dashboard", headers=headers)
        assert dashboard_response.status_code == 200
        
        dashboard_data = dashboard_response.json()
        assert "user" in dashboard_data
        assert "recent_markets" in dashboard_data
        assert "trending_markets" in dashboard_data
        assert "recent_trades" in dashboard_data
        assert "notifications" in dashboard_data
    
    def test_mobile_portfolio(self, setup_database, test_user):
        """Test mobile portfolio"""
        # Create user
        user_response = client.post("/api/v1/auth/register", json=test_user)
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test mobile portfolio
        portfolio_response = client.get("/api/v1/mobile/portfolio", headers=headers)
        assert portfolio_response.status_code == 200
        
        portfolio_data = portfolio_response.json()
        assert "user" in portfolio_data
        assert "portfolio_summary" in portfolio_data
        assert "positions" in portfolio_data
        assert "performance_chart" in portfolio_data
    
    def test_device_registration(self, setup_database, test_user):
        """Test mobile device registration"""
        # Create user
        user_response = client.post("/api/v1/auth/register", json=test_user)
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Test device registration
        device_data = {
            "push_token": "test_push_token_123",
            "device_info": {
                "platform": "ios",
                "version": "15.0",
                "model": "iPhone 13"
            }
        }
        
        register_response = client.post("/api/v1/mobile/device/register", 
                                      json=device_data, headers=headers)
        assert register_response.status_code == 200
        
        register_data = register_response.json()
        assert "success" in register_data
        assert "device_id" in register_data

class TestAdvancedOrders:
    """Test Advanced Order features"""
    
    def test_stop_loss_order(self, setup_database, test_user, test_market):
        """Test stop-loss order creation"""
        # Create user and market
        user_response = client.post("/api/v1/auth/register", json=test_user)
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        market_response = client.post("/api/v1/markets/", json=test_market, headers=headers)
        market_id = market_response.json()["id"]
        
        # Place a trade to create a position
        trade_data = {
            "market_id": market_id,
            "trade_type": "buy",
            "outcome": "outcome_a",
            "amount": 10.0
        }
        trade_response = client.post("/api/v1/trades/", json=trade_data, headers=headers)
        assert trade_response.status_code == 200
        
        # Test stop-loss order
        stop_loss_data = {
            "market_id": market_id,
            "outcome": "outcome_a",
            "shares": 5.0,
            "stop_price": 0.3
        }
        
        stop_loss_response = client.post("/api/v1/advanced-orders/stop-loss", 
                                       json=stop_loss_data, headers=headers)
        assert stop_loss_response.status_code == 200
        
        stop_loss_result = stop_loss_response.json()
        assert "success" in stop_loss_result
        assert "order_id" in stop_loss_result
    
    def test_take_profit_order(self, setup_database, test_user, test_market):
        """Test take-profit order creation"""
        # Create user and market
        user_response = client.post("/api/v1/auth/register", json=test_user)
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        market_response = client.post("/api/v1/markets/", json=test_market, headers=headers)
        market_id = market_response.json()["id"]
        
        # Place a trade to create a position
        trade_data = {
            "market_id": market_id,
            "trade_type": "buy",
            "outcome": "outcome_a",
            "amount": 10.0
        }
        client.post("/api/v1/trades/", json=trade_data, headers=headers)
        
        # Test take-profit order
        take_profit_data = {
            "market_id": market_id,
            "outcome": "outcome_a",
            "shares": 5.0,
            "take_profit_price": 0.8
        }
        
        take_profit_response = client.post("/api/v1/advanced-orders/take-profit", 
                                         json=take_profit_data, headers=headers)
        assert take_profit_response.status_code == 200
        
        take_profit_result = take_profit_response.json()
        assert "success" in take_profit_result
        assert "order_id" in take_profit_result
    
    def test_trailing_stop_order(self, setup_database, test_user, test_market):
        """Test trailing stop order creation"""
        # Create user and market
        user_response = client.post("/api/v1/auth/register", json=test_user)
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        market_response = client.post("/api/v1/markets/", json=test_market, headers=headers)
        market_id = market_response.json()["id"]
        
        # Place a trade to create a position
        trade_data = {
            "market_id": market_id,
            "trade_type": "buy",
            "outcome": "outcome_a",
            "amount": 10.0
        }
        client.post("/api/v1/trades/", json=trade_data, headers=headers)
        
        # Test trailing stop order
        trailing_stop_data = {
            "market_id": market_id,
            "outcome": "outcome_a",
            "shares": 5.0,
            "trailing_percentage": 10.0
        }
        
        trailing_stop_response = client.post("/api/v1/advanced-orders/trailing-stop", 
                                           json=trailing_stop_data, headers=headers)
        assert trailing_stop_response.status_code == 200
        
        trailing_stop_result = trailing_stop_response.json()
        assert "success" in trailing_stop_result
        assert "order_id" in trailing_stop_result
    
    def test_order_types(self, setup_database):
        """Test getting available order types"""
        order_types_response = client.get("/api/v1/advanced-orders/order-types")
        assert order_types_response.status_code == 200
        
        order_types_data = order_types_response.json()
        assert "order_types" in order_types_data
        
        # Check that all expected order types are present
        expected_types = ["stop_loss", "take_profit", "trailing_stop", "conditional", "bracket"]
        returned_types = [ot["type"] for ot in order_types_data["order_types"]]
        
        for expected_type in expected_types:
            assert expected_type in returned_types
    
    def test_risk_management_tips(self, setup_database):
        """Test risk management tips"""
        tips_response = client.get("/api/v1/advanced-orders/risk-management")
        assert tips_response.status_code == 200
        
        tips_data = tips_response.json()
        assert "tips" in tips_data
        assert "risk_warnings" in tips_data
        assert len(tips_data["tips"]) > 0
        assert len(tips_data["risk_warnings"]) > 0

class TestIntegration:
    """Integration tests for advanced features"""
    
    def test_complete_trading_workflow(self, setup_database, test_user, test_market):
        """Test complete trading workflow with advanced features"""
        # Create user
        user_response = client.post("/api/v1/auth/register", json=test_user)
        login_response = client.post("/api/v1/auth/login", data={
            "username": test_user["username"],
            "password": test_user["password"]
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Create market
        market_response = client.post("/api/v1/markets/", json=test_market, headers=headers)
        market_id = market_response.json()["id"]
        
        # Place trade
        trade_data = {
            "market_id": market_id,
            "trade_type": "buy",
            "outcome": "outcome_a",
            "amount": 10.0
        }
        trade_response = client.post("/api/v1/trades/", json=trade_data, headers=headers)
        assert trade_response.status_code == 200
        
        # Create stop-loss order
        stop_loss_data = {
            "market_id": market_id,
            "outcome": "outcome_a",
            "shares": 5.0,
            "stop_price": 0.3
        }
        stop_loss_response = client.post("/api/v1/advanced-orders/stop-loss", 
                                       json=stop_loss_data, headers=headers)
        assert stop_loss_response.status_code == 200
        
        # Check portfolio
        portfolio_response = client.get("/api/v1/positions/portfolio", headers=headers)
        assert portfolio_response.status_code == 200
        
        # Check AI insights
        insights_response = client.get("/api/v1/ai-analytics/user/insights", headers=headers)
        assert insights_response.status_code == 200
        
        # Check rewards
        rewards_response = client.get("/api/v1/rewards/achievements", headers=headers)
        assert rewards_response.status_code == 200
        
        # Check mobile dashboard
        mobile_response = client.get("/api/v1/mobile/dashboard", headers=headers)
        assert mobile_response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
