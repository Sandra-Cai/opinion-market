#!/usr/bin/env python3
"""
Test script for Opinion Market API
Run this script to test all API endpoints
"""

import requests
import json
import time
from datetime import datetime, timedelta

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

# Global variables to store test data
test_user_token = None
test_user_id = None
test_market_id = None
test_trade_id = None
test_position_id = None
test_dispute_id = None
test_order_id = None

def print_response(response, title=""):
    """Print API response in a formatted way"""
    print(f"\n{'='*50}")
    if title:
        print(f"📋 {title}")
    print(f"Status: {response.status_code}")
    print(f"URL: {response.url}")
    try:
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except:
        print(f"Text: {response.text}")
    print(f"{'='*50}\n")

def test_auth():
    """Test authentication endpoints"""
    global test_user_token, test_user_id
    
    print("🔐 Testing Authentication...")
    
    # Test user registration
    user_data = {
        "username": f"testuser_{int(time.time())}",
        "email": f"test{int(time.time())}@example.com",
        "password": "testpassword123"
    }
    
    response = requests.post(f"{API_BASE}/auth/register", json=user_data)
    print_response(response, "User Registration")
    
    if response.status_code == 201:
        test_user_id = response.json()["id"]
    
    # Test user login
    login_data = {
        "username": user_data["username"],
        "password": user_data["password"]
    }
    
    response = requests.post(f"{API_BASE}/auth/login", data=login_data)
    print_response(response, "User Login")
    
    if response.status_code == 200:
        test_user_token = response.json()["access_token"]
    
    # Test get current user
    headers = {"Authorization": f"Bearer {test_user_token}"}
    response = requests.get(f"{API_BASE}/users/me", headers=headers)
    print_response(response, "Get Current User")

def test_markets():
    """Test market endpoints"""
    global test_market_id
    
    print("📊 Testing Markets...")
    
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test create market
    market_data = {
        "title": "Will Bitcoin reach $100k by end of 2024?",
        "description": "A prediction market for Bitcoin's price target",
        "category": "crypto",
        "question": "Will Bitcoin reach $100,000 by December 31, 2024?",
        "outcome_a": "Yes",
        "outcome_b": "No",
        "closes_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
        "total_liquidity": 10000.0,
        "tags": ["bitcoin", "crypto", "price-prediction"]
    }
    
    response = requests.post(f"{API_BASE}/markets/", json=market_data, headers=headers)
    print_response(response, "Create Market")
    
    if response.status_code == 201:
        test_market_id = response.json()["id"]
    
    # Test get markets
    response = requests.get(f"{API_BASE}/markets/", headers=headers)
    print_response(response, "Get Markets")
    
    # Test get trending markets
    response = requests.get(f"{API_BASE}/markets/trending", headers=headers)
    print_response(response, "Get Trending Markets")
    
    # Test get market stats
    response = requests.get(f"{API_BASE}/markets/stats", headers=headers)
    print_response(response, "Get Market Stats")

def test_trades():
    """Test trade endpoints"""
    global test_trade_id
    
    print("💰 Testing Trades...")
    
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test create trade
    trade_data = {
        "market_id": test_market_id,
        "trade_type": "buy",
        "outcome": "outcome_a",
        "amount": 10.0
    }
    
    response = requests.post(f"{API_BASE}/trades/", json=trade_data, headers=headers)
    print_response(response, "Create Trade")
    
    if response.status_code == 201:
        test_trade_id = response.json()["id"]
    
    # Test get trades
    response = requests.get(f"{API_BASE}/trades/", headers=headers)
    print_response(response, "Get Trades")

def test_positions():
    """Test position endpoints"""
    global test_position_id
    
    print("📈 Testing Positions...")
    
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test get positions
    response = requests.get(f"{API_BASE}/positions/", headers=headers)
    print_response(response, "Get Positions")
    
    # Test get portfolio summary
    response = requests.get(f"{API_BASE}/positions/portfolio", headers=headers)
    print_response(response, "Get Portfolio Summary")
    
    if response.status_code == 200:
        positions = response.json()["positions"]
        if positions:
            test_position_id = positions[0]["id"]

def test_disputes():
    """Test dispute endpoints"""
    global test_dispute_id
    
    print("⚖️ Testing Disputes...")
    
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # First, we need to resolve a market to create a dispute
    # For testing, we'll just test the dispute creation endpoint
    dispute_data = {
        "market_id": test_market_id,
        "dispute_type": "incorrect_resolution",
        "reason": "The market resolution does not match the actual outcome",
        "evidence": "Official sources confirm the opposite result"
    }
    
    response = requests.post(f"{API_BASE}/disputes/", json=dispute_data, headers=headers)
    print_response(response, "Create Dispute")
    
    if response.status_code == 201:
        test_dispute_id = response.json()["id"]
    
    # Test get disputes
    response = requests.get(f"{API_BASE}/disputes/", headers=headers)
    print_response(response, "Get Disputes")

def test_notifications():
    """Test notification endpoints"""
    print("🔔 Testing Notifications...")
    
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test get notifications
    response = requests.get(f"{API_BASE}/notifications/", headers=headers)
    print_response(response, "Get Notifications")
    
    # Test get notification preferences
    response = requests.get(f"{API_BASE}/notifications/preferences", headers=headers)
    print_response(response, "Get Notification Preferences")
    
    # Test update notification preferences
    preference_update = {
        "email_enabled": True,
        "push_enabled": False,
        "price_alert_threshold": 0.03
    }
    
    response = requests.put(f"{API_BASE}/notifications/preferences", json=preference_update, headers=headers)
    print_response(response, "Update Notification Preferences")
    
    # Test get unread count
    response = requests.get(f"{API_BASE}/notifications/unread-count", headers=headers)
    print_response(response, "Get Unread Count")

def test_analytics():
    """Test analytics endpoints"""
    print("📊 Testing Analytics...")
    
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test get market analytics
    response = requests.get(f"{API_BASE}/analytics/market/{test_market_id}", headers=headers)
    print_response(response, "Get Market Analytics")
    
    # Test get user analytics
    response = requests.get(f"{API_BASE}/analytics/user/me", headers=headers)
    print_response(response, "Get User Analytics")
    
    # Test get platform analytics
    response = requests.get(f"{API_BASE}/analytics/platform", headers=headers)
    print_response(response, "Get Platform Analytics")
    
    # Test get market predictions
    response = requests.get(f"{API_BASE}/analytics/market/{test_market_id}/predictions", headers=headers)
    print_response(response, "Get Market Predictions")
    
    # Test get trending analytics
    response = requests.get(f"{API_BASE}/analytics/trending", headers=headers)
    print_response(response, "Get Trending Analytics")
    
    # Test get volume by category
    response = requests.get(f"{API_BASE}/analytics/volume-by-category?period=24h", headers=headers)
    print_response(response, "Get Volume by Category")
    
    # Test get price movements
    response = requests.get(f"{API_BASE}/analytics/price-movements?market_id={test_market_id}&hours=24", headers=headers)
    print_response(response, "Get Price Movements")
    
    # Test get sentiment analysis
    response = requests.get(f"{API_BASE}/analytics/sentiment-analysis?market_id={test_market_id}", headers=headers)
    print_response(response, "Get Sentiment Analysis")

def test_verification():
    """Test verification endpoints"""
    print("✅ Testing Verification...")
    
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test get pending verifications
    response = requests.get(f"{API_BASE}/verification/pending", headers=headers)
    print_response(response, "Get Pending Verifications")
    
    # Test get verification stats
    response = requests.get(f"{API_BASE}/verification/stats", headers=headers)
    print_response(response, "Get Verification Stats")
    
    # Test get my verifications
    response = requests.get(f"{API_BASE}/verification/my-verifications", headers=headers)
    print_response(response, "Get My Verifications")

def test_orders():
    """Test order book endpoints"""
    global test_order_id
    
    print("📋 Testing Orders...")
    
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test create limit order
    order_data = {
        "market_id": test_market_id,
        "order_type": "limit",
        "side": "buy",
        "outcome": "outcome_a",
        "original_amount": 5.0,
        "limit_price": 0.45
    }
    
    response = requests.post(f"{API_BASE}/orders/", json=order_data, headers=headers)
    print_response(response, "Create Limit Order")
    
    if response.status_code == 201:
        test_order_id = response.json()["id"]
    
    # Test create market order
    market_order_data = {
        "market_id": test_market_id,
        "order_type": "market",
        "side": "sell",
        "outcome": "outcome_b",
        "original_amount": 3.0
    }
    
    response = requests.post(f"{API_BASE}/orders/", json=market_order_data, headers=headers)
    print_response(response, "Create Market Order")
    
    # Test get orders
    response = requests.get(f"{API_BASE}/orders/", headers=headers)
    print_response(response, "Get Orders")
    
    # Test get order book
    response = requests.get(f"{API_BASE}/orders/market/{test_market_id}/orderbook?levels=5", headers=headers)
    print_response(response, "Get Order Book")
    
    # Test cancel order
    if test_order_id:
        response = requests.post(f"{API_BASE}/orders/{test_order_id}/cancel", headers=headers)
        print_response(response, "Cancel Order")

def test_leaderboard():
    """Test leaderboard endpoints"""
    print("🏆 Testing Leaderboards...")
    
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test get top traders
    response = requests.get(f"{API_BASE}/leaderboard/traders?period=7d", headers=headers)
    print_response(response, "Get Top Traders")
    
    # Test get top volume traders
    response = requests.get(f"{API_BASE}/leaderboard/volume?period=30d", headers=headers)
    print_response(response, "Get Top Volume Traders")
    
    # Test get top market creators
    response = requests.get(f"{API_BASE}/leaderboard/creators", headers=headers)
    print_response(response, "Get Top Market Creators")
    
    # Test get top win rate traders
    response = requests.get(f"{API_BASE}/leaderboard/win-rate?min_trades=5", headers=headers)
    print_response(response, "Get Top Win Rate Traders")

def test_websocket():
    """Test WebSocket endpoints"""
    print("🔌 Testing WebSocket...")
    
    # Note: WebSocket testing requires a WebSocket client
    # This is a placeholder for WebSocket testing
    print("WebSocket endpoints available:")
    print("- /ws/market/{market_id} - Market-specific updates")
    print("- /ws/user - User-specific updates")
    print("- /ws/global - Global market updates")

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting Opinion Market API Tests")
    print(f"Testing against: {BASE_URL}")
    
    try:
        test_auth()
        test_markets()
        test_trades()
        test_positions()
        test_disputes()
        test_notifications()
        test_analytics()
        test_verification()
        test_orders()
        test_leaderboard()
        test_websocket()
        
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
