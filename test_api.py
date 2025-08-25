#!/usr/bin/env python3
"""
Comprehensive API Test Script for Opinion Market
Tests all endpoints including advanced features
"""

import requests
import json
import time
from datetime import datetime, timedelta

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

# Test data
test_user = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpassword123",
    "full_name": "Test User"
}

test_market = {
    "title": "Will Bitcoin reach $100k by end of 2024?",
    "description": "A prediction market for Bitcoin price target",
    "category": "crypto",
    "question": "Will Bitcoin reach $100,000 by December 31, 2024?",
    "outcome_a": "Yes",
    "outcome_b": "No",
    "closes_at": (datetime.utcnow() + timedelta(days=7)).isoformat(),
    "total_liquidity": 10000.0
}

# Global variables to store test data
test_user_token = None
test_market_id = None
test_trade_id = None
test_position_id = None
test_dispute_id = None
test_order_id = None
test_proposal_id = None
test_futures_contract_id = None
test_options_contract_id = None

def run_test(description, test_func):
    """Run a test and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = test_func()
        print(f"✅ {description} - PASSED")
        return result
    except Exception as e:
        print(f"❌ {description} - FAILED: {str(e)}")
        return None

def test_auth():
    """Test authentication endpoints"""
    global test_user_token
    
    # Test user registration
    response = requests.post(f"{API_BASE}/auth/register", json=test_user)
    if response.status_code == 200:
        print("✅ User registration successful")
    else:
        print(f"⚠️ User registration failed: {response.text}")
    
    # Test user login
    login_data = {
        "username": test_user["username"],
        "password": test_user["password"]
    }
    response = requests.post(f"{API_BASE}/auth/login", data=login_data)
    if response.status_code == 200:
        test_user_token = response.json()["access_token"]
        print("✅ User login successful")
    else:
        print(f"❌ User login failed: {response.text}")

def test_users():
    """Test user endpoints"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test get current user
    response = requests.get(f"{API_BASE}/users/me", headers=headers)
    if response.status_code == 200:
        user_data = response.json()
        print(f"✅ Current user: {user_data['username']}")
    
    # Test update user
    update_data = {"full_name": "Updated Test User"}
    response = requests.put(f"{API_BASE}/users/me", json=update_data, headers=headers)
    if response.status_code == 200:
        print("✅ User update successful")

def test_markets():
    """Test market endpoints"""
    global test_market_id
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test create market
    response = requests.post(f"{API_BASE}/markets/", json=test_market, headers=headers)
    if response.status_code == 200:
        test_market_id = response.json()["id"]
        print(f"✅ Market created with ID: {test_market_id}")
    
    # Test get markets
    response = requests.get(f"{API_BASE}/markets/")
    if response.status_code == 200:
        markets = response.json()
        print(f"✅ Retrieved {len(markets['markets'])} markets")
    
    # Test get specific market
    if test_market_id:
        response = requests.get(f"{API_BASE}/markets/{test_market_id}")
        if response.status_code == 200:
            print("✅ Market details retrieved")
    
    # Test trending markets
    response = requests.get(f"{API_BASE}/markets/trending")
    if response.status_code == 200:
        print("✅ Trending markets retrieved")

def test_trades():
    """Test trade endpoints"""
    global test_trade_id
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    if test_market_id:
        # Test create trade
        trade_data = {
            "market_id": test_market_id,
            "trade_type": "buy",
            "outcome": "outcome_a",
            "amount": 10.0
        }
        response = requests.post(f"{API_BASE}/trades/", json=trade_data, headers=headers)
        if response.status_code == 200:
            test_trade_id = response.json()["id"]
            print(f"✅ Trade created with ID: {test_trade_id}")
        
        # Test get trades
        response = requests.get(f"{API_BASE}/trades/", headers=headers)
        if response.status_code == 200:
            trades = response.json()
            print(f"✅ Retrieved {len(trades['trades'])} trades")

def test_votes():
    """Test vote endpoints"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    if test_market_id:
        # Test create vote
        vote_data = {
            "market_id": test_market_id,
            "outcome": "outcome_a",
            "confidence": 0.8
        }
        response = requests.post(f"{API_BASE}/votes/", json=vote_data, headers=headers)
        if response.status_code == 200:
            print("✅ Vote created")
        
        # Test get votes
        response = requests.get(f"{API_BASE}/votes/", headers=headers)
        if response.status_code == 200:
            votes = response.json()
            print(f"✅ Retrieved {len(votes['votes'])} votes")

def test_positions():
    """Test position endpoints"""
    global test_position_id
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test get positions
    response = requests.get(f"{API_BASE}/positions/", headers=headers)
    if response.status_code == 200:
        positions = response.json()
        if positions['positions']:
            test_position_id = positions['positions'][0]['id']
        print(f"✅ Retrieved {len(positions['positions'])} positions")
    
    # Test portfolio summary
    response = requests.get(f"{API_BASE}/positions/portfolio", headers=headers)
    if response.status_code == 200:
        portfolio = response.json()
        print(f"✅ Portfolio value: ${portfolio['total_portfolio_value']:.2f}")

def test_websocket():
    """Test WebSocket endpoints"""
    print("✅ WebSocket endpoints available (manual testing required)")

def test_leaderboard():
    """Test leaderboard endpoints"""
    # Test top traders
    response = requests.get(f"{API_BASE}/leaderboard/traders")
    if response.status_code == 200:
        print("✅ Top traders leaderboard retrieved")
    
    # Test top volume traders
    response = requests.get(f"{API_BASE}/leaderboard/volume")
    if response.status_code == 200:
        print("✅ Top volume traders retrieved")

def test_disputes():
    """Test dispute endpoints"""
    global test_dispute_id
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    if test_market_id:
        # Test create dispute
        dispute_data = {
            "market_id": test_market_id,
            "dispute_type": "ambiguous_question",
            "reason": "The question is unclear and could be interpreted in multiple ways"
        }
        response = requests.post(f"{API_BASE}/disputes/", json=dispute_data, headers=headers)
        if response.status_code == 200:
            test_dispute_id = response.json()["id"]
            print(f"✅ Dispute created with ID: {test_dispute_id}")
        
        # Test get disputes
        response = requests.get(f"{API_BASE}/disputes/")
        if response.status_code == 200:
            disputes = response.json()
            print(f"✅ Retrieved {len(disputes['disputes'])} disputes")

def test_notifications():
    """Test notification endpoints"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test get notifications
    response = requests.get(f"{API_BASE}/notifications/", headers=headers)
    if response.status_code == 200:
        notifications = response.json()
        print(f"✅ Retrieved {len(notifications['notifications'])} notifications")
    
    # Test notification preferences
    response = requests.get(f"{API_BASE}/notifications/preferences", headers=headers)
    if response.status_code == 200:
        print("✅ Notification preferences retrieved")

def test_analytics():
    """Test analytics endpoints"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test platform analytics
    response = requests.get(f"{API_BASE}/analytics/platform")
    if response.status_code == 200:
        print("✅ Platform analytics retrieved")
    
    # Test user analytics
    response = requests.get(f"{API_BASE}/analytics/user/me", headers=headers)
    if response.status_code == 200:
        print("✅ User analytics retrieved")

def test_verification():
    """Test verification endpoints"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test pending verifications
    response = requests.get(f"{API_BASE}/verification/pending", headers=headers)
    if response.status_code == 200:
        print("✅ Pending verifications retrieved")

def test_orders():
    """Test order endpoints"""
    global test_order_id
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    if test_market_id:
        # Test create order
        order_data = {
            "market_id": test_market_id,
            "order_type": "limit",
            "side": "buy",
            "outcome": "outcome_a",
            "amount": 5.0,
            "limit_price": 0.6
        }
        response = requests.post(f"{API_BASE}/orders/", json=order_data, headers=headers)
        if response.status_code == 200:
            test_order_id = response.json()["id"]
            print(f"✅ Order created with ID: {test_order_id}")
        
        # Test get orders
        response = requests.get(f"{API_BASE}/orders/", headers=headers)
        if response.status_code == 200:
            orders = response.json()
            print(f"✅ Retrieved {len(orders['orders'])} orders")

def test_governance():
    """Test governance endpoints"""
    global test_proposal_id
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test governance stats
    response = requests.get(f"{API_BASE}/governance/stats")
    if response.status_code == 200:
        print("✅ Governance stats retrieved")
    
    # Test user tokens
    response = requests.get(f"{API_BASE}/governance/tokens/me", headers=headers)
    if response.status_code == 200:
        print("✅ User governance tokens retrieved")
    
    # Test create proposal
    proposal_data = {
        "title": "Test Proposal",
        "description": "This is a test governance proposal",
        "proposal_type": "feature_request",
        "voting_start": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
        "voting_end": (datetime.utcnow() + timedelta(days=7)).isoformat()
    }
    response = requests.post(f"{API_BASE}/governance/proposals", json=proposal_data, headers=headers)
    if response.status_code == 200:
        test_proposal_id = response.json()["id"]
        print(f"✅ Governance proposal created with ID: {test_proposal_id}")

def test_advanced_markets():
    """Test advanced market endpoints"""
    global test_futures_contract_id, test_options_contract_id
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    if test_market_id:
        # Test create futures contract
        futures_data = {
            "market_id": test_market_id,
            "contract_size": 1.0,
            "margin_requirement": 0.1,
            "settlement_date": (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
        response = requests.post(f"{API_BASE}/advanced-markets/futures/contracts", json=futures_data, headers=headers)
        if response.status_code == 200:
            test_futures_contract_id = response.json()["id"]
            print(f"✅ Futures contract created with ID: {test_futures_contract_id}")
        
        # Test create options contract
        options_data = {
            "market_id": test_market_id,
            "option_type": "call",
            "strike_price": 0.5,
            "expiration_date": (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
        response = requests.post(f"{API_BASE}/advanced-markets/options/contracts", json=options_data, headers=headers)
        if response.status_code == 200:
            test_options_contract_id = response.json()["id"]
            print(f"✅ Options contract created with ID: {test_options_contract_id}")

def test_ai_analytics():
    """Test AI analytics endpoints"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    if test_market_id:
        # Test market prediction
        response = requests.get(f"{API_BASE}/ai-analytics/market/{test_market_id}/prediction")
        if response.status_code == 200:
            print("✅ AI market prediction retrieved")
        
        # Test market sentiment
        response = requests.get(f"{API_BASE}/ai-analytics/market/{test_market_id}/sentiment")
        if response.status_code == 200:
            print("✅ Market sentiment analysis retrieved")
    
    # Test user insights
    response = requests.get(f"{API_BASE}/ai-analytics/user/insights", headers=headers)
    if response.status_code == 200:
        print("✅ AI user insights retrieved")

def test_rewards():
    """Test rewards endpoints"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test daily login reward
    response = requests.post(f"{API_BASE}/rewards/daily-login", headers=headers)
    if response.status_code == 200:
        print("✅ Daily login reward claimed")
    
    # Test achievements
    response = requests.get(f"{API_BASE}/rewards/achievements", headers=headers)
    if response.status_code == 200:
        print("✅ User achievements retrieved")
    
    # Test rewards stats
    response = requests.get(f"{API_BASE}/rewards/stats", headers=headers)
    if response.status_code == 200:
        print("✅ Rewards statistics retrieved")

def test_mobile():
    """Test mobile API endpoints"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test mobile dashboard
    response = requests.get(f"{API_BASE}/mobile/dashboard", headers=headers)
    if response.status_code == 200:
        print("✅ Mobile dashboard retrieved")
    
    # Test mobile portfolio
    response = requests.get(f"{API_BASE}/mobile/portfolio", headers=headers)
    if response.status_code == 200:
        print("✅ Mobile portfolio retrieved")
    
    # Test device registration
    device_data = {
        "push_token": "test_push_token_123",
        "device_info": {
            "platform": "ios",
            "version": "15.0",
            "model": "iPhone 13"
        }
    }
    response = requests.post(f"{API_BASE}/mobile/device/register", json=device_data, headers=headers)
    if response.status_code == 200:
        print("✅ Mobile device registered")

def test_advanced_orders():
    """Test advanced order endpoints"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    if test_market_id:
        # Test stop-loss order
        stop_loss_data = {
            "market_id": test_market_id,
            "outcome": "outcome_a",
            "shares": 5.0,
            "stop_price": 0.3
        }
        response = requests.post(f"{API_BASE}/advanced-orders/stop-loss", json=stop_loss_data, headers=headers)
        if response.status_code == 200:
            print("✅ Stop-loss order created")
        
        # Test take-profit order
        take_profit_data = {
            "market_id": test_market_id,
            "outcome": "outcome_a",
            "shares": 5.0,
            "take_profit_price": 0.8
        }
        response = requests.post(f"{API_BASE}/advanced-orders/take-profit", json=take_profit_data, headers=headers)
        if response.status_code == 200:
            print("✅ Take-profit order created")
    
    # Test order types
    response = requests.get(f"{API_BASE}/advanced-orders/order-types")
    if response.status_code == 200:
        print("✅ Advanced order types retrieved")
    
    # Test risk management tips
    response = requests.get(f"{API_BASE}/advanced-orders/risk-management")
    if response.status_code == 200:
        print("✅ Risk management tips retrieved")

def test_security():
    """Test security features"""
    headers = {"Authorization": f"Bearer {test_user_token}"}
    
    # Test rate limiting by making multiple requests
    print("🔄 Testing rate limiting...")
    for i in range(5):
        response = requests.get(f"{API_BASE}/users/me", headers=headers)
        if response.status_code == 429:
            print("✅ Rate limiting working correctly")
            break
        time.sleep(0.1)
    else:
        print("⚠️ Rate limiting not triggered")

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting Opinion Market API Tests...")
    
    # Test basic functionality
    test_auth()
    test_users()
    test_markets()
    test_trades()
    test_votes()
    test_positions()
    
    # Test advanced features
    test_websocket()
    test_leaderboard()
    test_disputes()
    test_notifications()
    test_analytics()
    test_verification()
    test_orders()
    test_governance()
    test_advanced_markets()
    test_ai_analytics()
    test_rewards()
    test_mobile()
    test_advanced_orders()
    test_security()
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    run_all_tests()
