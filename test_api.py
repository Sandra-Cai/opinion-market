#!/usr/bin/env python3
"""
Opinion Market - API Test Script
Simple script to test the API endpoints
"""

import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000/api/v1"

def test_api():
    """Test the API endpoints"""
    print("🧪 Testing Opinion Market API...")
    
    # Test health check
    print("\n1. Testing health check...")
    response = requests.get("http://localhost:8000/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test user registration
    print("\n2. Testing user registration...")
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User"
    }
    response = requests.post(f"{BASE_URL}/auth/register", json=user_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   User created: {response.json()['username']}")
    else:
        print(f"   Error: {response.json()}")
    
    # Test user login
    print("\n3. Testing user login...")
    login_data = {
        "username": "testuser",
        "password": "testpassword123"
    }
    response = requests.post(f"{BASE_URL}/auth/login", data=login_data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        token = response.json()["access_token"]
        print(f"   Token received: {token[:20]}...")
        headers = {"Authorization": f"Bearer {token}"}
    else:
        print(f"   Error: {response.json()}")
        return
    
    # Test market creation
    print("\n4. Testing market creation...")
    market_data = {
        "title": "Will it rain tomorrow?",
        "description": "A simple weather prediction market",
        "category": "science",
        "question": "Will it rain in New York City tomorrow?",
        "outcome_a": "Yes",
        "outcome_b": "No",
        "closes_at": (datetime.utcnow() + timedelta(days=1)).isoformat()
    }
    response = requests.post(f"{BASE_URL}/markets", json=market_data, headers=headers)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        market = response.json()
        market_id = market["id"]
        print(f"   Market created: {market['title']}")
    else:
        print(f"   Error: {response.json()}")
        return
    
    # Test getting markets
    print("\n5. Testing get markets...")
    response = requests.get(f"{BASE_URL}/markets")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        markets = response.json()
        print(f"   Found {markets['total']} markets")
    
    # Test creating a trade
    print("\n6. Testing trade creation...")
    trade_data = {
        "trade_type": "buy",
        "outcome": "outcome_a",
        "amount": 10.0,
        "price_per_share": 0.6,
        "market_id": market_id
    }
    response = requests.post(f"{BASE_URL}/trades", json=trade_data, headers=headers)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        trade = response.json()
        print(f"   Trade created: {trade['amount']} shares at ${trade['price_per_share']}")
    else:
        print(f"   Error: {response.json()}")
    
    # Test creating a vote
    print("\n7. Testing vote creation...")
    vote_data = {
        "outcome": "outcome_a",
        "confidence": 0.8,
        "market_id": market_id
    }
    response = requests.post(f"{BASE_URL}/votes", json=vote_data, headers=headers)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        vote = response.json()
        print(f"   Vote created: {vote['outcome']} with {vote['confidence']} confidence")
    else:
        print(f"   Error: {response.json()}")
    
    # Test getting user profile
    print("\n8. Testing get user profile...")
    response = requests.get(f"{BASE_URL}/users/me", headers=headers)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        user = response.json()
        print(f"   User: {user['username']} - {user['total_trades']} trades")
    
    print("\n✅ API testing completed!")

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API. Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
