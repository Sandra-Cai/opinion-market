# Opinion Market API Documentation

## Overview

The Opinion Market API is a comprehensive prediction market platform that allows users to create, trade, and manage prediction markets. It provides real-time trading, advanced analytics, and social features.

## Base URL

- **Production**: `https://api.opinionmarket.com`
- **Development**: `http://localhost:8000`

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_access_token>
```

### Getting Access Token

1. **Register** a new account or **login** with existing credentials
2. Use the returned `access_token` in subsequent requests
3. Use `refresh_token` to get new access tokens when they expire

## Rate Limiting

- **General API**: 100 requests per minute
- **Authentication**: 5 requests per minute
- **WebSocket**: 20 connections per minute

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

## Endpoints

### Authentication

#### Register User
```http
POST /api/v1/auth/register
```

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "full_name": "string",
  "bio": "string" // optional
}
```

**Response:**
```json
{
  "id": 1,
  "username": "string",
  "email": "string",
  "full_name": "string",
  "is_active": true,
  "is_verified": false,
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### Login
```http
POST /api/v1/auth/login
```

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
```

**Request Body:**
```json
{
  "refresh_token": "string"
}
```

#### Get Current User
```http
GET /api/v1/auth/me
```

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "id": 1,
  "username": "string",
  "email": "string",
  "full_name": "string",
  "bio": "string",
  "avatar_url": "string",
  "is_active": true,
  "is_verified": true,
  "is_premium": false,
  "preferences": {},
  "notification_settings": {},
  "created_at": "2024-01-01T00:00:00Z",
  "last_login": "2024-01-01T00:00:00Z"
}
```

#### Change Password
```http
POST /api/v1/auth/change-password
```

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "current_password": "string",
  "new_password": "string"
}
```

#### Logout
```http
POST /api/v1/auth/logout
```

**Headers:** `Authorization: Bearer <token>`

### Markets

#### Create Market
```http
POST /api/v1/markets/
```

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "title": "string",
  "description": "string",
  "question": "string",
  "category": "POLITICS|SPORTS|ECONOMICS|TECHNOLOGY|ENTERTAINMENT|SCIENCE|OTHER",
  "outcome_a": "string",
  "outcome_b": "string",
  "closes_at": "2024-01-01T00:00:00Z",
  "resolution_criteria": "string",
  "initial_liquidity": 1000.0,
  "trading_fee": 0.02
}
```

**Response:**
```json
{
  "id": 1,
  "title": "string",
  "description": "string",
  "question": "string",
  "category": "POLITICS",
  "outcome_a": "string",
  "outcome_b": "string",
  "creator_id": 1,
  "closes_at": "2024-01-01T00:00:00Z",
  "status": "OPEN",
  "price_a": 0.5,
  "price_b": 0.5,
  "volume_total": 0.0,
  "volume_24h": 0.0,
  "trending_score": 0.0,
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### List Markets
```http
GET /api/v1/markets/
```

**Query Parameters:**
- `category` - Filter by category
- `status` - Filter by status (OPEN, CLOSED, RESOLVED, CANCELLED)
- `trending` - Show only trending markets (true/false)
- `search` - Search in title and description
- `skip` - Number of records to skip (default: 0)
- `limit` - Number of records to return (default: 20, max: 100)

**Response:**
```json
{
  "markets": [
    {
      "id": 1,
      "title": "string",
      "description": "string",
      "question": "string",
      "category": "POLITICS",
      "outcome_a": "string",
      "outcome_b": "string",
      "creator_id": 1,
      "closes_at": "2024-01-01T00:00:00Z",
      "status": "OPEN",
      "price_a": 0.5,
      "price_b": 0.5,
      "volume_total": 0.0,
      "volume_24h": 0.0,
      "trending_score": 0.0,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 100,
  "skip": 0,
  "limit": 20
}
```

#### Get Market
```http
GET /api/v1/markets/{market_id}
```

**Response:**
```json
{
  "id": 1,
  "title": "string",
  "description": "string",
  "question": "string",
  "category": "POLITICS",
  "outcome_a": "string",
  "outcome_b": "string",
  "creator_id": 1,
  "closes_at": "2024-01-01T00:00:00Z",
  "status": "OPEN",
  "price_a": 0.5,
  "price_b": 0.5,
  "volume_total": 0.0,
  "volume_24h": 0.0,
  "trending_score": 0.0,
  "sentiment_score": 0.0,
  "liquidity_score": 0.0,
  "risk_score": 0.0,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

#### Get Trending Markets
```http
GET /api/v1/markets/trending
```

**Response:**
```json
{
  "markets": [
    {
      "id": 1,
      "title": "string",
      "trending_score": 0.95,
      "price_a": 0.5,
      "price_b": 0.5,
      "volume_24h": 1000.0
    }
  ]
}
```

#### Get Market Stats
```http
GET /api/v1/markets/stats
```

**Response:**
```json
{
  "total_markets": 1000,
  "active_markets": 500,
  "total_volume": 1000000.0,
  "volume_24h": 50000.0,
  "top_categories": [
    {
      "category": "POLITICS",
      "count": 300,
      "volume": 400000.0
    }
  ]
}
```

### Trading

#### Create Trade
```http
POST /api/v1/trades/
```

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "trade_type": "BUY|SELL",
  "outcome": "OUTCOME_A|OUTCOME_B",
  "amount": 10.0,
  "market_id": 1
}
```

**Response:**
```json
{
  "id": 1,
  "trade_type": "BUY",
  "outcome": "OUTCOME_A",
  "amount": 10.0,
  "price_per_share": 0.5,
  "total_value": 5.0,
  "fee": 0.1,
  "status": "COMPLETED",
  "market_id": 1,
  "user_id": 1,
  "trade_hash": "string",
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### Get User Trades
```http
GET /api/v1/trades/
```

**Headers:** `Authorization: Bearer <token>`

**Query Parameters:**
- `market_id` - Filter by market ID
- `status` - Filter by status
- `skip` - Number of records to skip
- `limit` - Number of records to return

**Response:**
```json
{
  "trades": [
    {
      "id": 1,
      "trade_type": "BUY",
      "outcome": "OUTCOME_A",
      "amount": 10.0,
      "price_per_share": 0.5,
      "total_value": 5.0,
      "fee": 0.1,
      "status": "COMPLETED",
      "market_id": 1,
      "trade_hash": "string",
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 50,
  "skip": 0,
  "limit": 20
}
```

#### Get Trade
```http
GET /api/v1/trades/{trade_id}
```

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "id": 1,
  "trade_type": "BUY",
  "outcome": "OUTCOME_A",
  "amount": 10.0,
  "price_a": 0.5,
  "price_b": 0.5,
  "price_per_share": 0.5,
  "total_value": 5.0,
  "fee": 0.1,
  "price_impact": 0.01,
  "slippage": 0.005,
  "status": "COMPLETED",
  "market_id": 1,
  "user_id": 1,
  "trade_hash": "string",
  "created_at": "2024-01-01T00:00:00Z",
  "executed_at": "2024-01-01T00:00:00Z"
}
```

### Orders

#### Create Order
```http
POST /api/v1/orders/
```

**Headers:** `Authorization: Bearer <token>`

**Request Body:**
```json
{
  "market_id": 1,
  "order_type": "MARKET|LIMIT|STOP|STOP_LIMIT",
  "trade_type": "BUY|SELL",
  "outcome": "OUTCOME_A|OUTCOME_B",
  "amount": 10.0,
  "price": 0.5,
  "stop_price": 0.4,
  "time_in_force": "GTC|IOC|FOK",
  "expires_at": "2024-01-01T00:00:00Z"
}
```

**Response:**
```json
{
  "id": 1,
  "market_id": 1,
  "order_type": "LIMIT",
  "trade_type": "BUY",
  "outcome": "OUTCOME_A",
  "amount": 10.0,
  "price": 0.5,
  "status": "PENDING",
  "time_in_force": "GTC",
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### Get User Orders
```http
GET /api/v1/orders/
```

**Headers:** `Authorization: Bearer <token>`

**Query Parameters:**
- `market_id` - Filter by market ID
- `status` - Filter by status
- `skip` - Number of records to skip
- `limit` - Number of records to return

**Response:**
```json
{
  "orders": [
    {
      "id": 1,
      "market_id": 1,
      "order_type": "LIMIT",
      "trade_type": "BUY",
      "outcome": "OUTCOME_A",
      "amount": 10.0,
      "price": 0.5,
      "status": "PENDING",
      "filled_amount": 0.0,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 25,
  "skip": 0,
  "limit": 20
}
```

#### Cancel Order
```http
DELETE /api/v1/orders/{order_id}
```

**Headers:** `Authorization: Bearer <token>`

**Response:**
```json
{
  "message": "Order cancelled successfully"
}
```

### WebSocket

#### Connect to WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
    
    // Subscribe to market updates
    ws.send(JSON.stringify({
        type: 'subscribe',
        channel: 'market_updates',
        market_id: 1
    }));
    
    // Subscribe to user trades
    ws.send(JSON.stringify({
        type: 'subscribe',
        channel: 'user_trades',
        token: 'your_access_token'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

ws.onclose = function(event) {
    console.log('WebSocket connection closed');
};
```

#### WebSocket Message Types

**Subscribe to Channel:**
```json
{
  "type": "subscribe",
  "channel": "market_updates|user_trades|price_updates",
  "market_id": 1,
  "token": "your_access_token"
}
```

**Unsubscribe from Channel:**
```json
{
  "type": "unsubscribe",
  "channel": "market_updates",
  "market_id": 1
}
```

**Ping:**
```json
{
  "type": "ping"
}
```

#### WebSocket Response Types

**Market Update:**
```json
{
  "type": "market_update",
  "market_id": 1,
  "price_a": 0.52,
  "price_b": 0.48,
  "volume_24h": 1000.0,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Trade Update:**
```json
{
  "type": "trade_update",
  "trade_id": 1,
  "market_id": 1,
  "trade_type": "BUY",
  "outcome": "OUTCOME_A",
  "amount": 10.0,
  "price_per_share": 0.5,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Price Update:**
```json
{
  "type": "price_update",
  "market_id": 1,
  "price_a": 0.52,
  "price_b": 0.48,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Pong:**
```json
{
  "type": "pong",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Data Models

### User
```json
{
  "id": 1,
  "username": "string",
  "email": "string",
  "full_name": "string",
  "bio": "string",
  "avatar_url": "string",
  "is_active": true,
  "is_verified": true,
  "is_premium": false,
  "preferences": {},
  "notification_settings": {},
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "last_login": "2024-01-01T00:00:00Z"
}
```

### Market
```json
{
  "id": 1,
  "title": "string",
  "description": "string",
  "question": "string",
  "category": "POLITICS|SPORTS|ECONOMICS|TECHNOLOGY|ENTERTAINMENT|SCIENCE|OTHER",
  "outcome_a": "string",
  "outcome_b": "string",
  "creator_id": 1,
  "closes_at": "2024-01-01T00:00:00Z",
  "resolved_at": "2024-01-01T00:00:00Z",
  "resolution_criteria": "string",
  "status": "OPEN|CLOSED|RESOLVED|CANCELLED",
  "price_a": 0.5,
  "price_b": 0.5,
  "volume_total": 0.0,
  "volume_24h": 0.0,
  "trending_score": 0.0,
  "sentiment_score": 0.0,
  "liquidity_score": 0.0,
  "risk_score": 0.0,
  "initial_liquidity": 1000.0,
  "trading_fee": 0.02,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "closed_at": "2024-01-01T00:00:00Z"
}
```

### Trade
```json
{
  "id": 1,
  "trade_type": "BUY|SELL",
  "outcome": "OUTCOME_A|OUTCOME_B",
  "amount": 10.0,
  "price_a": 0.5,
  "price_b": 0.5,
  "price_per_share": 0.5,
  "total_value": 5.0,
  "status": "PENDING|EXECUTED|COMPLETED|CANCELLED|FAILED|PARTIALLY_FILLED",
  "fee": 0.1,
  "price_impact": 0.01,
  "slippage": 0.005,
  "order_id": 1,
  "fill_amount": 10.0,
  "fill_price": 0.5,
  "market_id": 1,
  "user_id": 1,
  "trade_hash": "string",
  "gas_fee": 0.0,
  "additional_data": {},
  "created_at": "2024-01-01T00:00:00Z",
  "executed_at": "2024-01-01T00:00:00Z"
}
```

### Order
```json
{
  "id": 1,
  "user_id": 1,
  "market_id": 1,
  "order_type": "MARKET|LIMIT|STOP|STOP_LIMIT",
  "trade_type": "BUY|SELL",
  "outcome": "OUTCOME_A|OUTCOME_B",
  "amount": 10.0,
  "price": 0.5,
  "stop_price": 0.4,
  "status": "PENDING|PARTIALLY_FILLED|FILLED|CANCELLED|REJECTED",
  "time_in_force": "GTC|IOC|FOK",
  "expires_at": "2024-01-01T00:00:00Z",
  "filled_amount": 0.0,
  "average_price": 0.5,
  "total_fees": 0.0,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z",
  "filled_at": "2024-01-01T00:00:00Z"
}
```

## SDKs and Libraries

### Python
```python
import requests

class OpinionMarketAPI:
    def __init__(self, base_url, access_token=None):
        self.base_url = base_url
        self.access_token = access_token
        self.session = requests.Session()
        
        if access_token:
            self.session.headers.update({
                'Authorization': f'Bearer {access_token}'
            })
    
    def login(self, username, password):
        response = self.session.post(
            f'{self.base_url}/api/v1/auth/login',
            data={'username': username, 'password': password}
        )
        data = response.json()
        self.access_token = data['access_token']
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}'
        })
        return data
    
    def get_markets(self, **kwargs):
        response = self.session.get(
            f'{self.base_url}/api/v1/markets/',
            params=kwargs
        )
        return response.json()
    
    def create_trade(self, trade_data):
        response = self.session.post(
            f'{self.base_url}/api/v1/trades/',
            json=trade_data
        )
        return response.json()

# Usage
api = OpinionMarketAPI('http://localhost:8000')
api.login('username', 'password')
markets = api.get_markets(category='POLITICS')
```

### JavaScript
```javascript
class OpinionMarketAPI {
    constructor(baseUrl, accessToken = null) {
        this.baseUrl = baseUrl;
        this.accessToken = accessToken;
    }
    
    async login(username, password) {
        const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                username,
                password
            })
        });
        
        const data = await response.json();
        this.accessToken = data.access_token;
        return data;
    }
    
    async getMarkets(params = {}) {
        const url = new URL(`${this.baseUrl}/api/v1/markets/`);
        Object.keys(params).forEach(key => 
            url.searchParams.append(key, params[key])
        );
        
        const response = await fetch(url, {
            headers: {
                'Authorization': `Bearer ${this.accessToken}`
            }
        });
        
        return response.json();
    }
    
    async createTrade(tradeData) {
        const response = await fetch(`${this.baseUrl}/api/v1/trades/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.accessToken}`
            },
            body: JSON.stringify(tradeData)
        });
        
        return response.json();
    }
}

// Usage
const api = new OpinionMarketAPI('http://localhost:8000');
await api.login('username', 'password');
const markets = await api.getMarkets({ category: 'POLITICS' });
```

## Examples

### Complete Trading Flow

1. **Register and Login**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "trader1",
    "email": "trader1@example.com",
    "password": "SecurePassword123!",
    "full_name": "Trader One"
  }'

curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=trader1&password=SecurePassword123!"
```

2. **Browse Markets**
```bash
curl -X GET "http://localhost:8000/api/v1/markets/?category=POLITICS&limit=10" \
  -H "Authorization: Bearer <access_token>"
```

3. **Create a Trade**
```bash
curl -X POST "http://localhost:8000/api/v1/trades/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <access_token>" \
  -d '{
    "trade_type": "BUY",
    "outcome": "OUTCOME_A",
    "amount": 10.0,
    "market_id": 1
  }'
```

4. **Monitor with WebSocket**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws');
ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'subscribe',
        channel: 'market_updates',
        market_id: 1
    }));
};
```

## Support

For API support and questions:
- **Email**: api-support@opinionmarket.com
- **Documentation**: https://docs.opinionmarket.com
- **Status Page**: https://status.opinionmarket.com


