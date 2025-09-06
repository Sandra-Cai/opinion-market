from typing import Dict, List, Any
from datetime import datetime
import json


class APIDocumentation:
    """Comprehensive API documentation system"""

    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.api_version = "v1"
        self.api_base = f"{self.base_url}/api/{self.api_version}"

    def get_api_overview(self) -> Dict:
        """Get API overview and general information"""
        return {
            "title": "Opinion Market API",
            "version": "1.0.0",
            "description": "A comprehensive API for the Opinion Market prediction platform",
            "base_url": self.api_base,
            "authentication": {
                "type": "Bearer Token",
                "description": "All authenticated endpoints require a Bearer token in the Authorization header",
                "example": "Authorization: Bearer <your_jwt_token>",
            },
            "rate_limits": {
                "auth": "5 requests per 5 minutes",
                "trades": "100 requests per minute",
                "orders": "50 requests per minute",
                "default": "1000 requests per minute",
            },
            "response_format": {
                "success": {
                    "status": "HTTP status code",
                    "data": "Response data",
                    "message": "Success message (optional)",
                },
                "error": {
                    "status": "HTTP status code",
                    "error": "Error type",
                    "detail": "Error description",
                },
            },
            "features": [
                "User Authentication & Authorization",
                "Market Creation & Management",
                "Trading & Order Management",
                "Portfolio Management",
                "Real-time WebSocket Feeds",
                "AI-Powered Analytics",
                "Governance System",
                "Advanced Market Types",
                "Rewards & Gamification",
                "Mobile API Support",
                "Security & Fraud Detection",
            ],
        }

    def get_endpoint_documentation(self) -> Dict:
        """Get detailed endpoint documentation"""
        return {
            "authentication": {
                "description": "User authentication and authorization endpoints",
                "endpoints": {
                    "POST /auth/register": {
                        "description": "Register a new user account",
                        "request_body": {
                            "username": "string (required, 3-50 chars)",
                            "email": "string (required, valid email)",
                            "password": "string (required, min 8 chars)",
                            "full_name": "string (optional)",
                            "bio": "string (optional)",
                        },
                        "response": {
                            "id": "integer",
                            "username": "string",
                            "email": "string",
                            "full_name": "string",
                            "created_at": "datetime",
                        },
                        "example": {
                            "request": {
                                "username": "trader123",
                                "email": "trader@example.com",
                                "password": "securepassword123",
                                "full_name": "John Doe",
                            },
                            "response": {
                                "id": 1,
                                "username": "trader123",
                                "email": "trader@example.com",
                                "full_name": "John Doe",
                                "created_at": "2024-01-15T10:30:00Z",
                            },
                        },
                    },
                    "POST /auth/login": {
                        "description": "Login and get access token",
                        "request_body": {
                            "username": "string (required)",
                            "password": "string (required)",
                        },
                        "response": {
                            "access_token": "string",
                            "token_type": "bearer",
                            "expires_in": "integer (minutes)",
                        },
                    },
                },
            },
            "markets": {
                "description": "Market creation, management, and trading",
                "endpoints": {
                    "GET /markets/": {
                        "description": "Get list of markets with filtering and pagination",
                        "query_parameters": {
                            "skip": "integer (default: 0)",
                            "limit": "integer (default: 20, max: 100)",
                            "category": "string (optional)",
                            "status": "string (optional: open, closed, resolved)",
                            "search": "string (optional)",
                            "sort_by": "string (optional: volume, trending, created_at)",
                            "sort_order": "string (optional: asc, desc)",
                        },
                        "response": {
                            "markets": "array of market objects",
                            "total": "integer",
                            "page": "integer",
                            "per_page": "integer",
                        },
                    },
                    "POST /markets/": {
                        "description": "Create a new prediction market",
                        "request_body": {
                            "title": "string (required, 5-200 chars)",
                            "description": "string (required, 20-2000 chars)",
                            "category": "string (required)",
                            "question": "string (required)",
                            "outcome_a": "string (required)",
                            "outcome_b": "string (required)",
                            "closes_at": "datetime (required)",
                            "total_liquidity": "float (required, min: 100)",
                            "tags": "array of strings (optional)",
                        },
                        "response": "Market object with all details",
                    },
                    "GET /markets/{market_id}": {
                        "description": "Get detailed information about a specific market",
                        "path_parameters": {"market_id": "integer (required)"},
                        "response": "Complete market object with all properties",
                    },
                },
            },
            "trades": {
                "description": "Trading operations and trade history",
                "endpoints": {
                    "POST /trades/": {
                        "description": "Execute a trade on a market",
                        "request_body": {
                            "market_id": "integer (required)",
                            "trade_type": "string (required: buy, sell)",
                            "outcome": "string (required: outcome_a, outcome_b)",
                            "amount": "float (required, min: 1)",
                        },
                        "response": {
                            "id": "integer",
                            "market_id": "integer",
                            "user_id": "integer",
                            "trade_type": "string",
                            "outcome": "string",
                            "amount": "float",
                            "price_per_share": "float",
                            "total_value": "float",
                            "profit_loss": "float",
                            "created_at": "datetime",
                        },
                    },
                    "GET /trades/": {
                        "description": "Get user's trade history",
                        "query_parameters": {
                            "skip": "integer (default: 0)",
                            "limit": "integer (default: 20, max: 100)",
                            "market_id": "integer (optional)",
                            "trade_type": "string (optional: buy, sell)",
                        },
                        "response": {
                            "trades": "array of trade objects",
                            "total": "integer",
                            "page": "integer",
                            "per_page": "integer",
                        },
                    },
                },
            },
            "positions": {
                "description": "Portfolio and position management",
                "endpoints": {
                    "GET /positions/": {
                        "description": "Get user's portfolio positions",
                        "query_parameters": {
                            "skip": "integer (default: 0)",
                            "limit": "integer (default: 20, max: 100)",
                            "active_only": "boolean (default: true)",
                        },
                        "response": {
                            "positions": "array of position objects",
                            "total": "integer",
                            "page": "integer",
                            "per_page": "integer",
                        },
                    },
                    "GET /positions/portfolio": {
                        "description": "Get portfolio summary and statistics",
                        "response": {
                            "total_positions": "integer",
                            "active_positions": "integer",
                            "total_portfolio_value": "float",
                            "total_unrealized_pnl": "float",
                            "total_realized_pnl": "float",
                            "total_pnl": "float",
                            "portfolio_return_percentage": "float",
                        },
                    },
                },
            },
            "ai_analytics": {
                "description": "AI-powered analytics and insights",
                "endpoints": {
                    "GET /ai-analytics/market/{market_id}/prediction": {
                        "description": "Get AI-powered market prediction",
                        "response": {
                            "market_id": "integer",
                            "prediction": {
                                "short_term": "object",
                                "medium_term": "object",
                                "long_term": "object",
                            },
                            "confidence": "float",
                            "technical_indicators": "object",
                            "market_sentiment": "object",
                            "risk_assessment": "object",
                        },
                    },
                    "GET /ai-analytics/user/insights": {
                        "description": "Get AI-powered user insights",
                        "response": {
                            "performance_metrics": "object",
                            "trading_patterns": "object",
                            "risk_assessment": "object",
                            "recommendations": "array",
                        },
                    },
                },
            },
            "governance": {
                "description": "DAO governance and proposal system",
                "endpoints": {
                    "POST /governance/proposals": {
                        "description": "Create a new governance proposal",
                        "request_body": {
                            "title": "string (required, 5-200 chars)",
                            "description": "string (required, 20-2000 chars)",
                            "proposal_type": "string (required)",
                            "voting_start": "datetime (required)",
                            "voting_end": "datetime (required)",
                            "quorum_required": "float (default: 0.1)",
                            "majority_required": "float (default: 0.6)",
                        },
                    },
                    "POST /governance/proposals/{proposal_id}/vote": {
                        "description": "Vote on a governance proposal",
                        "request_body": {
                            "vote_type": "string (required: yes, no, abstain)",
                            "voting_power": "float (required)",
                            "reason": "string (optional)",
                        },
                    },
                },
            },
            "rewards": {
                "description": "Rewards and gamification system",
                "endpoints": {
                    "POST /rewards/daily-login": {
                        "description": "Claim daily login reward",
                        "response": {
                            "reward_type": "string",
                            "tokens_awarded": "integer",
                            "xp_gained": "integer",
                            "streak": "integer",
                        },
                    },
                    "GET /rewards/achievements": {
                        "description": "Get user's achievements and progress",
                        "response": {
                            "achievements": "array of achievement objects",
                            "total_unlocked": "integer",
                            "total_achievements": "integer",
                        },
                    },
                },
            },
            "mobile": {
                "description": "Mobile-optimized API endpoints",
                "endpoints": {
                    "GET /mobile/dashboard": {
                        "description": "Get mobile-optimized dashboard",
                        "response": {
                            "user": "object",
                            "recent_markets": "array",
                            "trending_markets": "array",
                            "recent_trades": "array",
                            "notifications": "object",
                        },
                    },
                    "GET /mobile/portfolio": {
                        "description": "Get mobile-optimized portfolio",
                        "response": {
                            "user": "object",
                            "portfolio_summary": "object",
                            "positions": "array",
                            "performance_chart": "array",
                        },
                    },
                },
            },
        }

    def get_code_examples(self) -> Dict:
        """Get code examples for different programming languages"""
        return {
            "python": {
                "authentication": {
                    "register": """
import requests

# Register a new user
response = requests.post('http://localhost:8000/api/v1/auth/register', json={
    'username': 'trader123',
    'email': 'trader@example.com',
    'password': 'securepassword123',
    'full_name': 'John Doe'
})

print(response.json())
""",
                    "login": """
import requests

# Login and get token
response = requests.post('http://localhost:8000/api/v1/auth/login', data={
    'username': 'trader123',
    'password': 'securepassword123'
})

token = response.json()['access_token']
headers = {'Authorization': f'Bearer {token}'}
""",
                    "create_market": """
# Create a new market
market_data = {
    'title': 'Will Bitcoin reach $100k by end of 2024?',
    'description': 'A prediction market for Bitcoin price target',
    'category': 'crypto',
    'question': 'Will Bitcoin reach $100,000 by December 31, 2024?',
    'outcome_a': 'Yes',
    'outcome_b': 'No',
    'closes_at': '2024-12-31T23:59:59Z',
    'total_liquidity': 10000.0
}

response = requests.post('http://localhost:8000/api/v1/markets/', 
                        json=market_data, headers=headers)
print(response.json())
""",
                    "place_trade": """
# Place a trade
trade_data = {
    'market_id': 1,
    'trade_type': 'buy',
    'outcome': 'outcome_a',
    'amount': 10.0
}

response = requests.post('http://localhost:8000/api/v1/trades/', 
                        json=trade_data, headers=headers)
print(response.json())
""",
                },
                "javascript": {
                    "authentication": """
// Register a new user
const registerUser = async () => {
    const response = await fetch('http://localhost:8000/api/v1/auth/register', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            username: 'trader123',
            email: 'trader@example.com',
            password: 'securepassword123',
            full_name: 'John Doe'
        })
    });
    
    const data = await response.json();
    console.log(data);
};

// Login and get token
const login = async () => {
    const response = await fetch('http://localhost:8000/api/v1/auth/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'username=trader123&password=securepassword123'
    });
    
    const data = await response.json();
    const token = data.access_token;
    return token;
};
""",
                    "trading": """
// Place a trade
const placeTrade = async (token) => {
    const response = await fetch('http://localhost:8000/api/v1/trades/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
            market_id: 1,
            trade_type: 'buy',
            outcome: 'outcome_a',
            amount: 10.0
        })
    });
    
    const data = await response.json();
    console.log(data);
};
""",
                },
                "curl": {
                    "authentication": """
# Register a new user
curl -X POST "http://localhost:8000/api/v1/auth/register" \\
     -H "Content-Type: application/json" \\
     -d '{
       "username": "trader123",
       "email": "trader@example.com",
       "password": "securepassword123",
       "full_name": "John Doe"
     }'

# Login and get token
curl -X POST "http://localhost:8000/api/v1/auth/login" \\
     -H "Content-Type: application/x-www-form-urlencoded" \\
     -d "username=trader123&password=securepassword123"

# Create a market (with token)
curl -X POST "http://localhost:8000/api/v1/markets/" \\
     -H "Content-Type: application/json" \\
     -H "Authorization: Bearer YOUR_TOKEN_HERE" \\
     -d '{
       "title": "Will Bitcoin reach $100k by end of 2024?",
       "description": "A prediction market for Bitcoin price target",
       "category": "crypto",
       "question": "Will Bitcoin reach $100,000 by December 31, 2024?",
       "outcome_a": "Yes",
       "outcome_b": "No",
       "closes_at": "2024-12-31T23:59:59Z",
       "total_liquidity": 10000.0
     }'
"""
                },
            },
            "websocket": {
                "python": """
import websockets
import json
import asyncio

async def connect_to_market_feed():
    uri = "ws://localhost:8000/api/v1/ws/market/1"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to market updates
        await websocket.send(json.dumps({
            "type": "subscribe",
            "market_id": 1
        }))
        
        # Listen for updates
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                print(f"Received: {data}")
            except websockets.exceptions.ConnectionClosed:
                break

# Run the WebSocket client
asyncio.run(connect_to_market_feed())
""",
                "javascript": """
// Connect to market WebSocket feed
const connectToMarketFeed = (marketId) => {
    const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/market/${marketId}`);
    
    ws.onopen = () => {
        console.log('Connected to market feed');
        // Subscribe to updates
        ws.send(JSON.stringify({
            type: 'subscribe',
            market_id: marketId
        }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received:', data);
        
        // Handle different message types
        switch(data.type) {
            case 'price_update':
                updatePriceDisplay(data);
                break;
            case 'trade_executed':
                updateTradeHistory(data);
                break;
            case 'market_resolved':
                handleMarketResolution(data);
                break;
        }
    };
    
    ws.onclose = () => {
        console.log('Disconnected from market feed');
    };
    
    return ws;
};

// Connect to user-specific feed
const connectToUserFeed = (token) => {
    const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/user?token=${token}`);
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('User update:', data);
        
        // Handle user-specific updates
        switch(data.type) {
            case 'portfolio_update':
                updatePortfolio(data);
                break;
            case 'trade_executed':
                updateUserTrades(data);
                break;
            case 'notification':
                showNotification(data);
                break;
        }
    };
};
""",
            },
        }

    def get_error_codes(self) -> Dict:
        """Get comprehensive error code documentation"""
        return {
            "400": {
                "title": "Bad Request",
                "description": "The request could not be understood or contained invalid parameters",
                "common_causes": [
                    "Missing required fields",
                    "Invalid data types",
                    "Validation errors",
                    "Insufficient balance for trade",
                ],
                "example": {
                    "status_code": 400,
                    "error": "ValidationError",
                    "detail": "Field 'username' is required",
                },
            },
            "401": {
                "title": "Unauthorized",
                "description": "Authentication is required and has failed or has not been provided",
                "common_causes": [
                    "Missing Authorization header",
                    "Invalid or expired token",
                    "Invalid credentials",
                ],
                "example": {
                    "status_code": 401,
                    "error": "Unauthorized",
                    "detail": "Invalid authentication credentials",
                },
            },
            "403": {
                "title": "Forbidden",
                "description": "The server understood the request but refuses to authorize it",
                "common_causes": [
                    "Insufficient permissions",
                    "Account suspended",
                    "Rate limit exceeded",
                ],
                "example": {
                    "status_code": 403,
                    "error": "Forbidden",
                    "detail": "Insufficient permissions to create markets",
                },
            },
            "404": {
                "title": "Not Found",
                "description": "The requested resource was not found",
                "common_causes": [
                    "Invalid market ID",
                    "User not found",
                    "Trade not found",
                ],
                "example": {
                    "status_code": 404,
                    "error": "NotFound",
                    "detail": "Market with ID 123 not found",
                },
            },
            "429": {
                "title": "Too Many Requests",
                "description": "Rate limit exceeded",
                "common_causes": [
                    "Too many requests per minute",
                    "API rate limit exceeded",
                ],
                "example": {
                    "status_code": 429,
                    "error": "RateLimitExceeded",
                    "detail": "Rate limit exceeded. Try again in 60 seconds.",
                },
            },
            "500": {
                "title": "Internal Server Error",
                "description": "An unexpected error occurred on the server",
                "common_causes": [
                    "Database connection issues",
                    "External service failures",
                    "System errors",
                ],
                "example": {
                    "status_code": 500,
                    "error": "InternalServerError",
                    "detail": "An unexpected error occurred. Please try again later.",
                },
            },
        }

    def get_rate_limits(self) -> Dict:
        """Get detailed rate limiting information"""
        return {
            "overview": "Rate limits are applied per IP address and user account to ensure fair usage and prevent abuse.",
            "limits": {
                "authentication": {
                    "requests": 5,
                    "window": "5 minutes",
                    "description": "Login and registration attempts",
                },
                "trades": {
                    "requests": 100,
                    "window": "1 minute",
                    "description": "Trade execution requests",
                },
                "orders": {
                    "requests": 50,
                    "window": "1 minute",
                    "description": "Order placement and management",
                },
                "market_creation": {
                    "requests": 10,
                    "window": "1 hour",
                    "description": "Market creation requests",
                },
                "api_requests": {
                    "requests": 1000,
                    "window": "1 minute",
                    "description": "General API requests",
                },
            },
            "headers": {
                "X-RateLimit-Limit": "Maximum requests allowed in the window",
                "X-RateLimit-Remaining": "Number of requests remaining in the current window",
                "X-RateLimit-Reset": "Time when the rate limit window resets (Unix timestamp)",
            },
            "best_practices": [
                "Implement exponential backoff for retries",
                "Cache responses when possible",
                "Use WebSocket connections for real-time data",
                "Monitor rate limit headers",
                "Implement proper error handling",
            ],
        }

    def get_webhook_documentation(self) -> Dict:
        """Get webhook documentation for integrations"""
        return {
            "overview": "Webhooks allow you to receive real-time notifications when events occur on the platform.",
            "setup": {
                "endpoint": "Your webhook endpoint URL",
                "events": "Array of events to subscribe to",
                "secret": "Optional secret for signature verification",
            },
            "events": {
                "market.created": {
                    "description": "Triggered when a new market is created",
                    "payload": {
                        "event": "market.created",
                        "market_id": "integer",
                        "title": "string",
                        "creator_id": "integer",
                        "created_at": "datetime",
                    },
                },
                "trade.executed": {
                    "description": "Triggered when a trade is executed",
                    "payload": {
                        "event": "trade.executed",
                        "trade_id": "integer",
                        "market_id": "integer",
                        "user_id": "integer",
                        "trade_type": "string",
                        "amount": "float",
                        "price_per_share": "float",
                        "executed_at": "datetime",
                    },
                },
                "market.resolved": {
                    "description": "Triggered when a market is resolved",
                    "payload": {
                        "event": "market.resolved",
                        "market_id": "integer",
                        "outcome": "string",
                        "resolved_at": "datetime",
                    },
                },
            },
            "security": {
                "signature_verification": "Webhook payloads include a signature for verification",
                "retry_policy": "Failed webhook deliveries are retried with exponential backoff",
                "timeout": "Webhook requests timeout after 30 seconds",
            },
        }


# Global API documentation instance
api_docs = APIDocumentation()


def get_api_documentation() -> APIDocumentation:
    """Get the global API documentation instance"""
    return api_docs
