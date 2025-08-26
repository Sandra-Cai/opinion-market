from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
from typing import Dict, Any

def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Opinion Market API",
        version="2.0.0",
        description="""
# Opinion Market API Documentation

## Overview
The Opinion Market API is a comprehensive prediction market platform that allows users to trade and vote on their opinions. This API provides real-time market data, advanced trading features, social networking, and AI-powered analytics.

## Key Features
- **Prediction Markets**: Create and trade on binary outcome markets
- **Real-time Data**: Live market feeds and WebSocket connections
- **Advanced Trading**: Stop-loss, take-profit, and conditional orders
- **Social Features**: User profiles, communities, and social trading
- **AI Analytics**: Machine learning-powered predictions and insights
- **Blockchain Integration**: Smart contracts and DeFi features
- **Mobile Support**: Optimized endpoints for mobile applications
- **Governance**: DAO-style voting and proposal system

## Authentication
Most endpoints require authentication using JWT tokens. Include the token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

## Rate Limiting
API requests are rate-limited to ensure fair usage:
- **Standard users**: 100 requests per minute
- **Premium users**: 500 requests per minute
- **Enterprise users**: 2000 requests per minute

## Error Handling
The API uses standard HTTP status codes:
- `200`: Success
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

## WebSocket Endpoints
Real-time data is available through WebSocket connections:
- `/ws/market-data`: Live market price updates
- `/ws/market-alerts`: Real-time market alerts
- `/ws/trades`: Live trade notifications

## Getting Started
1. Register an account using `/auth/register`
2. Obtain a JWT token via `/auth/login`
3. Start exploring markets with `/markets`
4. Begin trading with `/trades`

For more information, visit our [developer portal](https://docs.opinionmarket.com).
        """,
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://opinionmarket.com/logo.png"
    }
    
    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "https://api.opinionmarket.com/v1",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.opinionmarket.com/v1",
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000/v1",
            "description": "Local development server"
        }
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /auth/login"
        },
        "apiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for enterprise users"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [
        {"bearerAuth": []}
    ]
    
    # Add tags with descriptions
    openapi_schema["tags"] = [
        {
            "name": "authentication",
            "description": "User authentication and authorization endpoints"
        },
        {
            "name": "users",
            "description": "User management and profile operations"
        },
        {
            "name": "markets",
            "description": "Prediction market creation, management, and data"
        },
        {
            "name": "trades",
            "description": "Trading operations and order management"
        },
        {
            "name": "votes",
            "description": "Market outcome voting and resolution"
        },
        {
            "name": "positions",
            "description": "User position tracking and portfolio management"
        },
        {
            "name": "websocket",
            "description": "Real-time WebSocket connections for live data"
        },
        {
            "name": "leaderboard",
            "description": "User rankings and performance metrics"
        },
        {
            "name": "disputes",
            "description": "Market dispute resolution and moderation"
        },
        {
            "name": "notifications",
            "description": "User notification management and preferences"
        },
        {
            "name": "analytics",
            "description": "Platform analytics and reporting"
        },
        {
            "name": "verification",
            "description": "Market verification and quality control"
        },
        {
            "name": "orders",
            "description": "Advanced order types and management"
        },
        {
            "name": "governance",
            "description": "DAO governance and proposal system"
        },
        {
            "name": "advanced-markets",
            "description": "Advanced market types (futures, options, etc.)"
        },
        {
            "name": "ai-analytics",
            "description": "AI-powered analytics and insights"
        },
        {
            "name": "rewards",
            "description": "Gamification and reward system"
        },
        {
            "name": "mobile",
            "description": "Mobile-optimized API endpoints"
        },
        {
            "name": "advanced-orders",
            "description": "Advanced order types (stop-loss, take-profit, etc.)"
        },
        {
            "name": "market-data",
            "description": "Real-time market data feeds and alerts"
        },
        {
            "name": "ml-analytics",
            "description": "Machine learning analytics and predictions"
        },
        {
            "name": "social",
            "description": "Social features, profiles, and communities"
        }
    ]
    
    # Add examples
    openapi_schema["components"]["examples"] = {
        "MarketCreation": {
            "summary": "Create a new prediction market",
            "value": {
                "title": "Will Bitcoin reach $100,000 by end of 2024?",
                "description": "A prediction market on Bitcoin's price movement",
                "outcome_a": "Yes",
                "outcome_b": "No",
                "category": "cryptocurrency",
                "end_date": "2024-12-31T23:59:59Z",
                "initial_liquidity": 1000.0
            }
        },
        "TradeExecution": {
            "summary": "Execute a trade",
            "value": {
                "market_id": 1,
                "outcome": "outcome_a",
                "shares": 10.0,
                "order_type": "market"
            }
        },
        "UserProfile": {
            "summary": "User profile data",
            "value": {
                "display_name": "CryptoTrader",
                "bio": "Passionate about cryptocurrency trading and prediction markets",
                "avatar_url": "https://example.com/avatar.jpg",
                "location": "San Francisco, CA",
                "website": "https://cryptotrader.com",
                "social_links": {
                    "twitter": "@cryptotrader",
                    "linkedin": "linkedin.com/in/cryptotrader"
                }
            }
        },
        "SocialPost": {
            "summary": "Create a social post",
            "value": {
                "content": "Just made a bullish bet on Bitcoin! 🚀 #crypto #trading",
                "post_type": "trade_alert",
                "market_id": 1,
                "tags": ["crypto", "trading", "bitcoin"]
            }
        }
    }
    
    # Add response examples
    openapi_schema["components"]["responses"] = {
        "ValidationError": {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "title"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        },
        "UnauthorizedError": {
            "description": "Authentication required",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Not authenticated"
                    }
                }
            }
        },
        "RateLimitError": {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Rate limit exceeded. Try again in 60 seconds."
                    }
                }
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

def get_api_documentation_links() -> Dict[str, str]:
    """Get links to various API documentation resources"""
    return {
        "swagger_ui": "/docs",
        "redoc": "/redoc",
        "openapi_json": "/openapi.json",
        "developer_portal": "https://docs.opinionmarket.com",
        "api_status": "https://status.opinionmarket.com",
        "github": "https://github.com/opinionmarket/api",
        "support": "https://support.opinionmarket.com",
        "changelog": "https://docs.opinionmarket.com/changelog"
    }

def get_api_usage_examples() -> Dict[str, Any]:
    """Get comprehensive API usage examples"""
    return {
        "authentication": {
            "register": {
                "endpoint": "POST /auth/register",
                "description": "Register a new user account",
                "example": {
                    "username": "cryptotrader",
                    "email": "trader@example.com",
                    "password": "securepassword123"
                }
            },
            "login": {
                "endpoint": "POST /auth/login",
                "description": "Authenticate and get JWT token",
                "example": {
                    "username": "cryptotrader",
                    "password": "securepassword123"
                }
            }
        },
        "markets": {
            "create_market": {
                "endpoint": "POST /markets",
                "description": "Create a new prediction market",
                "example": {
                    "title": "Will Ethereum 2.0 launch in Q1 2024?",
                    "description": "Prediction market on Ethereum's upgrade timeline",
                    "outcome_a": "Yes",
                    "outcome_b": "No",
                    "category": "cryptocurrency",
                    "end_date": "2024-03-31T23:59:59Z",
                    "initial_liquidity": 500.0
                }
            },
            "get_markets": {
                "endpoint": "GET /markets",
                "description": "Get list of available markets",
                "example": {
                    "category": "politics",
                    "status": "active",
                    "limit": 20,
                    "offset": 0
                }
            }
        },
        "trading": {
            "place_trade": {
                "endpoint": "POST /trades",
                "description": "Execute a trade",
                "example": {
                    "market_id": 1,
                    "outcome": "outcome_a",
                    "shares": 5.0,
                    "order_type": "market"
                }
            },
            "get_positions": {
                "endpoint": "GET /positions",
                "description": "Get user's current positions",
                "example": {
                    "market_id": 1,
                    "include_closed": False
                }
            }
        },
        "social": {
            "create_post": {
                "endpoint": "POST /social/posts",
                "description": "Create a social post",
                "example": {
                    "content": "Just made a bullish bet on Tesla! 📈 #stocks #tesla",
                    "post_type": "trade_alert",
                    "market_id": 2,
                    "tags": ["stocks", "tesla", "trading"]
                }
            },
            "follow_user": {
                "endpoint": "POST /social/follow/{user_id}",
                "description": "Follow another user",
                "example": {
                    "user_id": 123
                }
            }
        },
        "analytics": {
            "get_market_prediction": {
                "endpoint": "GET /ml-analytics/market/{market_id}/prediction",
                "description": "Get AI-powered market prediction",
                "example": {
                    "market_id": 1,
                    "horizon": "24h"
                }
            },
            "get_user_insights": {
                "endpoint": "GET /ml-analytics/analytics/user-insights",
                "description": "Get personalized trading insights",
                "example": {}
            }
        },
        "websocket": {
            "market_data": {
                "endpoint": "WebSocket /ws/market-data",
                "description": "Subscribe to real-time market data",
                "example": {
                    "action": "subscribe",
                    "markets": [1, 2, 3]
                }
            }
        }
    }

def get_api_rate_limits() -> Dict[str, Any]:
    """Get API rate limiting information"""
    return {
        "standard_user": {
            "requests_per_minute": 100,
            "requests_per_hour": 5000,
            "requests_per_day": 100000
        },
        "premium_user": {
            "requests_per_minute": 500,
            "requests_per_hour": 25000,
            "requests_per_day": 500000
        },
        "enterprise_user": {
            "requests_per_minute": 2000,
            "requests_per_hour": 100000,
            "requests_per_day": 2000000
        },
        "websocket_connections": {
            "standard_user": 5,
            "premium_user": 20,
            "enterprise_user": 100
        }
    }

def get_api_error_codes() -> Dict[str, Any]:
    """Get comprehensive API error codes and descriptions"""
    return {
        "authentication_errors": {
            "AUTH_001": "Invalid credentials",
            "AUTH_002": "Token expired",
            "AUTH_003": "Insufficient permissions",
            "AUTH_004": "Account locked",
            "AUTH_005": "Two-factor authentication required"
        },
        "market_errors": {
            "MARKET_001": "Market not found",
            "MARKET_002": "Market closed for trading",
            "MARKET_003": "Insufficient liquidity",
            "MARKET_004": "Invalid market parameters",
            "MARKET_005": "Market resolution pending"
        },
        "trading_errors": {
            "TRADE_001": "Insufficient balance",
            "TRADE_002": "Invalid order parameters",
            "TRADE_003": "Order execution failed",
            "TRADE_004": "Position limit exceeded",
            "TRADE_005": "Market manipulation detected"
        },
        "user_errors": {
            "USER_001": "User not found",
            "USER_002": "Username already exists",
            "USER_003": "Email already registered",
            "USER_004": "Invalid user data",
            "USER_005": "Account verification required"
        },
        "system_errors": {
            "SYS_001": "Internal server error",
            "SYS_002": "Service temporarily unavailable",
            "SYS_003": "Database connection error",
            "SYS_004": "External service error",
            "SYS_005": "Rate limit exceeded"
        }
    }

def get_api_webhooks() -> Dict[str, Any]:
    """Get webhook documentation and examples"""
    return {
        "available_webhooks": {
            "trade_executed": {
                "description": "Triggered when a trade is executed",
                "endpoint": "POST /webhooks/trade-executed",
                "payload": {
                    "event": "trade_executed",
                    "trade_id": "trade_123",
                    "user_id": 456,
                    "market_id": 1,
                    "outcome": "outcome_a",
                    "shares": 10.0,
                    "price": 0.65,
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            },
            "market_resolved": {
                "description": "Triggered when a market is resolved",
                "endpoint": "POST /webhooks/market-resolved",
                "payload": {
                    "event": "market_resolved",
                    "market_id": 1,
                    "winning_outcome": "outcome_a",
                    "resolution_reason": "Official announcement",
                    "timestamp": "2024-01-15T23:59:59Z"
                }
            },
            "user_registered": {
                "description": "Triggered when a new user registers",
                "endpoint": "POST /webhooks/user-registered",
                "payload": {
                    "event": "user_registered",
                    "user_id": 789,
                    "username": "newuser",
                    "email": "user@example.com",
                    "timestamp": "2024-01-15T09:00:00Z"
                }
            }
        },
        "webhook_security": {
            "authentication": "Webhooks use HMAC-SHA256 signatures",
            "signature_header": "X-Webhook-Signature",
            "verification": "Verify signature using your webhook secret"
        }
    }

def get_api_sdks() -> Dict[str, Any]:
    """Get SDK and client library information"""
    return {
        "official_sdks": {
            "python": {
                "repository": "https://github.com/opinionmarket/python-sdk",
                "documentation": "https://docs.opinionmarket.com/python",
                "installation": "pip install opinionmarket-sdk",
                "version": "2.0.0"
            },
            "javascript": {
                "repository": "https://github.com/opinionmarket/javascript-sdk",
                "documentation": "https://docs.opinionmarket.com/javascript",
                "installation": "npm install @opinionmarket/sdk",
                "version": "2.0.0"
            },
            "java": {
                "repository": "https://github.com/opinionmarket/java-sdk",
                "documentation": "https://docs.opinionmarket.com/java",
                "installation": "Maven dependency available",
                "version": "2.0.0"
            }
        },
        "community_libraries": {
            "go": "https://github.com/community/opinionmarket-go",
            "rust": "https://github.com/community/opinionmarket-rust",
            "php": "https://github.com/community/opinionmarket-php"
        }
    }
