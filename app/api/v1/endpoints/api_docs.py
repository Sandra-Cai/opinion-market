"""
Enhanced API Documentation Endpoints
Provides comprehensive API documentation with examples and usage patterns
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List
import json
from datetime import datetime

from app.core.database import get_db
from app.core.enhanced_config import enhanced_config_manager

router = APIRouter()


@router.get("/docs/overview")
async def get_api_overview():
    """
    Get comprehensive API overview with all available endpoints
    """
    return {
        "api_name": "Opinion Market API",
        "version": "2.0.0",
        "description": "A comprehensive prediction market platform with advanced trading features",
        "base_url": "http://localhost:8000/api/v1",
        "documentation_url": "http://localhost:8000/docs",
        "openapi_schema": "http://localhost:8000/openapi.json",
        "last_updated": datetime.utcnow().isoformat(),
        "endpoints": {
            "authentication": {
                "description": "User authentication and authorization",
                "endpoints": [
                    "POST /auth/register - Register new user",
                    "POST /auth/login - User login",
                    "POST /auth/logout - User logout",
                    "POST /auth/refresh - Refresh access token",
                    "GET /auth/me - Get current user info"
                ]
            },
            "users": {
                "description": "User management and profiles",
                "endpoints": [
                    "GET /users/ - List all users",
                    "GET /users/{user_id} - Get user details",
                    "PUT /users/{user_id} - Update user profile",
                    "DELETE /users/{user_id} - Delete user account",
                    "GET /users/{user_id}/trades - Get user's trading history",
                    "GET /users/{user_id}/positions - Get user's positions"
                ]
            },
            "markets": {
                "description": "Prediction market management",
                "endpoints": [
                    "GET /markets/ - List all markets",
                    "POST /markets/ - Create new market",
                    "GET /markets/{market_id} - Get market details",
                    "PUT /markets/{market_id} - Update market",
                    "DELETE /markets/{market_id} - Delete market",
                    "GET /markets/{market_id}/trades - Get market trades",
                    "GET /markets/{market_id}/votes - Get market votes"
                ]
            },
            "trading": {
                "description": "Trading operations and order management",
                "endpoints": [
                    "POST /trades/ - Execute trade",
                    "GET /trades/ - List trades",
                    "GET /trades/{trade_id} - Get trade details",
                    "POST /orders/ - Create order",
                    "GET /orders/ - List orders",
                    "PUT /orders/{order_id} - Update order",
                    "DELETE /orders/{order_id} - Cancel order"
                ]
            },
            "analytics": {
                "description": "Market analytics and insights",
                "endpoints": [
                    "GET /analytics/market/{market_id} - Market analytics",
                    "GET /analytics/user/{user_id} - User analytics",
                    "GET /analytics/global - Global market analytics",
                    "GET /ai-analytics/predictions - AI predictions",
                    "GET /ai-analytics/trends - Market trends"
                ]
            },
            "health": {
                "description": "System health and monitoring",
                "endpoints": [
                    "GET /health - Basic health check",
                    "GET /health/detailed - Detailed health information",
                    "GET /health/readiness - Readiness probe",
                    "GET /health/liveness - Liveness probe",
                    "GET /health/metrics - System metrics"
                ]
            }
        },
        "features": [
            "Real-time market data",
            "Advanced order types",
            "AI-powered analytics",
            "Social trading features",
            "Derivatives trading",
            "Forex trading",
            "Blockchain integration",
            "Comprehensive monitoring",
            "Enterprise security",
            "Performance optimization"
        ]
    }


@router.get("/docs/examples")
async def get_api_examples():
    """
    Get comprehensive API usage examples
    """
    return {
        "examples": {
            "authentication": {
                "register_user": {
                    "endpoint": "POST /api/v1/auth/register",
                    "description": "Register a new user account",
                    "request_body": {
                        "username": "john_doe",
                        "email": "john@example.com",
                        "password": "SecurePassword123!",
                        "full_name": "John Doe"
                    },
                    "response": {
                        "user_id": 123,
                        "username": "john_doe",
                        "email": "john@example.com",
                        "created_at": "2024-01-15T10:30:00Z",
                        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                        "token_type": "bearer"
                    }
                },
                "login": {
                    "endpoint": "POST /api/v1/auth/login",
                    "description": "Authenticate user and get access token",
                    "request_body": {
                        "username": "john_doe",
                        "password": "SecurePassword123!"
                    },
                    "response": {
                        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                        "token_type": "bearer",
                        "expires_in": 3600,
                        "user": {
                            "id": 123,
                            "username": "john_doe",
                            "email": "john@example.com"
                        }
                    }
                }
            },
            "markets": {
                "create_market": {
                    "endpoint": "POST /api/v1/markets/",
                    "description": "Create a new prediction market",
                    "request_body": {
                        "title": "Will Bitcoin reach $100,000 by end of 2024?",
                        "description": "Prediction market for Bitcoin price target",
                        "outcome_a": "Yes, Bitcoin will reach $100,000",
                        "outcome_b": "No, Bitcoin will not reach $100,000",
                        "end_date": "2024-12-31T23:59:59Z",
                        "category": "cryptocurrency",
                        "tags": ["bitcoin", "cryptocurrency", "price-prediction"]
                    },
                    "response": {
                        "market_id": 456,
                        "title": "Will Bitcoin reach $100,000 by end of 2024?",
                        "status": "open",
                        "created_at": "2024-01-15T10:30:00Z",
                        "creator_id": 123,
                        "current_price_a": 0.45,
                        "current_price_b": 0.55
                    }
                },
                "get_market": {
                    "endpoint": "GET /api/v1/markets/456",
                    "description": "Get detailed market information",
                    "response": {
                        "market_id": 456,
                        "title": "Will Bitcoin reach $100,000 by end of 2024?",
                        "description": "Prediction market for Bitcoin price target",
                        "outcome_a": "Yes, Bitcoin will reach $100,000",
                        "outcome_b": "No, Bitcoin will not reach $100,000",
                        "status": "open",
                        "current_price_a": 0.45,
                        "current_price_b": 0.55,
                        "volume": 125000.50,
                        "trades_count": 1250,
                        "votes_count": 890,
                        "created_at": "2024-01-15T10:30:00Z",
                        "end_date": "2024-12-31T23:59:59Z"
                    }
                }
            },
            "trading": {
                "execute_trade": {
                    "endpoint": "POST /api/v1/trades/",
                    "description": "Execute a trade in a prediction market",
                    "request_body": {
                        "market_id": 456,
                        "trade_type": "buy",
                        "outcome": "outcome_a",
                        "amount": 100.0,
                        "price_per_share": 0.45
                    },
                    "response": {
                        "trade_id": 789,
                        "market_id": 456,
                        "user_id": 123,
                        "trade_type": "buy",
                        "outcome": "outcome_a",
                        "amount": 100.0,
                        "price_per_share": 0.45,
                        "total_value": 45.0,
                        "status": "executed",
                        "executed_at": "2024-01-15T10:35:00Z"
                    }
                },
                "create_order": {
                    "endpoint": "POST /api/v1/orders/",
                    "description": "Create a limit order",
                    "request_body": {
                        "market_id": 456,
                        "order_type": "limit",
                        "side": "buy",
                        "outcome": "outcome_a",
                        "amount": 50.0,
                        "limit_price": 0.40
                    },
                    "response": {
                        "order_id": 101,
                        "market_id": 456,
                        "user_id": 123,
                        "order_type": "limit",
                        "side": "buy",
                        "outcome": "outcome_a",
                        "amount": 50.0,
                        "limit_price": 0.40,
                        "status": "pending",
                        "created_at": "2024-01-15T10:40:00Z"
                    }
                }
            },
            "analytics": {
                "market_analytics": {
                    "endpoint": "GET /api/v1/analytics/market/456",
                    "description": "Get comprehensive market analytics",
                    "response": {
                        "market_id": 456,
                        "price_history": [
                            {"timestamp": "2024-01-15T10:00:00Z", "price_a": 0.42, "price_b": 0.58},
                            {"timestamp": "2024-01-15T10:30:00Z", "price_a": 0.45, "price_b": 0.55}
                        ],
                        "volume_history": [
                            {"timestamp": "2024-01-15T10:00:00Z", "volume": 50000.0},
                            {"timestamp": "2024-01-15T10:30:00Z", "volume": 125000.0}
                        ],
                        "trading_activity": {
                            "total_trades": 1250,
                            "total_volume": 125000.50,
                            "avg_trade_size": 100.0,
                            "most_active_hour": "14:00-15:00"
                        },
                        "sentiment_analysis": {
                            "bullish_sentiment": 0.65,
                            "bearish_sentiment": 0.35,
                            "confidence_score": 0.78
                        }
                    }
                }
            }
        }
    }


@router.get("/docs/status-codes")
async def get_status_codes():
    """
    Get comprehensive HTTP status codes documentation
    """
    return {
        "status_codes": {
            "success": {
                "200": {
                    "description": "OK - Request successful",
                    "examples": ["GET /markets/", "GET /users/123"]
                },
                "201": {
                    "description": "Created - Resource created successfully",
                    "examples": ["POST /markets/", "POST /trades/"]
                },
                "202": {
                    "description": "Accepted - Request accepted for processing",
                    "examples": ["POST /orders/", "POST /auth/register"]
                }
            },
            "client_errors": {
                "400": {
                    "description": "Bad Request - Invalid request data",
                    "examples": ["Missing required fields", "Invalid data format"]
                },
                "401": {
                    "description": "Unauthorized - Authentication required",
                    "examples": ["Missing access token", "Invalid credentials"]
                },
                "403": {
                    "description": "Forbidden - Access denied",
                    "examples": ["Insufficient permissions", "Account suspended"]
                },
                "404": {
                    "description": "Not Found - Resource not found",
                    "examples": ["GET /markets/999", "GET /users/999"]
                },
                "409": {
                    "description": "Conflict - Resource conflict",
                    "examples": ["Username already exists", "Market already closed"]
                },
                "422": {
                    "description": "Unprocessable Entity - Validation error",
                    "examples": ["Invalid email format", "Password too weak"]
                }
            },
            "server_errors": {
                "500": {
                    "description": "Internal Server Error - Server error",
                    "examples": ["Database connection failed", "Unexpected error"]
                },
                "502": {
                    "description": "Bad Gateway - Upstream server error",
                    "examples": ["External API unavailable", "Service timeout"]
                },
                "503": {
                    "description": "Service Unavailable - Service temporarily unavailable",
                    "examples": ["Maintenance mode", "High load"]
                }
            }
        }
    }


@router.get("/docs/authentication")
async def get_authentication_guide():
    """
    Get comprehensive authentication guide
    """
    return {
        "authentication": {
            "overview": "The Opinion Market API uses JWT (JSON Web Token) based authentication",
            "flow": [
                "1. Register a new account or login with existing credentials",
                "2. Receive an access token in the response",
                "3. Include the token in the Authorization header for protected endpoints",
                "4. Token expires after 1 hour, use refresh endpoint to get new token"
            ],
            "token_format": "Bearer <access_token>",
            "header_example": "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
            "endpoints": {
                "public": [
                    "POST /auth/register",
                    "POST /auth/login",
                    "GET /markets/",
                    "GET /markets/{market_id}",
                    "GET /health"
                ],
                "protected": [
                    "GET /auth/me",
                    "POST /auth/logout",
                    "POST /trades/",
                    "POST /orders/",
                    "GET /users/{user_id}",
                    "POST /markets/"
                ]
            },
            "security_features": [
                "JWT token-based authentication",
                "Password hashing with bcrypt",
                "Rate limiting on authentication endpoints",
                "Account lockout after failed attempts",
                "Session management",
                "Token refresh mechanism"
            ]
        }
    }


@router.get("/docs/rate-limits")
async def get_rate_limits():
    """
    Get API rate limiting information
    """
    return {
        "rate_limits": {
            "overview": "API rate limiting is implemented to ensure fair usage and system stability",
            "limits": {
                "authentication": {
                    "login": "5 requests per minute per IP",
                    "register": "3 requests per minute per IP",
                    "refresh": "10 requests per minute per user"
                },
                "trading": {
                    "trades": "100 requests per minute per user",
                    "orders": "50 requests per minute per user"
                },
                "general": {
                    "read_operations": "1000 requests per minute per user",
                    "write_operations": "100 requests per minute per user"
                }
            },
            "headers": {
                "rate_limit": "X-RateLimit-Limit",
                "rate_remaining": "X-RateLimit-Remaining",
                "rate_reset": "X-RateLimit-Reset"
            },
            "error_response": {
                "status_code": 429,
                "body": {
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": 60
                }
            }
        }
    }


@router.get("/docs/interactive", response_class=HTMLResponse)
async def get_interactive_docs():
    """
    Get interactive API documentation page
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Opinion Market API - Interactive Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .method { display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; font-weight: bold; margin-right: 10px; }
            .get { background-color: #27ae60; }
            .post { background-color: #e74c3c; }
            .put { background-color: #f39c12; }
            .delete { background-color: #e67e22; }
            .url { font-family: monospace; background: #2c3e50; color: white; padding: 5px 10px; border-radius: 3px; }
            .description { margin-top: 10px; color: #7f8c8d; }
            .example { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border: 1px solid #dee2e6; }
            .code { font-family: monospace; background: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 3px; overflow-x: auto; }
            .status-code { display: inline-block; padding: 3px 8px; border-radius: 3px; font-weight: bold; margin-right: 5px; }
            .status-200 { background-color: #27ae60; color: white; }
            .status-400 { background-color: #e74c3c; color: white; }
            .status-401 { background-color: #f39c12; color: white; }
            .status-404 { background-color: #95a5a6; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Opinion Market API - Interactive Documentation</h1>
            
            <h2>📋 Quick Start</h2>
            <div class="example">
                <p><strong>Base URL:</strong> <span class="url">http://localhost:8000/api/v1</span></p>
                <p><strong>Authentication:</strong> JWT Bearer Token</p>
                <p><strong>Content-Type:</strong> application/json</p>
            </div>

            <h2>🔐 Authentication</h2>
            <div class="endpoint">
                <span class="method post">POST</span>
                <span class="url">/auth/register</span>
                <div class="description">Register a new user account</div>
                <div class="example">
                    <strong>Request Body:</strong>
                    <div class="code">
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePassword123!",
  "full_name": "John Doe"
}
                    </div>
                    <strong>Response:</strong> <span class="status-code status-201">201 Created</span>
                </div>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span>
                <span class="url">/auth/login</span>
                <div class="description">Authenticate user and get access token</div>
                <div class="example">
                    <strong>Request Body:</strong>
                    <div class="code">
{
  "username": "john_doe",
  "password": "SecurePassword123!"
}
                    </div>
                    <strong>Response:</strong> <span class="status-code status-200">200 OK</span>
                </div>
            </div>

            <h2>📊 Markets</h2>
            <div class="endpoint">
                <span class="method get">GET</span>
                <span class="url">/markets/</span>
                <div class="description">List all prediction markets</div>
                <div class="example">
                    <strong>Response:</strong> <span class="status-code status-200">200 OK</span>
                </div>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span>
                <span class="url">/markets/</span>
                <div class="description">Create a new prediction market</div>
                <div class="example">
                    <strong>Request Body:</strong>
                    <div class="code">
{
  "title": "Will Bitcoin reach $100,000 by end of 2024?",
  "description": "Prediction market for Bitcoin price target",
  "outcome_a": "Yes, Bitcoin will reach $100,000",
  "outcome_b": "No, Bitcoin will not reach $100,000",
  "end_date": "2024-12-31T23:59:59Z",
  "category": "cryptocurrency"
}
                    </div>
                    <strong>Response:</strong> <span class="status-code status-201">201 Created</span>
                </div>
            </div>

            <h2>💰 Trading</h2>
            <div class="endpoint">
                <span class="method post">POST</span>
                <span class="url">/trades/</span>
                <div class="description">Execute a trade in a prediction market</div>
                <div class="example">
                    <strong>Request Body:</strong>
                    <div class="code">
{
  "market_id": 456,
  "trade_type": "buy",
  "outcome": "outcome_a",
  "amount": 100.0,
  "price_per_share": 0.45
}
                    </div>
                    <strong>Response:</strong> <span class="status-code status-201">201 Created</span>
                </div>
            </div>

            <h2>📈 Analytics</h2>
            <div class="endpoint">
                <span class="method get">GET</span>
                <span class="url">/analytics/market/{market_id}</span>
                <div class="description">Get comprehensive market analytics</div>
                <div class="example">
                    <strong>Response:</strong> <span class="status-code status-200">200 OK</span>
                </div>
            </div>

            <h2>🏥 Health & Monitoring</h2>
            <div class="endpoint">
                <span class="method get">GET</span>
                <span class="url">/health</span>
                <div class="description">Basic health check</div>
                <div class="example">
                    <strong>Response:</strong> <span class="status-code status-200">200 OK</span>
                </div>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <span class="url">/health/detailed</span>
                <div class="description">Detailed system health information</div>
                <div class="example">
                    <strong>Response:</strong> <span class="status-code status-200">200 OK</span>
                </div>
            </div>

            <h2>🔗 Useful Links</h2>
            <div class="example">
                <p><a href="/docs" target="_blank">📚 Swagger UI Documentation</a></p>
                <p><a href="/openapi.json" target="_blank">📋 OpenAPI Schema</a></p>
                <p><a href="/api/v1/health" target="_blank">🏥 Health Check</a></p>
                <p><a href="/api/v1/docs/overview" target="_blank">📊 API Overview</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/docs/errors")
async def get_error_handling_guide():
    """
    Get comprehensive error handling guide
    """
    return {
        "error_handling": {
            "overview": "The API uses standard HTTP status codes and returns detailed error information",
            "error_format": {
                "error": "Error type",
                "message": "Human-readable error message",
                "details": "Additional error details",
                "timestamp": "Error timestamp",
                "request_id": "Unique request identifier"
            },
            "common_errors": {
                "validation_error": {
                    "status_code": 422,
                    "example": {
                        "error": "ValidationError",
                        "message": "Invalid input data",
                        "details": {
                            "field": "email",
                            "issue": "Invalid email format"
                        }
                    }
                },
                "authentication_error": {
                    "status_code": 401,
                    "example": {
                        "error": "AuthenticationError",
                        "message": "Invalid or expired token",
                        "details": "Please login again"
                    }
                },
                "authorization_error": {
                    "status_code": 403,
                    "example": {
                        "error": "AuthorizationError",
                        "message": "Insufficient permissions",
                        "details": "You don't have permission to access this resource"
                    }
                },
                "not_found_error": {
                    "status_code": 404,
                    "example": {
                        "error": "NotFoundError",
                        "message": "Resource not found",
                        "details": "Market with ID 999 does not exist"
                    }
                },
                "rate_limit_error": {
                    "status_code": 429,
                    "example": {
                        "error": "RateLimitError",
                        "message": "Rate limit exceeded",
                        "details": "Too many requests. Please try again later.",
                        "retry_after": 60
                    }
                }
            }
        }
    }
