"""
Enhanced API Documentation System
Comprehensive OpenAPI schema with examples, validation, and interactive documentation
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from enum import Enum


class APIStatus(Enum):
    """API status levels"""
    STABLE = "stable"
    BETA = "beta"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


class APICategory(Enum):
    """API categories for organization"""
    AUTHENTICATION = "Authentication"
    MARKETS = "Markets"
    TRADING = "Trading"
    ANALYTICS = "Analytics"
    ADMIN = "Administration"
    SECURITY = "Security"
    MONITORING = "Monitoring"


def create_enhanced_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """Create enhanced OpenAPI schema with comprehensive documentation"""
    
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Opinion Market API",
        version="2.0.0",
        description="""
        # Opinion Market API - Advanced Prediction Market Platform
        
        A comprehensive prediction market platform that allows users to trade and vote on their opinions about real-world events.
        
        ## Key Features
        
        - **Prediction Markets**: Create and trade on binary and multiple-choice markets
        - **Advanced Trading**: Market orders, limit orders, and stop orders
        - **Real-time Analytics**: AI-powered market predictions and sentiment analysis
        - **Social Features**: User profiles, leaderboards, and community interactions
        - **Blockchain Integration**: Smart contract integration and DeFi features
        - **Enterprise Security**: Advanced authentication, authorization, and monitoring
        
        ## Authentication
        
        The API uses JWT (JSON Web Token) for authentication. Include the token in the Authorization header:
        
        ```
        Authorization: Bearer <your-jwt-token>
        ```
        
        ## Rate Limiting
        
        API requests are rate limited to ensure fair usage:
        - **Standard users**: 100 requests per minute
        - **Premium users**: 500 requests per minute
        - **Admin users**: 1000 requests per minute
        
        ## Error Handling
        
        The API uses standard HTTP status codes and returns detailed error information:
        
        - `400 Bad Request`: Invalid request data
        - `401 Unauthorized`: Authentication required
        - `403 Forbidden`: Insufficient permissions
        - `404 Not Found`: Resource not found
        - `429 Too Many Requests`: Rate limit exceeded
        - `500 Internal Server Error`: Server error
        
        ## WebSocket Support
        
        Real-time updates are available via WebSocket connections:
        - Market price updates
        - Trade executions
        - System notifications
        - Live analytics data
        
        ## Support
        
        For API support and questions:
        - **Documentation**: Check the interactive docs at `/docs`
        - **Issues**: Report bugs on GitHub
        - **Community**: Join our Discord server
        """,
        routes=app.routes,
    )

    # Enhanced schema with additional metadata
    openapi_schema["info"].update({
        "contact": {
            "name": "Opinion Market API Support",
            "url": "https://opinionmarket.com/support",
            "email": "api-support@opinionmarket.com"
        },
        "license": {
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        "termsOfService": "https://opinionmarket.com/terms",
        "x-logo": {
            "url": "https://opinionmarket.com/logo.png",
            "altText": "Opinion Market Logo"
        }
    })

    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "https://api.opinionmarket.com",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.opinionmarket.com",
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ]

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from the /auth/login endpoint"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for programmatic access"
        }
    }

    # Add global security
    openapi_schema["security"] = [
        {"BearerAuth": []},
        {"ApiKeyAuth": []}
    ]

    # Enhanced components with examples
    openapi_schema["components"]["schemas"].update({
        "User": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Unique user identifier",
                    "example": "123e4567-e89b-12d3-a456-426614174000"
                },
                "username": {
                    "type": "string",
                    "description": "User's username",
                    "example": "trader123",
                    "minLength": 3,
                    "maxLength": 50
                },
                "email": {
                    "type": "string",
                    "format": "email",
                    "description": "User's email address",
                    "example": "user@example.com"
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "premium", "admin"],
                    "description": "User's role",
                    "example": "user"
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Account creation timestamp",
                    "example": "2024-01-15T10:30:00Z"
                },
                "is_verified": {
                    "type": "boolean",
                    "description": "Email verification status",
                    "example": True
                },
                "trading_stats": {
                    "$ref": "#/components/schemas/TradingStats"
                }
            },
            "required": ["id", "username", "email", "role", "created_at"]
        },
        "TradingStats": {
            "type": "object",
            "properties": {
                "total_trades": {
                    "type": "integer",
                    "description": "Total number of trades",
                    "example": 150
                },
                "total_volume": {
                    "type": "number",
                    "format": "float",
                    "description": "Total trading volume",
                    "example": 12500.50
                },
                "win_rate": {
                    "type": "number",
                    "format": "float",
                    "description": "Percentage of winning trades",
                    "example": 0.65
                },
                "profit_loss": {
                    "type": "number",
                    "format": "float",
                    "description": "Total profit/loss",
                    "example": 2500.75
                }
            }
        },
        "Market": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Unique market identifier",
                    "example": "456e7890-e89b-12d3-a456-426614174001"
                },
                "title": {
                    "type": "string",
                    "description": "Market title",
                    "example": "Will Bitcoin reach $100,000 by end of 2024?",
                    "maxLength": 200
                },
                "description": {
                    "type": "string",
                    "description": "Detailed market description",
                    "example": "This market resolves to Yes if Bitcoin reaches $100,000 or higher by December 31, 2024."
                },
                "category": {
                    "type": "string",
                    "enum": ["politics", "sports", "economics", "technology", "entertainment"],
                    "description": "Market category",
                    "example": "economics"
                },
                "market_type": {
                    "type": "string",
                    "enum": ["binary", "multiple", "scalar"],
                    "description": "Type of prediction market",
                    "example": "binary"
                },
                "outcomes": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Available market outcomes",
                    "example": ["Yes", "No"]
                },
                "current_prices": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "format": "float"
                    },
                    "description": "Current prices for each outcome",
                    "example": [0.65, 0.35]
                },
                "total_volume": {
                    "type": "number",
                    "format": "float",
                    "description": "Total trading volume",
                    "example": 50000.00
                },
                "participant_count": {
                    "type": "integer",
                    "description": "Number of participants",
                    "example": 1250
                },
                "end_date": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Market end date",
                    "example": "2024-12-31T23:59:59Z"
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "resolved", "cancelled", "paused"],
                    "description": "Market status",
                    "example": "active"
                },
                "created_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Market creation timestamp",
                    "example": "2024-01-15T10:30:00Z"
                }
            },
            "required": ["id", "title", "description", "category", "market_type", "outcomes", "status"]
        },
        "Trade": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Unique trade identifier",
                    "example": "789e0123-e89b-12d3-a456-426614174002"
                },
                "market_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "Market identifier",
                    "example": "456e7890-e89b-12d3-a456-426614174001"
                },
                "user_id": {
                    "type": "string",
                    "format": "uuid",
                    "description": "User identifier",
                    "example": "123e4567-e89b-12d3-a456-426614174000"
                },
                "outcome": {
                    "type": "string",
                    "description": "Traded outcome",
                    "example": "Yes"
                },
                "shares": {
                    "type": "number",
                    "format": "float",
                    "description": "Number of shares traded",
                    "example": 10.5
                },
                "price": {
                    "type": "number",
                    "format": "float",
                    "description": "Price per share",
                    "example": 0.65
                },
                "total_cost": {
                    "type": "number",
                    "format": "float",
                    "description": "Total cost of trade",
                    "example": 6.825
                },
                "trade_type": {
                    "type": "string",
                    "enum": ["buy", "sell"],
                    "description": "Type of trade",
                    "example": "buy"
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "completed", "cancelled", "failed"],
                    "description": "Trade status",
                    "example": "completed"
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Trade timestamp",
                    "example": "2024-01-15T14:30:00Z"
                }
            },
            "required": ["id", "market_id", "user_id", "outcome", "shares", "price", "trade_type", "status"]
        },
        "Error": {
            "type": "object",
            "properties": {
                "error_id": {
                    "type": "string",
                    "description": "Unique error identifier",
                    "example": "ERR_1705312200000"
                },
                "message": {
                    "type": "string",
                    "description": "Error message",
                    "example": "Invalid request parameters"
                },
                "status_code": {
                    "type": "integer",
                    "description": "HTTP status code",
                    "example": 400
                },
                "timestamp": {
                    "type": "number",
                    "format": "float",
                    "description": "Error timestamp",
                    "example": 1705312200.0
                },
                "details": {
                    "type": "object",
                    "description": "Additional error details",
                    "example": {
                        "field": "email",
                        "reason": "Invalid email format"
                    }
                }
            },
            "required": ["error_id", "message", "status_code", "timestamp"]
        },
        "Pagination": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "description": "Current page number",
                    "example": 1,
                    "minimum": 1
                },
                "per_page": {
                    "type": "integer",
                    "description": "Items per page",
                    "example": 20,
                    "minimum": 1,
                    "maximum": 100
                },
                "total": {
                    "type": "integer",
                    "description": "Total number of items",
                    "example": 150
                },
                "pages": {
                    "type": "integer",
                    "description": "Total number of pages",
                    "example": 8
                }
            },
            "required": ["page", "per_page", "total", "pages"]
        }
    })

    # Add response examples
    openapi_schema["components"]["responses"] = {
        "Success": {
            "description": "Successful operation",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "example": True
                            },
                            "data": {
                                "type": "object"
                            },
                            "message": {
                                "type": "string",
                                "example": "Operation completed successfully"
                            }
                        }
                    }
                }
            }
        },
        "BadRequest": {
            "description": "Bad request - invalid parameters",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/Error"
                    },
                    "example": {
                        "error_id": "ERR_1705312200000",
                        "message": "Invalid request parameters",
                        "status_code": 400,
                        "timestamp": 1705312200.0,
                        "details": {
                            "field": "email",
                            "reason": "Invalid email format"
                        }
                    }
                }
            }
        },
        "Unauthorized": {
            "description": "Unauthorized - authentication required",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/Error"
                    },
                    "example": {
                        "error_id": "ERR_1705312200001",
                        "message": "Authentication required",
                        "status_code": 401,
                        "timestamp": 1705312200.0
                    }
                }
            }
        },
        "Forbidden": {
            "description": "Forbidden - insufficient permissions",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/Error"
                    },
                    "example": {
                        "error_id": "ERR_1705312200002",
                        "message": "Insufficient permissions",
                        "status_code": 403,
                        "timestamp": 1705312200.0
                    }
                }
            }
        },
        "NotFound": {
            "description": "Resource not found",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/Error"
                    },
                    "example": {
                        "error_id": "ERR_1705312200003",
                        "message": "Resource not found",
                        "status_code": 404,
                        "timestamp": 1705312200.0
                    }
                }
            }
        },
        "RateLimitExceeded": {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/Error"
                    },
                    "example": {
                        "error_id": "ERR_1705312200004",
                        "message": "Rate limit exceeded. Try again later.",
                        "status_code": 429,
                        "timestamp": 1705312200.0,
                        "details": {
                            "retry_after": 60
                        }
                    }
                }
            }
        },
        "InternalServerError": {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/Error"
                    },
                    "example": {
                        "error_id": "ERR_1705312200005",
                        "message": "An internal error occurred",
                        "status_code": 500,
                        "timestamp": 1705312200.0
                    }
                }
            }
        }
    }

    # Add tags for better organization
    openapi_schema["tags"] = [
        {
            "name": "Authentication",
            "description": "User authentication and authorization endpoints",
            "x-category": "Authentication"
        },
        {
            "name": "Users",
            "description": "User management and profile operations",
            "x-category": "Authentication"
        },
        {
            "name": "Markets",
            "description": "Prediction market operations",
            "x-category": "Markets"
        },
        {
            "name": "Trading",
            "description": "Trading operations and order management",
            "x-category": "Trading"
        },
        {
            "name": "Analytics",
            "description": "Market analytics and insights",
            "x-category": "Analytics"
        },
        {
            "name": "AI/ML",
            "description": "AI-powered predictions and machine learning",
            "x-category": "Analytics"
        },
        {
            "name": "Social",
            "description": "Social features and community interactions",
            "x-category": "Social"
        },
        {
            "name": "Admin",
            "description": "Administrative operations",
            "x-category": "Administration"
        },
        {
            "name": "Security",
            "description": "Security monitoring and threat detection",
            "x-category": "Security"
        },
        {
            "name": "Monitoring",
            "description": "System monitoring and health checks",
            "x-category": "Monitoring"
        },
        {
            "name": "WebSocket",
            "description": "Real-time WebSocket connections",
            "x-category": "Real-time"
        }
    ]

    # Add extensions for additional metadata
    openapi_schema["x-api-status"] = "stable"
    openapi_schema["x-version"] = "2.0.0"
    openapi_schema["x-last-updated"] = datetime.utcnow().isoformat()
    openapi_schema["x-features"] = [
        "JWT Authentication",
        "Real-time WebSocket",
        "AI Predictions",
        "Blockchain Integration",
        "Advanced Analytics",
        "Social Features",
        "Enterprise Security"
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def create_api_documentation_endpoint(app: FastAPI):
    """Create custom API documentation endpoint"""
    
    @app.get("/api-docs", include_in_schema=False)
    async def get_api_documentation():
        """Get comprehensive API documentation"""
        return {
            "title": "Opinion Market API Documentation",
            "version": "2.0.0",
            "description": "Comprehensive API documentation for the Opinion Market platform",
            "endpoints": {
                "authentication": {
                    "description": "User authentication and authorization",
                    "endpoints": [
                        "POST /api/v1/auth/register",
                        "POST /api/v1/auth/login",
                        "POST /api/v1/auth/refresh",
                        "POST /api/v1/auth/logout"
                    ]
                },
                "markets": {
                    "description": "Prediction market operations",
                    "endpoints": [
                        "GET /api/v1/markets",
                        "POST /api/v1/markets",
                        "GET /api/v1/markets/{id}",
                        "PUT /api/v1/markets/{id}",
                        "DELETE /api/v1/markets/{id}"
                    ]
                },
                "trading": {
                    "description": "Trading operations",
                    "endpoints": [
                        "POST /api/v1/trades",
                        "GET /api/v1/trades",
                        "GET /api/v1/orders",
                        "POST /api/v1/orders",
                        "PUT /api/v1/orders/{id}",
                        "DELETE /api/v1/orders/{id}"
                    ]
                },
                "analytics": {
                    "description": "Market analytics and insights",
                    "endpoints": [
                        "GET /api/v1/analytics/market/{id}",
                        "GET /api/v1/analytics/user/me",
                        "GET /api/v1/analytics/platform",
                        "GET /api/v1/analytics/predictions"
                    ]
                }
            },
            "websocket": {
                "description": "Real-time WebSocket connections",
                "endpoints": [
                    "ws://api.opinionmarket.com/ws/market/{id}",
                    "ws://api.opinionmarket.com/ws/user",
                    "ws://api.opinionmarket.com/ws/global"
                ]
            },
            "rate_limits": {
                "standard": "100 requests/minute",
                "premium": "500 requests/minute",
                "admin": "1000 requests/minute"
            },
            "authentication": {
                "type": "JWT Bearer Token",
                "header": "Authorization: Bearer <token>",
                "token_expiry": "24 hours"
            }
        }
