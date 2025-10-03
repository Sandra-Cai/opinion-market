"""
Advanced API Gateway
Comprehensive API gateway with rate limiting, authentication, and advanced features
"""

import asyncio
import time
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import secrets
import jwt
import httpx
from urllib.parse import urljoin, urlparse

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class GatewayRoute(Enum):
    """Gateway route enumeration"""
    API_V1 = "api_v1"
    API_V2 = "api_v2"
    WEBSOCKET = "websocket"
    STATIC = "static"
    ADMIN = "admin"
    MONITORING = "monitoring"


class AuthenticationMethod(Enum):
    """Authentication method enumeration"""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    CUSTOM = "custom"


class RateLimitType(Enum):
    """Rate limit type enumeration"""
    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_ENDPOINT = "per_endpoint"
    GLOBAL = "global"


@dataclass
class APIRoute:
    """API route data structure"""
    route_id: str
    name: str
    path: str
    methods: List[str]
    target_service: str
    target_url: str
    authentication_required: bool
    authentication_method: AuthenticationMethod
    rate_limit: Dict[str, Any]
    middleware: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RateLimitRule:
    """Rate limit rule data structure"""
    rule_id: str
    name: str
    rate_limit_type: RateLimitType
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    window_size: int  # seconds
    scope: str  # user_id, ip_address, endpoint
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class APIRequest:
    """API request data structure"""
    request_id: str
    client_ip: str
    user_id: Optional[str]
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, Any]
    body: Optional[bytes]
    timestamp: datetime
    response_time: Optional[float] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class AuthenticationToken:
    """Authentication token data structure"""
    token_id: str
    user_id: str
    token_type: str
    token_value: str
    expires_at: datetime
    scopes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class AdvancedAPIGateway:
    """Advanced API Gateway with comprehensive features"""
    
    def __init__(self):
        self.routes: Dict[str, APIRoute] = {}
        self.rate_limit_rules: Dict[str, RateLimitRule] = {}
        self.authentication_tokens: Dict[str, AuthenticationToken] = {}
        self.request_logs: List[APIRequest] = []
        self.rate_limit_counters: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        
        # Configuration
        self.config = {
            "gateway_enabled": True,
            "rate_limiting_enabled": True,
            "authentication_enabled": True,
            "request_logging_enabled": True,
            "caching_enabled": True,
            "load_balancing_enabled": True,
            "circuit_breaker_enabled": True,
            "request_timeout": 30,  # seconds
            "max_request_size": 10 * 1024 * 1024,  # 10MB
            "default_rate_limit": {
                "requests_per_minute": 100,
                "requests_per_hour": 1000,
                "requests_per_day": 10000
            },
            "jwt_secret": "your-secret-key",
            "jwt_algorithm": "HS256",
            "token_expiry_hours": 24,
            "cache_ttl": 300,  # 5 minutes
            "health_check_interval": 30,  # seconds
            "metrics_collection_interval": 60  # seconds
        }
        
        # Default routes
        self.default_routes = {
            GatewayRoute.API_V1: {
                "name": "API V1",
                "path": "/api/v1/*",
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "target_service": "api-service",
                "target_url": "http://localhost:8000",
                "authentication_required": True,
                "authentication_method": AuthenticationMethod.JWT
            },
            GatewayRoute.WEBSOCKET: {
                "name": "WebSocket",
                "path": "/ws/*",
                "methods": ["GET"],
                "target_service": "websocket-service",
                "target_url": "ws://localhost:8001",
                "authentication_required": True,
                "authentication_method": AuthenticationMethod.JWT
            },
            GatewayRoute.ADMIN: {
                "name": "Admin Panel",
                "path": "/admin/*",
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "target_service": "admin-service",
                "target_url": "http://localhost:8002",
                "authentication_required": True,
                "authentication_method": AuthenticationMethod.API_KEY
            },
            GatewayRoute.MONITORING: {
                "name": "Monitoring",
                "path": "/monitoring/*",
                "methods": ["GET"],
                "target_service": "monitoring-service",
                "target_url": "http://localhost:8003",
                "authentication_required": False,
                "authentication_method": AuthenticationMethod.BASIC
            }
        }
        
        # Middleware
        self.middleware = {
            "rate_limiting": self._rate_limiting_middleware,
            "authentication": self._authentication_middleware,
            "logging": self._logging_middleware,
            "caching": self._caching_middleware,
            "load_balancing": self._load_balancing_middleware,
            "circuit_breaker": self._circuit_breaker_middleware
        }
        
        # Monitoring
        self.gateway_active = False
        self.gateway_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.gateway_stats = {
            "requests_processed": 0,
            "requests_blocked": 0,
            "authentication_failures": 0,
            "rate_limit_hits": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "circuit_breaker_trips": 0,
            "average_response_time": 0.0
        }
        
    async def start_gateway(self):
        """Start the API gateway"""
        if self.gateway_active:
            logger.warning("API gateway already active")
            return
            
        self.gateway_active = True
        self.gateway_task = asyncio.create_task(self._gateway_processing_loop())
        
        # Initialize default routes
        await self._initialize_default_routes()
        
        logger.info("Advanced API Gateway started")
        
    async def stop_gateway(self):
        """Stop the API gateway"""
        self.gateway_active = False
        if self.gateway_task:
            self.gateway_task.cancel()
            try:
                await self.gateway_task
            except asyncio.CancelledError:
                pass
        logger.info("Advanced API Gateway stopped")
        
    async def _gateway_processing_loop(self):
        """Main gateway processing loop"""
        while self.gateway_active:
            try:
                # Clean up expired tokens
                await self._cleanup_expired_tokens()
                
                # Clean up old request logs
                await self._cleanup_old_logs()
                
                # Update rate limit counters
                await self._update_rate_limit_counters()
                
                # Wait before next cycle
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in gateway processing loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _initialize_default_routes(self):
        """Initialize default routes"""
        try:
            for route_type, route_config in self.default_routes.items():
                route_id = f"route_{route_type.value}"
                
                route = APIRoute(
                    route_id=route_id,
                    name=route_config["name"],
                    path=route_config["path"],
                    methods=route_config["methods"],
                    target_service=route_config["target_service"],
                    target_url=route_config["target_url"],
                    authentication_required=route_config["authentication_required"],
                    authentication_method=route_config["authentication_method"],
                    rate_limit=self.config["default_rate_limit"]
                )
                
                self.routes[route_id] = route
                
            logger.info(f"Initialized {len(self.routes)} default routes")
            
        except Exception as e:
            logger.error(f"Error initializing default routes: {e}")
            
    async def add_route(self, route_data: Dict[str, Any]) -> APIRoute:
        """Add a new API route"""
        try:
            route_id = f"route_{int(time.time())}_{secrets.token_hex(4)}"
            
            route = APIRoute(
                route_id=route_id,
                name=route_data.get("name", "Custom Route"),
                path=route_data.get("path", "/*"),
                methods=route_data.get("methods", ["GET"]),
                target_service=route_data.get("target_service", "default-service"),
                target_url=route_data.get("target_url", "http://localhost:8000"),
                authentication_required=route_data.get("authentication_required", True),
                authentication_method=AuthenticationMethod(route_data.get("authentication_method", "jwt")),
                rate_limit=route_data.get("rate_limit", self.config["default_rate_limit"]),
                middleware=route_data.get("middleware", ["rate_limiting", "authentication", "logging"])
            )
            
            self.routes[route_id] = route
            
            logger.info(f"API route added: {route_id}")
            return route
            
        except Exception as e:
            logger.error(f"Error adding API route: {e}")
            raise
            
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an API request through the gateway"""
        try:
            request_id = f"req_{int(time.time())}_{secrets.token_hex(4)}"
            start_time = time.time()
            
            # Create request object
            request = APIRequest(
                request_id=request_id,
                client_ip=request_data.get("client_ip", "127.0.0.1"),
                user_id=request_data.get("user_id"),
                method=request_data.get("method", "GET"),
                path=request_data.get("path", "/"),
                headers=request_data.get("headers", {}),
                query_params=request_data.get("query_params", {}),
                body=request_data.get("body"),
                timestamp=datetime.now()
            )
            
            # Find matching route
            route = await self._find_matching_route(request.path, request.method)
            if not route:
                return await self._create_error_response(404, "Route not found")
                
            # Apply middleware
            middleware_result = await self._apply_middleware(request, route)
            if not middleware_result["success"]:
                return await self._create_error_response(
                    middleware_result["status_code"], 
                    middleware_result["message"]
                )
                
            # Forward request to target service
            response = await self._forward_request(request, route)
            
            # Update request with response data
            request.response_time = (time.time() - start_time) * 1000  # milliseconds
            request.status_code = response.get("status_code", 200)
            
            # Log request
            if self.config["request_logging_enabled"]:
                self.request_logs.append(request)
                
            # Update statistics
            self.gateway_stats["requests_processed"] += 1
            self.gateway_stats["average_response_time"] = (
                (self.gateway_stats["average_response_time"] * (self.gateway_stats["requests_processed"] - 1) + 
                 request.response_time) / self.gateway_stats["requests_processed"]
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return await self._create_error_response(500, "Internal server error")
            
    async def _find_matching_route(self, path: str, method: str) -> Optional[APIRoute]:
        """Find matching route for request"""
        try:
            for route in self.routes.values():
                if method in route.methods:
                    # Simple path matching (in real implementation, would use more sophisticated routing)
                    if path.startswith(route.path.replace("*", "")):
                        return route
            return None
            
        except Exception as e:
            logger.error(f"Error finding matching route: {e}")
            return None
            
    async def _apply_middleware(self, request: APIRequest, route: APIRoute) -> Dict[str, Any]:
        """Apply middleware to request"""
        try:
            for middleware_name in route.middleware:
                if middleware_name in self.middleware:
                    middleware_func = self.middleware[middleware_name]
                    result = await middleware_func(request, route)
                    
                    if not result["success"]:
                        return result
                        
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error applying middleware: {e}")
            return {"success": False, "status_code": 500, "message": "Middleware error"}
            
    async def _rate_limiting_middleware(self, request: APIRequest, route: APIRoute) -> Dict[str, Any]:
        """Rate limiting middleware"""
        try:
            if not self.config["rate_limiting_enabled"]:
                return {"success": True}
                
            # Check rate limits
            rate_limit_key = f"{request.client_ip}:{route.route_id}"
            current_time = time.time()
            
            # Clean old entries
            if rate_limit_key in self.rate_limit_counters:
                while (self.rate_limit_counters[rate_limit_key] and 
                       current_time - self.rate_limit_counters[rate_limit_key][0] > 60):
                    self.rate_limit_counters[rate_limit_key].popleft()
                    
            # Check rate limit
            if rate_limit_key in self.rate_limit_counters:
                if len(self.rate_limit_counters[rate_limit_key]) >= route.rate_limit["requests_per_minute"]:
                    self.gateway_stats["rate_limit_hits"] += 1
                    return {
                        "success": False,
                        "status_code": 429,
                        "message": "Rate limit exceeded"
                    }
                    
            # Add current request
            if rate_limit_key not in self.rate_limit_counters:
                self.rate_limit_counters[rate_limit_key] = deque()
            self.rate_limit_counters[rate_limit_key].append(current_time)
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error in rate limiting middleware: {e}")
            return {"success": False, "status_code": 500, "message": "Rate limiting error"}
            
    async def _authentication_middleware(self, request: APIRequest, route: APIRoute) -> Dict[str, Any]:
        """Authentication middleware"""
        try:
            if not route.authentication_required:
                return {"success": True}
                
            if not self.config["authentication_enabled"]:
                return {"success": True}
                
            # Extract token
            token = None
            if route.authentication_method == AuthenticationMethod.JWT:
                token = request.headers.get("Authorization", "").replace("Bearer ", "")
            elif route.authentication_method == AuthenticationMethod.API_KEY:
                token = request.headers.get("X-API-Key")
                
            if not token:
                self.gateway_stats["authentication_failures"] += 1
                return {
                    "success": False,
                    "status_code": 401,
                    "message": "Authentication required"
                }
                
            # Validate token
            if not await self._validate_token(token, route.authentication_method):
                self.gateway_stats["authentication_failures"] += 1
                return {
                    "success": False,
                    "status_code": 401,
                    "message": "Invalid token"
                }
                
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error in authentication middleware: {e}")
            return {"success": False, "status_code": 500, "message": "Authentication error"}
            
    async def _logging_middleware(self, request: APIRequest, route: APIRoute) -> Dict[str, Any]:
        """Logging middleware"""
        try:
            if self.config["request_logging_enabled"]:
                logger.info(f"Request: {request.method} {request.path} - Route: {route.name}")
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error in logging middleware: {e}")
            return {"success": True}  # Don't fail request for logging errors
            
    async def _caching_middleware(self, request: APIRequest, route: APIRoute) -> Dict[str, Any]:
        """Caching middleware"""
        try:
            if not self.config["caching_enabled"] or request.method != "GET":
                return {"success": True}
                
            # Check cache
            cache_key = f"gateway:{request.path}:{hash(str(request.query_params))}"
            cached_response = await enhanced_cache.get(cache_key)
            
            if cached_response:
                self.gateway_stats["cache_hits"] += 1
                return {
                    "success": True,
                    "cached": True,
                    "response": cached_response
                }
            else:
                self.gateway_stats["cache_misses"] += 1
                
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error in caching middleware: {e}")
            return {"success": True}  # Don't fail request for caching errors
            
    async def _load_balancing_middleware(self, request: APIRequest, route: APIRoute) -> Dict[str, Any]:
        """Load balancing middleware"""
        try:
            if not self.config["load_balancing_enabled"]:
                return {"success": True}
                
            # Simple load balancing logic (in real implementation, would use more sophisticated algorithms)
            # For now, just return success
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error in load balancing middleware: {e}")
            return {"success": True}
            
    async def _circuit_breaker_middleware(self, request: APIRequest, route: APIRoute) -> Dict[str, Any]:
        """Circuit breaker middleware"""
        try:
            if not self.config["circuit_breaker_enabled"]:
                return {"success": True}
                
            # Simple circuit breaker logic (in real implementation, would track failures)
            # For now, just return success
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error in circuit breaker middleware: {e}")
            return {"success": True}
            
    async def _forward_request(self, request: APIRequest, route: APIRoute) -> Dict[str, Any]:
        """Forward request to target service"""
        try:
            # Build target URL
            target_url = urljoin(route.target_url, request.path)
            
            # Simulate request forwarding
            response = {
                "status_code": 200,
                "headers": {"Content-Type": "application/json"},
                "body": {"message": "Request processed successfully", "route": route.name},
                "request_id": request.request_id
            }
            
            # Cache response if applicable
            if (self.config["caching_enabled"] and 
                request.method == "GET" and 
                response["status_code"] == 200):
                cache_key = f"gateway:{request.path}:{hash(str(request.query_params))}"
                await enhanced_cache.set(cache_key, response, ttl=self.config["cache_ttl"])
                
            return response
            
        except Exception as e:
            logger.error(f"Error forwarding request: {e}")
            return await self._create_error_response(500, "Service unavailable")
            
    async def _validate_token(self, token: str, auth_method: AuthenticationMethod) -> bool:
        """Validate authentication token"""
        try:
            if auth_method == AuthenticationMethod.JWT:
                # Decode JWT token
                payload = jwt.decode(token, self.config["jwt_secret"], algorithms=[self.config["jwt_algorithm"]])
                return True
            elif auth_method == AuthenticationMethod.API_KEY:
                # Check API key
                return token in self.authentication_tokens
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return False
            
    async def _create_error_response(self, status_code: int, message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "status_code": status_code,
            "headers": {"Content-Type": "application/json"},
            "body": {"error": message, "status_code": status_code}
        }
        
    async def _cleanup_expired_tokens(self):
        """Clean up expired authentication tokens"""
        try:
            current_time = datetime.now()
            expired_tokens = []
            
            for token_id, token in self.authentication_tokens.items():
                if token.expires_at < current_time:
                    expired_tokens.append(token_id)
                    
            for token_id in expired_tokens:
                del self.authentication_tokens[token_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up expired tokens: {e}")
            
    async def _cleanup_old_logs(self):
        """Clean up old request logs"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=24)
            
            # Keep only recent logs
            self.request_logs = [
                log for log in self.request_logs
                if log.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
            
    async def _update_rate_limit_counters(self):
        """Update rate limit counters"""
        try:
            current_time = time.time()
            
            # Clean up old rate limit entries
            for key in list(self.rate_limit_counters.keys()):
                if key in self.rate_limit_counters:
                    while (self.rate_limit_counters[key] and 
                           current_time - self.rate_limit_counters[key][0] > 3600):  # 1 hour
                        self.rate_limit_counters[key].popleft()
                        
        except Exception as e:
            logger.error(f"Error updating rate limit counters: {e}")
            
    def get_gateway_summary(self) -> Dict[str, Any]:
        """Get comprehensive gateway summary"""
        try:
            # Calculate route statistics
            routes_by_method = defaultdict(int)
            for route in self.routes.values():
                for method in route.methods:
                    routes_by_method[method] += 1
                    
            # Calculate request statistics
            recent_requests = self.request_logs[-100:] if self.request_logs else []
            requests_by_status = defaultdict(int)
            for request in recent_requests:
                if request.status_code:
                    requests_by_status[request.status_code] += 1
                    
            return {
                "timestamp": datetime.now().isoformat(),
                "gateway_active": self.gateway_active,
                "total_routes": len(self.routes),
                "routes_by_method": dict(routes_by_method),
                "total_requests": len(self.request_logs),
                "recent_requests": len(recent_requests),
                "requests_by_status": dict(requests_by_status),
                "active_tokens": len(self.authentication_tokens),
                "rate_limit_entries": len(self.rate_limit_counters),
                "stats": self.gateway_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting gateway summary: {e}")
            return {"error": str(e)}


# Global instance
advanced_api_gateway = AdvancedAPIGateway()
