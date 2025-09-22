"""
Rate Limiting Middleware for FastAPI
Integrates advanced rate limiting with FastAPI applications
"""

import time
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

from app.core.advanced_rate_limiter import (
    advanced_rate_limiter,
    RateLimitScope,
    RateLimit,
    RateLimitAlgorithm
)

logger = logging.getLogger(__name__)

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""
    
    def __init__(
        self,
        app: ASGIApp,
        enabled: bool = True,
        skip_paths: Optional[list] = None,
        custom_limits: Optional[dict] = None,
        error_message: str = "Rate limit exceeded",
        error_code: int = status.HTTP_429_TOO_MANY_REQUESTS
    ):
        super().__init__(app)
        self.enabled = enabled
        self.skip_paths = skip_paths or ["/health", "/docs", "/openapi.json", "/redoc"]
        self.custom_limits = custom_limits or {}
        self.error_message = error_message
        self.error_code = error_code
        
        # Setup default rate limits for different endpoint types
        self._setup_endpoint_limits()
    
    def _setup_endpoint_limits(self):
        """Setup default rate limits for different endpoint types"""
        # Authentication endpoints - more restrictive
        auth_limits = RateLimit(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000,
            burst_limit=5,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.ENDPOINT
        )
        
        # API endpoints - moderate limits
        api_limits = RateLimit(
            requests_per_minute=100,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_limit=20,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            scope=RateLimitScope.ENDPOINT
        )
        
        # Read-only endpoints - higher limits
        read_limits = RateLimit(
            requests_per_minute=200,
            requests_per_hour=2000,
            requests_per_day=20000,
            burst_limit=50,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.ENDPOINT
        )
        
        # Set default limits
        self.custom_limits.update({
            "/auth/login": auth_limits,
            "/auth/register": auth_limits,
            "/auth/refresh": auth_limits,
            "/api/v1/markets": read_limits,
            "/api/v1/users": api_limits,
            "/api/v1/trades": api_limits,
            "/api/v1/orders": api_limits,
        })
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting middleware"""
        
        # Skip rate limiting if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for certain paths
        if self._should_skip_path(request.url.path):
            return await call_next(request)
        
        # Get identifiers for rate limiting
        ip_address = self._get_client_ip(request)
        user_id = self._get_user_id(request)
        api_key = self._get_api_key(request)
        endpoint = request.url.path
        
        # Check rate limits in order of specificity
        rate_limit_result = None
        
        # 1. Check API key limits (most specific)
        if api_key:
            rate_limit_result = advanced_rate_limiter.check_rate_limit(
                RateLimitScope.API_KEY, api_key
            )
            if not rate_limit_result.allowed:
                return self._create_rate_limit_response(rate_limit_result, "API key")
        
        # 2. Check user limits
        if user_id and (not rate_limit_result or rate_limit_result.allowed):
            rate_limit_result = advanced_rate_limiter.check_rate_limit(
                RateLimitScope.USER, user_id
            )
            if not rate_limit_result.allowed:
                return self._create_rate_limit_response(rate_limit_result, "user")
        
        # 3. Check endpoint limits
        if not rate_limit_result or rate_limit_result.allowed:
            endpoint_limit = self._get_endpoint_limit(endpoint)
            if endpoint_limit:
                rate_limit_result = advanced_rate_limiter.check_rate_limit(
                    RateLimitScope.ENDPOINT, endpoint, endpoint_limit
                )
                if not rate_limit_result.allowed:
                    return self._create_rate_limit_response(rate_limit_result, "endpoint")
        
        # 4. Check IP limits
        if not rate_limit_result or rate_limit_result.allowed:
            rate_limit_result = advanced_rate_limiter.check_rate_limit(
                RateLimitScope.IP, ip_address
            )
            if not rate_limit_result.allowed:
                return self._create_rate_limit_response(rate_limit_result, "IP address")
        
        # 5. Check global limits
        if not rate_limit_result or rate_limit_result.allowed:
            rate_limit_result = advanced_rate_limiter.check_rate_limit(
                RateLimitScope.GLOBAL, "global"
            )
            if not rate_limit_result.allowed:
                return self._create_rate_limit_response(rate_limit_result, "global")
        
        # Add rate limit headers to response
        response = await call_next(request)
        self._add_rate_limit_headers(response, rate_limit_result)
        
        return response
    
    def _should_skip_path(self, path: str) -> bool:
        """Check if path should be skipped for rate limiting"""
        return any(path.startswith(skip_path) for skip_path in self.skip_paths)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Get user ID from request"""
        # Try to get user ID from JWT token or session
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            # In a real implementation, you'd decode the JWT token here
            # For now, we'll extract a simple user ID from the token
            token = authorization.split(" ")[1]
            # This is a simplified example - implement proper JWT decoding
            return f"user_{hash(token) % 10000}"
        
        return None
    
    def _get_api_key(self, request: Request) -> Optional[str]:
        """Get API key from request"""
        # Check X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key
        
        # Check query parameter
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key
        
        return None
    
    def _get_endpoint_limit(self, endpoint: str) -> Optional[RateLimit]:
        """Get custom rate limit for endpoint"""
        # Check for exact match first
        if endpoint in self.custom_limits:
            return self.custom_limits[endpoint]
        
        # Check for prefix match
        for path_prefix, limit in self.custom_limits.items():
            if endpoint.startswith(path_prefix):
                return limit
        
        return None
    
    def _create_rate_limit_response(self, rate_limit_result, limit_type: str) -> Response:
        """Create rate limit exceeded response"""
        headers = {
            "X-RateLimit-Limit": str(rate_limit_result.limit),
            "X-RateLimit-Remaining": str(rate_limit_result.remaining),
            "X-RateLimit-Reset": str(int(rate_limit_result.reset_time)),
        }
        
        if rate_limit_result.retry_after:
            headers["Retry-After"] = str(int(rate_limit_result.retry_after))
        
        error_detail = {
            "error": "Rate limit exceeded",
            "message": f"Rate limit exceeded for {limit_type}",
            "limit_type": limit_type,
            "limit": rate_limit_result.limit,
            "remaining": rate_limit_result.remaining,
            "reset_time": rate_limit_result.reset_time,
            "retry_after": rate_limit_result.retry_after
        }
        
        logger.warning(f"Rate limit exceeded for {limit_type}: {error_detail}")
        
        return JSONResponse(
            status_code=self.error_code,
            content=error_detail,
            headers=headers
        )
    
    def _add_rate_limit_headers(self, response: Response, rate_limit_result):
        """Add rate limit headers to successful response"""
        if rate_limit_result:
            response.headers["X-RateLimit-Limit"] = str(rate_limit_result.limit)
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_result.remaining)
            response.headers["X-RateLimit-Reset"] = str(int(rate_limit_result.reset_time))

class RateLimitConfig:
    """Configuration for rate limiting middleware"""
    
    def __init__(
        self,
        enabled: bool = True,
        skip_paths: Optional[list] = None,
        custom_limits: Optional[dict] = None,
        error_message: str = "Rate limit exceeded",
        error_code: int = status.HTTP_429_TOO_MANY_REQUESTS
    ):
        self.enabled = enabled
        self.skip_paths = skip_paths or []
        self.custom_limits = custom_limits or {}
        self.error_message = error_message
        self.error_code = error_code

def create_rate_limiting_middleware(config: RateLimitConfig) -> RateLimitingMiddleware:
    """Create rate limiting middleware with configuration"""
    return RateLimitingMiddleware(
        app=None,  # Will be set by FastAPI
        enabled=config.enabled,
        skip_paths=config.skip_paths,
        custom_limits=config.custom_limits,
        error_message=config.error_message,
        error_code=config.error_code
    )

# Default configuration
default_rate_limit_config = RateLimitConfig(
    enabled=True,
    skip_paths=["/health", "/docs", "/openapi.json", "/redoc", "/metrics"],
    custom_limits={
        "/auth/login": RateLimit(
            requests_per_minute=5,
            requests_per_hour=50,
            requests_per_day=500,
            burst_limit=3,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.ENDPOINT
        ),
        "/auth/register": RateLimit(
            requests_per_minute=3,
            requests_per_hour=30,
            requests_per_day=300,
            burst_limit=2,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.ENDPOINT
        ),
        "/api/v1/markets": RateLimit(
            requests_per_minute=300,
            requests_per_hour=3000,
            requests_per_day=30000,
            burst_limit=100,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            scope=RateLimitScope.ENDPOINT
        ),
        "/api/v1/trades": RateLimit(
            requests_per_minute=100,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_limit=20,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.ENDPOINT
        )
    }
)
