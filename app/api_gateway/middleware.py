"""
API Gateway Middleware
Provides cross-cutting concerns for microservices
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse

from app.core.security_manager import security_manager
from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class BaseMiddleware:
    """Base middleware class"""
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request through middleware"""
        # Pre-processing
        await self.before_request(request)
        
        # Process request
        response = await call_next(request)
        
        # Post-processing
        await self.after_request(request, response)
        
        return response
    
    async def before_request(self, request: Request):
        """Pre-request processing"""
        pass
    
    async def after_request(self, request: Request, response: Response):
        """Post-request processing"""
        pass


class SecurityMiddleware(BaseMiddleware):
    """Security middleware for threat detection"""
    
    async def before_request(self, request: Request):
        """Check for security threats before processing"""
        try:
            # Get client IP
            client_ip = request.client.host if request.client else "unknown"
            
            # Check if IP is blocked
            if security_manager.is_blocked(client_ip):
                raise HTTPException(status_code=403, detail="IP address blocked")
            
            # Detect threats in request data
            request_data = {
                "client_ip": client_ip,
                "path": str(request.url.path),
                "method": request.method,
                "headers": dict(request.headers)
            }
            
            # Get request body for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        request_data["body"] = body.decode("utf-8")
                except Exception:
                    pass
            
            # Detect threats
            threats = await security_manager.detect_threats(request_data)
            
            if threats:
                # Log security threat
                for threat in threats:
                    logger.warning(f"Security threat detected: {threat.event_type} from {client_ip}")
                
                # Block high-risk threats
                high_risk_threats = [t for t in threats if t.risk_score > 80]
                if high_risk_threats:
                    raise HTTPException(status_code=403, detail="Security threat detected")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in security middleware: {e}")


class RateLimitMiddleware(BaseMiddleware):
    """Rate limiting middleware"""
    
    async def before_request(self, request: Request):
        """Check rate limits before processing"""
        try:
            client_ip = request.client.host if request.client else "unknown"
            
            # Check rate limit based on endpoint
            endpoint = str(request.url.path)
            if "/api/v1/auth/login" in endpoint:
                limit_type = "login_attempts"
            elif "/api/v1/auth/password-reset" in endpoint:
                limit_type = "password_reset"
            else:
                limit_type = "api_requests"
            
            # Check rate limit
            allowed, details = await security_manager.check_rate_limit(client_ip, limit_type)
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Retry after {details.get('retry_after', 60)} seconds"
                )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in rate limit middleware: {e}")


class AuthenticationMiddleware(BaseMiddleware):
    """Authentication middleware"""
    
    async def before_request(self, request: Request):
        """Authenticate requests"""
        try:
            # Skip authentication for public endpoints
            public_endpoints = [
                "/health",
                "/ready",
                "/docs",
                "/openapi.json",
                "/api/v1/auth/login",
                "/api/v1/auth/register",
                "/api/v1/auth/refresh"
            ]
            
            if any(request.url.path.startswith(endpoint) for endpoint in public_endpoints):
                return
            
            # Get authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
            
            token = auth_header.split(" ")[1]
            
            # Validate token (simplified - in real implementation, verify JWT)
            if not await self._validate_token(token):
                raise HTTPException(status_code=401, detail="Invalid or expired token")
            
            # Add user info to request state
            request.state.user = await self._get_user_from_token(token)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in authentication middleware: {e}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def _validate_token(self, token: str) -> bool:
        """Validate JWT token"""
        try:
            # In a real implementation, this would verify the JWT signature
            # For now, we'll do a simple check
            return len(token) > 10
            
        except Exception:
            return False
    
    async def _get_user_from_token(self, token: str) -> Dict[str, Any]:
        """Get user information from token"""
        try:
            # In a real implementation, this would decode the JWT
            # For now, return mock user data
            return {
                "user_id": "user_123",
                "username": "test_user",
                "email": "test@example.com"
            }
            
        except Exception:
            return {}


class LoggingMiddleware(BaseMiddleware):
    """Request/response logging middleware"""
    
    async def before_request(self, request: Request):
        """Log incoming request"""
        try:
            request.state.start_time = time.time()
            
            # Log request details
            logger.info(f"Request: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
            
        except Exception as e:
            logger.error(f"Error in logging middleware (before): {e}")
    
    async def after_request(self, request: Request, response: Response):
        """Log response details"""
        try:
            # Calculate response time
            response_time = time.time() - getattr(request.state, 'start_time', time.time())
            
            # Log response details
            logger.info(f"Response: {response.status_code} for {request.method} {request.url.path} "
                       f"in {response_time:.3f}s")
            
            # Add response time header
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            
        except Exception as e:
            logger.error(f"Error in logging middleware (after): {e}")


class MetricsMiddleware(BaseMiddleware):
    """Metrics collection middleware"""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
    
    async def before_request(self, request: Request):
        """Collect request metrics"""
        try:
            self.request_count += 1
            request.state.start_time = time.time()
            
        except Exception as e:
            logger.error(f"Error in metrics middleware (before): {e}")
    
    async def after_request(self, request: Request, response: Response):
        """Collect response metrics"""
        try:
            # Calculate response time
            response_time = time.time() - getattr(request.state, 'start_time', time.time())
            self.response_times.append(response_time)
            
            # Keep only last 1000 response times
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
            
            # Count errors
            if response.status_code >= 400:
                self.error_count += 1
            
            # Update metrics in cache
            await self._update_metrics()
            
        except Exception as e:
            logger.error(f"Error in metrics middleware (after): {e}")
    
    async def _update_metrics(self):
        """Update metrics in cache"""
        try:
            metrics = {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "average_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                "last_updated": time.time()
            }
            
            await enhanced_cache.set(
                "gateway_metrics",
                metrics,
                ttl=300,
                tags=["metrics", "gateway"]
            )
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "response_times_count": len(self.response_times)
        }
