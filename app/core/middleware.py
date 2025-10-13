"""
Advanced Middleware System for Opinion Market
Handles monitoring, security, performance, and request/response processing
"""

import time
import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from contextlib import asynccontextmanager
import uuid
import gzip
import base64
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
import uvicorn

from app.core.config import settings
from app.core.database import get_redis_client
from app.core.logging import log_api_call, log_system_metric, log_security_event
from app.core.security import security_manager, get_client_ip
from app.core.cache import cache
from app.core.validation import input_validator
from app.core.security_audit import security_auditor, SecurityEventType, SecuritySeverity


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and optimization"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.redis_client = get_redis_client()
        self.performance_metrics = defaultdict(list)
        self.slow_queries = deque(maxlen=1000)
        self.request_times = deque(maxlen=10000)
        
    async def dispatch(self, request: Request, call_next):
        """Process request with performance monitoring"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Log request start
        log_api_call(
            endpoint=request.url.path,
            method=request.method,
            user_id=getattr(request.state, "user_id", None)
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Timestamp"] = datetime.utcnow().isoformat()
            
            # Record performance metrics
            await self._record_performance_metrics(request, response, process_time)
            
            # Check for slow requests
            if process_time > 2.0:  # Requests taking more than 2 seconds
                await self._handle_slow_request(request, process_time)
            
            return response
            
        except Exception as e:
            # Record error metrics
            process_time = time.time() - start_time
            await self._record_error_metrics(request, e, process_time)
            raise
    
    async def _record_performance_metrics(self, request: Request, response: Response, process_time: float):
        """Record performance metrics with enhanced error handling"""
        try:
            metrics_data = {
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "process_time": process_time,
                "timestamp": datetime.utcnow().isoformat(),
                "user_agent": request.headers.get("user-agent", "")[:200],  # Truncate long user agents
                "content_length": response.headers.get("content-length", 0),
                "request_id": getattr(request.state, "request_id", None)
            }
            
            # Store in Redis for real-time monitoring (with error handling)
            if self.redis_client:
                try:
                    metrics_key = f"perf_metrics:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"
                    self.redis_client.lpush(metrics_key, json.dumps(metrics_data))
                    self.redis_client.expire(metrics_key, 24 * 3600)  # 24 hours retention
                except Exception as redis_error:
                    # Log Redis error but don't fail the request
                    log_system_metric("redis_error", 1, {"error": str(redis_error)})
            
            # Store in memory for quick access (with size limits)
            endpoint_metrics = self.performance_metrics[request.url.path]
            if len(endpoint_metrics) >= 1000:  # Limit memory usage
                endpoint_metrics.pop(0)  # Remove oldest entry
            endpoint_metrics.append(metrics_data)
            
            # Limit request times buffer
            if len(self.request_times) >= 10000:
                self.request_times.popleft()
            self.request_times.append(process_time)
            
            # Log system metric
            log_system_metric("request_duration", process_time, {
                "endpoint": request.url.path,
                "method": request.method,
                "status_code": response.status_code
            })
            
        except Exception as e:
            # Don't let metrics collection break the request
            log_system_metric("metrics_error", 1, {"error": str(e)})
    
    async def _handle_slow_request(self, request: Request, process_time: float):
        """Handle slow requests"""
        slow_request_data = {
            "endpoint": request.url.path,
            "method": request.method,
            "process_time": process_time,
            "timestamp": datetime.utcnow().isoformat(),
            "query_params": dict(request.query_params),
            "headers": dict(request.headers)
        }
        
        self.slow_queries.append(slow_request_data)
        
        # Log slow request
        log_system_metric("slow_request", process_time, {
            "endpoint": request.url.path,
            "method": request.method,
            "threshold": 2.0
        })
        
        # Alert if too many slow requests
        recent_slow_requests = [
            req for req in self.slow_queries
            if (datetime.utcnow() - datetime.fromisoformat(req["timestamp"])).total_seconds() < 300
        ]
        
        if len(recent_slow_requests) > 10:  # More than 10 slow requests in 5 minutes
            await security_auditor.log_security_event(
                SecurityEventType.SYSTEM_COMPROMISE,
                SecuritySeverity.HIGH,
                get_client_ip(request),
                {
                    "slow_requests_count": len(recent_slow_requests),
                    "time_window": "5_minutes",
                    "average_process_time": sum(req["process_time"] for req in recent_slow_requests) / len(recent_slow_requests)
                }
            )
    
    async def _record_error_metrics(self, request: Request, error: Exception, process_time: float):
        """Record error metrics"""
        error_data = {
            "endpoint": request.url.path,
            "method": request.method,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "process_time": process_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        log_system_metric("request_error", 1, {
            "endpoint": request.url.path,
            "method": request.method,
            "error_type": type(error).__name__
        })
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.request_times:
            return {"error": "No performance data available"}
        
        request_times_list = list(self.request_times)
        return {
            "total_requests": len(request_times_list),
            "average_response_time": sum(request_times_list) / len(request_times_list),
            "min_response_time": min(request_times_list),
            "max_response_time": max(request_times_list),
            "p95_response_time": sorted(request_times_list)[int(len(request_times_list) * 0.95)],
            "p99_response_time": sorted(request_times_list)[int(len(request_times_list) * 0.99)],
            "slow_requests_count": len(self.slow_queries),
            "endpoints_tracked": len(self.performance_metrics),
            "memory_usage": {
                "performance_metrics_size": sum(len(metrics) for metrics in self.performance_metrics.values()),
                "slow_queries_size": len(self.slow_queries),
                "request_times_size": len(self.request_times)
            }
        }


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security monitoring and enforcement"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.redis_client = get_redis_client()
        self.suspicious_ips = defaultdict(int)
        self.blocked_ips = set()
        self.rate_limit_cache = {}
        
    async def dispatch(self, request: Request, call_next):
        """Process request with security checks"""
        client_ip = get_client_ip(request)
        
        # Check if IP is blocked
        if security_manager.is_ip_blocked(client_ip):
            await self._handle_blocked_ip(request, client_ip)
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "IP address is blocked"}
            )
        
        # Check for suspicious activity
        if await self._is_suspicious_request(request, client_ip):
            await self._handle_suspicious_request(request, client_ip)
        
        # Rate limiting
        if not await self._check_rate_limit(request, client_ip):
            await self._handle_rate_limit_exceeded(request, client_ip)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"}
            )
        
        # Input validation for sensitive endpoints
        if await self._requires_input_validation(request):
            validation_result = await self._validate_request_input(request)
            if not validation_result["is_valid"]:
                await self._handle_invalid_input(request, client_ip, validation_result)
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "Invalid input data", "errors": validation_result["errors"]}
                )
        
        # Process request
        response = await call_next(request)
        
        # Post-request security checks
        await self._post_request_security_checks(request, response, client_ip)
        
        return response
    
    async def _is_suspicious_request(self, request: Request, client_ip: str) -> bool:
        """Check if request is suspicious"""
        suspicious_indicators = 0
        
        # Check for suspicious headers
        suspicious_headers = ["x-forwarded-for", "x-real-ip", "x-cluster-client-ip"]
        for header in suspicious_headers:
            if header in request.headers and len(request.headers[header]) > 100:
                suspicious_indicators += 1
        
        # Check for suspicious user agent
        user_agent = request.headers.get("user-agent", "")
        if len(user_agent) > 500 or not user_agent:
            suspicious_indicators += 1
        
        # Check for suspicious query parameters
        for param, value in request.query_params.items():
            if len(value) > 1000:  # Very long parameter values
                suspicious_indicators += 1
        
        # Check for suspicious path
        if len(request.url.path) > 2000:  # Very long paths
            suspicious_indicators += 1
        
        # Check request frequency
        if client_ip in self.suspicious_ips:
            self.suspicious_ips[client_ip] += 1
            if self.suspicious_ips[client_ip] > 100:  # More than 100 requests
                suspicious_indicators += 2
        
        return suspicious_indicators >= 2
    
    async def _handle_suspicious_request(self, request: Request, client_ip: str):
        """Handle suspicious request"""
        await security_auditor.log_security_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            SecuritySeverity.MEDIUM,
            client_ip,
            {
                "endpoint": request.url.path,
                "method": request.method,
                "user_agent": request.headers.get("user-agent", ""),
                "headers_count": len(request.headers),
                "query_params_count": len(request.query_params)
            }
        )
        
        # Mark IP as suspicious
        security_manager.mark_ip_suspicious(client_ip, "Suspicious request patterns")
    
    async def _check_rate_limit(self, request: Request, client_ip: str) -> bool:
        """Check rate limiting"""
        if not settings.RATE_LIMIT_ENABLED:
            return True
        
        # Generate rate limit key
        rate_limit_key = f"rate_limit:{client_ip}:{request.url.path}"
        
        # Check current rate
        if self.redis_client:
            current_requests = self.redis_client.incr(rate_limit_key)
            if current_requests == 1:
                self.redis_client.expire(rate_limit_key, settings.RATE_LIMIT_WINDOW)
            
            return current_requests <= settings.RATE_LIMIT_REQUESTS
        
        # Fallback to in-memory rate limiting
        if rate_limit_key not in self.rate_limit_cache:
            self.rate_limit_cache[rate_limit_key] = {"count": 0, "reset_time": time.time() + settings.RATE_LIMIT_WINDOW}
        
        rate_data = self.rate_limit_cache[rate_limit_key]
        
        # Reset if window expired
        if time.time() > rate_data["reset_time"]:
            rate_data["count"] = 0
            rate_data["reset_time"] = time.time() + settings.RATE_LIMIT_WINDOW
        
        rate_data["count"] += 1
        return rate_data["count"] <= settings.RATE_LIMIT_REQUESTS
    
    async def _handle_rate_limit_exceeded(self, request: Request, client_ip: str):
        """Handle rate limit exceeded"""
        await security_auditor.log_security_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            SecuritySeverity.MEDIUM,
            client_ip,
            {
                "endpoint": request.url.path,
                "method": request.method,
                "limit": settings.RATE_LIMIT_REQUESTS,
                "window": settings.RATE_LIMIT_WINDOW
            }
        )
    
    async def _requires_input_validation(self, request: Request) -> bool:
        """Check if request requires input validation"""
        sensitive_endpoints = [
            "/api/v1/auth/",
            "/api/v1/markets/",
            "/api/v1/trades/",
            "/api/v1/users/"
        ]
        
        return any(request.url.path.startswith(endpoint) for endpoint in sensitive_endpoints)
    
    async def _validate_request_input(self, request: Request) -> Dict[str, Any]:
        """Validate request input with enhanced security checks"""
        validation_result = {"is_valid": True, "errors": [], "warnings": []}
        
        try:
            # Validate query parameters
            for param, value in request.query_params.items():
                if len(value) > 1000:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"Query parameter '{param}' is too long")
                
                # Check for potential SQL injection patterns
                sql_patterns = ["'", '"', ";", "--", "/*", "*/", "xp_", "sp_"]
                if any(pattern in value.lower() for pattern in sql_patterns):
                    validation_result["warnings"].append(f"Query parameter '{param}' contains suspicious characters")
                
                # Check for potential XSS patterns
                xss_patterns = ["<script", "javascript:", "onload=", "onerror="]
                if any(pattern in value.lower() for pattern in xss_patterns):
                    validation_result["warnings"].append(f"Query parameter '{param}' contains potential XSS patterns")
            
            # Validate headers
            for header, value in request.headers.items():
                if len(value) > 1000:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"Header '{header}' is too long")
                
                # Check for suspicious header values
                if header.lower() in ["user-agent", "referer", "origin"]:
                    if len(value) > 500:
                        validation_result["warnings"].append(f"Header '{header}' is unusually long")
            
            # Validate request path
            if len(request.url.path) > 2000:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Request path is too long")
            
            # Check for path traversal attempts
            if ".." in request.url.path or "//" in request.url.path:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Invalid path detected")
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    async def _handle_invalid_input(self, request: Request, client_ip: str, validation_result: Dict[str, Any]):
        """Handle invalid input"""
        await security_auditor.log_security_event(
            SecurityEventType.API_ABUSE,
            SecuritySeverity.MEDIUM,
            client_ip,
            {
                "endpoint": request.url.path,
                "method": request.method,
                "validation_errors": validation_result["errors"]
            }
        )
    
    async def _post_request_security_checks(self, request: Request, response: Response, client_ip: str):
        """Post-request security checks"""
        # Check for sensitive data in response
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                # Check response size
                content_length = response.headers.get("content-length", "0")
                if int(content_length) > 10 * 1024 * 1024:  # 10MB
                    await security_auditor.log_security_event(
                        SecurityEventType.DATA_BREACH_ATTEMPT,
                        SecuritySeverity.HIGH,
                        client_ip,
                        {
                            "endpoint": request.url.path,
                            "response_size": content_length,
                            "threshold": "10MB"
                        }
                    )


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive monitoring and observability"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.redis_client = get_redis_client()
        self.metrics_buffer = deque(maxlen=1000)
        self.health_checks = {}
        
    async def dispatch(self, request: Request, call_next):
        """Process request with monitoring"""
        # Health check endpoints
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)
        
        # Record request start
        start_time = time.time()
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # Collect request metrics
        request_metrics = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": {k: v for k, v in request.headers.items() if k.lower() not in ["authorization", "cookie"]},
            "client_ip": get_client_ip(request),
            "user_agent": request.headers.get("user-agent", ""),
            "content_length": request.headers.get("content-length", 0)
        }
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Collect response metrics
            response_metrics = {
                "status_code": response.status_code,
                "process_time": process_time,
                "response_headers": dict(response.headers),
                "content_length": response.headers.get("content-length", 0)
            }
            
            # Combine metrics
            full_metrics = {**request_metrics, **response_metrics}
            
            # Store metrics
            await self._store_metrics(full_metrics)
            
            # Update health checks
            await self._update_health_checks(request, response, process_time)
            
            return response
            
        except Exception as e:
            # Record error metrics
            error_metrics = {
                **request_metrics,
                "error": str(e),
                "error_type": type(e).__name__,
                "process_time": time.time() - start_time
            }
            
            await self._store_metrics(error_metrics)
            raise
    
    async def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in Redis and buffer"""
        # Add to buffer
        self.metrics_buffer.append(metrics)
        
        # Store in Redis
        if self.redis_client:
            metrics_key = f"monitoring_metrics:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"
            self.redis_client.lpush(metrics_key, json.dumps(metrics))
            self.redis_client.expire(metrics_key, 7 * 24 * 3600)  # 7 days retention
        
        # Log system metric
        log_system_metric("request_processed", 1, {
            "endpoint": metrics.get("path", ""),
            "method": metrics.get("method", ""),
            "status_code": metrics.get("status_code", 0),
            "process_time": metrics.get("process_time", 0)
        })
    
    async def _update_health_checks(self, request: Request, response: Response, process_time: float):
        """Update health check metrics"""
        endpoint = request.url.path
        
        if endpoint not in self.health_checks:
            self.health_checks[endpoint] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0.0,
                "last_request": None
            }
        
        health_data = self.health_checks[endpoint]
        health_data["total_requests"] += 1
        health_data["last_request"] = datetime.utcnow().isoformat()
        
        if 200 <= response.status_code < 400:
            health_data["successful_requests"] += 1
        else:
            health_data["failed_requests"] += 1
        
        # Update average response time
        total_time = health_data["average_response_time"] * (health_data["total_requests"] - 1)
        health_data["average_response_time"] = (total_time + process_time) / health_data["total_requests"]


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware for response compression"""
    
    def __init__(self, app: ASGIApp, minimum_size: int = 1000):
        super().__init__(app)
        self.minimum_size = minimum_size
        
    async def dispatch(self, request: Request, call_next):
        """Process request with compression"""
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return await call_next(request)
        
        # Process request
        response = await call_next(request)
        
        # Check if response should be compressed
        content_length = response.headers.get("content-length", "0")
        if int(content_length) < self.minimum_size:
            return response
        
        # Check content type
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type for ct in ["application/json", "text/", "application/javascript"]):
            return response
        
        # Compress response
        if hasattr(response, "body"):
            body = response.body
            if isinstance(body, bytes):
                compressed_body = gzip.compress(body)
                response.headers["content-encoding"] = "gzip"
                response.headers["content-length"] = str(len(compressed_body))
                response.body = compressed_body
        
        return response


class CacheMiddleware(BaseHTTPMiddleware):
    """Middleware for response caching"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.cache = cache
        self.cacheable_methods = ["GET"]
        self.cacheable_endpoints = [
            "/api/v1/markets/",
            "/api/v1/users/",
            "/api/v1/stats/"
        ]
        
    async def dispatch(self, request: Request, call_next):
        """Process request with caching"""
        # Check if request is cacheable
        if not self._is_cacheable(request):
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Try to get from cache
        cached_response = self.cache.get(cache_key)
        if cached_response:
            # Return cached response
            response = JSONResponse(content=cached_response)
            response.headers["X-Cache"] = "HIT"
            response.headers["X-Cache-Key"] = cache_key
            return response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            try:
                # Get response body
                if hasattr(response, "body"):
                    body = response.body
                    if isinstance(body, bytes):
                        try:
                            content = json.loads(body.decode())
                            # Cache for 5 minutes
                            self.cache.set(cache_key, content, ttl=300)
                            response.headers["X-Cache"] = "MISS"
                            response.headers["X-Cache-Key"] = cache_key
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
        
        return response
    
    def _is_cacheable(self, request: Request) -> bool:
        """Check if request is cacheable"""
        # Only GET requests
        if request.method not in self.cacheable_methods:
            return False
        
        # Check endpoint
        if not any(request.url.path.startswith(endpoint) for endpoint in self.cacheable_endpoints):
            return False
        
        # Check for cache control headers
        cache_control = request.headers.get("cache-control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False
        
        return True
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""
        key_data = {
            "method": request.method,
            "path": request.url.path,
            "query": dict(request.query_params),
            "headers": {k: v for k, v in request.headers.items() if k.lower() in ["accept", "accept-language"]}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"cache:{hashlib.md5(key_string.encode()).hexdigest()}"


class MiddlewareManager:
    """Manager for all middleware components"""
    
    def __init__(self, app: ASGIApp):
        self.app = app
        self.middleware_stack = []
        
    def add_middleware(self, middleware_class: type, **kwargs):
        """Add middleware to the stack"""
        self.middleware_stack.append((middleware_class, kwargs))
    
    def build_middleware_stack(self):
        """Build the complete middleware stack"""
        # Add middleware in reverse order (last added is first executed)
        for middleware_class, kwargs in reversed(self.middleware_stack):
            self.app = middleware_class(self.app, **kwargs)
        
        return self.app
    
    def get_middleware_info(self) -> Dict[str, Any]:
        """Get information about loaded middleware"""
        return {
            "middleware_count": len(self.middleware_stack),
            "middleware_list": [
                {
                    "class": middleware_class.__name__,
                    "kwargs": kwargs
                }
                for middleware_class, kwargs in self.middleware_stack
            ]
        }


# Global middleware manager
middleware_manager = MiddlewareManager(None)
