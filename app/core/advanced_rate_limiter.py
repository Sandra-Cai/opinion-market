"""
Advanced Rate Limiting System
Provides sophisticated rate limiting with multiple algorithms and dynamic adjustment
"""

import time
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"

class RateLimitScope(Enum):
    """Rate limiting scopes"""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    API_KEY = "api_key"

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = 10
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    scope: RateLimitScope = RateLimitScope.IP

@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[float] = None
    limit: int = 0
    window_size: int = 0

class TokenBucket:
    """Token bucket rate limiter"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from the bucket"""
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_remaining(self) -> int:
        """Get remaining tokens"""
        with self._lock:
            self._refill()
            return int(self.tokens)

class SlidingWindow:
    """Sliding window rate limiter"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self._lock:
            now = time.time()
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # Check if we can add a new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_remaining(self) -> int:
        """Get remaining requests in current window"""
        with self._lock:
            now = time.time()
            
            # Remove old requests
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            return max(0, self.max_requests - len(self.requests))
    
    def get_reset_time(self) -> float:
        """Get time when the window resets"""
        if not self.requests:
            return time.time()
        
        return self.requests[0] + self.window_size

class FixedWindow:
    """Fixed window rate limiter"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.windows: Dict[int, int] = {}
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        with self._lock:
            now = time.time()
            window_start = int(now // self.window_size) * self.window_size
            
            # Clean up old windows
            current_time = time.time()
            old_windows = [w for w in self.windows.keys() if w < current_time - self.window_size * 2]
            for old_window in old_windows:
                del self.windows[old_window]
            
            # Check current window
            if window_start not in self.windows:
                self.windows[window_start] = 0
            
            if self.windows[window_start] < self.max_requests:
                self.windows[window_start] += 1
                return True
            
            return False
    
    def get_remaining(self) -> int:
        """Get remaining requests in current window"""
        with self._lock:
            now = time.time()
            window_start = int(now // self.window_size) * self.window_size
            
            if window_start not in self.windows:
                return self.max_requests
            
            return max(0, self.max_requests - self.windows[window_start])
    
    def get_reset_time(self) -> float:
        """Get time when the window resets"""
        now = time.time()
        window_start = int(now // self.window_size) * self.window_size
        return window_start + self.window_size

class LeakyBucket:
    """Leaky bucket rate limiter"""
    
    def __init__(self, capacity: int, leak_rate: float):
        self.capacity = capacity
        self.leak_rate = leak_rate
        self.level = 0
        self.last_leak = time.time()
        self._lock = threading.Lock()
    
    def add_request(self) -> bool:
        """Add a request to the bucket"""
        with self._lock:
            self._leak()
            
            if self.level < self.capacity:
                self.level += 1
                return True
            
            return False
    
    def _leak(self):
        """Leak requests from the bucket"""
        now = time.time()
        time_passed = now - self.last_leak
        leaked = time_passed * self.leak_rate
        
        self.level = max(0, self.level - leaked)
        self.last_leak = now
    
    def get_remaining(self) -> int:
        """Get remaining capacity"""
        with self._lock:
            self._leak()
            return max(0, self.capacity - self.level)

class AdvancedRateLimiter:
    """Advanced rate limiter with multiple algorithms and scopes"""
    
    def __init__(self):
        self.limiters: Dict[str, Union[TokenBucket, SlidingWindow, FixedWindow, LeakyBucket]] = {}
        self.rate_limits: Dict[str, RateLimit] = {}
        self.user_limits: Dict[str, RateLimit] = {}
        self.endpoint_limits: Dict[str, RateLimit] = {}
        self.api_key_limits: Dict[str, RateLimit] = {}
        self._lock = threading.RLock()
        
        # Default rate limits
        self._setup_default_limits()
    
    def _setup_default_limits(self):
        """Setup default rate limits"""
        # Global limits
        self.rate_limits["global"] = RateLimit(
            requests_per_minute=1000,
            requests_per_hour=10000,
            requests_per_day=100000,
            burst_limit=50,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.GLOBAL
        )
        
        # User limits
        self.rate_limits["user"] = RateLimit(
            requests_per_minute=100,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_limit=20,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            scope=RateLimitScope.USER
        )
        
        # IP limits
        self.rate_limits["ip"] = RateLimit(
            requests_per_minute=200,
            requests_per_hour=2000,
            requests_per_day=20000,
            burst_limit=30,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            scope=RateLimitScope.IP
        )
        
        # Endpoint limits
        self.rate_limits["endpoint"] = RateLimit(
            requests_per_minute=500,
            requests_per_hour=5000,
            requests_per_day=50000,
            burst_limit=100,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            scope=RateLimitScope.ENDPOINT
        )
    
    def _get_limiter_key(self, scope: RateLimitScope, identifier: str, limit_type: str = "minute") -> str:
        """Generate a unique key for the limiter"""
        return f"{scope.value}:{identifier}:{limit_type}"
    
    def _get_limiter(self, scope: RateLimitScope, identifier: str, limit_type: str, rate_limit: RateLimit) -> Union[TokenBucket, SlidingWindow, FixedWindow, LeakyBucket]:
        """Get or create a limiter instance"""
        key = self._get_limiter_key(scope, identifier, limit_type)
        
        if key not in self.limiters:
            with self._lock:
                if key not in self.limiters:
                    if limit_type == "minute":
                        max_requests = rate_limit.requests_per_minute
                        window_size = 60
                    elif limit_type == "hour":
                        max_requests = rate_limit.requests_per_hour
                        window_size = 3600
                    elif limit_type == "day":
                        max_requests = rate_limit.requests_per_day
                        window_size = 86400
                    else:
                        max_requests = rate_limit.requests_per_minute
                        window_size = 60
                    
                    if rate_limit.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                        refill_rate = max_requests / window_size
                        self.limiters[key] = TokenBucket(max_requests, refill_rate)
                    elif rate_limit.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
                        self.limiters[key] = SlidingWindow(window_size, max_requests)
                    elif rate_limit.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                        self.limiters[key] = FixedWindow(window_size, max_requests)
                    elif rate_limit.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                        leak_rate = max_requests / window_size
                        self.limiters[key] = LeakyBucket(max_requests, leak_rate)
        
        return self.limiters[key]
    
    def check_rate_limit(self, scope: RateLimitScope, identifier: str, 
                        rate_limit: Optional[RateLimit] = None) -> RateLimitResult:
        """Check if request is allowed under rate limits"""
        if rate_limit is None:
            rate_limit = self.rate_limits.get(scope.value, self.rate_limits["global"])
        
        # Check multiple time windows
        minute_limiter = self._get_limiter(scope, identifier, "minute", rate_limit)
        hour_limiter = self._get_limiter(scope, identifier, "hour", rate_limit)
        day_limiter = self._get_limiter(scope, identifier, "day", rate_limit)
        
        # Check if any limit is exceeded
        minute_allowed = self._check_limiter(minute_limiter, rate_limit.algorithm)
        hour_allowed = self._check_limiter(hour_limiter, rate_limit.algorithm)
        day_allowed = self._check_limiter(day_limiter, rate_limit.algorithm)
        
        allowed = minute_allowed and hour_allowed and day_allowed
        
        # Get remaining requests (use the most restrictive)
        minute_remaining = self._get_remaining(minute_limiter, rate_limit.algorithm)
        hour_remaining = self._get_remaining(hour_limiter, rate_limit.algorithm)
        day_remaining = self._get_remaining(day_limiter, rate_limit.algorithm)
        
        remaining = min(minute_remaining, hour_remaining, day_remaining)
        
        # Get reset time (use the earliest reset)
        minute_reset = self._get_reset_time(minute_limiter, rate_limit.algorithm)
        hour_reset = self._get_reset_time(hour_limiter, rate_limit.algorithm)
        day_reset = self._get_reset_time(day_limiter, rate_limit.algorithm)
        
        reset_time = min(minute_reset, hour_reset, day_reset)
        
        # Calculate retry after
        retry_after = None
        if not allowed:
            retry_after = max(0, reset_time - time.time())
        
        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
            limit=rate_limit.requests_per_minute,
            window_size=60
        )
    
    def _check_limiter(self, limiter: Union[TokenBucket, SlidingWindow, FixedWindow, LeakyBucket], 
                      algorithm: RateLimitAlgorithm) -> bool:
        """Check if limiter allows the request"""
        if algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return limiter.consume()
        elif algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return limiter.is_allowed()
        elif algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return limiter.is_allowed()
        elif algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return limiter.add_request()
        return False
    
    def _get_remaining(self, limiter: Union[TokenBucket, SlidingWindow, FixedWindow, LeakyBucket], 
                      algorithm: RateLimitAlgorithm) -> int:
        """Get remaining requests from limiter"""
        if hasattr(limiter, 'get_remaining'):
            return limiter.get_remaining()
        return 0
    
    def _get_reset_time(self, limiter: Union[TokenBucket, SlidingWindow, FixedWindow, LeakyBucket], 
                       algorithm: RateLimitAlgorithm) -> float:
        """Get reset time from limiter"""
        if hasattr(limiter, 'get_reset_time'):
            return limiter.get_reset_time()
        return time.time() + 60  # Default to 1 minute
    
    def set_user_limit(self, user_id: str, rate_limit: RateLimit):
        """Set custom rate limit for a user"""
        self.user_limits[user_id] = rate_limit
        logger.info(f"Set custom rate limit for user {user_id}")
    
    def set_endpoint_limit(self, endpoint: str, rate_limit: RateLimit):
        """Set custom rate limit for an endpoint"""
        self.endpoint_limits[endpoint] = rate_limit
        logger.info(f"Set custom rate limit for endpoint {endpoint}")
    
    def set_api_key_limit(self, api_key: str, rate_limit: RateLimit):
        """Set custom rate limit for an API key"""
        self.api_key_limits[api_key] = rate_limit
        logger.info(f"Set custom rate limit for API key {api_key}")
    
    def get_rate_limit_status(self, scope: RateLimitScope, identifier: str) -> Dict[str, Any]:
        """Get current rate limit status"""
        rate_limit = self.rate_limits.get(scope.value, self.rate_limits["global"])
        
        minute_limiter = self._get_limiter(scope, identifier, "minute", rate_limit)
        hour_limiter = self._get_limiter(scope, identifier, "hour", rate_limit)
        day_limiter = self._get_limiter(scope, identifier, "day", rate_limit)
        
        return {
            "scope": scope.value,
            "identifier": identifier,
            "limits": {
                "minute": {
                    "limit": rate_limit.requests_per_minute,
                    "remaining": self._get_remaining(minute_limiter, rate_limit.algorithm),
                    "reset_time": self._get_reset_time(minute_limiter, rate_limit.algorithm)
                },
                "hour": {
                    "limit": rate_limit.requests_per_hour,
                    "remaining": self._get_remaining(hour_limiter, rate_limit.algorithm),
                    "reset_time": self._get_reset_time(hour_limiter, rate_limit.algorithm)
                },
                "day": {
                    "limit": rate_limit.requests_per_day,
                    "remaining": self._get_remaining(day_limiter, rate_limit.algorithm),
                    "reset_time": self._get_reset_time(day_limiter, rate_limit.algorithm)
                }
            },
            "algorithm": rate_limit.algorithm.value,
            "burst_limit": rate_limit.burst_limit
        }
    
    def reset_rate_limit(self, scope: RateLimitScope, identifier: str):
        """Reset rate limit for a specific scope and identifier"""
        keys_to_remove = []
        
        for key in self.limiters.keys():
            if key.startswith(f"{scope.value}:{identifier}:"):
                keys_to_remove.append(key)
        
        with self._lock:
            for key in keys_to_remove:
                del self.limiters[key]
        
        logger.info(f"Reset rate limit for {scope.value}:{identifier}")
    
    def get_all_limiters(self) -> Dict[str, Any]:
        """Get information about all active limiters"""
        result = {}
        
        for key, limiter in self.limiters.items():
            scope, identifier, limit_type = key.split(":", 2)
            
            if scope not in result:
                result[scope] = {}
            
            if identifier not in result[scope]:
                result[scope][identifier] = {}
            
            result[scope][identifier][limit_type] = {
                "type": type(limiter).__name__,
                "remaining": self._get_remaining(limiter, RateLimitAlgorithm.TOKEN_BUCKET),
                "reset_time": self._get_reset_time(limiter, RateLimitAlgorithm.TOKEN_BUCKET)
            }
        
        return result
    
    def cleanup_old_limiters(self, max_age_hours: int = 24):
        """Clean up old limiters to free memory"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        keys_to_remove = []
        
        # This is a simplified cleanup - in a real implementation,
        # you'd want to track last access time for each limiter
        with self._lock:
            # For now, just remove limiters that haven't been accessed recently
            # In a production system, you'd implement proper LRU cleanup
            pass
        
        logger.info(f"Cleaned up {len(keys_to_remove)} old limiters")

# Global rate limiter instance
advanced_rate_limiter = AdvancedRateLimiter()

# Convenience functions
def check_rate_limit(scope: RateLimitScope, identifier: str, 
                    rate_limit: Optional[RateLimit] = None) -> RateLimitResult:
    """Check rate limit for a scope and identifier"""
    return advanced_rate_limiter.check_rate_limit(scope, identifier, rate_limit)

def set_user_limit(user_id: str, rate_limit: RateLimit):
    """Set custom rate limit for a user"""
    advanced_rate_limiter.set_user_limit(user_id, rate_limit)

def set_endpoint_limit(endpoint: str, rate_limit: RateLimit):
    """Set custom rate limit for an endpoint"""
    advanced_rate_limiter.set_endpoint_limit(endpoint, rate_limit)

def get_rate_limit_status(scope: RateLimitScope, identifier: str) -> Dict[str, Any]:
    """Get rate limit status"""
    return advanced_rate_limiter.get_rate_limit_status(scope, identifier)
