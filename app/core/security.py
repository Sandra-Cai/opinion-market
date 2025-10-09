"""
Advanced security module for Opinion Market
Provides authentication, authorization, rate limiting, and security utilities
"""

import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
from functools import wraps
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import redis
from redis.exceptions import RedisError
import ipaddress
import re
from app.core.config import settings
from app.core.database import get_db, get_redis_client
from app.core.logging import log_security_event, log_api_call
from app.models.user import User


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handling
security = HTTPBearer(auto_error=False)


class SecurityManager:
    """Centralized security management"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.blocked_ips = set()
        self.suspicious_ips = set()
        self.rate_limit_cache = {}
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def is_password_strong(self, password: str) -> Dict[str, Any]:
        """Check password strength"""
        result = {
            "is_strong": True,
            "score": 0,
            "issues": []
        }
        
        if len(password) < settings.PASSWORD_MIN_LENGTH:
            result["issues"].append(f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters")
            result["is_strong"] = False
        
        if not re.search(r"[A-Z]", password):
            result["issues"].append("Password must contain at least one uppercase letter")
            result["score"] += 1
        
        if not re.search(r"[a-z]", password):
            result["issues"].append("Password must contain at least one lowercase letter")
            result["score"] += 1
        
        if not re.search(r"\d", password):
            result["issues"].append("Password must contain at least one number")
            result["score"] += 1
        
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            result["issues"].append("Password must contain at least one special character")
            result["score"] += 1
        
        # Check for common patterns
        common_patterns = [
            r"123", r"abc", r"qwe", r"password", r"admin", r"user"
        ]
        for pattern in common_patterns:
            if re.search(pattern, password.lower()):
                result["issues"].append("Password contains common patterns")
                result["score"] += 2
                break
        
        result["is_strong"] = result["score"] <= 1 and len(result["issues"]) == 0
        return result
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for storage"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        if self.redis_client:
            try:
                return self.redis_client.get(f"blocked_ip:{ip}") is not None
            except RedisError:
                pass
        return ip in self.blocked_ips
    
    def block_ip(self, ip: str, duration: int = 3600):
        """Block IP address"""
        if self.redis_client:
            try:
                self.redis_client.setex(f"blocked_ip:{ip}", duration, "1")
            except RedisError:
                pass
        self.blocked_ips.add(ip)
        log_security_event("ip_blocked", {"ip": ip, "duration": duration})
    
    def is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP is suspicious"""
        if self.redis_client:
            try:
                return self.redis_client.get(f"suspicious_ip:{ip}") is not None
            except RedisError:
                pass
        return ip in self.suspicious_ips
    
    def mark_ip_suspicious(self, ip: str, reason: str):
        """Mark IP as suspicious"""
        if self.redis_client:
            try:
                self.redis_client.setex(f"suspicious_ip:{ip}", 86400, reason)  # 24 hours
            except RedisError:
                pass
        self.suspicious_ips.add(ip)
        log_security_event("ip_marked_suspicious", {"ip": ip, "reason": reason})
    
    def check_rate_limit(self, identifier: str, limit: int = None, window: int = None) -> Dict[str, Any]:
        """Check rate limit for identifier"""
        if not settings.RATE_LIMIT_ENABLED:
            return {"allowed": True, "remaining": 0, "reset_time": 0}
        
        limit = limit or settings.RATE_LIMIT_REQUESTS
        window = window or settings.RATE_LIMIT_WINDOW
        
        current_time = int(time.time())
        window_start = current_time - window
        
        if self.redis_client:
            try:
                # Use Redis for distributed rate limiting
                key = f"rate_limit:{identifier}"
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zcard(key)
                pipe.zadd(key, {str(current_time): current_time})
                pipe.expire(key, window)
                results = pipe.execute()
                
                current_count = results[1]
                if current_count >= limit:
                    return {
                        "allowed": False,
                        "remaining": 0,
                        "reset_time": current_time + window,
                        "limit": limit
                    }
                
                return {
                    "allowed": True,
                    "remaining": limit - current_count - 1,
                    "reset_time": current_time + window,
                    "limit": limit
                }
            except RedisError:
                pass
        
        # Fallback to in-memory rate limiting
        if identifier not in self.rate_limit_cache:
            self.rate_limit_cache[identifier] = []
        
        # Clean old entries
        self.rate_limit_cache[identifier] = [
            timestamp for timestamp in self.rate_limit_cache[identifier]
            if timestamp > window_start
        ]
        
        if len(self.rate_limit_cache[identifier]) >= limit:
            return {
                "allowed": False,
                "remaining": 0,
                "reset_time": current_time + window,
                "limit": limit
            }
        
        self.rate_limit_cache[identifier].append(current_time)
        return {
            "allowed": True,
            "remaining": limit - len(self.rate_limit_cache[identifier]),
            "reset_time": current_time + window,
            "limit": limit
        }
    
    def validate_input(self, data: str, max_length: int = 1000) -> Dict[str, Any]:
        """Validate and sanitize input data"""
        result = {
            "is_valid": True,
            "sanitized": data,
            "issues": []
        }
        
        if len(data) > max_length:
            result["issues"].append(f"Input too long (max {max_length} characters)")
            result["is_valid"] = False
        
        # Check for SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|\/\*|\*\/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                result["issues"].append("Potential SQL injection detected")
                result["is_valid"] = False
                break
        
        # Check for XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                result["issues"].append("Potential XSS detected")
                result["is_valid"] = False
                break
        
        # Sanitize HTML
        if result["is_valid"]:
            import html
            result["sanitized"] = html.escape(data)
        
        return result


# Global security manager instance
security_manager = SecurityManager()


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = security_manager.verify_token(credentials.credentials)
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def get_current_verified_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current verified user"""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not verified"
        )
    return current_user


def get_current_premium_user(current_user: User = Depends(get_current_active_user)) -> User:
    """Get current premium user"""
    if not current_user.is_premium:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    return current_user


def rate_limit(requests: int = None, window: int = None):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get client identifier (IP + User ID if authenticated)
            client_ip = request.client.host
            identifier = client_ip
            
            # Add user ID if authenticated
            if hasattr(request, "user") and request.user:
                identifier = f"{client_ip}:{request.user.id}"
            
            # Check rate limit
            rate_limit_result = security_manager.check_rate_limit(identifier, requests, window)
            
            if not rate_limit_result["allowed"]:
                log_security_event("rate_limit_exceeded", {
                    "identifier": identifier,
                    "limit": rate_limit_result["limit"],
                    "endpoint": request.url.path
                })
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(rate_limit_result["limit"]),
                        "X-RateLimit-Remaining": str(rate_limit_result["remaining"]),
                        "X-RateLimit-Reset": str(rate_limit_result["reset_time"])
                    }
                )
            
            # Add rate limit headers to response
            response = await func(request, *args, **kwargs)
            if hasattr(response, "headers"):
                response.headers["X-RateLimit-Limit"] = str(rate_limit_result["limit"])
                response.headers["X-RateLimit-Remaining"] = str(rate_limit_result["remaining"])
                response.headers["X-RateLimit-Reset"] = str(rate_limit_result["reset_time"])
            
            return response
        return wrapper
    return decorator


def require_permissions(permissions: List[str]):
    """Require specific permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(current_user: User = Depends(get_current_active_user), *args, **kwargs):
            # Check if user has required permissions
            user_permissions = current_user.preferences.get("permissions", [])
            if not all(perm in user_permissions for perm in permissions):
                log_security_event("insufficient_permissions", {
                    "user_id": current_user.id,
                    "required": permissions,
                    "user_permissions": user_permissions
                })
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return await func(current_user, *args, **kwargs)
        return wrapper
    return decorator


def validate_request_data(max_length: int = 1000):
    """Validate request data"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Validate all string arguments
            for key, value in kwargs.items():
                if isinstance(value, str):
                    validation_result = security_manager.validate_input(value, max_length)
                    if not validation_result["is_valid"]:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid input in {key}: {', '.join(validation_result['issues'])}"
                        )
                    kwargs[key] = validation_result["sanitized"]
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def log_security_events(event_type: str):
    """Log security events decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                log_security_event(event_type, {"status": "success"})
                return result
            except Exception as e:
                log_security_event(event_type, {"status": "failed", "error": str(e)})
                raise
        return wrapper
    return decorator


def get_client_ip(request: Request) -> str:
    """Get client IP address from request"""
    # Check for forwarded headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host


def is_valid_ip(ip: str) -> bool:
    """Check if IP address is valid"""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def is_private_ip(ip: str) -> bool:
    """Check if IP address is private"""
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def generate_csrf_token() -> str:
    """Generate CSRF token"""
    return security_manager.generate_secure_token()


def verify_csrf_token(token: str, session_token: str) -> bool:
    """Verify CSRF token"""
    return token == session_token and len(token) >= 32


# Export commonly used functions
__all__ = [
    "SecurityManager",
    "security_manager",
    "get_current_user",
    "get_current_active_user",
    "get_current_verified_user",
    "get_current_premium_user",
    "rate_limit",
    "require_permissions",
    "validate_request_data",
    "log_security_events",
    "get_client_ip",
    "is_valid_ip",
    "is_private_ip",
    "generate_csrf_token",
    "verify_csrf_token",
]