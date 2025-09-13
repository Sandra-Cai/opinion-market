"""
Advanced security utilities and authentication
"""

import hashlib
import secrets
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
import bcrypt
import logging

logger = logging.getLogger(__name__)


class SecurityManager:
    """Advanced security management system"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.failed_attempts: Dict[str, int] = {}
        self.locked_accounts: Dict[str, datetime] = {}
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Generate JWT token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32)  # Unique token ID
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if user_id in self.locked_accounts:
            lockout_time = self.locked_accounts[user_id]
            if datetime.utcnow() < lockout_time:
                return True
            else:
                # Remove expired lockout
                del self.locked_accounts[user_id]
                if user_id in self.failed_attempts:
                    del self.failed_attempts[user_id]
        return False
    
    def record_failed_attempt(self, user_id: str) -> bool:
        """Record failed login attempt and check if account should be locked"""
        self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
        
        if self.failed_attempts[user_id] >= self.max_attempts:
            self.locked_accounts[user_id] = datetime.utcnow() + timedelta(seconds=self.lockout_duration)
            logger.warning(f"Account {user_id} locked due to {self.max_attempts} failed attempts")
            return True
        return False
    
    def reset_failed_attempts(self, user_id: str) -> None:
        """Reset failed attempts for successful login"""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
        if user_id in self.locked_accounts:
            del self.locked_accounts[user_id]
    
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)
    
    def generate_csrf_token(self) -> str:
        """Generate CSRF token"""
        return secrets.token_urlsafe(32)
    
    def verify_csrf_token(self, token: str, session_token: str) -> bool:
        """Verify CSRF token"""
        return token == session_token


class InputValidator:
    """Advanced input validation and sanitization"""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            return ""
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in input_str if ord(char) >= 32 or char in '\t\n\r')
        
        # Limit length
        return sanitized[:max_length]
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength"""
        result = {
            "is_valid": True,
            "score": 0,
            "issues": []
        }
        
        if len(password) < 8:
            result["issues"].append("Password must be at least 8 characters long")
            result["is_valid"] = False
        
        if not any(c.isupper() for c in password):
            result["issues"].append("Password must contain at least one uppercase letter")
            result["is_valid"] = False
        
        if not any(c.islower() for c in password):
            result["issues"].append("Password must contain at least one lowercase letter")
            result["is_valid"] = False
        
        if not any(c.isdigit() for c in password):
            result["issues"].append("Password must contain at least one digit")
            result["is_valid"] = False
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            result["issues"].append("Password must contain at least one special character")
            result["is_valid"] = False
        
        # Calculate strength score
        result["score"] = min(100, len(password) * 2 + len([c for c in password if c.isalnum()]) * 2)
        
        return result
    
    @staticmethod
    def validate_sql_injection(input_str: str) -> bool:
        """Basic SQL injection detection"""
        dangerous_patterns = [
            "';", "--", "/*", "*/", "xp_", "sp_", "exec", "execute",
            "union", "select", "insert", "update", "delete", "drop",
            "create", "alter", "truncate"
        ]
        
        input_lower = input_str.lower()
        return not any(pattern in input_lower for pattern in dangerous_patterns)


class SecurityHeaders:
    """Security headers management"""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get comprehensive security headers"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }