"""
Unit tests for security module
"""

import pytest
from datetime import datetime, timedelta
from fastapi import HTTPException

from app.core.security import (
    security_manager,
    get_current_user,
    get_current_active_user,
    rate_limit,
    validate_request_data,
    get_client_ip
)
from app.models.user import User


class TestSecurityManager:
    """Test cases for SecurityManager class"""
    
    def test_hash_password(self):
        """Test password hashing"""
        password = "testpassword123"
        hashed = security_manager.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert isinstance(hashed, str)
    
    def test_verify_password(self):
        """Test password verification"""
        password = "testpassword123"
        hashed = security_manager.hash_password(password)
        
        assert security_manager.verify_password(password, hashed) is True
        assert security_manager.verify_password("wrongpassword", hashed) is False
    
    def test_create_access_token(self):
        """Test access token creation"""
        data = {"sub": 1, "username": "testuser"}
        token = security_manager.create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_refresh_token(self):
        """Test refresh token creation"""
        data = {"sub": 1, "username": "testuser"}
        token = security_manager.create_refresh_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token(self):
        """Test token verification"""
        data = {"sub": 1, "username": "testuser"}
        token = security_manager.create_access_token(data)
        
        payload = security_manager.verify_token(token)
        assert payload["sub"] == 1
        assert payload["username"] == "testuser"
    
    def test_verify_invalid_token(self):
        """Test invalid token verification"""
        with pytest.raises(HTTPException):
            security_manager.verify_token("invalid_token")
    
    def test_is_password_strong(self):
        """Test password strength validation"""
        # Strong password
        strong_password = "StrongPassword123!"
        result = security_manager.is_password_strong(strong_password)
        assert result["is_strong"] is True
        assert len(result["issues"]) == 0
        
        # Weak password
        weak_password = "weak"
        result = security_manager.is_password_strong(weak_password)
        assert result["is_strong"] is False
        assert len(result["issues"]) > 0
    
    def test_block_ip(self):
        """Test IP blocking"""
        ip = "192.168.1.1"
        
        # Initially not blocked
        assert security_manager.is_ip_blocked(ip) is False
        
        # Block IP
        security_manager.block_ip(ip, duration_minutes=1)
        assert security_manager.is_ip_blocked(ip) is True
    
    def test_mark_ip_suspicious(self):
        """Test marking IP as suspicious"""
        ip = "192.168.1.2"
        
        # Initially not suspicious
        assert security_manager.is_suspicious_ip(ip) is False
        
        # Mark as suspicious
        security_manager.mark_ip_suspicious(ip, "Test reason", duration_minutes=1)
        assert security_manager.is_suspicious_ip(ip) is True
    
    def test_rate_limit_key_generation(self):
        """Test rate limit key generation"""
        from fastapi import Request
        from unittest.mock import Mock
        
        # Mock request
        request = Mock(spec=Request)
        request.url.path = "/api/v1/test"
        request.headers = {}
        
        # Mock get_client_ip
        with pytest.patch('app.core.security.get_client_ip', return_value="192.168.1.1"):
            key = security_manager.get_rate_limit_key(request)
            assert key == "rate_limit:192.168.1.1:/api/v1/test"


class TestPasswordStrength:
    """Test cases for password strength validation"""
    
    def test_minimum_length(self):
        """Test minimum password length"""
        short_password = "1234567"  # 7 characters
        result = security_manager.is_password_strong(short_password)
        assert result["is_strong"] is False
        assert any("at least 8 characters" in issue for issue in result["issues"])
    
    def test_uppercase_requirement(self):
        """Test uppercase letter requirement"""
        no_uppercase = "lowercase123!"
        result = security_manager.is_password_strong(no_uppercase)
        assert result["is_strong"] is False
        assert any("uppercase letter" in issue for issue in result["issues"])
    
    def test_lowercase_requirement(self):
        """Test lowercase letter requirement"""
        no_lowercase = "UPPERCASE123!"
        result = security_manager.is_password_strong(no_lowercase)
        assert result["is_strong"] is False
        assert any("lowercase letter" in issue for issue in result["issues"])
    
    def test_digit_requirement(self):
        """Test digit requirement"""
        no_digit = "NoDigits!"
        result = security_manager.is_password_strong(no_digit)
        assert result["is_strong"] is False
        assert any("digit" in issue for issue in result["issues"])
    
    def test_special_character_requirement(self):
        """Test special character requirement"""
        no_special = "NoSpecial123"
        result = security_manager.is_password_strong(no_special)
        assert result["is_strong"] is False
        assert any("special character" in issue for issue in result["issues"])
    
    def test_strong_password(self):
        """Test strong password passes all checks"""
        strong_password = "StrongPassword123!"
        result = security_manager.is_password_strong(strong_password)
        assert result["is_strong"] is True
        assert len(result["issues"]) == 0


class TestTokenExpiration:
    """Test cases for token expiration"""
    
    def test_access_token_expiration(self):
        """Test access token expiration"""
        data = {"sub": 1, "username": "testuser"}
        
        # Create token with short expiration
        token = security_manager.create_access_token(
            data, expires_delta=timedelta(seconds=1)
        )
        
        # Verify token is valid initially
        payload = security_manager.verify_token(token)
        assert payload["sub"] == 1
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Token should be invalid now
        with pytest.raises(HTTPException):
            security_manager.verify_token(token)
    
    def test_refresh_token_expiration(self):
        """Test refresh token expiration"""
        data = {"sub": 1, "username": "testuser"}
        
        # Create refresh token with short expiration
        token = security_manager.create_refresh_token(
            data, expires_delta=timedelta(seconds=1)
        )
        
        # Verify token is valid initially
        payload = security_manager.verify_token(token)
        assert payload["sub"] == 1
        assert payload.get("type") == "refresh"
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Token should be invalid now
        with pytest.raises(HTTPException):
            security_manager.verify_token(token)


class TestIPBlocking:
    """Test cases for IP blocking functionality"""
    
    def test_ip_blocking_expiration(self):
        """Test IP blocking expiration"""
        ip = "192.168.1.3"
        
        # Block IP for 1 second
        security_manager.block_ip(ip, duration_minutes=1/60)  # 1 second
        
        # Should be blocked initially
        assert security_manager.is_ip_blocked(ip) is True
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Should not be blocked anymore
        assert security_manager.is_ip_blocked(ip) is False
    
    def test_suspicious_ip_expiration(self):
        """Test suspicious IP expiration"""
        ip = "192.168.1.4"
        
        # Mark as suspicious for 1 second
        security_manager.mark_ip_suspicious(ip, "Test reason", duration_minutes=1/60)
        
        # Should be suspicious initially
        assert security_manager.is_suspicious_ip(ip) is True
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Should not be suspicious anymore
        assert security_manager.is_suspicious_ip(ip) is False


class TestRateLimiting:
    """Test cases for rate limiting functionality"""
    
    def test_rate_limit_check(self):
        """Test rate limit checking"""
        key = "test_rate_limit_key"
        
        # Should allow requests within limit
        for i in range(5):
            result = security_manager.check_rate_limit(key, requests=10, window=60)
            assert result is True
        
        # Should block requests over limit
        for i in range(6):
            result = security_manager.check_rate_limit(key, requests=10, window=60)
            if i < 10:
                assert result is True
            else:
                assert result is False


class TestClientIP:
    """Test cases for client IP extraction"""
    
    def test_get_client_ip_from_headers(self):
        """Test getting client IP from headers"""
        from fastapi import Request
        from unittest.mock import Mock
        
        # Test X-Forwarded-For header
        request = Mock(spec=Request)
        request.headers = {"x-forwarded-for": "192.168.1.1, 10.0.0.1"}
        request.client = None
        
        ip = get_client_ip(request)
        assert ip == "192.168.1.1"
        
        # Test X-Real-IP header
        request.headers = {"x-real-ip": "192.168.1.2"}
        ip = get_client_ip(request)
        assert ip == "192.168.1.2"
    
    def test_get_client_ip_from_client(self):
        """Test getting client IP from request client"""
        from fastapi import Request
        from unittest.mock import Mock
        
        request = Mock(spec=Request)
        request.headers = {}
        request.client = Mock()
        request.client.host = "192.168.1.3"
        
        ip = get_client_ip(request)
        assert ip == "192.168.1.3"
    
    def test_get_client_ip_unknown(self):
        """Test getting unknown client IP"""
        from fastapi import Request
        from unittest.mock import Mock
        
        request = Mock(spec=Request)
        request.headers = {}
        request.client = None
        
        ip = get_client_ip(request)
        assert ip == "unknown"


class TestSecurityDecorators:
    """Test cases for security decorators"""
    
    def test_rate_limit_decorator(self):
        """Test rate limit decorator"""
        from fastapi import Request
        from unittest.mock import Mock
        
        # Mock request
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.headers = {}
        
        # Mock get_client_ip
        with pytest.patch('app.core.security.get_client_ip', return_value="192.168.1.1"):
            # Create decorated function
            @rate_limit(requests=5, window=60)
            async def test_function(req: Request):
                return {"message": "success"}
            
            # Should work within rate limit
            result = asyncio.run(test_function(request))
            assert result["message"] == "success"
    
    def test_validate_request_data_decorator(self):
        """Test request data validation decorator"""
        from fastapi import Request
        from unittest.mock import Mock
        
        # Mock request
        request = Mock(spec=Request)
        request.query_params = {"param": "normal_value"}
        request.headers = {"header": "normal_value"}
        
        # Create decorated function
        @validate_request_data()
        async def test_function(req: Request):
            return {"message": "success"}
        
        # Should work with normal data
        result = asyncio.run(test_function(request))
        assert result["message"] == "success"
        
        # Test with suspicious data
        request.query_params = {"param": "'; DROP TABLE users; --"}
        
        # Should raise HTTPException
        with pytest.raises(HTTPException):
            asyncio.run(test_function(request))
