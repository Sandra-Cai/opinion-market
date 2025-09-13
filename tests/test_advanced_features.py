"""
Comprehensive test suite for advanced features
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main_simple import app
from app.core.security import SecurityManager, InputValidator
from app.core.config_manager import config_manager
from app.core.metrics import metrics_collector
from app.core.caching import memory_cache


class TestSecurityFeatures:
    """Test security features"""
    
    def test_password_hashing(self):
        """Test password hashing and verification"""
        security_manager = SecurityManager("test-secret")
        
        password = "test_password_123"
        hashed = security_manager.hash_password(password)
        
        assert hashed != password
        assert security_manager.verify_password(password, hashed)
        assert not security_manager.verify_password("wrong_password", hashed)
    
    def test_token_generation(self):
        """Test JWT token generation and verification"""
        security_manager = SecurityManager("test-secret")
        
        user_id = "test_user"
        token = security_manager.generate_token(user_id)
        
        assert token is not None
        assert isinstance(token, str)
        
        payload = security_manager.verify_token(token)
        assert payload is not None
        assert payload["user_id"] == user_id
    
    def test_input_validation(self):
        """Test input validation"""
        # Test email validation
        assert InputValidator.validate_email("test@example.com")
        assert not InputValidator.validate_email("invalid-email")
        
        # Test password strength
        weak_password = "123"
        strong_password = "StrongPass123!"
        
        weak_result = InputValidator.validate_password_strength(weak_password)
        strong_result = InputValidator.validate_password_strength(strong_password)
        
        assert not weak_result["is_valid"]
        assert strong_result["is_valid"]
        assert strong_result["score"] > weak_result["score"]


class TestAPIIntegration:
    """Test API integration"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "features" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "overall_status" in data
        assert "components" in data
        assert "timestamp" in data
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        client = TestClient(app)
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        assert "counters" in data
        assert "gauges" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
