"""
Integration tests for authentication endpoints
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


class TestAuthEndpoints:
    """Test cases for authentication endpoints"""
    
    def test_register_user_success(self, client: TestClient, sample_user_data):
        """Test successful user registration"""
        response = client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == sample_user_data["username"]
        assert data["email"] == sample_user_data["email"]
        assert data["full_name"] == sample_user_data["full_name"]
        assert "id" in data
        assert "hashed_password" not in data  # Password should not be returned
    
    def test_register_user_duplicate_username(self, client: TestClient, test_user, sample_user_data):
        """Test registration with duplicate username"""
        sample_user_data["username"] = test_user.username
        
        response = client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
    
    def test_register_user_duplicate_email(self, client: TestClient, test_user, sample_user_data):
        """Test registration with duplicate email"""
        sample_user_data["email"] = test_user.email
        
        response = client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
    
    def test_register_user_weak_password(self, client: TestClient, sample_user_data):
        """Test registration with weak password"""
        sample_user_data["password"] = "weak"
        
        response = client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == 400
        assert "too weak" in response.json()["detail"]
    
    def test_register_user_invalid_email(self, client: TestClient, sample_user_data):
        """Test registration with invalid email"""
        sample_user_data["email"] = "invalid-email"
        
        response = client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == 400
        assert "Invalid email format" in response.json()["detail"]
    
    def test_register_user_invalid_username(self, client: TestClient, sample_user_data):
        """Test registration with invalid username"""
        sample_user_data["username"] = "ab"  # Too short
        
        response = client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == 400
        assert "3-20 characters" in response.json()["detail"]
    
    def test_login_success(self, client: TestClient, test_user):
        """Test successful login"""
        login_data = {
            "username": test_user.username,
            "password": "testpassword"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_login_invalid_username(self, client: TestClient):
        """Test login with invalid username"""
        login_data = {
            "username": "nonexistent",
            "password": "password"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_login_invalid_password(self, client: TestClient, test_user):
        """Test login with invalid password"""
        login_data = {
            "username": test_user.username,
            "password": "wrongpassword"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_login_inactive_user(self, client: TestClient, db_session: Session):
        """Test login with inactive user"""
        from app.models.user import User
        from app.core.security import security_manager
        
        # Create inactive user
        inactive_user = User(
            username="inactive",
            email="inactive@example.com",
            hashed_password=security_manager.hash_password("password"),
            is_active=False
        )
        db_session.add(inactive_user)
        db_session.commit()
        
        login_data = {
            "username": "inactive",
            "password": "password"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        assert response.status_code == 400
        assert "Account is deactivated" in response.json()["detail"]
    
    def test_refresh_token_success(self, client: TestClient, test_user):
        """Test successful token refresh"""
        from app.core.security import security_manager
        
        # Create refresh token
        refresh_token = security_manager.create_refresh_token(
            data={"sub": test_user.id, "username": test_user.username}
        )
        
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_refresh_token_invalid(self, client: TestClient):
        """Test refresh with invalid token"""
        response = client.post("/api/v1/auth/refresh", json={"refresh_token": "invalid_token"})
        
        assert response.status_code == 401
        assert "Invalid refresh token" in response.json()["detail"]
    
    def test_get_current_user_success(self, client: TestClient, test_user, auth_headers):
        """Test getting current user info"""
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_user.id
        assert data["username"] == test_user.username
        assert data["email"] == test_user.email
    
    def test_get_current_user_unauthorized(self, client: TestClient):
        """Test getting current user without authentication"""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == 401
    
    def test_get_current_user_invalid_token(self, client: TestClient):
        """Test getting current user with invalid token"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/auth/me", headers=headers)
        
        assert response.status_code == 401
    
    def test_change_password_success(self, client: TestClient, test_user, auth_headers):
        """Test successful password change"""
        password_data = {
            "current_password": "testpassword",
            "new_password": "NewPassword123!"
        }
        
        response = client.post("/api/v1/auth/change-password", json=password_data, headers=auth_headers)
        
        assert response.status_code == 200
        assert "Password changed successfully" in response.json()["message"]
    
    def test_change_password_wrong_current(self, client: TestClient, test_user, auth_headers):
        """Test password change with wrong current password"""
        password_data = {
            "current_password": "wrongpassword",
            "new_password": "NewPassword123!"
        }
        
        response = client.post("/api/v1/auth/change-password", json=password_data, headers=auth_headers)
        
        assert response.status_code == 400
        assert "Current password is incorrect" in response.json()["detail"]
    
    def test_change_password_weak_new(self, client: TestClient, test_user, auth_headers):
        """Test password change with weak new password"""
        password_data = {
            "current_password": "testpassword",
            "new_password": "weak"
        }
        
        response = client.post("/api/v1/auth/change-password", json=password_data, headers=auth_headers)
        
        assert response.status_code == 400
        assert "too weak" in response.json()["detail"]
    
    def test_logout_success(self, client: TestClient, test_user, auth_headers):
        """Test successful logout"""
        response = client.post("/api/v1/auth/logout", headers=auth_headers)
        
        assert response.status_code == 200
        assert "Successfully logged out" in response.json()["message"]
    
    def test_logout_unauthorized(self, client: TestClient):
        """Test logout without authentication"""
        response = client.post("/api/v1/auth/logout")
        
        assert response.status_code == 401


class TestAuthRateLimiting:
    """Test cases for authentication rate limiting"""
    
    def test_register_rate_limit(self, client: TestClient, sample_user_data):
        """Test registration rate limiting"""
        # Make multiple registration attempts
        for i in range(6):  # Exceed rate limit of 5
            sample_user_data["username"] = f"user{i}"
            sample_user_data["email"] = f"user{i}@example.com"
            response = client.post("/api/v1/auth/register", json=sample_user_data)
            
            if i < 5:
                assert response.status_code in [200, 400]  # Success or validation error
            else:
                assert response.status_code == 429  # Rate limit exceeded
    
    def test_login_rate_limit(self, client: TestClient, test_user):
        """Test login rate limiting"""
        login_data = {
            "username": test_user.username,
            "password": "wrongpassword"
        }
        
        # Make multiple failed login attempts
        for i in range(11):  # Exceed rate limit of 10
            response = client.post("/api/v1/auth/login", data=login_data)
            
            if i < 10:
                assert response.status_code == 401  # Unauthorized
            else:
                assert response.status_code == 429  # Rate limit exceeded


class TestAuthSecurity:
    """Test cases for authentication security"""
    
    def test_sql_injection_in_login(self, client: TestClient):
        """Test SQL injection attempt in login"""
        login_data = {
            "username": "admin'; DROP TABLE users; --",
            "password": "password"
        }
        
        response = client.post("/api/v1/auth/login", data=login_data)
        
        # Should not crash and should return 401
        assert response.status_code == 401
    
    def test_xss_in_registration(self, client: TestClient, sample_user_data):
        """Test XSS attempt in registration"""
        sample_user_data["full_name"] = "<script>alert('xss')</script>"
        
        response = client.post("/api/v1/auth/register", json=sample_user_data)
        
        # Should be rejected due to XSS detection
        assert response.status_code == 400
    
    def test_path_traversal_in_registration(self, client: TestClient, sample_user_data):
        """Test path traversal attempt in registration"""
        sample_user_data["bio"] = "../../../etc/passwd"
        
        response = client.post("/api/v1/auth/register", json=sample_user_data)
        
        # Should be rejected due to path traversal detection
        assert response.status_code == 400
    
    def test_command_injection_in_registration(self, client: TestClient, sample_user_data):
        """Test command injection attempt in registration"""
        sample_user_data["bio"] = "test; rm -rf /"
        
        response = client.post("/api/v1/auth/register", json=sample_user_data)
        
        # Should be rejected due to command injection detection
        assert response.status_code == 400


class TestAuthValidation:
    """Test cases for authentication input validation"""
    
    def test_register_missing_fields(self, client: TestClient):
        """Test registration with missing required fields"""
        incomplete_data = {
            "username": "testuser"
            # Missing email, password, etc.
        }
        
        response = client.post("/api/v1/auth/register", json=incomplete_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_register_empty_fields(self, client: TestClient):
        """Test registration with empty fields"""
        empty_data = {
            "username": "",
            "email": "",
            "password": "",
            "full_name": ""
        }
        
        response = client.post("/api/v1/auth/register", json=empty_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_register_long_fields(self, client: TestClient, sample_user_data):
        """Test registration with fields that are too long"""
        sample_user_data["username"] = "a" * 51  # Too long
        sample_user_data["email"] = "a" * 100 + "@example.com"  # Too long
        sample_user_data["full_name"] = "a" * 101  # Too long
        sample_user_data["bio"] = "a" * 501  # Too long
        
        response = client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_login_missing_fields(self, client: TestClient):
        """Test login with missing fields"""
        incomplete_data = {
            "username": "testuser"
            # Missing password
        }
        
        response = client.post("/api/v1/auth/login", data=incomplete_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_change_password_missing_fields(self, client: TestClient, auth_headers):
        """Test password change with missing fields"""
        incomplete_data = {
            "current_password": "testpassword"
            # Missing new_password
        }
        
        response = client.post("/api/v1/auth/change-password", json=incomplete_data, headers=auth_headers)
        
        assert response.status_code == 422  # Validation error
