"""
Security endpoints and authentication
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import logging

from app.core.security import SecurityManager, InputValidator, SecurityHeaders
from app.core.config_manager import config_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Security dependencies
security = HTTPBearer()
security_manager = SecurityManager(
    secret_key=config_manager.get_config().security.secret_key,
    algorithm=config_manager.get_config().security.algorithm
)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = security_manager.verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {
        "user_id": payload.get("user_id"),
        "token_id": payload.get("jti"),
        "expires_at": payload.get("exp")
    }


@router.post("/auth/login")
async def login(
    request: Request,
    username: str,
    password: str
):
    """User login with security checks"""
    # Check if account is locked
    if security_manager.is_account_locked(username):
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail="Account is temporarily locked due to too many failed attempts"
        )
    
    # Validate input
    if not InputValidator.validate_sql_injection(username) or not InputValidator.validate_sql_injection(password):
        security_manager.record_failed_attempt(username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input detected"
        )
    
    # Simulate password verification (in real app, check against database)
    # For demo purposes, accept any password
    if len(password) < 8:
        security_manager.record_failed_attempt(username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Reset failed attempts on successful login
    security_manager.reset_failed_attempts(username)
    
    # Generate token
    token = security_manager.generate_token(username)
    
    # Log security event
    logger.info(f"User {username} logged in successfully")
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": config_manager.get_config().security.access_token_expire_minutes * 60
    }


@router.post("/auth/register")
async def register(
    username: str,
    email: str,
    password: str
):
    """User registration with validation"""
    # Validate email
    if not InputValidator.validate_email(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    # Validate password strength
    password_validation = InputValidator.validate_password_strength(password)
    if not password_validation["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password validation failed: {', '.join(password_validation['issues'])}"
        )
    
    # Sanitize inputs
    username = InputValidator.sanitize_string(username, 50)
    email = InputValidator.sanitize_string(email, 100)
    
    # Hash password
    hashed_password = security_manager.hash_password(password)
    
    # In a real app, save to database
    logger.info(f"User {username} registered successfully")
    
    return {
        "message": "User registered successfully",
        "username": username,
        "email": email
    }


@router.get("/auth/me")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information"""
    return {
        "user_id": current_user["user_id"],
        "authenticated": True,
        "token_expires_at": current_user["expires_at"]
    }


@router.post("/auth/refresh")
async def refresh_token(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Refresh authentication token"""
    new_token = security_manager.generate_token(current_user["user_id"])
    
    return {
        "access_token": new_token,
        "token_type": "bearer",
        "expires_in": config_manager.get_config().security.access_token_expire_minutes * 60
    }


@router.post("/auth/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """User logout"""
    # In a real app, invalidate token in database
    logger.info(f"User {current_user['user_id']} logged out")
    
    return {"message": "Logged out successfully"}


@router.get("/security/headers")
async def get_security_headers():
    """Get security headers configuration"""
    return SecurityHeaders.get_security_headers()


@router.get("/security/status")
async def get_security_status():
    """Get security system status"""
    config = config_manager.get_config()
    
    return {
        "authentication_enabled": True,
        "rate_limiting_enabled": True,
        "input_validation_enabled": True,
        "password_policy": {
            "min_length": config.security.password_min_length,
            "require_special_chars": config.security.require_special_chars
        },
        "session_settings": {
            "timeout_minutes": config.security.session_timeout_minutes,
            "max_login_attempts": config.security.max_login_attempts,
            "lockout_duration_minutes": config.security.lockout_duration_minutes
        }
    }


@router.post("/security/validate-password")
async def validate_password_strength(password: str):
    """Validate password strength"""
    validation_result = InputValidator.validate_password_strength(password)
    
    return {
        "password": password[:3] + "*" * (len(password) - 3) if len(password) > 3 else "***",
        "is_valid": validation_result["is_valid"],
        "score": validation_result["score"],
        "issues": validation_result["issues"]
    }
