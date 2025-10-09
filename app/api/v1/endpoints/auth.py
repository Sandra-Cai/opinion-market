from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Optional
import re

from app.core.database import get_db
from app.core.config import settings
from app.core.security import (
    security_manager, 
    get_current_user, 
    get_current_active_user,
    rate_limit,
    validate_request_data,
    log_security_event,
    get_client_ip
)
from app.core.cache import cache
from app.core.logging import log_trading_event
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse, Token, UserLogin, PasswordReset, PasswordChange

router = APIRouter()


@router.post("/register", response_model=UserResponse)
@rate_limit(requests=5, window=300)  # 5 requests per 5 minutes
@validate_request_data()
async def register(
    user_data: UserCreate, 
    request: Request,
    db: Session = Depends(get_db)
):
    """Register a new user with enhanced security and validation"""
    client_ip = get_client_ip(request)
    
    # Validate password strength
    password_check = security_manager.is_password_strong(user_data.password)
    if not password_check["is_strong"]:
        log_security_event("weak_password_attempt", {
            "ip": client_ip,
            "username": user_data.username,
            "issues": password_check["issues"]
        })
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password is too weak: {', '.join(password_check['issues'])}"
        )
    
    # Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    # Validate username format
    username_pattern = r'^[a-zA-Z0-9_-]{3,20}$'
    if not re.match(username_pattern, user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username must be 3-20 characters long and contain only letters, numbers, underscores, and hyphens"
        )
    
    # Check if user already exists
    existing_user = (
        db.query(User)
        .filter((User.username == user_data.username) | (User.email == user_data.email))
        .first()
    )

    if existing_user:
        log_security_event("duplicate_registration_attempt", {
            "ip": client_ip,
            "username": user_data.username,
            "email": user_data.email
        })
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered",
        )

    # Create new user
    hashed_password = security_manager.hash_password(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        bio=user_data.bio,
        is_active=True,
        is_verified=False,  # Require email verification
        preferences={},
        notification_settings={
            "email_notifications": True,
            "push_notifications": True,
            "market_updates": True,
            "price_alerts": False
        }
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Log successful registration
    log_security_event("user_registered", {
        "user_id": db_user.id,
        "username": db_user.username,
        "ip": client_ip
    })
    
    # Clear any cached user data
    if cache:
        cache.delete(f"user:{db_user.id}")
        cache.delete(f"user_by_username:{db_user.username}")
        cache.delete(f"user_by_email:{db_user.email}")

    return db_user


@router.post("/login", response_model=Token)
@rate_limit(requests=10, window=300)  # 10 requests per 5 minutes
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    request: Request = None,
    db: Session = Depends(get_db)
):
    """Enhanced login with security monitoring and rate limiting"""
    client_ip = get_client_ip(request) if request else "unknown"
    
    # Check for suspicious login attempts
    login_attempts_key = f"login_attempts:{client_ip}"
    if cache:
        attempts = cache.l2_cache.increment(login_attempts_key, ttl=3600) or 0
        if attempts > 10:  # More than 10 failed attempts in 1 hour
            security_manager.mark_ip_suspicious(client_ip, "Excessive login attempts")
            log_security_event("suspicious_login_activity", {
                "ip": client_ip,
                "username": form_data.username,
                "attempts": attempts
            })
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please try again later."
            )
    
    # Find user by username or email
    user = db.query(User).filter(
        (User.username == form_data.username) | (User.email == form_data.username)
    ).first()

    if not user or not security_manager.verify_password(form_data.password, user.hashed_password):
        # Increment failed attempts
        if cache:
            cache.l2_cache.increment(login_attempts_key, ttl=3600)
        
        log_security_event("failed_login_attempt", {
            "ip": client_ip,
            "username": form_data.username,
            "user_exists": user is not None
        })
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        log_security_event("inactive_user_login_attempt", {
            "ip": client_ip,
            "user_id": user.id,
            "username": user.username
        })
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Account is deactivated. Please contact support."
        )

    # Clear failed attempts on successful login
    if cache:
        cache.l2_cache.delete(login_attempts_key)
    
    # Update last login
    user.update_last_login()
    db.commit()

    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security_manager.create_access_token(
        data={"sub": user.id, "username": user.username}, 
        expires_delta=access_token_expires
    )
    
    # Create refresh token
    refresh_token = security_manager.create_refresh_token(
        data={"sub": user.id, "username": user.username}
    )
    
    # Log successful login
    log_security_event("successful_login", {
        "user_id": user.id,
        "username": user.username,
        "ip": client_ip
    })

    return {
        "access_token": access_token, 
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }


@router.post("/refresh", response_model=Token)
@rate_limit(requests=20, window=300)
async def refresh_token(
    refresh_token: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token"""
    client_ip = get_client_ip(request)
    
    try:
        # Verify refresh token
        payload = security_manager.verify_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        user_id = payload.get("sub")
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = security_manager.create_access_token(
            data={"sub": user.id, "username": user.username},
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except Exception as e:
        log_security_event("invalid_refresh_token", {
            "ip": client_ip,
            "error": str(e)
        })
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user),
    request: Request = None
):
    """Logout user and invalidate tokens"""
    client_ip = get_client_ip(request) if request else "unknown"
    
    # In a real implementation, you would add the token to a blacklist
    # For now, we'll just log the logout
    log_security_event("user_logout", {
        "user_id": current_user.id,
        "username": current_user.username,
        "ip": client_ip
    })
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information"""
    return current_user


@router.post("/change-password")
@rate_limit(requests=5, window=300)
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """Change user password"""
    client_ip = get_client_ip(request) if request else "unknown"
    
    # Verify current password
    if not security_manager.verify_password(password_data.current_password, current_user.hashed_password):
        log_security_event("incorrect_password_change", {
            "user_id": current_user.id,
            "ip": client_ip
        })
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Validate new password strength
    password_check = security_manager.is_password_strong(password_data.new_password)
    if not password_check["is_strong"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"New password is too weak: {', '.join(password_check['issues'])}"
        )
    
    # Update password
    current_user.hashed_password = security_manager.hash_password(password_data.new_password)
    db.commit()
    
    # Clear cached user data
    if cache:
        cache.delete(f"user:{current_user.id}")
    
    log_security_event("password_changed", {
        "user_id": current_user.id,
        "ip": client_ip
    })
    
    return {"message": "Password changed successfully"}
