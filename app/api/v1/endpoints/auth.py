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
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    # Find user by username
    user = db.query(User).filter(User.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}
