import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from sqlalchemy import desc

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.decorators import handle_errors, log_execution_time
from app.models.user import User
from app.schemas.user import UserResponse, UserUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/me", response_model=UserResponse)
@handle_errors(default_message="Failed to retrieve user profile")
@log_execution_time
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    """
    Get current authenticated user's profile.
    
    Args:
        current_user: Current authenticated user from dependency
        
    Returns:
        User profile response
    """
    return current_user


@router.put("/me", response_model=UserResponse)
@handle_errors(default_message="Failed to update user profile")
@log_execution_time
async def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserResponse:
    """
    Update current user's profile.
    
    Args:
        user_update: User update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated user profile response
        
    Raises:
        HTTPException: If validation fails
    """
    # Update user fields
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:  # Only update non-None values
            setattr(current_user, field, value)

    db.commit()
    db.refresh(current_user)

    logger.info(f"User profile updated: {current_user.id}")
    return current_user


@router.get("/{user_id}", response_model=UserResponse)
@handle_errors(default_message="Failed to retrieve user profile")
@log_execution_time
async def get_user_profile(
    user_id: int, 
    db: Session = Depends(get_db)
) -> UserResponse:
    """
    Get a specific user's profile by ID.
    
    Args:
        user_id: User ID to retrieve
        db: Database session
        
    Returns:
        User profile response
        
    Raises:
        HTTPException: If user not found
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="User not found"
        )
    return user


@router.get("/", response_model=List[UserResponse])
@handle_errors(default_message="Failed to retrieve users")
@log_execution_time
async def get_users(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum number of records to return"),
    db: Session = Depends(get_db)
) -> List[UserResponse]:
    """
    Get list of users with pagination.
    
    Args:
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return
        db: Database session
        
    Returns:
        List of user responses
    """
    users = db.query(User).order_by(desc(User.created_at)).offset(skip).limit(limit).all()
    return users
