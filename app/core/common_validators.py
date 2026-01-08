"""
Common validation utilities and validators for reuse across the application
"""

import re
from typing import Any, Optional, List, Dict
from pydantic import validator, BaseModel
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PaginationParams(BaseModel):
    """Common pagination parameters"""
    skip: int = 0
    limit: int = 20
    
    @validator('skip')
    def validate_skip(cls, v):
        if v < 0:
            raise ValueError('skip must be non-negative')
        return v
    
    @validator('limit')
    def validate_limit(cls, v):
        if v < 1:
            raise ValueError('limit must be at least 1')
        if v > 100:
            raise ValueError('limit cannot exceed 100')
        return v


class DateRangeParams(BaseModel):
    """Common date range parameters"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v and values.get('start_date') and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class SearchParams(BaseModel):
    """Common search parameters"""
    search: Optional[str] = None
    search_fields: Optional[List[str]] = None
    
    @validator('search')
    def validate_search(cls, v):
        if v and len(v) < 2:
            raise ValueError('search term must be at least 2 characters')
        if v and len(v) > 100:
            raise ValueError('search term cannot exceed 100 characters')
        return v


class SortParams(BaseModel):
    """Common sorting parameters"""
    order_by: Optional[str] = None
    order_direction: str = "desc"
    
    @validator('order_direction')
    def validate_order_direction(cls, v):
        if v not in ["asc", "desc"]:
            raise ValueError('order_direction must be "asc" or "desc"')
        return v


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_username(username: str) -> bool:
    """
    Validate username format.
    
    Args:
        username: Username to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not username:
        return False
    
    # Username: 3-50 characters, alphanumeric and underscores only
    pattern = r'^[a-zA-Z0-9_]{3,50}$'
    return bool(re.match(pattern, username))


def validate_password_strength(password: str) -> tuple[bool, List[str]]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    
    if not re.search(r'[A-Z]', password):
        issues.append("Password must contain at least one uppercase letter")
    
    if not re.search(r'[a-z]', password):
        issues.append("Password must contain at least one lowercase letter")
    
    if not re.search(r'\d', password):
        issues.append("Password must contain at least one number")
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        issues.append("Password must contain at least one special character")
    
    return len(issues) == 0, issues


def validate_amount(amount: float, min_amount: float = 0.01, max_amount: float = 1000000.0) -> bool:
    """
    Validate monetary amount.
    
    Args:
        amount: Amount to validate
        min_amount: Minimum allowed amount
        max_amount: Maximum allowed amount
        
    Returns:
        True if valid, False otherwise
    """
    if amount < min_amount:
        logger.warning(f"Amount {amount} below minimum {min_amount}")
        return False
    
    if amount > max_amount:
        logger.warning(f"Amount {amount} above maximum {max_amount}")
        return False
    
    return True


def validate_percentage(value: float, min_value: float = 0.0, max_value: float = 100.0) -> bool:
    """
    Validate percentage value.
    
    Args:
        value: Percentage to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        True if valid, False otherwise
    """
    if value < min_value or value > max_value:
        return False
    return True


def validate_date_in_future(date: datetime, min_days_ahead: int = 0) -> bool:
    """
    Validate that a date is in the future.
    
    Args:
        date: Date to validate
        min_days_ahead: Minimum number of days in the future
        
    Returns:
        True if valid, False otherwise
    """
    min_date = datetime.utcnow() + timedelta(days=min_days_ahead)
    return date > min_date


def validate_date_in_past(date: datetime, max_days_ago: Optional[int] = None) -> bool:
    """
    Validate that a date is in the past.
    
    Args:
        date: Date to validate
        max_days_ago: Maximum number of days in the past (None for no limit)
        
    Returns:
        True if valid, False otherwise
    """
    if date > datetime.utcnow():
        return False
    
    if max_days_ago:
        max_date = datetime.utcnow() - timedelta(days=max_days_ago)
        if date < max_date:
            return False
    
    return True


def validate_id(id_value: Any, min_value: int = 1) -> bool:
    """
    Validate ID value.
    
    Args:
        id_value: ID to validate
        min_value: Minimum allowed value
        
    Returns:
        True if valid, False otherwise
    """
    try:
        id_int = int(id_value)
        return id_int >= min_value
    except (ValueError, TypeError):
        return False


def validate_enum_value(value: str, allowed_values: List[str], case_sensitive: bool = False) -> bool:
    """
    Validate enum-like value.
    
    Args:
        value: Value to validate
        allowed_values: List of allowed values
        case_sensitive: Whether comparison should be case-sensitive
        
    Returns:
        True if valid, False otherwise
    """
    if not case_sensitive:
        value = value.lower()
        allowed_values = [v.lower() for v in allowed_values]
    
    return value in allowed_values

