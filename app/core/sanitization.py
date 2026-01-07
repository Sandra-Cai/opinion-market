"""
Input sanitization utilities for security and data cleaning
"""

import re
import html
import logging
from typing import Any, Dict, List, Optional, Union
from app.core.validation import input_validator

logger = logging.getLogger(__name__)


def sanitize_string(
    value: str,
    max_length: int = 1000,
    strip_whitespace: bool = True,
    remove_html: bool = True,
    remove_sql_patterns: bool = True,
    remove_xss_patterns: bool = True
) -> str:
    """
    Sanitize a string input for safe storage and display.
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        strip_whitespace: Remove leading/trailing whitespace
        remove_html: Remove HTML tags
        remove_sql_patterns: Remove SQL injection patterns
        remove_xss_patterns: Remove XSS patterns
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        value = str(value)
    
    # Strip whitespace
    if strip_whitespace:
        value = value.strip()
    
    # Truncate if too long
    if len(value) > max_length:
        value = value[:max_length]
        logger.warning(f"String truncated to {max_length} characters")
    
    # Remove HTML
    if remove_html:
        # First escape HTML entities
        value = html.escape(value)
        # Remove any remaining HTML tags
        value = re.sub(r'<[^>]+>', '', value)
    
    # Remove SQL injection patterns
    if remove_sql_patterns:
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|\/\*|\*\/)",
        ]
        for pattern in sql_patterns:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE)
    
    # Remove XSS patterns
    if remove_xss_patterns:
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        for pattern in xss_patterns:
            value = re.sub(pattern, '', value, flags=re.IGNORECASE)
    
    return value


def sanitize_dict(
    data: Dict[str, Any],
    string_fields: Optional[List[str]] = None,
    max_length: int = 1000
) -> Dict[str, Any]:
    """
    Sanitize all string values in a dictionary.
    
    Args:
        data: Dictionary to sanitize
        string_fields: Optional list of field names to sanitize (if None, sanitize all strings)
        max_length: Maximum length for string fields
        
    Returns:
        Sanitized dictionary
    """
    sanitized = {}
    
    for key, value in data.items():
        # Only sanitize specified fields if list provided
        if string_fields and key not in string_fields:
            sanitized[key] = value
            continue
        
        if isinstance(value, str):
            sanitized[key] = sanitize_string(value, max_length=max_length)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, string_fields, max_length)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_string(item, max_length=max_length) if isinstance(item, str)
                else sanitize_dict(item, string_fields, max_length) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


def sanitize_email(email: str) -> str:
    """
    Sanitize an email address.
    
    Args:
        email: Email address to sanitize
        
    Returns:
        Sanitized email address
    """
    if not email:
        return ""
    
    # Remove whitespace and convert to lowercase
    email = email.strip().lower()
    
    # Basic email validation pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        logger.warning(f"Invalid email format: {email}")
        return ""
    
    return email


def sanitize_url(url: str, allowed_schemes: Optional[List[str]] = None) -> str:
    """
    Sanitize a URL.
    
    Args:
        url: URL to sanitize
        allowed_schemes: List of allowed URL schemes (default: http, https)
        
    Returns:
        Sanitized URL or empty string if invalid
    """
    if not url:
        return ""
    
    if allowed_schemes is None:
        allowed_schemes = ["http", "https"]
    
    url = url.strip()
    
    # Check scheme
    for scheme in allowed_schemes:
        if url.startswith(f"{scheme}://"):
            # Basic URL validation
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            if re.match(url_pattern, url):
                return url
    
    logger.warning(f"Invalid URL scheme or format: {url}")
    return ""


def sanitize_number(
    value: Union[int, float, str],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    integer_only: bool = False
) -> Optional[Union[int, float]]:
    """
    Sanitize and validate a number.
    
    Args:
        value: Number to sanitize
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        integer_only: If True, only allow integers
        
    Returns:
        Sanitized number or None if invalid
    """
    try:
        if integer_only:
            num = int(value)
        else:
            num = float(value)
        
        if min_value is not None and num < min_value:
            logger.warning(f"Number {num} below minimum {min_value}")
            return None
        
        if max_value is not None and num > max_value:
            logger.warning(f"Number {num} above maximum {max_value}")
            return None
        
        return num
    
    except (ValueError, TypeError):
        logger.warning(f"Invalid number: {value}")
        return None

