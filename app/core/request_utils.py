"""
Request utility functions for tracking and processing requests
"""

import uuid
import logging
from typing import Optional
from fastapi import Request

logger = logging.getLogger(__name__)


def get_request_id(request: Request) -> str:
    """
    Get or create a request ID for tracking.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Request ID string
    """
    # Check if request ID already exists in state
    if hasattr(request.state, "request_id"):
        return request.state.request_id
    
    # Check if request ID is in headers (from upstream services)
    request_id = request.headers.get("X-Request-ID")
    if request_id:
        request.state.request_id = request_id
        return request_id
    
    # Generate new request ID
    request_id = f"req_{uuid.uuid4().hex[:12]}_{int(__import__('time').time())}"
    request.state.request_id = request_id
    
    return request_id


def get_client_info(request: Request) -> dict:
    """
    Extract client information from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Dict containing client information
    """
    client_ip = getattr(request.state, "client_ip", None)
    if not client_ip:
        # Try to get from headers (for proxied requests)
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
            request.headers.get("X-Real-IP") or
            request.client.host if request.client else "unknown"
        )
        request.state.client_ip = client_ip
    
    return {
        "ip": client_ip,
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "referer": request.headers.get("Referer"),
        "origin": request.headers.get("Origin"),
    }


def log_request_info(request: Request, endpoint: str, user_id: Optional[int] = None) -> None:
    """
    Log request information for tracking and debugging.
    
    Args:
        request: FastAPI request object
        endpoint: Endpoint being called
        user_id: Optional user ID
    """
    request_id = get_request_id(request)
    client_info = get_client_info(request)
    
    logger.info(
        f"Request: {request.method} {endpoint}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "endpoint": endpoint,
            "user_id": user_id,
            "client_ip": client_info["ip"],
            "user_agent": client_info["user_agent"],
        }
    )

