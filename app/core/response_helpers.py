"""
Response helper utilities for consistent API responses
"""

from typing import Any, Dict, Optional
from datetime import datetime
from fastapi.responses import JSONResponse
from fastapi import status


def success_response(
    data: Any,
    message: Optional[str] = None,
    status_code: int = status.HTTP_200_OK
) -> JSONResponse:
    """
    Create a standardized success response.
    
    Args:
        data: The response data
        message: Optional success message
        status_code: HTTP status code (default: 200)
        
    Returns:
        JSONResponse with standardized format
    """
    response_data: Dict[str, Any] = {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if message:
        response_data["message"] = message
    
    return JSONResponse(content=response_data, status_code=status_code)


def error_response(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = status.HTTP_400_BAD_REQUEST
) -> JSONResponse:
    """
    Create a standardized error response.
    
    Args:
        message: Human-readable error message
        error_code: Optional error code for programmatic handling
        details: Optional additional error details
        status_code: HTTP status code (default: 400)
        
    Returns:
        JSONResponse with standardized error format
    """
    response_data: Dict[str, Any] = {
        "success": False,
        "error": {
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    if error_code:
        response_data["error"]["code"] = error_code
    
    if details:
        response_data["error"]["details"] = details
    
    return JSONResponse(content=response_data, status_code=status_code)


def paginated_response(
    items: list[Any],
    total: int,
    page: int = 1,
    page_size: int = 20,
    message: Optional[str] = None
) -> JSONResponse:
    """
    Create a standardized paginated response.
    
    Args:
        items: List of items for current page
        total: Total number of items
        page: Current page number (1-indexed)
        page_size: Number of items per page
        message: Optional message
        
    Returns:
        JSONResponse with paginated data
    """
    total_pages = (total + page_size - 1) // page_size
    
    response_data: Dict[str, Any] = {
        "success": True,
        "data": {
            "items": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if message:
        response_data["message"] = message
    
    return JSONResponse(content=response_data, status_code=status.HTTP_200_OK)

