"""
Decorators for API endpoints and common functionality
"""

import asyncio
import functools
import logging
import traceback
from typing import Any, Callable, TypeVar, ParamSpec
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

from app.core.response_helpers import error_response
from app.core.logging import log_system_metric

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def handle_errors(
    default_message: str = "An error occurred processing your request",
    log_error: bool = True
) -> Callable[[Callable[P, T]], Callable[P, T | JSONResponse]]:
    """
    Decorator to handle errors consistently across API endpoints.
    Supports both sync and async functions.
    
    Args:
        default_message: Default error message if exception doesn't have one
        log_error: Whether to log the error
        
    Example:
        @handle_errors(default_message="Failed to create trade")
        async def create_trade(...):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T | JSONResponse]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T | JSONResponse:
                try:
                    return await func(*args, **kwargs)
                except HTTPException:
                    raise
                except ValueError as e:
                    if log_error:
                        logger.warning(f"Validation error in {func.__name__}: {e}", exc_info=True)
                    return error_response(
                        message=str(e) or "Invalid input data",
                        error_code="VALIDATION_ERROR",
                        status_code=status.HTTP_400_BAD_REQUEST
                    )
                except KeyError as e:
                    if log_error:
                        logger.warning(f"Missing field in {func.__name__}: {e}", exc_info=True)
                    return error_response(
                        message=f"Missing required field: {e}",
                        error_code="MISSING_FIELD",
                        status_code=status.HTTP_400_BAD_REQUEST
                    )
                except Exception as e:
                    if log_error:
                        logger.error(
                            f"Unexpected error in {func.__name__}: {e}",
                            exc_info=True,
                            extra={"traceback": traceback.format_exc()}
                        )
                        log_system_metric("api_error", 1, {
                            "endpoint": func.__name__,
                            "error_type": type(e).__name__
                        })
                    return error_response(
                        message=default_message,
                        error_code="INTERNAL_ERROR",
                        details={"error_type": type(e).__name__} if logger.isEnabledFor(logging.DEBUG) else None,
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T | JSONResponse:
                try:
                    return func(*args, **kwargs)
                except HTTPException:
                    raise
                except ValueError as e:
                    if log_error:
                        logger.warning(f"Validation error in {func.__name__}: {e}", exc_info=True)
                    return error_response(
                        message=str(e) or "Invalid input data",
                        error_code="VALIDATION_ERROR",
                        status_code=status.HTTP_400_BAD_REQUEST
                    )
                except KeyError as e:
                    if log_error:
                        logger.warning(f"Missing field in {func.__name__}: {e}", exc_info=True)
                    return error_response(
                        message=f"Missing required field: {e}",
                        error_code="MISSING_FIELD",
                        status_code=status.HTTP_400_BAD_REQUEST
                    )
                except Exception as e:
                    if log_error:
                        logger.error(
                            f"Unexpected error in {func.__name__}: {e}",
                            exc_info=True,
                            extra={"traceback": traceback.format_exc()}
                        )
                        log_system_metric("api_error", 1, {
                            "endpoint": func.__name__,
                            "error_type": type(e).__name__
                        })
                    return error_response(
                        message=default_message,
                        error_code="INTERNAL_ERROR",
                        details={"error_type": type(e).__name__} if logger.isEnabledFor(logging.DEBUG) else None,
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            return sync_wrapper
    return decorator


def validate_input(
    required_fields: list[str] | None = None,
    max_length: dict[str, int] | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to validate input data.
    
    Args:
        required_fields: List of required field names
        max_length: Dict mapping field names to max lengths
        
    Example:
        @validate_input(
            required_fields=["name", "email"],
            max_length={"name": 100, "email": 255}
        )
        async def create_user(data: UserCreate):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Validate required fields if data object is in kwargs
            if required_fields:
                for arg in args:
                    if hasattr(arg, '__dict__'):
                        for field in required_fields:
                            if not hasattr(arg, field) or getattr(arg, field) is None:
                                raise ValueError(f"Required field '{field}' is missing")
            
            # Validate max lengths
            if max_length:
                for arg in args:
                    if hasattr(arg, '__dict__'):
                        for field, max_len in max_length.items():
                            if hasattr(arg, field):
                                value = getattr(arg, field)
                                if isinstance(value, str) and len(value) > max_len:
                                    raise ValueError(
                                        f"Field '{field}' exceeds maximum length of {max_len}"
                                    )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_execution_time(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to log function execution time.
    Supports both sync and async functions.
    
    Example:
        @log_execution_time
        async def expensive_operation():
            ...
    """
    import time
    
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > 1.0:  # Log slow operations
                    logger.warning(
                        f"Slow operation: {func.__name__} took {duration:.2f}s",
                        extra={"duration": duration, "function": func.__name__}
                    )
                else:
                    logger.debug(
                        f"Operation completed: {func.__name__} took {duration:.2f}s",
                        extra={"duration": duration, "function": func.__name__}
                    )
                
                log_system_metric("function_execution_time", duration, {
                    "function": func.__name__
                })
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation failed: {func.__name__} failed after {duration:.2f}s",
                    exc_info=True,
                    extra={"duration": duration, "function": func.__name__}
                )
                raise
        
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if duration > 1.0:  # Log slow operations
                    logger.warning(
                        f"Slow operation: {func.__name__} took {duration:.2f}s",
                        extra={"duration": duration, "function": func.__name__}
                    )
                else:
                    logger.debug(
                        f"Operation completed: {func.__name__} took {duration:.2f}s",
                        extra={"duration": duration, "function": func.__name__}
                    )
                
                log_system_metric("function_execution_time", duration, {
                    "function": func.__name__
                })
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Operation failed: {func.__name__} failed after {duration:.2f}s",
                    exc_info=True,
                    extra={"duration": duration, "function": func.__name__}
                )
                raise
        
        return sync_wrapper

