"""
Enhanced error handling and logging utilities
"""

import logging
import traceback
import uuid
from typing import Any, Dict, Optional, Union, Callable
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from pydantic import ValidationError
import time
import asyncio
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    SECURITY = "security"


class ErrorHandler:
    """Centralized error handling for the application"""

    def __init__(self):
        self.error_counts = {}
        self.error_history = []

    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        return f"ERR_{uuid.uuid4().hex[:8].upper()}_{int(time.time() * 1000)}"

    def _categorize_error(self, exc: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Categorize error type and severity"""
        if isinstance(exc, RequestValidationError):
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW
        elif isinstance(exc, HTTPException):
            if exc.status_code == 401:
                return ErrorCategory.AUTHENTICATION, ErrorSeverity.MEDIUM
            elif exc.status_code == 403:
                return ErrorCategory.AUTHORIZATION, ErrorSeverity.MEDIUM
            elif exc.status_code >= 500:
                return ErrorCategory.SYSTEM, ErrorSeverity.HIGH
            else:
                return ErrorCategory.BUSINESS_LOGIC, ErrorSeverity.LOW
        elif isinstance(exc, (SQLAlchemyError, IntegrityError)):
            return ErrorCategory.DATABASE, ErrorSeverity.HIGH
        elif isinstance(exc, ValidationError):
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW
        else:
            return ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL

    def _should_alert(self, category: ErrorCategory, severity: ErrorSeverity) -> bool:
        """Determine if error should trigger alerts"""
        return severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]

    def handle_api_error(
        self,
        request: Request,
        exc: Exception,
        status_code: int = 500,
        detail: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> JSONResponse:
        """Handle API errors with comprehensive logging and response"""

        error_id = self._generate_error_id()
        category, severity = self._categorize_error(exc)
        
        # Track error frequency
        error_key = f"{type(exc).__name__}:{request.url.path}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Prepare error context
        error_context = {
            "error_id": error_id,
            "category": category.value,
            "severity": severity.value,
            "path": str(request.url),
            "method": request.method,
            "user_id": user_id,
            "client_ip": getattr(request.state, "client_ip", None),
            "user_agent": request.headers.get("user-agent", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "error_count": self.error_counts[error_key]
        }

        # Log the error with appropriate level
        log_level = logging.ERROR if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else logging.WARNING
        logger.log(
            log_level,
            f"API Error {error_id}: {str(exc)}",
            extra={
                **error_context,
                "traceback": traceback.format_exc(),
            },
        )

        # Store error in history (keep last 1000 errors)
        self.error_history.append(error_context)
        if len(self.error_history) > 1000:
            self.error_history.pop(0)

        # Trigger alerts for critical errors
        if self._should_alert(category, severity):
            self._trigger_alert(error_context, exc)

        # Prepare error response
        error_response = {
            "error_id": error_id,
            "message": detail or self._get_user_friendly_message(exc, category),
            "status_code": status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "category": category.value,
        }

        # Include additional details in development
        if hasattr(request.app.state, "debug") and request.app.state.debug:
            error_response["details"] = {
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
                "context": error_context
            }

        return JSONResponse(status_code=status_code, content=error_response)

    def _get_user_friendly_message(self, exc: Exception, category: ErrorCategory) -> str:
        """Get user-friendly error message"""
        if isinstance(exc, HTTPException):
            return exc.detail
        elif category == ErrorCategory.VALIDATION:
            return "Invalid input data provided"
        elif category == ErrorCategory.AUTHENTICATION:
            return "Authentication required"
        elif category == ErrorCategory.AUTHORIZATION:
            return "Insufficient permissions"
        elif category == ErrorCategory.DATABASE:
            return "Database operation failed"
        else:
            return "An internal error occurred"

    def _trigger_alert(self, error_context: Dict[str, Any], exc: Exception):
        """Trigger alert for critical errors"""
        try:
            # This would integrate with your alerting system
            logger.critical(
                f"CRITICAL ERROR ALERT: {error_context['error_id']}",
                extra=error_context
            )
            # Here you could send to Slack, email, PagerDuty, etc.
        except Exception as alert_error:
            logger.error(f"Failed to send alert: {alert_error}")

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts,
            "recent_errors": self.error_history[-10:] if self.error_history else [],
            "error_categories": {
                category.value: len([e for e in self.error_history if e.get("category") == category.value])
                for category in ErrorCategory
            }
        }

    @staticmethod
    def handle_validation_error(exc: Exception) -> JSONResponse:
        """Handle validation errors"""
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "message": str(exc),
                "timestamp": time.time(),
            },
        )

    @staticmethod
    def handle_database_error(exc: Exception) -> JSONResponse:
        """Handle database errors"""
        error_id = f"DB_ERR_{int(time.time() * 1000)}"

        logger.error(
            f"Database Error {error_id}: {str(exc)}",
            extra={"error_id": error_id, "traceback": traceback.format_exc()},
        )

        return JSONResponse(
            status_code=500,
            content={
                "error_id": error_id,
                "message": "Database operation failed",
                "timestamp": time.time(),
            },
        )


class RetryHandler:
    """Utility for retrying operations with exponential backoff"""

    @staticmethod
    async def retry_async(
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        *args,
        **kwargs,
    ):
        """Retry an async function with exponential backoff"""
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    break

                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                )
                import asyncio

                await asyncio.sleep(delay)

        raise last_exception

    @staticmethod
    def retry_sync(
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        *args,
        **kwargs,
    ):
        """Retry a sync function with exponential backoff"""
        import time

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    break

                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                )
                time.sleep(delay)

        raise last_exception


class ErrorHandlingDecorator:
    """Decorator for automatic error handling"""

    @staticmethod
    def handle_errors(
        default_status_code: int = 500,
        default_message: str = "An error occurred",
        log_errors: bool = True,
        reraise: bool = False
    ):
        """Decorator to handle errors in API endpoints"""
        def decorator(func: Callable):
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except HTTPException:
                    raise  # Re-raise HTTP exceptions as-is
                except Exception as e:
                    if log_errors:
                        logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                    
                    if reraise:
                        raise
                    
                    # Try to extract request from args
                    request = None
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break
                    
                    if request:
                        error_handler = ErrorHandler()
                        return error_handler.handle_api_error(
                            request, e, default_status_code, default_message
                        )
                    else:
                        raise HTTPException(
                            status_code=default_status_code,
                            detail=default_message
                        )
            
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except HTTPException:
                    raise  # Re-raise HTTP exceptions as-is
                except Exception as e:
                    if log_errors:
                        logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                    
                    if reraise:
                        raise
                    
                    # Try to extract request from args
                    request = None
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break
                    
                    if request:
                        error_handler = ErrorHandler()
                        return error_handler.handle_api_error(
                            request, e, default_status_code, default_message
                        )
                    else:
                        raise HTTPException(
                            status_code=default_status_code,
                            detail=default_message
                        )
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator


# Global error handler instance
error_handler = ErrorHandler()
retry_handler = RetryHandler()
error_decorator = ErrorHandlingDecorator()
