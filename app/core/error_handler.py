"""
Enhanced error handling and logging utilities
"""

import logging
import traceback
from typing import Any, Dict, Optional
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import time

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling for the application"""

    @staticmethod
    def handle_api_error(
        request: Request,
        exc: Exception,
        status_code: int = 500,
        detail: Optional[str] = None,
    ) -> JSONResponse:
        """Handle API errors with proper logging and response"""

        error_id = f"ERR_{int(time.time() * 1000)}"

        # Log the error
        logger.error(
            f"API Error {error_id}: {str(exc)}",
            extra={
                "error_id": error_id,
                "path": str(request.url),
                "method": request.method,
                "traceback": traceback.format_exc(),
            },
        )

        # Prepare error response
        error_response = {
            "error_id": error_id,
            "message": detail or "An internal error occurred",
            "status_code": status_code,
            "timestamp": time.time(),
        }

        return JSONResponse(status_code=status_code, content=error_response)

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


# Global error handler instance
error_handler = ErrorHandler()
retry_handler = RetryHandler()
