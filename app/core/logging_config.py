"""
Advanced structured logging configuration
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import traceback
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "duration"):
            log_entry["duration"] = record.duration
        if hasattr(record, "status_code"):
            log_entry["status_code"] = record.status_code

        return json.dumps(log_entry, default=str)


class StructuredLogger:
    """Advanced structured logging system"""

    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """Setup log handlers"""
        # Console handler with JSON formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)

        # File handler for application logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)

        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "error.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(error_handler)

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration: float,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """Log HTTP request"""
        self.logger.info(
            f"{method} {path}",
            extra={
                "request_id": request_id,
                "user_id": user_id,
                "status_code": status_code,
                "duration": duration,
                "event_type": "http_request",
            },
        )

    def log_database_query(self, query: str, duration: float, rows_affected: int = 0):
        """Log database query"""
        self.logger.debug(
            "Database query executed",
            extra={
                "query": query[:200] + "..." if len(query) > 200 else query,
                "duration": duration,
                "rows_affected": rows_affected,
                "event_type": "database_query",
            },
        )

    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log security-related events"""
        self.logger.warning(
            f"Security event: {event_type}",
            extra={
                "user_id": user_id,
                "ip_address": ip_address,
                "event_type": "security",
                "security_event": event_type,
                "details": details or {},
            },
        )

    def log_performance_metric(self, metric_name: str, value: float, unit: str = "ms"):
        """Log performance metrics"""
        self.logger.info(
            f"Performance metric: {metric_name}",
            extra={
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "event_type": "performance",
            },
        )

    def log_business_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log business events"""
        self.logger.info(
            f"Business event: {event_type}",
            extra={
                "user_id": user_id,
                "event_type": "business",
                "business_event": event_type,
                "details": details or {},
            },
        )


class LoggingMiddleware:
    """FastAPI middleware for request logging"""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    async def __call__(self, request, call_next):
        start_time = datetime.utcnow()
        import uuid
        import time

        request_id = (
            str(uuid.uuid4()) if "uuid" in globals() else f"req_{int(time.time())}"
        )

        # Log request start
        self.logger.logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "event_type": "request_start",
            },
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Log request completion
        self.logger.log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            request_id=request_id,
        )

        return response


def setup_logging(log_level: str = "INFO") -> StructuredLogger:
    """Setup application logging"""
    # Configure root logger
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    # Create structured logger
    logger = StructuredLogger("opinion_market", log_level)

    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    return logger


# Global logger instance
app_logger = setup_logging()
