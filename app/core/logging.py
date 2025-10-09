"""
Advanced logging configuration for Opinion Market
Provides structured logging with multiple outputs and monitoring
"""

import logging
import logging.handlers
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import structlog
from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "getMessage", "exc_info",
                "exc_text", "stack_info"
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for development"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class PerformanceFilter(logging.Filter):
    """Filter to track performance metrics"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add performance tracking
        if hasattr(record, 'duration'):
            if record.duration > 1.0:
                record.levelno = logging.WARNING
                record.levelname = 'WARNING'
                record.msg = f"SLOW: {record.msg}"
        return True


class SecurityFilter(logging.Filter):
    """Filter to enhance security-related logs"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Enhance security logs
        if 'security' in record.name.lower() or 'auth' in record.name.lower():
            record.levelno = max(record.levelno, logging.INFO)
            # Add security context
            if not hasattr(record, 'security_context'):
                record.security_context = True
        return True


def setup_logging():
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" 
            else structlog.dev.ConsoleRenderer(colors=True)
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.value))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if settings.LOG_FORMAT == "json":
        console_formatter = JSONFormatter()
    else:
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(PerformanceFilter())
    console_handler.addFilter(SecurityFilter())
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    if settings.LOG_FILE:
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / settings.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(PerformanceFilter())
        file_handler.addFilter(SecurityFilter())
        root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "error.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = JSONFormatter()
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)
    
    # Performance log handler
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / "performance.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    perf_handler.setLevel(logging.WARNING)
    perf_formatter = JSONFormatter()
    perf_handler.setFormatter(perf_formatter)
    perf_handler.addFilter(PerformanceFilter())
    root_logger.addHandler(perf_handler)
    
    # Security log handler
    security_handler = logging.handlers.RotatingFileHandler(
        log_dir / "security.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    security_handler.setLevel(logging.INFO)
    security_formatter = JSONFormatter()
    security_handler.setFormatter(security_formatter)
    security_handler.addFilter(SecurityFilter())
    root_logger.addHandler(security_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.DEBUG else logging.WARNING
    )
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Create structured logger
    logger = structlog.get_logger("opinion_market")
    logger.info("Logging system initialized", 
                level=settings.LOG_LEVEL.value,
                format=settings.LOG_FORMAT)
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self):
        """Get structured logger for this class"""
        return structlog.get_logger(self.__class__.__name__)


class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, operation: str, logger: Optional[structlog.BoundLogger] = None):
        self.operation = operation
        self.logger = logger or structlog.get_logger("performance")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug("Operation started", operation=self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            self.logger.error("Operation failed",
                            operation=self.operation,
                            duration=duration,
                            error=str(exc_val))
        else:
            self.logger.info("Operation completed",
                           operation=self.operation,
                           duration=duration)


def log_function_call(func):
    """Decorator to log function calls with performance metrics"""
    def wrapper(*args, **kwargs):
        logger = structlog.get_logger(func.__module__)
        start_time = time.time()
        
        logger.debug("Function called",
                    function=func.__name__,
                    args_count=len(args),
                    kwargs_count=len(kwargs))
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug("Function completed",
                        function=func.__name__,
                        duration=duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error("Function failed",
                        function=func.__name__,
                        duration=duration,
                        error=str(e))
            raise
    
    return wrapper


def log_api_call(endpoint: str, method: str, user_id: Optional[int] = None):
    """Log API calls with context"""
    logger = structlog.get_logger("api")
    logger.info("API call",
               endpoint=endpoint,
               method=method,
               user_id=user_id,
               timestamp=datetime.utcnow().isoformat())


def log_security_event(event_type: str, details: Dict[str, Any], user_id: Optional[int] = None):
    """Log security-related events"""
    logger = structlog.get_logger("security")
    logger.warning("Security event",
                  event_type=event_type,
                  details=details,
                  user_id=user_id,
                  timestamp=datetime.utcnow().isoformat())


def log_trading_event(event_type: str, market_id: int, user_id: int, amount: float, **kwargs):
    """Log trading-related events"""
    logger = structlog.get_logger("trading")
    logger.info("Trading event",
               event_type=event_type,
               market_id=market_id,
               user_id=user_id,
               amount=amount,
               **kwargs,
               timestamp=datetime.utcnow().isoformat())


def log_system_metric(metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Log system metrics for monitoring"""
    logger = structlog.get_logger("metrics")
    logger.info("System metric",
               metric_name=metric_name,
               value=value,
               tags=tags or {},
               timestamp=datetime.utcnow().isoformat())


# Initialize logging system
main_logger = setup_logging()
