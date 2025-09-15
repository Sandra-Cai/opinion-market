"""
Enhanced Error Handling System
Provides comprehensive error handling, logging, and recovery mechanisms
"""

import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass
import json
import functools
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    DATABASE = "database"
    NETWORK = "network"
    EXTERNAL_API = "external_api"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class ErrorContext:
    """Context information for errors"""
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = None
    additional_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.additional_data is None:
            self.additional_data = {}


@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    stack_trace: Optional[str] = None
    recovery_suggestions: List[str] = None
    is_recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.recovery_suggestions is None:
            self.recovery_suggestions = []


class EnhancedErrorHandler:
    """Enhanced error handling system with comprehensive logging and recovery"""

    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.error_patterns: Dict[str, int] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.alert_thresholds = {
            ErrorSeverity.CRITICAL: 1,
            ErrorSeverity.HIGH: 5,
            ErrorSeverity.MEDIUM: 20,
            ErrorSeverity.LOW: 100
        }
        self.rate_limits: Dict[str, datetime] = {}

    def register_recovery_strategy(
        self, 
        error_pattern: str, 
        strategy: Callable,
        max_retries: int = 3
    ):
        """Register a recovery strategy for specific error patterns"""
        self.recovery_strategies[error_pattern] = {
            "strategy": strategy,
            "max_retries": max_retries
        }

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        additional_info: Dict[str, Any] = None
    ) -> ErrorInfo:
        """Handle an error with comprehensive logging and analysis"""
        
        # Create error info
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            category=category,
            context=context,
            stack_trace=traceback.format_exc(),
            is_recoverable=self._is_recoverable(error),
            recovery_suggestions=self._get_recovery_suggestions(error, category)
        )

        # Add additional info to context
        if additional_info:
            error_info.context.additional_data.update(additional_info)

        # Log the error
        self._log_error(error_info)

        # Track error patterns
        self._track_error_pattern(error_info)

        # Check for alert conditions
        self._check_alert_conditions(error_info)

        # Attempt recovery if applicable
        if error_info.is_recoverable:
            self._attempt_recovery(error_info)

        # Store in history
        self.error_history.append(error_info)

        return error_info

    def _is_recoverable(self, error: Exception) -> bool:
        """Determine if an error is recoverable"""
        recoverable_errors = (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError
        )
        return isinstance(error, recoverable_errors)

    def _get_recovery_suggestions(
        self, 
        error: Exception, 
        category: ErrorCategory
    ) -> List[str]:
        """Get recovery suggestions based on error type and category"""
        suggestions = []

        if category == ErrorCategory.DATABASE:
            suggestions.extend([
                "Check database connection",
                "Verify database credentials",
                "Check database server status",
                "Review query syntax"
            ])
        elif category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check network connectivity",
                "Verify endpoint URL",
                "Check firewall settings",
                "Retry with exponential backoff"
            ])
        elif category == ErrorCategory.AUTHENTICATION:
            suggestions.extend([
                "Verify authentication credentials",
                "Check token expiration",
                "Refresh authentication token",
                "Contact administrator"
            ])
        elif category == ErrorCategory.VALIDATION:
            suggestions.extend([
                "Validate input data",
                "Check data format",
                "Verify required fields",
                "Review data constraints"
            ])

        return suggestions

    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level based on severity"""
        log_data = {
            "error_type": error_info.error_type,
            "message": error_info.message,
            "severity": error_info.severity.value,
            "category": error_info.category.value,
            "context": {
                "user_id": error_info.context.user_id,
                "request_id": error_info.context.request_id,
                "endpoint": error_info.context.endpoint,
                "method": error_info.context.method,
                "timestamp": error_info.context.timestamp.isoformat(),
                "additional_data": error_info.context.additional_data
            },
            "is_recoverable": error_info.is_recoverable,
            "retry_count": error_info.retry_count
        }

        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {json.dumps(log_data)}")
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR: {json.dumps(log_data)}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR: {json.dumps(log_data)}")
        else:
            logger.info(f"LOW SEVERITY ERROR: {json.dumps(log_data)}")

        # Log stack trace for debugging
        if error_info.stack_trace:
            logger.debug(f"Stack trace: {error_info.stack_trace}")

    def _track_error_pattern(self, error_info: ErrorInfo):
        """Track error patterns for analysis"""
        pattern_key = f"{error_info.error_type}:{error_info.category.value}"
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1

    def _check_alert_conditions(self, error_info: ErrorInfo):
        """Check if error conditions warrant alerts"""
        threshold = self.alert_thresholds.get(error_info.severity, 0)
        pattern_key = f"{error_info.error_type}:{error_info.category.value}"
        
        if self.error_patterns.get(pattern_key, 0) >= threshold:
            self._send_alert(error_info, pattern_key)

    def _send_alert(self, error_info: ErrorInfo, pattern_key: str):
        """Send alert for error conditions"""
        alert_data = {
            "type": "error_alert",
            "pattern": pattern_key,
            "count": self.error_patterns[pattern_key],
            "severity": error_info.severity.value,
            "last_error": {
                "message": error_info.message,
                "timestamp": error_info.context.timestamp.isoformat(),
                "context": error_info.context.additional_data
            }
        }
        
        logger.critical(f"ALERT: {json.dumps(alert_data)}")
        # In production, this would send to alerting system (PagerDuty, Slack, etc.)

    def _attempt_recovery(self, error_info: ErrorInfo):
        """Attempt to recover from error using registered strategies"""
        for pattern, strategy_info in self.recovery_strategies.items():
            if pattern in error_info.error_type or pattern in error_info.message:
                if error_info.retry_count < strategy_info["max_retries"]:
                    try:
                        strategy_info["strategy"](error_info)
                        error_info.retry_count += 1
                        logger.info(f"Recovery strategy executed for {error_info.error_type}")
                    except Exception as recovery_error:
                        logger.error(f"Recovery strategy failed: {recovery_error}")

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [
            e for e in self.error_history 
            if e.context.timestamp >= cutoff_time
        ]

        stats = {
            "total_errors": len(recent_errors),
            "by_severity": {},
            "by_category": {},
            "by_type": {},
            "recovery_rate": 0,
            "top_patterns": []
        }

        # Count by severity
        for severity in ErrorSeverity:
            count = len([e for e in recent_errors if e.severity == severity])
            stats["by_severity"][severity.value] = count

        # Count by category
        for category in ErrorCategory:
            count = len([e for e in recent_errors if e.category == category])
            stats["by_category"][category.value] = count

        # Count by type
        for error in recent_errors:
            error_type = error.error_type
            stats["by_type"][error_type] = stats["by_type"].get(error_type, 0) + 1

        # Calculate recovery rate
        recovered_errors = len([e for e in recent_errors if e.retry_count > 0])
        if recent_errors:
            stats["recovery_rate"] = (recovered_errors / len(recent_errors)) * 100

        # Top error patterns
        sorted_patterns = sorted(
            self.error_patterns.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        stats["top_patterns"] = sorted_patterns[:10]

        return stats

    def cleanup_old_errors(self, days: int = 7):
        """Clean up old error history"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        self.error_history = [
            e for e in self.error_history 
            if e.context.timestamp >= cutoff_time
        ]


# Global error handler instance
enhanced_error_handler = EnhancedErrorHandler()


# Decorators for error handling
def handle_errors(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    reraise: bool = True
):
    """Decorator for automatic error handling"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    endpoint=getattr(func, '__name__', 'unknown'),
                    additional_data={"args": str(args), "kwargs": str(kwargs)}
                )
                
                error_info = enhanced_error_handler.handle_error(
                    e, context, severity, category
                )
                
                if reraise:
                    raise
                return None

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    endpoint=getattr(func, '__name__', 'unknown'),
                    additional_data={"args": str(args), "kwargs": str(kwargs)}
                )
                
                error_info = enhanced_error_handler.handle_error(
                    e, context, severity, category
                )
                
                if reraise:
                    raise
                return None

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@asynccontextmanager
async def error_context(
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    method: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
):
    """Context manager for error handling with specific context"""
    context = ErrorContext(
        user_id=user_id,
        request_id=request_id,
        endpoint=endpoint,
        method=method,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    try:
        yield context
    except Exception as e:
        enhanced_error_handler.handle_error(
            e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM
        )
        raise


# Recovery strategies
def database_recovery_strategy(error_info: ErrorInfo):
    """Recovery strategy for database errors"""
    logger.info("Attempting database recovery...")
    # In production, this would attempt to reconnect to database
    # or switch to read replica, etc.


def network_recovery_strategy(error_info: ErrorInfo):
    """Recovery strategy for network errors"""
    logger.info("Attempting network recovery...")
    # In production, this would implement retry logic with backoff


# Register default recovery strategies
enhanced_error_handler.register_recovery_strategy(
    "DatabaseError", database_recovery_strategy
)
enhanced_error_handler.register_recovery_strategy(
    "ConnectionError", network_recovery_strategy
)
enhanced_error_handler.register_recovery_strategy(
    "TimeoutError", network_recovery_strategy
)
