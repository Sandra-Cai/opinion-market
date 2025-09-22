"""
Advanced Request/Response Logging System
Provides comprehensive logging with performance metrics and analytics
"""

import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
from collections import defaultdict, deque
import threading

from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class LogCategory(Enum):
    """Log categories"""
    REQUEST = "request"
    RESPONSE = "response"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"
    SYSTEM = "system"

@dataclass
class LogEntry:
    """Represents a log entry"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    request_id: str
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    user_agent: Optional[str] = None
    referer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics for logging"""
    response_time: float
    cpu_usage: float
    memory_usage: float
    database_queries: int
    cache_hits: int
    cache_misses: int
    external_api_calls: int
    errors: int
    warnings: int

class AdvancedLogger:
    """Advanced logging system with analytics and performance tracking"""
    
    def __init__(self, max_entries: int = 10000, enable_analytics: bool = True):
        self.max_entries = max_entries
        self.enable_analytics = enable_analytics
        self.log_entries: deque = deque(maxlen=max_entries)
        self.performance_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.user_analytics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.endpoint_analytics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        
        # Setup loggers
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup loggers for different categories"""
        self.loggers = {
            LogCategory.REQUEST: logging.getLogger("request"),
            LogCategory.RESPONSE: logging.getLogger("response"),
            LogCategory.PERFORMANCE: logging.getLogger("performance"),
            LogCategory.SECURITY: logging.getLogger("security"),
            LogCategory.BUSINESS: logging.getLogger("business"),
            LogCategory.SYSTEM: logging.getLogger("system")
        }
        
        # Configure loggers
        for category, logger in self.loggers.items():
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {category.value.upper()} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    
    def log_request(self, request: Request, request_id: str, user_id: Optional[str] = None) -> LogEntry:
        """Log incoming request"""
        try:
            # Extract request information
            ip_address = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")
            referer = request.headers.get("Referer", "")
            request_size = self._get_request_size(request)
            
            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level=LogLevel.INFO,
                category=LogCategory.REQUEST,
                message=f"Request received: {request.method} {request.url.path}",
                request_id=request_id,
                user_id=user_id,
                ip_address=ip_address,
                endpoint=request.url.path,
                method=request.method,
                request_size=request_size,
                user_agent=user_agent,
                referer=referer,
                metadata={
                    "query_params": dict(request.query_params),
                    "headers": dict(request.headers),
                    "url": str(request.url)
                }
            )
            
            # Store log entry
            self._store_log_entry(log_entry)
            
            # Log to category logger
            self.loggers[LogCategory.REQUEST].info(
                f"Request {request_id}: {request.method} {request.url.path} from {ip_address}"
            )
            
            return log_entry
            
        except Exception as e:
            logger.error(f"Error logging request: {e}")
            return None
    
    def log_response(self, request: Request, response: Response, request_id: str, 
                    response_time: float, user_id: Optional[str] = None) -> LogEntry:
        """Log outgoing response"""
        try:
            # Extract response information
            ip_address = self._get_client_ip(request)
            response_size = self._get_response_size(response)
            
            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level=LogLevel.INFO,
                category=LogCategory.RESPONSE,
                message=f"Response sent: {response.status_code} for {request.method} {request.url.path}",
                request_id=request_id,
                user_id=user_id,
                ip_address=ip_address,
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time=response_time,
                response_size=response_size,
                metadata={
                    "response_headers": dict(response.headers),
                    "status_code": response.status_code
                }
            )
            
            # Store log entry
            self._store_log_entry(log_entry)
            
            # Log to category logger
            self.loggers[LogCategory.RESPONSE].info(
                f"Response {request_id}: {response.status_code} in {response_time:.3f}s"
            )
            
            return log_entry
            
        except Exception as e:
            logger.error(f"Error logging response: {e}")
            return None
    
    def log_performance(self, request_id: str, metrics: PerformanceMetrics, 
                       endpoint: str, user_id: Optional[str] = None):
        """Log performance metrics"""
        try:
            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level=LogLevel.INFO,
                category=LogCategory.PERFORMANCE,
                message=f"Performance metrics for {endpoint}",
                request_id=request_id,
                user_id=user_id,
                endpoint=endpoint,
                response_time=metrics.response_time,
                metadata={
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "database_queries": metrics.database_queries,
                    "cache_hits": metrics.cache_hits,
                    "cache_misses": metrics.cache_misses,
                    "external_api_calls": metrics.external_api_calls,
                    "errors": metrics.errors,
                    "warnings": metrics.warnings
                }
            )
            
            # Store log entry
            self._store_log_entry(log_entry)
            
            # Store performance metrics
            with self._lock:
                self.performance_metrics[endpoint].append(metrics)
                # Keep only last 100 metrics per endpoint
                if len(self.performance_metrics[endpoint]) > 100:
                    self.performance_metrics[endpoint] = self.performance_metrics[endpoint][-100:]
            
            # Log to category logger
            self.loggers[LogCategory.PERFORMANCE].info(
                f"Performance {request_id}: {metrics.response_time:.3f}s, "
                f"{metrics.database_queries} queries, {metrics.cache_hits} cache hits"
            )
            
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    def log_security_event(self, event_type: str, request: Request, request_id: str, 
                          details: Dict[str, Any], user_id: Optional[str] = None):
        """Log security events"""
        try:
            ip_address = self._get_client_ip(request)
            
            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level=LogLevel.WARNING,
                category=LogCategory.SECURITY,
                message=f"Security event: {event_type}",
                request_id=request_id,
                user_id=user_id,
                ip_address=ip_address,
                endpoint=request.url.path,
                method=request.method,
                metadata={
                    "event_type": event_type,
                    "details": details,
                    "user_agent": request.headers.get("User-Agent", ""),
                    "referer": request.headers.get("Referer", "")
                }
            )
            
            # Store log entry
            self._store_log_entry(log_entry)
            
            # Track error patterns
            with self._lock:
                self.error_patterns[event_type] += 1
            
            # Log to category logger
            self.loggers[LogCategory.SECURITY].warning(
                f"Security event {request_id}: {event_type} from {ip_address}"
            )
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    def log_business_event(self, event_type: str, request_id: str, user_id: str, 
                          details: Dict[str, Any], endpoint: Optional[str] = None):
        """Log business events"""
        try:
            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level=LogLevel.INFO,
                category=LogCategory.BUSINESS,
                message=f"Business event: {event_type}",
                request_id=request_id,
                user_id=user_id,
                endpoint=endpoint,
                metadata={
                    "event_type": event_type,
                    "details": details
                }
            )
            
            # Store log entry
            self._store_log_entry(log_entry)
            
            # Log to category logger
            self.loggers[LogCategory.BUSINESS].info(
                f"Business event {request_id}: {event_type} for user {user_id}"
            )
            
        except Exception as e:
            logger.error(f"Error logging business event: {e}")
    
    def log_system_event(self, event_type: str, message: str, level: LogLevel = LogLevel.INFO, 
                        details: Dict[str, Any] = None):
        """Log system events"""
        try:
            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.utcnow(),
                level=level,
                category=LogCategory.SYSTEM,
                message=f"System event: {message}",
                request_id=f"system_{int(time.time())}",
                metadata={
                    "event_type": event_type,
                    "details": details or {}
                }
            )
            
            # Store log entry
            self._store_log_entry(log_entry)
            
            # Log to category logger
            self.loggers[LogCategory.SYSTEM].log(
                getattr(logging, level.value.upper()),
                f"System event: {message}"
            )
            
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
    
    def _store_log_entry(self, log_entry: LogEntry):
        """Store log entry with analytics"""
        with self._lock:
            self.log_entries.append(log_entry)
            
            if self.enable_analytics:
                self._update_analytics(log_entry)
    
    def _update_analytics(self, log_entry: LogEntry):
        """Update analytics based on log entry"""
        # Update user analytics
        if log_entry.user_id:
            if log_entry.user_id not in self.user_analytics:
                self.user_analytics[log_entry.user_id] = {
                    "total_requests": 0,
                    "total_errors": 0,
                    "total_response_time": 0.0,
                    "endpoints_used": set(),
                    "last_activity": None
                }
            
            analytics = self.user_analytics[log_entry.user_id]
            analytics["total_requests"] += 1
            analytics["last_activity"] = log_entry.timestamp
            
            if log_entry.endpoint:
                analytics["endpoints_used"].add(log_entry.endpoint)
            
            if log_entry.response_time:
                analytics["total_response_time"] += log_entry.response_time
            
            if log_entry.status_code and log_entry.status_code >= 400:
                analytics["total_errors"] += 1
        
        # Update endpoint analytics
        if log_entry.endpoint:
            if log_entry.endpoint not in self.endpoint_analytics:
                self.endpoint_analytics[log_entry.endpoint] = {
                    "total_requests": 0,
                    "total_errors": 0,
                    "total_response_time": 0.0,
                    "unique_users": set(),
                    "last_accessed": None
                }
            
            analytics = self.endpoint_analytics[log_entry.endpoint]
            analytics["total_requests"] += 1
            analytics["last_accessed"] = log_entry.timestamp
            
            if log_entry.user_id:
                analytics["unique_users"].add(log_entry.user_id)
            
            if log_entry.response_time:
                analytics["total_response_time"] += log_entry.response_time
            
            if log_entry.status_code and log_entry.status_code >= 400:
                analytics["total_errors"] += 1
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _get_request_size(self, request: Request) -> int:
        """Get request size in bytes"""
        try:
            content_length = request.headers.get("Content-Length")
            if content_length:
                return int(content_length)
            
            # Estimate from headers
            headers_size = sum(len(k) + len(v) + 4 for k, v in request.headers.items())
            return headers_size
        except:
            return 0
    
    def _get_response_size(self, response: Response) -> int:
        """Get response size in bytes"""
        try:
            content_length = response.headers.get("Content-Length")
            if content_length:
                return int(content_length)
            
            # Estimate from headers
            headers_size = sum(len(k) + len(v) + 4 for k, v in response.headers.items())
            return headers_size
        except:
            return 0
    
    def get_analytics(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics data"""
        with self._lock:
            if category == "users":
                return {
                    user_id: {
                        "total_requests": data["total_requests"],
                        "total_errors": data["total_errors"],
                        "avg_response_time": data["total_response_time"] / data["total_requests"] if data["total_requests"] > 0 else 0,
                        "endpoints_used": list(data["endpoints_used"]),
                        "last_activity": data["last_activity"].isoformat() if data["last_activity"] else None
                    }
                    for user_id, data in self.user_analytics.items()
                }
            elif category == "endpoints":
                return {
                    endpoint: {
                        "total_requests": data["total_requests"],
                        "total_errors": data["total_errors"],
                        "avg_response_time": data["total_response_time"] / data["total_requests"] if data["total_requests"] > 0 else 0,
                        "unique_users": len(data["unique_users"]),
                        "last_accessed": data["last_accessed"].isoformat() if data["last_accessed"] else None
                    }
                    for endpoint, data in self.endpoint_analytics.items()
                }
            elif category == "performance":
                return {
                    endpoint: {
                        "avg_response_time": sum(m.response_time for m in metrics) / len(metrics) if metrics else 0,
                        "avg_cpu_usage": sum(m.cpu_usage for m in metrics) / len(metrics) if metrics else 0,
                        "avg_memory_usage": sum(m.memory_usage for m in metrics) / len(metrics) if metrics else 0,
                        "avg_database_queries": sum(m.database_queries for m in metrics) / len(metrics) if metrics else 0,
                        "avg_cache_hits": sum(m.cache_hits for m in metrics) / len(metrics) if metrics else 0,
                        "total_requests": len(metrics)
                    }
                    for endpoint, metrics in self.performance_metrics.items()
                }
            elif category == "errors":
                return dict(self.error_patterns)
            else:
                return {
                    "users": self.get_analytics("users"),
                    "endpoints": self.get_analytics("endpoints"),
                    "performance": self.get_analytics("performance"),
                    "errors": self.get_analytics("errors"),
                    "total_log_entries": len(self.log_entries)
                }
    
    def get_recent_logs(self, limit: int = 100, category: Optional[LogCategory] = None) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        with self._lock:
            logs = list(self.log_entries)
            
            if category:
                logs = [log for log in logs if log.category == category]
            
            # Return most recent logs
            recent_logs = logs[-limit:] if len(logs) > limit else logs
            
            return [
                {
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level.value,
                    "category": log.category.value,
                    "message": log.message,
                    "request_id": log.request_id,
                    "user_id": log.user_id,
                    "ip_address": log.ip_address,
                    "endpoint": log.endpoint,
                    "method": log.method,
                    "status_code": log.status_code,
                    "response_time": log.response_time,
                    "metadata": log.metadata
                }
                for log in recent_logs
            ]

# Global advanced logger instance
advanced_logger = AdvancedLogger()

# Convenience functions
def log_request(request: Request, request_id: str, user_id: Optional[str] = None) -> LogEntry:
    """Log incoming request"""
    return advanced_logger.log_request(request, request_id, user_id)

def log_response(request: Request, response: Response, request_id: str, 
                response_time: float, user_id: Optional[str] = None) -> LogEntry:
    """Log outgoing response"""
    return advanced_logger.log_response(request, response, request_id, response_time, user_id)

def log_performance(request_id: str, metrics: PerformanceMetrics, 
                   endpoint: str, user_id: Optional[str] = None):
    """Log performance metrics"""
    advanced_logger.log_performance(request_id, metrics, endpoint, user_id)

def log_security_event(event_type: str, request: Request, request_id: str, 
                      details: Dict[str, Any], user_id: Optional[str] = None):
    """Log security events"""
    advanced_logger.log_security_event(event_type, request, request_id, details, user_id)

def log_business_event(event_type: str, request_id: str, user_id: str, 
                      details: Dict[str, Any], endpoint: Optional[str] = None):
    """Log business events"""
    advanced_logger.log_business_event(event_type, request_id, user_id, details, endpoint)

def get_analytics(category: Optional[str] = None) -> Dict[str, Any]:
    """Get analytics data"""
    return advanced_logger.get_analytics(category)

def get_recent_logs(limit: int = 100, category: Optional[LogCategory] = None) -> List[Dict[str, Any]]:
    """Get recent log entries"""
    return advanced_logger.get_recent_logs(limit, category)
