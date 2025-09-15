"""
Advanced Security System
Comprehensive security features including threat detection, rate limiting, and audit logging
"""

import hashlib
import hmac
import secrets
import time
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict, deque
import asyncio
import re

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Security event types"""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    ACCOUNT_LOCKED = "account_locked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INVALID_TOKEN = "invalid_token"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"


@dataclass
class SecurityAlert:
    """Security alert information"""
    alert_id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    description: str
    timestamp: datetime
    additional_data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class UserSecurityProfile:
    """User security profile for risk assessment"""
    user_id: str
    risk_score: float = 0.0
    failed_login_attempts: int = 0
    last_failed_login: Optional[datetime] = None
    last_successful_login: Optional[datetime] = None
    suspicious_activities: List[str] = field(default_factory=list)
    ip_addresses: Set[str] = field(default_factory=set)
    user_agents: Set[str] = field(default_factory=set)
    account_locked_until: Optional[datetime] = None
    two_factor_enabled: bool = False
    security_questions_answered: bool = False


class AdvancedSecurityManager:
    """Advanced security management system with threat detection"""

    def __init__(self):
        self.user_profiles: Dict[str, UserSecurityProfile] = {}
        self.security_alerts: List[SecurityAlert] = []
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        self.security_rules: List[Dict[str, Any]] = []
        
        # Security thresholds
        self.max_failed_logins = 5
        self.account_lockout_duration = 300  # 5 minutes
        self.rate_limit_window = 60  # 1 minute
        self.rate_limit_max_requests = 100
        self.suspicious_activity_threshold = 3
        
        # Initialize security rules
        self._initialize_security_rules()

    def _initialize_security_rules(self):
        """Initialize security rules and patterns"""
        self.security_rules = [
            {
                "name": "sql_injection_detection",
                "pattern": r"(union|select|insert|update|delete|drop|create|alter|exec|execute)",
                "case_sensitive": False,
                "threat_level": ThreatLevel.HIGH
            },
            {
                "name": "xss_detection",
                "pattern": r"<script|javascript:|onload=|onerror=|onclick=",
                "case_sensitive": False,
                "threat_level": ThreatLevel.MEDIUM
            },
            {
                "name": "path_traversal_detection",
                "pattern": r"\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c",
                "case_sensitive": False,
                "threat_level": ThreatLevel.HIGH
            },
            {
                "name": "command_injection_detection",
                "pattern": r"[;&|`$(){}[\]\\]",
                "case_sensitive": False,
                "threat_level": ThreatLevel.HIGH
            }
        ]

    def get_user_profile(self, user_id: str) -> UserSecurityProfile:
        """Get or create user security profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserSecurityProfile(user_id=user_id)
        return self.user_profiles[user_id]

    def analyze_input_security(self, input_data: str, user_id: Optional[str] = None) -> List[SecurityAlert]:
        """Analyze input data for security threats"""
        alerts = []
        
        for rule in self.security_rules:
            pattern = re.compile(rule["pattern"], re.IGNORECASE if not rule["case_sensitive"] else 0)
            if pattern.search(input_data):
                alert = SecurityAlert(
                    alert_id=f"SEC_{int(time.time() * 1000)}",
                    event_type=self._get_event_type_from_rule(rule["name"]),
                    threat_level=rule["threat_level"],
                    user_id=user_id,
                    ip_address=None,
                    user_agent=None,
                    description=f"Potential {rule['name']} detected in input",
                    timestamp=datetime.utcnow(),
                    additional_data={
                        "rule_name": rule["name"],
                        "pattern": rule["pattern"],
                        "input_sample": input_data[:100]  # First 100 chars
                    }
                )
                alerts.append(alert)
                self.security_alerts.append(alert)
                
                # Update suspicious patterns
                self.suspicious_patterns[rule["name"]] += 1
                
                logger.warning(f"Security threat detected: {rule['name']} by user {user_id}")
        
        return alerts

    def _get_event_type_from_rule(self, rule_name: str) -> SecurityEvent:
        """Map rule name to security event type"""
        mapping = {
            "sql_injection_detection": SecurityEvent.SQL_INJECTION_ATTEMPT,
            "xss_detection": SecurityEvent.XSS_ATTEMPT,
            "path_traversal_detection": SecurityEvent.UNAUTHORIZED_ACCESS,
            "command_injection_detection": SecurityEvent.UNAUTHORIZED_ACCESS
        }
        return mapping.get(rule_name, SecurityEvent.SUSPICIOUS_ACTIVITY)

    def check_rate_limit(self, identifier: str, limit: int = None) -> Tuple[bool, int]:
        """Check if rate limit is exceeded"""
        if limit is None:
            limit = self.rate_limit_max_requests
        
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.rate_limit_window)
        
        # Clean old entries
        while self.rate_limits[identifier] and self.rate_limits[identifier][0] < window_start:
            self.rate_limits[identifier].popleft()
        
        # Check if limit exceeded
        current_count = len(self.rate_limits[identifier])
        if current_count >= limit:
            # Log rate limit exceeded
            alert = SecurityAlert(
                alert_id=f"RATE_{int(time.time() * 1000)}",
                event_type=SecurityEvent.RATE_LIMIT_EXCEEDED,
                threat_level=ThreatLevel.MEDIUM,
                user_id=None,
                ip_address=identifier if self._is_ip_address(identifier) else None,
                user_agent=None,
                description=f"Rate limit exceeded for {identifier}",
                timestamp=now,
                additional_data={
                    "current_count": current_count,
                    "limit": limit,
                    "window_seconds": self.rate_limit_window
                }
            )
            self.security_alerts.append(alert)
            logger.warning(f"Rate limit exceeded for {identifier}: {current_count}/{limit}")
            return False, current_count
        
        # Add current request
        self.rate_limits[identifier].append(now)
        return True, current_count + 1

    def _is_ip_address(self, identifier: str) -> bool:
        """Check if identifier is an IP address"""
        try:
            ipaddress.ip_address(identifier)
            return True
        except ValueError:
            return False

    def handle_login_attempt(self, user_id: str, ip_address: str, user_agent: str, success: bool) -> List[SecurityAlert]:
        """Handle login attempt and assess security risk"""
        alerts = []
        profile = self.get_user_profile(user_id)
        
        # Update profile
        profile.ip_addresses.add(ip_address)
        profile.user_agents.add(user_agent)
        
        if success:
            profile.failed_login_attempts = 0
            profile.last_successful_login = datetime.utcnow()
            profile.account_locked_until = None
            
            # Log successful login
            alert = SecurityAlert(
                alert_id=f"LOGIN_{int(time.time() * 1000)}",
                event_type=SecurityEvent.LOGIN_SUCCESS,
                threat_level=ThreatLevel.LOW,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                description=f"Successful login for user {user_id}",
                timestamp=datetime.utcnow()
            )
            alerts.append(alert)
            
        else:
            profile.failed_login_attempts += 1
            profile.last_failed_login = datetime.utcnow()
            
            # Check for brute force
            if profile.failed_login_attempts >= self.max_failed_logins:
                profile.account_locked_until = datetime.utcnow() + timedelta(seconds=self.account_lockout_duration)
                
                alert = SecurityAlert(
                    alert_id=f"LOCK_{int(time.time() * 1000)}",
                    event_type=SecurityEvent.ACCOUNT_LOCKED,
                    threat_level=ThreatLevel.HIGH,
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    description=f"Account locked due to {profile.failed_login_attempts} failed attempts",
                    timestamp=datetime.utcnow(),
                    additional_data={
                        "failed_attempts": profile.failed_login_attempts,
                        "locked_until": profile.account_locked_until.isoformat()
                    }
                )
                alerts.append(alert)
                logger.critical(f"Account locked for user {user_id} due to brute force")
            
            else:
                alert = SecurityAlert(
                    alert_id=f"FAIL_{int(time.time() * 1000)}",
                    event_type=SecurityEvent.LOGIN_FAILURE,
                    threat_level=ThreatLevel.MEDIUM,
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    description=f"Failed login attempt {profile.failed_login_attempts}/{self.max_failed_logins}",
                    timestamp=datetime.utcnow(),
                    additional_data={
                        "failed_attempts": profile.failed_login_attempts,
                        "max_attempts": self.max_failed_logins
                    }
                )
                alerts.append(alert)
        
        # Update risk score
        self._update_risk_score(profile)
        
        # Store alerts
        self.security_alerts.extend(alerts)
        return alerts

    def _update_risk_score(self, profile: UserSecurityProfile):
        """Update user risk score based on various factors"""
        risk_score = 0.0
        
        # Failed login attempts
        risk_score += profile.failed_login_attempts * 10
        
        # Account lockout
        if profile.account_locked_until and profile.account_locked_until > datetime.utcnow():
            risk_score += 50
        
        # Multiple IP addresses (potential account sharing/compromise)
        if len(profile.ip_addresses) > 3:
            risk_score += 20
        
        # Multiple user agents (potential account sharing)
        if len(profile.user_agents) > 2:
            risk_score += 15
        
        # No 2FA
        if not profile.two_factor_enabled:
            risk_score += 10
        
        # No security questions
        if not profile.security_questions_answered:
            risk_score += 5
        
        # Recent suspicious activities
        risk_score += len(profile.suspicious_activities) * 5
        
        profile.risk_score = min(risk_score, 100.0)  # Cap at 100

    def is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked"""
        profile = self.get_user_profile(user_id)
        if profile.account_locked_until:
            if profile.account_locked_until > datetime.utcnow():
                return True
            else:
                # Unlock expired account
                profile.account_locked_until = None
                profile.failed_login_attempts = 0
        return False

    def block_ip_address(self, ip_address: str, reason: str, duration_hours: int = 24):
        """Block an IP address"""
        self.blocked_ips.add(ip_address)
        
        alert = SecurityAlert(
            alert_id=f"BLOCK_{int(time.time() * 1000)}",
            event_type=SecurityEvent.UNAUTHORIZED_ACCESS,
            threat_level=ThreatLevel.HIGH,
            user_id=None,
            ip_address=ip_address,
            user_agent=None,
            description=f"IP address blocked: {reason}",
            timestamp=datetime.utcnow(),
            additional_data={
                "reason": reason,
                "duration_hours": duration_hours,
                "blocked_until": (datetime.utcnow() + timedelta(hours=duration_hours)).isoformat()
            }
        )
        self.security_alerts.append(alert)
        logger.critical(f"IP address {ip_address} blocked: {reason}")

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        return ip_address in self.blocked_ips

    def generate_security_token(self, user_id: str, additional_data: str = "") -> str:
        """Generate secure token with HMAC"""
        timestamp = str(int(time.time()))
        data = f"{user_id}:{timestamp}:{additional_data}"
        secret_key = "your-secret-key"  # In production, use environment variable
        token = hmac.new(
            secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{timestamp}:{token}"

    def verify_security_token(self, token: str, user_id: str, additional_data: str = "", max_age_seconds: int = 3600) -> bool:
        """Verify security token"""
        try:
            parts = token.split(":")
            if len(parts) != 2:
                return False
            
            timestamp_str, token_hash = parts
            timestamp = int(timestamp_str)
            
            # Check token age
            if time.time() - timestamp > max_age_seconds:
                return False
            
            # Regenerate token and compare
            data = f"{user_id}:{timestamp_str}:{additional_data}"
            secret_key = "your-secret-key"  # In production, use environment variable
            expected_token = hmac.new(
                secret_key.encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(token_hash, expected_token)
            
        except (ValueError, IndexError):
            return False

    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.security_alerts if alert.timestamp >= cutoff_time]
        
        summary = {
            "time_period_hours": hours,
            "total_alerts": len(recent_alerts),
            "alerts_by_type": {},
            "alerts_by_threat_level": {},
            "top_threats": [],
            "blocked_ips_count": len(self.blocked_ips),
            "locked_accounts_count": 0,
            "high_risk_users": []
        }
        
        # Count alerts by type
        for alert in recent_alerts:
            event_type = alert.event_type.value
            summary["alerts_by_type"][event_type] = summary["alerts_by_type"].get(event_type, 0) + 1
            
            threat_level = alert.threat_level.value
            summary["alerts_by_threat_level"][threat_level] = summary["alerts_by_threat_level"].get(threat_level, 0) + 1
        
        # Top threats
        threat_counts = defaultdict(int)
        for alert in recent_alerts:
            if alert.additional_data.get("rule_name"):
                threat_counts[alert.additional_data["rule_name"]] += 1
        
        summary["top_threats"] = sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Count locked accounts
        for profile in self.user_profiles.values():
            if profile.account_locked_until and profile.account_locked_until > datetime.utcnow():
                summary["locked_accounts_count"] += 1
            
            # High risk users
            if profile.risk_score > 70:
                summary["high_risk_users"].append({
                    "user_id": profile.user_id,
                    "risk_score": profile.risk_score,
                    "failed_attempts": profile.failed_login_attempts,
                    "locked": profile.account_locked_until is not None
                })
        
        return summary

    def cleanup_old_data(self, days: int = 30):
        """Clean up old security data"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Clean old alerts
        self.security_alerts = [alert for alert in self.security_alerts if alert.timestamp >= cutoff_time]
        
        # Clean old rate limit data
        for identifier in list(self.rate_limits.keys()):
            while self.rate_limits[identifier] and self.rate_limits[identifier][0] < cutoff_time:
                self.rate_limits[identifier].popleft()
            
            # Remove empty rate limit entries
            if not self.rate_limits[identifier]:
                del self.rate_limits[identifier]


# Global security manager instance
advanced_security_manager = AdvancedSecurityManager()


# Security decorators and utilities
def require_security_check(func):
    """Decorator to add security checks to functions"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Extract user_id and input data from arguments
        user_id = kwargs.get('user_id') or (args[0] if args else None)
        input_data = str(kwargs.get('data', '')) + str(kwargs.get('input', ''))
        
        # Check for security threats
        if input_data:
            alerts = advanced_security_manager.analyze_input_security(input_data, user_id)
            if alerts:
                # Log security alerts
                for alert in alerts:
                    logger.warning(f"Security alert: {alert.description}")
        
        return await func(*args, **kwargs)
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Extract user_id and input data from arguments
        user_id = kwargs.get('user_id') or (args[0] if args else None)
        input_data = str(kwargs.get('data', '')) + str(kwargs.get('input', ''))
        
        # Check for security threats
        if input_data:
            alerts = advanced_security_manager.analyze_input_security(input_data, user_id)
            if alerts:
                # Log security alerts
                for alert in alerts:
                    logger.warning(f"Security alert: {alert.description}")
        
        return func(*args, **kwargs)
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def rate_limit_check(identifier_func: Callable = None, limit: int = None):
    """Decorator for rate limiting"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get identifier (IP, user_id, etc.)
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = kwargs.get('ip_address') or kwargs.get('user_id') or 'default'
            
            # Check rate limit
            allowed, count = advanced_security_manager.check_rate_limit(identifier, limit)
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again later."
                )
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get identifier (IP, user_id, etc.)
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = kwargs.get('ip_address') or kwargs.get('user_id') or 'default'
            
            # Check rate limit
            allowed, count = advanced_security_manager.check_rate_limit(identifier, limit)
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again later."
                )
            
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
