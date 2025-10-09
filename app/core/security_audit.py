"""
Security audit and compliance system for Opinion Market
Provides comprehensive security monitoring, audit logging, and compliance reporting
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import ipaddress
import re

from app.core.logging import log_security_event
from app.core.config import settings
from app.core.database import get_db_session
from app.core.cache import cache


class SecurityEventType(str, Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    ACCOUNT_LOCKED = "account_locked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    ADMIN_ACTION = "admin_action"
    SECURITY_VIOLATION = "security_violation"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    event_type: SecurityEventType
    user_id: Optional[int]
    ip_address: str
    user_agent: Optional[str]
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    name: str
    description: str
    enabled: bool = True
    severity: RiskLevel = RiskLevel.MEDIUM
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)


class SecurityAuditor:
    """Comprehensive security audit system"""
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Set[str] = set()
        self.user_failed_attempts: Dict[int, List[datetime]] = {}
        self.ip_failed_attempts: Dict[str, List[datetime]] = {}
        self.running = False
        
        # Initialize security policies
        self._setup_default_policies()
    
    def _setup_default_policies(self):
        """Setup default security policies"""
        self.security_policies = {
            "multiple_failed_logins": SecurityPolicy(
                name="Multiple Failed Logins",
                description="Detect multiple failed login attempts",
                conditions={
                    "max_attempts": 5,
                    "time_window": 300,  # 5 minutes
                    "action": "temporary_lock"
                },
                actions=["log_event", "temporary_lock", "notify_admin"],
                severity=RiskLevel.HIGH
            ),
            "suspicious_ip_activity": SecurityPolicy(
                name="Suspicious IP Activity",
                description="Detect suspicious activity from IP addresses",
                conditions={
                    "max_requests": 100,
                    "time_window": 60,  # 1 minute
                    "action": "rate_limit"
                },
                actions=["log_event", "rate_limit", "monitor"],
                severity=RiskLevel.MEDIUM
            ),
            "unusual_access_pattern": SecurityPolicy(
                name="Unusual Access Pattern",
                description="Detect unusual access patterns",
                conditions={
                    "max_different_ips": 5,
                    "time_window": 3600,  # 1 hour
                    "action": "investigate"
                },
                actions=["log_event", "investigate", "notify_user"],
                severity=RiskLevel.MEDIUM
            ),
            "admin_action_monitoring": SecurityPolicy(
                name="Admin Action Monitoring",
                description="Monitor all admin actions",
                conditions={
                    "require_approval": True,
                    "log_all": True
                },
                actions=["log_event", "require_approval", "audit_trail"],
                severity=RiskLevel.HIGH
            ),
            "data_breach_detection": SecurityPolicy(
                name="Data Breach Detection",
                description="Detect potential data breaches",
                conditions={
                    "max_data_access": 1000,
                    "time_window": 300,  # 5 minutes
                    "action": "immediate_alert"
                },
                actions=["log_event", "immediate_alert", "block_access"],
                severity=RiskLevel.CRITICAL
            )
        }
    
    async def start(self):
        """Start security audit system"""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting security audit system")
        
        # Start monitoring tasks
        asyncio.create_task(self._cleanup_old_events())
        asyncio.create_task(self._analyze_security_events())
    
    async def stop(self):
        """Stop security audit system"""
        self.running = False
        self.logger.info("Stopped security audit system")
    
    async def _cleanup_old_events(self):
        """Cleanup old security events"""
        while self.running:
            try:
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                self.security_events = [
                    event for event in self.security_events
                    if event.timestamp > cutoff_time
                ]
                
                # Cleanup old failed attempts
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                for user_id in list(self.user_failed_attempts.keys()):
                    self.user_failed_attempts[user_id] = [
                        attempt for attempt in self.user_failed_attempts[user_id]
                        if attempt > cutoff_time
                    ]
                    if not self.user_failed_attempts[user_id]:
                        del self.user_failed_attempts[user_id]
                
                for ip in list(self.ip_failed_attempts.keys()):
                    self.ip_failed_attempts[ip] = [
                        attempt for attempt in self.ip_failed_attempts[ip]
                        if attempt > cutoff_time
                    ]
                    if not self.ip_failed_attempts[ip]:
                        del self.ip_failed_attempts[ip]
                
                await asyncio.sleep(3600)  # Run every hour
            
        except Exception as e:
                self.logger.error(f"Error in security event cleanup: {e}")
                await asyncio.sleep(3600)
    
    async def _analyze_security_events(self):
        """Analyze security events for patterns and threats"""
        while self.running:
            try:
                # Analyze recent events for security patterns
                recent_events = [
                    event for event in self.security_events
                    if event.timestamp > datetime.utcnow() - timedelta(hours=1)
                ]
                
                # Check for policy violations
                await self._check_policy_violations(recent_events)
                
                # Check for suspicious patterns
                await self._detect_suspicious_patterns(recent_events)
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in security event analysis: {e}")
                await asyncio.sleep(300)
    
    async def _check_policy_violations(self, events: List[SecurityEvent]):
        """Check for security policy violations"""
        for policy_name, policy in self.security_policies.items():
            if not policy.enabled:
                continue
            
            try:
                if policy_name == "multiple_failed_logins":
                    await self._check_failed_login_policy(events, policy)
                elif policy_name == "suspicious_ip_activity":
                    await self._check_suspicious_ip_policy(events, policy)
                elif policy_name == "unusual_access_pattern":
                    await self._check_unusual_access_policy(events, policy)
                elif policy_name == "data_breach_detection":
                    await self._check_data_breach_policy(events, policy)
                        
        except Exception as e:
                self.logger.error(f"Error checking policy {policy_name}: {e}")
    
    async def _check_failed_login_policy(self, events: List[SecurityEvent], policy: SecurityPolicy):
        """Check for multiple failed login attempts"""
        max_attempts = policy.conditions["max_attempts"]
        time_window = policy.conditions["time_window"]
        
        # Group by user and IP
        user_attempts = {}
        ip_attempts = {}
        
        for event in events:
            if event.event_type == SecurityEventType.LOGIN_FAILURE:
                if event.user_id:
                    if event.user_id not in user_attempts:
                        user_attempts[event.user_id] = []
                    user_attempts[event.user_id].append(event.timestamp)
                
                if event.ip_address not in ip_attempts:
                    ip_attempts[event.ip_address] = []
                ip_attempts[event.ip_address].append(event.timestamp)
        
        # Check user attempts
        for user_id, attempts in user_attempts.items():
            recent_attempts = [
                attempt for attempt in attempts
                if attempt > datetime.utcnow() - timedelta(seconds=time_window)
            ]
            
            if len(recent_attempts) >= max_attempts:
                await self._trigger_policy_violation(
                    policy, f"User {user_id} has {len(recent_attempts)} failed login attempts",
                    {"user_id": user_id, "attempts": len(recent_attempts)}
                )
        
        # Check IP attempts
        for ip, attempts in ip_attempts.items():
            recent_attempts = [
                attempt for attempt in attempts
                if attempt > datetime.utcnow() - timedelta(seconds=time_window)
            ]
            
            if len(recent_attempts) >= max_attempts:
                await self._trigger_policy_violation(
                    policy, f"IP {ip} has {len(recent_attempts)} failed login attempts",
                    {"ip_address": ip, "attempts": len(recent_attempts)}
                )
    
    async def _check_suspicious_ip_policy(self, events: List[SecurityEvent], policy: SecurityPolicy):
        """Check for suspicious IP activity"""
        max_requests = policy.conditions["max_requests"]
        time_window = policy.conditions["time_window"]
        
        # Count requests by IP
        ip_requests = {}
        for event in events:
            if event.ip_address not in ip_requests:
                ip_requests[event.ip_address] = []
            ip_requests[event.ip_address].append(event.timestamp)
        
        # Check for high request rates
        for ip, requests in ip_requests.items():
            recent_requests = [
                req for req in requests
                if req > datetime.utcnow() - timedelta(seconds=time_window)
            ]
            
            if len(recent_requests) >= max_requests:
                await self._trigger_policy_violation(
                    policy, f"IP {ip} has high request rate: {len(recent_requests)} requests",
                    {"ip_address": ip, "requests": len(recent_requests)}
                )
    
    async def _check_unusual_access_policy(self, events: List[SecurityEvent], policy: SecurityPolicy):
        """Check for unusual access patterns"""
        max_different_ips = policy.conditions["max_different_ips"]
        time_window = policy.conditions["time_window"]
        
        # Group by user
        user_ips = {}
        for event in events:
            if event.user_id and event.user_id not in user_ips:
                user_ips[event.user_id] = set()
            if event.user_id:
                user_ips[event.user_id].add(event.ip_address)
        
        # Check for users accessing from many different IPs
        for user_id, ips in user_ips.items():
            if len(ips) >= max_different_ips:
                await self._trigger_policy_violation(
                    policy, f"User {user_id} accessed from {len(ips)} different IPs",
                    {"user_id": user_id, "ip_count": len(ips), "ips": list(ips)}
                )
    
    async def _check_data_breach_policy(self, events: List[SecurityEvent], policy: SecurityPolicy):
        """Check for potential data breaches"""
        max_data_access = policy.conditions["max_data_access"]
        time_window = policy.conditions["time_window"]
        
        # Count data access events
        data_access_events = [
            event for event in events
            if event.event_type == SecurityEventType.DATA_ACCESS
            and event.timestamp > datetime.utcnow() - timedelta(seconds=time_window)
        ]
        
        if len(data_access_events) >= max_data_access:
            await self._trigger_policy_violation(
                policy, f"High volume data access detected: {len(data_access_events)} events",
                {"event_count": len(data_access_events), "events": data_access_events}
            )
    
    async def _detect_suspicious_patterns(self, events: List[SecurityEvent]):
        """Detect suspicious patterns in security events"""
        # Detect brute force attacks
        await self._detect_brute_force_attacks(events)
        
        # Detect credential stuffing
        await self._detect_credential_stuffing(events)
        
        # Detect account takeover attempts
        await self._detect_account_takeover_attempts(events)
    
    async def _detect_brute_force_attacks(self, events: List[SecurityEvent]):
        """Detect brute force attacks"""
        # Group failed logins by IP
        ip_failures = {}
        for event in events:
            if event.event_type == SecurityEventType.LOGIN_FAILURE:
                if event.ip_address not in ip_failures:
                    ip_failures[event.ip_address] = []
                ip_failures[event.ip_address].append(event)
        
        # Check for brute force patterns
        for ip, failures in ip_failures.items():
            if len(failures) >= 10:  # 10 failed attempts
                # Check if attempts are against different users
                unique_users = set(event.user_id for event in failures if event.user_id)
                
                if len(unique_users) >= 3:  # 3 or more different users
                    await self._create_security_event(
                        SecurityEventType.SUSPICIOUS_ACTIVITY,
                        None, ip, None,
                        {
                            "attack_type": "brute_force",
                            "attempts": len(failures),
                            "unique_users": len(unique_users),
                            "users": list(unique_users)
                        },
                        RiskLevel.HIGH
                    )
    
    async def _detect_credential_stuffing(self, events: List[SecurityEvent]):
        """Detect credential stuffing attacks"""
        # Look for patterns of failed logins with similar usernames
        username_patterns = {}
        
        for event in events:
            if event.event_type == SecurityEventType.LOGIN_FAILURE:
                username = event.details.get("username", "")
                if username:
                    # Extract common patterns (e.g., email domains)
                    if "@" in username:
                        domain = username.split("@")[1]
                        if domain not in username_patterns:
                            username_patterns[domain] = []
                        username_patterns[domain].append(event)
        
        # Check for credential stuffing patterns
        for domain, events_list in username_patterns.items():
            if len(events_list) >= 5:  # 5 attempts with same domain
                unique_ips = set(event.ip_address for event in events_list)
                if len(unique_ips) >= 2:  # From multiple IPs
                    await self._create_security_event(
                        SecurityEventType.SUSPICIOUS_ACTIVITY,
                        None, events_list[0].ip_address, None,
                        {
                            "attack_type": "credential_stuffing",
                            "domain": domain,
                            "attempts": len(events_list),
                            "unique_ips": len(unique_ips)
                        },
                        RiskLevel.HIGH
                    )
    
    async def _detect_account_takeover_attempts(self, events: List[SecurityEvent]):
        """Detect account takeover attempts"""
        # Look for patterns of password change attempts
        password_change_events = [
            event for event in events
            if event.event_type == SecurityEventType.PASSWORD_CHANGE
        ]
        
        # Group by user
        user_password_changes = {}
        for event in password_change_events:
            if event.user_id not in user_password_changes:
                user_password_changes[event.user_id] = []
            user_password_changes[event.user_id].append(event)
        
        # Check for suspicious password change patterns
        for user_id, changes in user_password_changes.items():
            if len(changes) >= 3:  # 3 password changes in short time
                unique_ips = set(event.ip_address for event in changes)
                if len(unique_ips) >= 2:  # From multiple IPs
                    await self._create_security_event(
                        SecurityEventType.SUSPICIOUS_ACTIVITY,
                        user_id, changes[0].ip_address, None,
                        {
                            "attack_type": "account_takeover",
                            "user_id": user_id,
                            "password_changes": len(changes),
                            "unique_ips": len(unique_ips)
                        },
                        RiskLevel.CRITICAL
                    )
    
    async def _trigger_policy_violation(self, policy: SecurityPolicy, message: str, details: Dict[str, Any]):
        """Trigger a security policy violation"""
        # Create security event
        await self._create_security_event(
            SecurityEventType.SECURITY_VIOLATION,
            details.get("user_id"), details.get("ip_address"), None,
            {
                "policy_name": policy.name,
                "message": message,
                "details": details
            },
            policy.severity
        )
        
        # Execute policy actions
        for action in policy.actions:
            if action == "temporary_lock":
                await self._temporary_lock_user(details.get("user_id"))
            elif action == "rate_limit":
                await self._rate_limit_ip(details.get("ip_address"))
            elif action == "notify_admin":
                await self._notify_admin(policy.name, message, details)
            elif action == "block_access":
                await self._block_ip(details.get("ip_address"))
    
    async def _create_security_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[int],
        ip_address: str,
        user_agent: Optional[str],
        details: Dict[str, Any],
        risk_level: RiskLevel = RiskLevel.LOW
    ) -> SecurityEvent:
        """Create a security event"""
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            details=details,
            risk_level=risk_level
        )
        
        self.security_events.append(event)
        
        # Log the security event
        log_security_event(event_type.value, {
            "event_id": event.event_id,
            "user_id": user_id,
            "ip_address": ip_address,
            "risk_level": risk_level.value,
            **details
        })
        
        return event
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_part = secrets.token_hex(8)
        return f"SEC_{timestamp}_{random_part}"
    
    async def _temporary_lock_user(self, user_id: Optional[int]):
        """Temporarily lock a user account"""
        if user_id:
            # This would integrate with the user management system
            self.logger.warning(f"Temporarily locking user {user_id}")
    
    async def _rate_limit_ip(self, ip_address: Optional[str]):
        """Apply rate limiting to an IP address"""
        if ip_address:
            self.suspicious_ips.add(ip_address)
            self.logger.warning(f"Rate limiting IP {ip_address}")
    
    async def _block_ip(self, ip_address: Optional[str]):
        """Block an IP address"""
        if ip_address:
            self.blocked_ips.add(ip_address)
            self.logger.warning(f"Blocking IP {ip_address}")
    
    async def _notify_admin(self, policy_name: str, message: str, details: Dict[str, Any]):
        """Notify administrators of security events"""
        # This would integrate with notification systems
        self.logger.critical(f"SECURITY ALERT: {policy_name} - {message}", **details)
    
    def log_security_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[int],
        ip_address: str,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log a security event"""
        asyncio.create_task(self._create_security_event(
            event_type, user_id, ip_address, user_agent, details or {}
        ))
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips
    
    def is_ip_suspicious(self, ip_address: str) -> bool:
        """Check if IP is suspicious"""
        return ip_address in self.suspicious_ips
    
    def get_security_events(
        self,
        event_type: Optional[SecurityEventType] = None,
        user_id: Optional[int] = None,
        risk_level: Optional[RiskLevel] = None,
        hours: int = 24
    ) -> List[SecurityEvent]:
        """Get security events with filters"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        events = [
            event for event in self.security_events
            if event.timestamp > cutoff_time
        ]
        
        if event_type:
            events = [event for event in events if event.event_type == event_type]
        
        if user_id:
            events = [event for event in events if event.user_id == user_id]
        
        if risk_level:
            events = [event for event in events if event.risk_level == risk_level]
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary"""
        events = self.get_security_events(hours=hours)
        
        summary = {
            "total_events": len(events),
            "events_by_type": {},
            "events_by_risk_level": {},
            "top_ips": {},
            "top_users": {},
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips": len(self.suspicious_ips),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Count by type
        for event in events:
            event_type = event.event_type.value
            summary["events_by_type"][event_type] = summary["events_by_type"].get(event_type, 0) + 1
        
        # Count by risk level
        for event in events:
            risk_level = event.risk_level.value
            summary["events_by_risk_level"][risk_level] = summary["events_by_risk_level"].get(risk_level, 0) + 1
        
        # Top IPs
        ip_counts = {}
        for event in events:
            ip_counts[event.ip_address] = ip_counts.get(event.ip_address, 0) + 1
        summary["top_ips"] = dict(sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Top users
        user_counts = {}
        for event in events:
            if event.user_id:
                user_counts[event.user_id] = user_counts.get(event.user_id, 0) + 1
        summary["top_users"] = dict(sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return summary


# Global security auditor instance
security_auditor = SecurityAuditor()