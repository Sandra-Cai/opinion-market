"""
Advanced Security V2
Enhanced security features with AI-powered threat detection and prevention
"""

import asyncio
import time
import hashlib
import hmac
import secrets
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import ipaddress
import re
import json

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Security event types"""
    LOGIN_ATTEMPT = "login_attempt"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    CSRF_ATTEMPT = "csrf_attempt"
    BRUTE_FORCE = "brute_force"
    DDoS_ATTEMPT = "ddos_attempt"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class SecurityThreat:
    """Security threat data structure"""
    threat_id: str
    threat_type: SecurityEvent
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    details: Dict[str, Any]
    detected_at: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    mitigation_action: Optional[str] = None


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    policy_id: str
    name: str
    description: str
    enabled: bool
    conditions: Dict[str, Any]
    actions: List[str]
    severity: ThreatLevel
    cooldown_period: int = 300  # seconds
    created_at: datetime = field(default_factory=datetime.now)


class AdvancedSecurityV2:
    """Advanced security system with AI-powered threat detection"""
    
    def __init__(self):
        self.threats: List[SecurityThreat] = []
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, int] = defaultdict(int)
        self.user_activity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.ip_activity: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Security configuration
        self.config = {
            "max_login_attempts": 5,
            "login_window": 300,  # 5 minutes
            "rate_limit_window": 60,  # 1 minute
            "max_requests_per_minute": 100,
            "suspicious_threshold": 10,
            "block_duration": 3600,  # 1 hour
            "ai_detection_enabled": True,
            "auto_block_enabled": True
        }
        
        # Threat patterns
        self.threat_patterns = {
            "sql_injection": [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
                r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
                r"(\b(OR|AND)\s+'.*'\s*=\s*'.*')",
                r"(\b(OR|AND)\s+\".*\"\s*=\s*\".*\")",
                r"(\b(OR|AND)\s+1\s*=\s*1)",
                r"(\b(OR|AND)\s+true\s*=\s*true)"
            ],
            "xss_attempt": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>",
                r"<link[^>]*>",
                r"<meta[^>]*>"
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c",
                r"\.\.%2f",
                r"\.\.%5c"
            ],
            "command_injection": [
                r"[;&|`$()]",
                r"\b(cat|ls|pwd|whoami|id|uname|ps|netstat|ifconfig)\b",
                r"\b(cmd|command|exec|system|shell_exec)\b"
            ]
        }
        
        # Security monitoring
        self.security_active = False
        self.security_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.security_stats = {
            "threats_detected": 0,
            "threats_blocked": 0,
            "false_positives": 0,
            "security_events": 0,
            "ips_blocked": 0,
            "users_blocked": 0
        }
        
        # Initialize default security policies
        self._initialize_default_policies()
        
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        default_policies = [
            SecurityPolicy(
                policy_id="brute_force_protection",
                name="Brute Force Protection",
                description="Detect and prevent brute force attacks",
                enabled=True,
                conditions={
                    "max_attempts": 5,
                    "time_window": 300,
                    "action": "block_ip"
                },
                actions=["block_ip", "alert_admin"],
                severity=ThreatLevel.HIGH
            ),
            SecurityPolicy(
                policy_id="rate_limiting",
                name="Rate Limiting",
                description="Prevent rate limit abuse",
                enabled=True,
                conditions={
                    "max_requests": 100,
                    "time_window": 60,
                    "action": "throttle"
                },
                actions=["throttle", "alert_admin"],
                severity=ThreatLevel.MEDIUM
            ),
            SecurityPolicy(
                policy_id="sql_injection_detection",
                name="SQL Injection Detection",
                description="Detect SQL injection attempts",
                enabled=True,
                conditions={
                    "patterns": self.threat_patterns["sql_injection"],
                    "action": "block_request"
                },
                actions=["block_request", "alert_admin", "log_incident"],
                severity=ThreatLevel.CRITICAL
            ),
            SecurityPolicy(
                policy_id="xss_detection",
                name="XSS Detection",
                description="Detect XSS attempts",
                enabled=True,
                conditions={
                    "patterns": self.threat_patterns["xss_attempt"],
                    "action": "block_request"
                },
                actions=["block_request", "alert_admin", "log_incident"],
                severity=ThreatLevel.HIGH
            ),
            SecurityPolicy(
                policy_id="suspicious_activity",
                name="Suspicious Activity Detection",
                description="Detect suspicious user behavior",
                enabled=True,
                conditions={
                    "threshold": 10,
                    "time_window": 300,
                    "action": "investigate"
                },
                actions=["investigate", "alert_admin"],
                severity=ThreatLevel.MEDIUM
            )
        ]
        
        for policy in default_policies:
            self.security_policies[policy.policy_id] = policy
            
    async def start_security_monitoring(self):
        """Start the security monitoring system"""
        if self.security_active:
            logger.warning("Security monitoring already active")
            return
            
        self.security_active = True
        self.security_task = asyncio.create_task(self._security_monitoring_loop())
        logger.info("Advanced Security V2 monitoring started")
        
    async def stop_security_monitoring(self):
        """Stop the security monitoring system"""
        self.security_active = False
        if self.security_task:
            self.security_task.cancel()
            try:
                await self.security_task
            except asyncio.CancelledError:
                pass
        logger.info("Advanced Security V2 monitoring stopped")
        
    async def _security_monitoring_loop(self):
        """Main security monitoring loop"""
        while self.security_active:
            try:
                # Analyze user activity
                await self._analyze_user_activity()
                
                # Analyze IP activity
                await self._analyze_ip_activity()
                
                # Check for suspicious patterns
                await self._detect_suspicious_patterns()
                
                # Update threat intelligence
                await self._update_threat_intelligence()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next cycle
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in security monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a request for security threats"""
        try:
            threat_detected = False
            detected_threats = []
            
            # Extract request information
            source_ip = request_data.get("source_ip", "unknown")
            user_id = request_data.get("user_id")
            method = request_data.get("method", "GET")
            path = request_data.get("path", "")
            headers = request_data.get("headers", {})
            body = request_data.get("body", "")
            query_params = request_data.get("query_params", {})
            
            # Check if IP is blocked
            if source_ip in self.blocked_ips:
                return {
                    "threat_detected": True,
                    "threat_level": ThreatLevel.CRITICAL.value,
                    "threat_type": "blocked_ip",
                    "action": "block",
                    "message": "IP address is blocked"
                }
            
            # Check rate limiting
            if await self._check_rate_limit(source_ip):
                threat_detected = True
                detected_threats.append({
                    "type": SecurityEvent.RATE_LIMIT_EXCEEDED,
                    "level": ThreatLevel.MEDIUM,
                    "description": "Rate limit exceeded"
                })
            
            # Check for SQL injection
            if await self._detect_sql_injection(path, body, query_params):
                threat_detected = True
                detected_threats.append({
                    "type": SecurityEvent.SQL_INJECTION,
                    "level": ThreatLevel.CRITICAL,
                    "description": "SQL injection attempt detected"
                })
            
            # Check for XSS
            if await self._detect_xss(path, body, query_params):
                threat_detected = True
                detected_threats.append({
                    "type": SecurityEvent.XSS_ATTEMPT,
                    "level": ThreatLevel.HIGH,
                    "description": "XSS attempt detected"
                })
            
            # Check for path traversal
            if await self._detect_path_traversal(path):
                threat_detected = True
                detected_threats.append({
                    "type": SecurityEvent.SUSPICIOUS_ACTIVITY,
                    "level": ThreatLevel.HIGH,
                    "description": "Path traversal attempt detected"
                })
            
            # Check for command injection
            if await self._detect_command_injection(path, body, query_params):
                threat_detected = True
                detected_threats.append({
                    "type": SecurityEvent.SUSPICIOUS_ACTIVITY,
                    "level": ThreatLevel.CRITICAL,
                    "description": "Command injection attempt detected"
                })
            
            # Record activity
            await self._record_activity(source_ip, user_id, request_data)
            
            # Process detected threats
            if threat_detected:
                await self._process_threats(source_ip, user_id, detected_threats)
                
                return {
                    "threat_detected": True,
                    "threat_level": max(t["level"].value for t in detected_threats),
                    "threat_type": detected_threats[0]["type"].value,
                    "action": "block" if any(t["level"] == ThreatLevel.CRITICAL for t in detected_threats) else "monitor",
                    "message": "; ".join(t["description"] for t in detected_threats)
                }
            
            return {
                "threat_detected": False,
                "threat_level": ThreatLevel.LOW.value,
                "action": "allow",
                "message": "Request is safe"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            return {
                "threat_detected": False,
                "threat_level": ThreatLevel.LOW.value,
                "action": "allow",
                "message": "Analysis error"
            }
            
    async def _check_rate_limit(self, source_ip: str) -> bool:
        """Check if source IP has exceeded rate limit"""
        try:
            current_time = time.time()
            window_start = current_time - self.config["rate_limit_window"]
            
            # Get recent requests for this IP
            if source_ip in self.ip_activity:
                recent_requests = [
                    req for req in self.ip_activity[source_ip]
                    if req.get("timestamp", 0) > window_start
                ]
                
                if len(recent_requests) > self.config["max_requests_per_minute"]:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
            
    async def _detect_sql_injection(self, path: str, body: str, query_params: Dict[str, Any]) -> bool:
        """Detect SQL injection attempts"""
        try:
            # Combine all input data
            input_data = f"{path} {body} {json.dumps(query_params)}"
            
            # Check against SQL injection patterns
            for pattern in self.threat_patterns["sql_injection"]:
                if re.search(pattern, input_data, re.IGNORECASE):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting SQL injection: {e}")
            return False
            
    async def _detect_xss(self, path: str, body: str, query_params: Dict[str, Any]) -> bool:
        """Detect XSS attempts"""
        try:
            # Combine all input data
            input_data = f"{path} {body} {json.dumps(query_params)}"
            
            # Check against XSS patterns
            for pattern in self.threat_patterns["xss_attempt"]:
                if re.search(pattern, input_data, re.IGNORECASE):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting XSS: {e}")
            return False
            
    async def _detect_path_traversal(self, path: str) -> bool:
        """Detect path traversal attempts"""
        try:
            # Check against path traversal patterns
            for pattern in self.threat_patterns["path_traversal"]:
                if re.search(pattern, path, re.IGNORECASE):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting path traversal: {e}")
            return False
            
    async def _detect_command_injection(self, path: str, body: str, query_params: Dict[str, Any]) -> bool:
        """Detect command injection attempts"""
        try:
            # Combine all input data
            input_data = f"{path} {body} {json.dumps(query_params)}"
            
            # Check against command injection patterns
            for pattern in self.threat_patterns["command_injection"]:
                if re.search(pattern, input_data, re.IGNORECASE):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting command injection: {e}")
            return False
            
    async def _record_activity(self, source_ip: str, user_id: Optional[str], request_data: Dict[str, Any]):
        """Record user and IP activity"""
        try:
            activity_data = {
                "timestamp": time.time(),
                "source_ip": source_ip,
                "user_id": user_id,
                "request_data": request_data
            }
            
            # Record IP activity
            self.ip_activity[source_ip].append(activity_data)
            
            # Record user activity if user_id is provided
            if user_id:
                self.user_activity[user_id].append(activity_data)
                
        except Exception as e:
            logger.error(f"Error recording activity: {e}")
            
    async def _process_threats(self, source_ip: str, user_id: Optional[str], threats: List[Dict[str, Any]]):
        """Process detected threats"""
        try:
            for threat_info in threats:
                # Create threat record
                threat = SecurityThreat(
                    threat_id=f"threat_{int(time.time())}_{secrets.token_hex(4)}",
                    threat_type=threat_info["type"],
                    threat_level=threat_info["level"],
                    source_ip=source_ip,
                    user_id=user_id,
                    description=threat_info["description"],
                    details={"threats": threats},
                    detected_at=datetime.now()
                )
                
                # Add to threats list
                self.threats.append(threat)
                
                # Update statistics
                self.security_stats["threats_detected"] += 1
                self.security_stats["security_events"] += 1
                
                # Take mitigation action
                await self._take_mitigation_action(threat)
                
        except Exception as e:
            logger.error(f"Error processing threats: {e}")
            
    async def _take_mitigation_action(self, threat: SecurityThreat):
        """Take mitigation action for a threat"""
        try:
            if threat.threat_level == ThreatLevel.CRITICAL:
                # Block IP for critical threats
                await self._block_ip(threat.source_ip, "Critical threat detected")
                threat.mitigation_action = "ip_blocked"
                
            elif threat.threat_level == ThreatLevel.HIGH:
                # Add to suspicious IPs
                self.suspicious_ips[threat.source_ip] += 1
                threat.mitigation_action = "marked_suspicious"
                
            # Log the threat
            logger.warning(f"Security threat detected: {threat.description} from {threat.source_ip}")
            
        except Exception as e:
            logger.error(f"Error taking mitigation action: {e}")
            
    async def _block_ip(self, ip_address: str, reason: str):
        """Block an IP address"""
        try:
            self.blocked_ips.add(ip_address)
            self.security_stats["ips_blocked"] += 1
            
            # Store in cache for persistence
            await enhanced_cache.set(f"blocked_ip_{ip_address}", {
                "blocked_at": datetime.now().isoformat(),
                "reason": reason
            }, ttl=self.config["block_duration"])
            
            logger.warning(f"IP address {ip_address} blocked: {reason}")
            
        except Exception as e:
            logger.error(f"Error blocking IP: {e}")
            
    async def _analyze_user_activity(self):
        """Analyze user activity for suspicious patterns"""
        try:
            for user_id, activity in self.user_activity.items():
                if len(activity) < 10:
                    continue
                    
                # Analyze activity patterns
                recent_activity = list(activity)[-10:]
                
                # Check for rapid requests
                timestamps = [req["timestamp"] for req in recent_activity]
                if len(timestamps) > 1:
                    time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                    avg_time_diff = sum(time_diffs) / len(time_diffs)
                    
                    if avg_time_diff < 1.0:  # Less than 1 second between requests
                        # Mark as suspicious
                        self.suspicious_ips[user_id] += 1
                        
        except Exception as e:
            logger.error(f"Error analyzing user activity: {e}")
            
    async def _analyze_ip_activity(self):
        """Analyze IP activity for suspicious patterns"""
        try:
            for ip_address, activity in self.ip_activity.items():
                if len(activity) < 20:
                    continue
                    
                # Analyze activity patterns
                recent_activity = list(activity)[-20:]
                
                # Check for rapid requests
                timestamps = [req["timestamp"] for req in recent_activity]
                if len(timestamps) > 1:
                    time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                    avg_time_diff = sum(time_diffs) / len(time_diffs)
                    
                    if avg_time_diff < 0.5:  # Less than 0.5 seconds between requests
                        # Mark as suspicious
                        self.suspicious_ips[ip_address] += 1
                        
                        # Block if too suspicious
                        if self.suspicious_ips[ip_address] > self.config["suspicious_threshold"]:
                            await self._block_ip(ip_address, "Suspicious activity pattern")
                            
        except Exception as e:
            logger.error(f"Error analyzing IP activity: {e}")
            
    async def _detect_suspicious_patterns(self):
        """Detect suspicious patterns in activity"""
        try:
            # This would implement more sophisticated pattern detection
            # For now, we'll use basic heuristics
            
            current_time = time.time()
            
            # Check for DDoS patterns
            for ip_address, activity in self.ip_activity.items():
                if len(activity) < 50:
                    continue
                    
                recent_activity = [
                    req for req in activity
                    if req.get("timestamp", 0) > current_time - 60  # Last minute
                ]
                
                if len(recent_activity) > 50:  # More than 50 requests per minute
                    await self._block_ip(ip_address, "DDoS pattern detected")
                    
        except Exception as e:
            logger.error(f"Error detecting suspicious patterns: {e}")
            
    async def _update_threat_intelligence(self):
        """Update threat intelligence data"""
        try:
            # This would integrate with external threat intelligence feeds
            # For now, we'll update internal patterns
            
            # Update threat patterns based on recent threats
            recent_threats = [
                threat for threat in self.threats
                if (datetime.now() - threat.detected_at).total_seconds() < 3600
            ]
            
            # Analyze patterns in recent threats
            if len(recent_threats) > 10:
                # Update threat patterns based on analysis
                pass
                
        except Exception as e:
            logger.error(f"Error updating threat intelligence: {e}")
            
    async def _cleanup_old_data(self):
        """Clean up old security data"""
        try:
            current_time = time.time()
            cutoff_time = current_time - 86400  # 24 hours
            
            # Clean up old threats
            self.threats = [
                threat for threat in self.threats
                if threat.detected_at.timestamp() > cutoff_time
            ]
            
            # Clean up old activity data
            for ip_address in list(self.ip_activity.keys()):
                self.ip_activity[ip_address] = deque([
                    activity for activity in self.ip_activity[ip_address]
                    if activity.get("timestamp", 0) > cutoff_time
                ], maxlen=1000)
                
            for user_id in list(self.user_activity.keys()):
                self.user_activity[user_id] = deque([
                    activity for activity in self.user_activity[user_id]
                    if activity.get("timestamp", 0) > cutoff_time
                ], maxlen=1000)
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    def add_security_policy(self, policy: SecurityPolicy):
        """Add a new security policy"""
        self.security_policies[policy.policy_id] = policy
        logger.info(f"Security policy added: {policy.name}")
        
    def update_security_policy(self, policy_id: str, **kwargs):
        """Update an existing security policy"""
        if policy_id in self.security_policies:
            policy = self.security_policies[policy_id]
            for key, value in kwargs.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            logger.info(f"Security policy updated: {policy.name}")
        else:
            logger.warning(f"Security policy not found: {policy_id}")
            
    def remove_security_policy(self, policy_id: str):
        """Remove a security policy"""
        if policy_id in self.security_policies:
            policy = self.security_policies[policy_id]
            del self.security_policies[policy_id]
            logger.info(f"Security policy removed: {policy.name}")
        else:
            logger.warning(f"Security policy not found: {policy_id}")
            
    def unblock_ip(self, ip_address: str):
        """Unblock an IP address"""
        try:
            self.blocked_ips.discard(ip_address)
            await enhanced_cache.delete(f"blocked_ip_{ip_address}")
            logger.info(f"IP address {ip_address} unblocked")
        except Exception as e:
            logger.error(f"Error unblocking IP: {e}")
            
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary"""
        try:
            # Get recent threats
            recent_threats = [
                threat for threat in self.threats
                if (datetime.now() - threat.detected_at).total_seconds() < 3600
            ]
            
            # Get threats by level
            threats_by_level = defaultdict(int)
            for threat in recent_threats:
                threats_by_level[threat.threat_level.value] += 1
                
            return {
                "timestamp": datetime.now().isoformat(),
                "security_active": self.security_active,
                "recent_threats": len(recent_threats),
                "threats_by_level": dict(threats_by_level),
                "blocked_ips": len(self.blocked_ips),
                "suspicious_ips": len(self.suspicious_ips),
                "active_policies": len([p for p in self.security_policies.values() if p.enabled]),
                "total_policies": len(self.security_policies),
                "stats": self.security_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting security summary: {e}")
            return {"error": str(e)}


# Global instance
advanced_security_v2 = AdvancedSecurityV2()
