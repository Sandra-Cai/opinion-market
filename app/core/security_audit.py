"""
Comprehensive Security Audit System for Opinion Market
Monitors security events, tracks vulnerabilities, and ensures compliance
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import ipaddress
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import get_redis_client
from app.core.logging import log_security_event, log_system_metric
from app.core.validation import input_validator


class SecurityEventType(str, Enum):
    """Types of security events"""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    COMMAND_INJECTION_ATTEMPT = "command_injection_attempt"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    MALICIOUS_FILE_UPLOAD = "malicious_file_upload"
    API_ABUSE = "api_abuse"
    SESSION_HIJACKING = "session_hijacking"
    CSRF_ATTEMPT = "csrf_attempt"
    DDOS_ATTEMPT = "ddos_attempt"
    IP_BLOCKED = "ip_blocked"
    USER_BLOCKED = "user_blocked"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_COMPROMISE = "system_compromise"


class SecuritySeverity(str, Enum):
    """Security event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(str, Enum):
    """Compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    event_type: SecurityEventType
    severity: SecuritySeverity
    timestamp: datetime
    source_ip: str
    user_id: Optional[int]
    session_id: Optional[str]
    endpoint: Optional[str]
    user_agent: Optional[str]
    details: Dict[str, Any]
    risk_score: float
    compliance_impact: List[ComplianceFramework]
    remediation_actions: List[str]
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


@dataclass
class SecurityMetrics:
    """Security metrics data structure"""
    timestamp: datetime
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    top_source_ips: List[Dict[str, Any]]
    top_endpoints: List[Dict[str, Any]]
    risk_score_trend: List[float]
    compliance_violations: Dict[str, int]
    blocked_ips: int
    blocked_users: int
    active_threats: int


class ThreatIntelligence:
    """Threat intelligence and pattern recognition"""
    
    def __init__(self):
        self.malicious_ips = set()
        self.suspicious_patterns = {
            "sql_injection": [
                r"union\s+select",
                r"drop\s+table",
                r"insert\s+into",
                r"delete\s+from",
                r"update\s+set",
                r"exec\s*\(",
                r"xp_cmdshell",
                r"sp_executesql"
            ],
            "xss": [
                r"<script[^>]*>",
                r"javascript:",
                r"vbscript:",
                r"on\w+\s*=",
                r"expression\s*\(",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>"
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
                r"[;&|`$]",
                r"\|\|",
                r"&&",
                r"`.*`",
                r"\$\(.*\)",
                r"<.*>",
                r">.*<"
            ]
        }
        self.behavioral_patterns = {
            "rapid_requests": {"threshold": 100, "window": 60},  # 100 requests per minute
            "failed_logins": {"threshold": 5, "window": 300},    # 5 failed logins per 5 minutes
            "unusual_endpoints": {"threshold": 10, "window": 3600},  # 10 different endpoints per hour
            "large_payloads": {"threshold": 1024 * 1024, "window": 60}  # 1MB payloads
        }
    
    def analyze_threat(self, event: SecurityEvent) -> Dict[str, Any]:
        """Analyze threat patterns and return intelligence"""
        threat_analysis = {
            "is_known_threat": False,
            "threat_category": None,
            "confidence_score": 0.0,
            "recommended_actions": [],
            "similar_incidents": [],
            "geolocation": self._get_ip_geolocation(event.source_ip),
            "reputation_score": self._get_ip_reputation(event.source_ip)
        }
        
        # Check against known malicious IPs
        if event.source_ip in self.malicious_ips:
            threat_analysis["is_known_threat"] = True
            threat_analysis["confidence_score"] = 0.9
            threat_analysis["recommended_actions"].append("block_ip_permanently")
        
        # Pattern analysis
        for category, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if any(pattern.lower() in str(detail).lower() for detail in event.details.values()):
                    threat_analysis["threat_category"] = category
                    threat_analysis["confidence_score"] = max(threat_analysis["confidence_score"], 0.7)
                    threat_analysis["recommended_actions"].append(f"investigate_{category}")
        
        # Behavioral analysis
        behavioral_analysis = self._analyze_behavioral_patterns(event)
        threat_analysis.update(behavioral_analysis)
        
        return threat_analysis
    
    def _get_ip_geolocation(self, ip: str) -> Dict[str, Any]:
        """Get IP geolocation information"""
        try:
            # In a real implementation, this would use a geolocation service
            # For now, return basic information
            ip_obj = ipaddress.ip_address(ip)
            return {
                "country": "Unknown",
                "region": "Unknown",
                "city": "Unknown",
                "is_private": ip_obj.is_private,
                "is_reserved": ip_obj.is_reserved
            }
        except ValueError:
            return {"error": "Invalid IP address"}
    
    def _get_ip_reputation(self, ip: str) -> float:
        """Get IP reputation score (0.0 = malicious, 1.0 = clean)"""
        # In a real implementation, this would query threat intelligence feeds
        if ip in self.malicious_ips:
            return 0.0
        
        # Check if it's a private/reserved IP
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private or ip_obj.is_reserved:
                return 0.5  # Neutral score for private IPs
        except ValueError:
            return 0.0
        
        return 0.8  # Default clean score
    
    def _analyze_behavioral_patterns(self, event: SecurityEvent) -> Dict[str, Any]:
        """Analyze behavioral patterns for anomalies"""
        return {
            "behavioral_anomalies": [],
            "risk_indicators": [],
            "recommended_monitoring": []
        }


class SecurityAuditor:
    """Main security audit system"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.threat_intelligence = ThreatIntelligence()
        self.event_queue = deque(maxlen=10000)
        self.metrics_cache = {}
        self.alert_handlers = []
        self.compliance_rules = self._load_compliance_rules()
        self.audit_lock = threading.Lock()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _load_compliance_rules(self) -> Dict[str, List[Dict]]:
        """Load compliance rules for different frameworks"""
        return {
            ComplianceFramework.GDPR: [
                {
                    "rule": "data_access_logging",
                    "description": "All data access must be logged",
                    "severity": SecuritySeverity.HIGH,
                    "check": self._check_data_access_logging
                },
                {
                    "rule": "consent_management",
                    "description": "User consent must be properly managed",
                    "severity": SecuritySeverity.HIGH,
                    "check": self._check_consent_management
                }
            ],
            ComplianceFramework.PCI_DSS: [
                {
                    "rule": "card_data_protection",
                    "description": "Card data must be encrypted",
                    "severity": SecuritySeverity.CRITICAL,
                    "check": self._check_card_data_protection
                },
                {
                    "rule": "access_control",
                    "description": "Access to card data must be restricted",
                    "severity": SecuritySeverity.HIGH,
                    "check": self._check_access_control
                }
            ]
        }
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Start metrics collection
        asyncio.create_task(self._collect_metrics_periodically())
        
        # Start compliance monitoring
        asyncio.create_task(self._monitor_compliance_periodically())
        
        # Start threat analysis
        asyncio.create_task(self._analyze_threats_periodically())
    
    async def log_security_event(
        self,
        event_type: SecurityEventType,
        severity: SecuritySeverity,
        source_ip: str,
        details: Dict[str, Any],
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> SecurityEvent:
        """Log a security event with comprehensive analysis"""
        
        # Generate unique event ID
        event_id = self._generate_event_id()
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(event_type, severity, details)
        
        # Determine compliance impact
        compliance_impact = self._determine_compliance_impact(event_type, details)
        
        # Generate remediation actions
        remediation_actions = self._generate_remediation_actions(event_type, severity, details)
        
        # Create security event
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            session_id=session_id,
            endpoint=endpoint,
            user_agent=user_agent,
            details=details,
            risk_score=risk_score,
            compliance_impact=compliance_impact,
            remediation_actions=remediation_actions
        )
        
        # Store event
        await self._store_event(event)
        
        # Perform threat analysis
        threat_analysis = self.threat_intelligence.analyze_threat(event)
        
        # Trigger alerts if necessary
        await self._check_and_trigger_alerts(event, threat_analysis)
        
        # Log to system
        log_security_event(event_type.value, {
            "event_id": event_id,
            "severity": severity.value,
            "risk_score": risk_score,
            "source_ip": source_ip,
            "threat_analysis": threat_analysis
        })
        
        return event
    
    async def _store_event(self, event: SecurityEvent):
        """Store security event in Redis and local queue"""
        with self.audit_lock:
            # Add to local queue
            self.event_queue.append(event)
            
            # Store in Redis
            if self.redis_client:
                event_key = f"security_event:{event.event_id}"
                event_data = {
                    "event": json.dumps(asdict(event), default=str),
                    "timestamp": event.timestamp.isoformat(),
                    "severity": event.severity.value,
                    "risk_score": event.risk_score
                }
                
                # Store with TTL (30 days)
                self.redis_client.hset(event_key, mapping=event_data)
                self.redis_client.expire(event_key, 30 * 24 * 3600)
                
                # Add to time-series for metrics
                metrics_key = f"security_metrics:{event.timestamp.strftime('%Y-%m-%d')}"
                self.redis_client.hincrby(metrics_key, f"events:{event.event_type.value}", 1)
                self.redis_client.hincrby(metrics_key, f"severity:{event.severity.value}", 1)
                self.redis_client.expire(metrics_key, 90 * 24 * 3600)  # 90 days retention
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.md5(f"{timestamp}{time.time()}".encode()).hexdigest()[:8]
        return f"sec_{timestamp}_{random_part}"
    
    def _calculate_risk_score(
        self,
        event_type: SecurityEventType,
        severity: SecuritySeverity,
        details: Dict[str, Any]
    ) -> float:
        """Calculate risk score for security event"""
        base_scores = {
            SecuritySeverity.LOW: 0.2,
            SecuritySeverity.MEDIUM: 0.4,
            SecuritySeverity.HIGH: 0.7,
            SecuritySeverity.CRITICAL: 0.9
        }
        
        base_score = base_scores.get(severity, 0.5)
        
        # Adjust based on event type
        event_multipliers = {
            SecurityEventType.SQL_INJECTION_ATTEMPT: 1.5,
            SecurityEventType.XSS_ATTEMPT: 1.3,
            SecurityEventType.BRUTE_FORCE_ATTEMPT: 1.2,
            SecurityEventType.DDOS_ATTEMPT: 1.4,
            SecurityEventType.SYSTEM_COMPROMISE: 2.0
        }
        
        multiplier = event_multipliers.get(event_type, 1.0)
        
        # Adjust based on details
        if "admin" in str(details).lower():
            multiplier *= 1.2
        
        if "database" in str(details).lower():
            multiplier *= 1.3
        
        return min(1.0, base_score * multiplier)
    
    def _determine_compliance_impact(
        self,
        event_type: SecurityEventType,
        details: Dict[str, Any]
    ) -> List[ComplianceFramework]:
        """Determine which compliance frameworks are impacted"""
        impact = []
        
        # GDPR impact
        if any(keyword in str(details).lower() for keyword in ["personal", "data", "user", "email"]):
            impact.append(ComplianceFramework.GDPR)
        
        # PCI DSS impact
        if any(keyword in str(details).lower() for keyword in ["payment", "card", "financial", "transaction"]):
            impact.append(ComplianceFramework.PCI_DSS)
        
        # SOX impact
        if any(keyword in str(details).lower() for keyword in ["financial", "audit", "accounting"]):
            impact.append(ComplianceFramework.SOX)
        
        return impact
    
    def _generate_remediation_actions(
        self,
        event_type: SecurityEventType,
        severity: SecuritySeverity,
        details: Dict[str, Any]
    ) -> List[str]:
        """Generate recommended remediation actions"""
        actions = []
        
        # Base actions by event type
        if event_type == SecurityEventType.BRUTE_FORCE_ATTEMPT:
            actions.extend([
                "temporarily_block_ip",
                "increase_rate_limiting",
                "require_captcha",
                "notify_user_of_suspicious_activity"
            ])
        
        elif event_type == SecurityEventType.SQL_INJECTION_ATTEMPT:
            actions.extend([
                "block_ip_permanently",
                "review_input_validation",
                "audit_database_queries",
                "update_waf_rules"
            ])
        
        elif event_type == SecurityEventType.XSS_ATTEMPT:
            actions.extend([
                "block_ip_temporarily",
                "review_output_encoding",
                "update_csp_headers",
                "audit_user_input_handling"
            ])
        
        # Severity-based actions
        if severity in [SecuritySeverity.HIGH, SecuritySeverity.CRITICAL]:
            actions.extend([
                "immediate_incident_response",
                "notify_security_team",
                "escalate_to_management"
            ])
        
        return actions
    
    async def _check_and_trigger_alerts(self, event: SecurityEvent, threat_analysis: Dict[str, Any]):
        """Check if alerts should be triggered and execute them"""
        should_alert = False
        
        # Alert conditions
        if event.severity in [SecuritySeverity.HIGH, SecuritySeverity.CRITICAL]:
            should_alert = True
        
        if event.risk_score > 0.8:
            should_alert = True
        
        if threat_analysis.get("is_known_threat", False):
            should_alert = True
        
        if should_alert:
            await self._trigger_alert(event, threat_analysis)
    
    async def _trigger_alert(self, event: SecurityEvent, threat_analysis: Dict[str, Any]):
        """Trigger security alert"""
        alert_data = {
            "event": asdict(event),
            "threat_analysis": threat_analysis,
            "alert_timestamp": datetime.utcnow().isoformat(),
            "alert_id": f"alert_{event.event_id}"
        }
        
        # Execute alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                log_system_metric("alert_handler_error", 1, {"error": str(e)})
        
        # Log alert
        log_security_event("security_alert_triggered", alert_data)
    
    async def _collect_metrics_periodically(self):
        """Collect security metrics periodically"""
        while True:
            try:
                await self._collect_metrics()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                log_system_metric("metrics_collection_error", 1, {"error": str(e)})
                await asyncio.sleep(60)  # Retry in 1 minute
    
    async def _collect_metrics(self):
        """Collect current security metrics"""
        now = datetime.utcnow()
        metrics = SecurityMetrics(
            timestamp=now,
            total_events=len(self.event_queue),
            events_by_type={},
            events_by_severity={},
            top_source_ips=[],
            top_endpoints=[],
            risk_score_trend=[],
            compliance_violations={},
            blocked_ips=0,
            blocked_users=0,
            active_threats=0
        )
        
        # Analyze recent events
        recent_events = [e for e in self.event_queue if (now - e.timestamp).total_seconds() < 3600]
        
        # Count by type and severity
        for event in recent_events:
            metrics.events_by_type[event.event_type.value] = metrics.events_by_type.get(event.event_type.value, 0) + 1
            metrics.events_by_severity[event.severity.value] = metrics.events_by_severity.get(event.severity.value, 0) + 1
        
        # Store metrics
        self.metrics_cache[now] = metrics
        
        # Log metrics
        log_system_metric("security_metrics_collected", 1, {
            "total_events": metrics.total_events,
            "high_severity": metrics.events_by_severity.get("high", 0),
            "critical_severity": metrics.events_by_severity.get("critical", 0)
        })
    
    async def _monitor_compliance_periodically(self):
        """Monitor compliance rules periodically"""
        while True:
            try:
                await self._check_compliance()
                await asyncio.sleep(3600)  # Every hour
            except Exception as e:
                log_system_metric("compliance_monitoring_error", 1, {"error": str(e)})
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _check_compliance(self):
        """Check compliance rules"""
        for framework, rules in self.compliance_rules.items():
            for rule in rules:
                try:
                    is_compliant = await rule["check"]()
                    if not is_compliant:
                        await self.log_security_event(
                            SecurityEventType.COMPLIANCE_VIOLATION,
                            rule["severity"],
                            "system",
                            {
                                "framework": framework.value,
                                "rule": rule["rule"],
                                "description": rule["description"]
                            }
                        )
                except Exception as e:
                    log_system_metric("compliance_check_error", 1, {
                        "framework": framework.value,
                        "rule": rule["rule"],
                        "error": str(e)
                    })
    
    async def _analyze_threats_periodically(self):
        """Analyze threats periodically"""
        while True:
            try:
                await self._analyze_threats()
                await asyncio.sleep(1800)  # Every 30 minutes
            except Exception as e:
                log_system_metric("threat_analysis_error", 1, {"error": str(e)})
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _analyze_threats(self):
        """Analyze current threats and update threat intelligence"""
        # Analyze recent events for patterns
        recent_events = [e for e in self.event_queue if (datetime.utcnow() - e.timestamp).total_seconds() < 3600]
        
        # Group by source IP
        ip_events = defaultdict(list)
        for event in recent_events:
            ip_events[event.source_ip].append(event)
        
        # Identify suspicious IPs
        for ip, events in ip_events.items():
            if len(events) > 10:  # More than 10 events in 1 hour
                self.threat_intelligence.malicious_ips.add(ip)
                await self.log_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    SecuritySeverity.HIGH,
                    ip,
                    {
                        "event_count": len(events),
                        "time_window": "1_hour",
                        "action": "marked_as_malicious"
                    }
                )
    
    # Compliance check methods
    async def _check_data_access_logging(self) -> bool:
        """Check if data access is properly logged"""
        # Implementation would check if all data access is logged
        return True
    
    async def _check_consent_management(self) -> bool:
        """Check if user consent is properly managed"""
        # Implementation would check consent management system
        return True
    
    async def _check_card_data_protection(self) -> bool:
        """Check if card data is properly protected"""
        # Implementation would check encryption and protection measures
        return True
    
    async def _check_access_control(self) -> bool:
        """Check if access control is properly implemented"""
        # Implementation would check access control mechanisms
        return True
    
    def add_alert_handler(self, handler: Callable):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
    
    def get_security_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get security metrics for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = {k: v for k, v in self.metrics_cache.items() if k > cutoff_time}
        
        return {
            "time_period_hours": hours,
            "metrics_count": len(recent_metrics),
            "latest_metrics": recent_metrics.get(max(recent_metrics.keys())) if recent_metrics else None,
            "trend_data": list(recent_metrics.values())
        }
    
    def get_threat_intelligence(self) -> Dict[str, Any]:
        """Get current threat intelligence"""
        return {
            "malicious_ips_count": len(self.threat_intelligence.malicious_ips),
            "malicious_ips": list(self.threat_intelligence.malicious_ips),
            "behavioral_patterns": self.threat_intelligence.behavioral_patterns,
            "suspicious_patterns": self.threat_intelligence.suspicious_patterns
        }


# Global security auditor instance
security_auditor = SecurityAuditor()