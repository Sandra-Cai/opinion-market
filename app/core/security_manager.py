"""
Advanced Security Manager
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    event_type: str
    severity: str
    source_ip: str
    timestamp: float
    details: Dict[str, Any]


class SecurityManager:
    def __init__(self):
        self.security_events = deque(maxlen=1000)
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        self.blocked_ips = set()
        
        self.config = {
            "rate_limits": {
                "login": {"max": 5, "window": 300},
                "api": {"max": 100, "window": 3600}
            },
            "patterns": {
                "sql_injection": [r"('|;|union|select)"],
                "xss": [r"<script", r"javascript:"]
            }
        }
    
    async def check_rate_limit(self, ip: str, limit_type: str) -> bool:
        """Check rate limit"""
        config = self.config["rate_limits"].get(limit_type)
        if not config:
            return True
        
        current_time = time.time()
        key = f"{ip}:{limit_type}"
        requests = self.rate_limits[key]
        
        # Remove old requests
        window_start = current_time - config["window"]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        if len(requests) >= config["max"]:
            await self._log_event("rate_limit_exceeded", "medium", ip, {"limit_type": limit_type})
            return False
        
        requests.append(current_time)
        return True
    
    async def detect_threats(self, data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect security threats"""
        threats = []
        
        for key, value in data.items():
            if isinstance(value, str):
                # Check SQL injection
                for pattern in self.config["patterns"]["sql_injection"]:
                    if re.search(pattern, value, re.IGNORECASE):
                        threats.append(SecurityEvent(
                            event_type="sql_injection",
                            severity="high",
                            source_ip=data.get("ip", "unknown"),
                            timestamp=time.time(),
                            details={"field": key}
                        ))
                        break
                
                # Check XSS
                for pattern in self.config["patterns"]["xss"]:
                    if re.search(pattern, value, re.IGNORECASE):
                        threats.append(SecurityEvent(
                            event_type="xss_attempt",
                            severity="high",
                            source_ip=data.get("ip", "unknown"),
                            timestamp=time.time(),
                            details={"field": key}
                        ))
                        break
        
        return threats
    
    async def _log_event(self, event_type: str, severity: str, ip: str, details: Dict[str, Any]):
        """Log security event"""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source_ip=ip,
            timestamp=time.time(),
            details=details
        )
        self.security_events.append(event)
        logger.info(f"Security event: {event_type} from {ip}")
    
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        return ip in self.blocked_ips
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        return {
            "events_count": len(self.security_events),
            "blocked_ips": len(self.blocked_ips)
        }


# Global instance
security_manager = SecurityManager()
