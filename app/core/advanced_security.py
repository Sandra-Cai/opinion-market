"""
Advanced Security Module for Opinion Market
Provides enhanced security features including threat detection, anomaly detection, and advanced authentication
"""

import hashlib
import secrets
import time
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple
from functools import wraps
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import redis
from redis.exceptions import RedisError
import ipaddress
import re
import asyncio
from collections import defaultdict, deque
import logging

from app.core.config import settings
from app.core.database import get_db, get_redis_client
from app.core.logging import log_security_event, log_api_call
from app.models.user import User

logger = logging.getLogger(__name__)

# Enhanced password hashing with multiple schemes
pwd_context = CryptContext(
    schemes=["bcrypt", "argon2"],
    deprecated="auto",
    bcrypt__rounds=12,
    argon2__memory_cost=65536,
    argon2__time_cost=3,
    argon2__parallelism=4
)

# JWT token handling
security = HTTPBearer(auto_error=False)


class ThreatDetector:
    """Advanced threat detection system"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.suspicious_patterns = deque(maxlen=10000)
        self.attack_attempts = defaultdict(int)
        self.geo_blocked_countries = set()
        self.suspicious_user_agents = set()
        self.known_bad_ips = set()
        
    def detect_brute_force(self, identifier: str, max_attempts: int = 5, window: int = 300) -> bool:
        """Detect brute force attacks"""
        if self.redis_client:
            try:
                key = f"brute_force:{identifier}"
                attempts = self.redis_client.incr(key)
                if attempts == 1:
                    self.redis_client.expire(key, window)
                
                if attempts >= max_attempts:
                    self.redis_client.setex(f"blocked_brute_force:{identifier}", 3600, "1")
                    return True
            except RedisError:
                pass
        
        # Fallback to in-memory tracking
        current_time = time.time()
        if identifier not in self.attack_attempts:
            self.attack_attempts[identifier] = []
        
        # Clean old attempts
        self.attack_attempts[identifier] = [
            attempt_time for attempt_time in self.attack_attempts[identifier]
            if current_time - attempt_time < window
        ]
        
        self.attack_attempts[identifier].append(current_time)
        return len(self.attack_attempts[identifier]) >= max_attempts
    
    def detect_anomalous_behavior(self, user_id: int, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalous user behavior"""
        anomaly_score = 0
        anomalies = []
        
        # Check for unusual request patterns
        if self.redis_client:
            try:
                # Check request frequency
                key = f"user_requests:{user_id}"
                current_hour = datetime.utcnow().strftime('%Y-%m-%d-%H')
                hourly_key = f"{key}:{current_hour}"
                
                requests_this_hour = self.redis_client.incr(hourly_key)
                if requests_this_hour == 1:
                    self.redis_client.expire(hourly_key, 3600)
                
                if requests_this_hour > 1000:  # More than 1000 requests per hour
                    anomaly_score += 30
                    anomalies.append("Unusually high request frequency")
                
                # Check for unusual endpoints
                endpoint_key = f"user_endpoints:{user_id}"
                endpoint = request_data.get("endpoint", "")
                self.redis_client.sadd(endpoint_key, endpoint)
                self.redis_client.expire(endpoint_key, 86400)  # 24 hours
                
                unique_endpoints = self.redis_client.scard(endpoint_key)
                if unique_endpoints > 50:  # More than 50 unique endpoints in 24h
                    anomaly_score += 20
                    anomalies.append("Accessing too many different endpoints")
                
            except RedisError:
                pass
        
        # Check for suspicious patterns
        user_agent = request_data.get("user_agent", "")
        if any(pattern in user_agent.lower() for pattern in ["bot", "crawler", "scraper", "spider"]):
            anomaly_score += 15
            anomalies.append("Suspicious user agent")
        
        # Check for rapid location changes (if IP geolocation is available)
        client_ip = request_data.get("client_ip", "")
        if self._is_rapid_location_change(user_id, client_ip):
            anomaly_score += 25
            anomalies.append("Rapid location changes detected")
        
        return {
            "anomaly_score": anomaly_score,
            "anomalies": anomalies,
            "is_suspicious": anomaly_score > 50
        }
    
    def _is_rapid_location_change(self, user_id: int, current_ip: str) -> bool:
        """Check for rapid location changes"""
        if not self.redis_client:
            return False
        
        try:
            key = f"user_locations:{user_id}"
            recent_locations = self.redis_client.lrange(key, 0, 4)  # Last 5 locations
            
            if current_ip not in recent_locations:
                self.redis_client.lpush(key, current_ip)
                self.redis_client.ltrim(key, 0, 9)  # Keep last 10
                self.redis_client.expire(key, 3600)  # 1 hour
                
                # If we have 5 different locations in the last hour, it's suspicious
                if len(set(recent_locations)) >= 4:
                    return True
        except RedisError:
            pass
        
        return False
    
    def detect_ddos_attack(self, client_ip: str, window: int = 60) -> bool:
        """Detect DDoS attacks"""
        if self.redis_client:
            try:
                key = f"ddos_check:{client_ip}"
                requests = self.redis_client.incr(key)
                if requests == 1:
                    self.redis_client.expire(key, window)
                
                # More than 100 requests per minute is suspicious
                if requests > 100:
                    self.redis_client.setex(f"ddos_blocked:{client_ip}", 3600, "1")
                    return True
            except RedisError:
                pass
        
        return False
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked for various reasons"""
        if self.redis_client:
            try:
                # Check various block reasons
                block_reasons = [
                    f"blocked_ip:{ip}",
                    f"blocked_brute_force:{ip}",
                    f"ddos_blocked:{ip}",
                    f"suspicious_ip:{ip}"
                ]
                
                for reason in block_reasons:
                    if self.redis_client.get(reason):
                        return True
            except RedisError:
                pass
        
        return ip in self.known_bad_ips


class AdvancedSecurityManager:
    """Enhanced security manager with advanced features"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.threat_detector = ThreatDetector()
        self.session_tokens = {}  # In-memory session storage
        self.device_fingerprints = {}
        self.security_events = deque(maxlen=10000)
        
    def create_secure_session(self, user_id: int, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a secure session with device fingerprinting"""
        session_id = secrets.token_urlsafe(32)
        device_fingerprint = self._generate_device_fingerprint(device_info)
        
        session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "device_fingerprint": device_fingerprint,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "ip_address": device_info.get("ip_address"),
            "user_agent": device_info.get("user_agent"),
            "is_secure": True
        }
        
        # Store session in Redis
        if self.redis_client:
            try:
                session_key = f"session:{session_id}"
                self.redis_client.setex(
                    session_key, 
                    settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60, 
                    json.dumps(session_data)
                )
            except RedisError:
                pass
        
        # Store in memory as backup
        self.session_tokens[session_id] = session_data
        self.device_fingerprints[user_id] = device_fingerprint
        
        return session_data
    
    def _generate_device_fingerprint(self, device_info: Dict[str, Any]) -> str:
        """Generate device fingerprint for security"""
        fingerprint_data = {
            "user_agent": device_info.get("user_agent", ""),
            "accept_language": device_info.get("accept_language", ""),
            "accept_encoding": device_info.get("accept_encoding", ""),
            "screen_resolution": device_info.get("screen_resolution", ""),
            "timezone": device_info.get("timezone", ""),
            "platform": device_info.get("platform", "")
        }
        
        fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()
    
    def validate_session(self, session_id: str, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate session and check for anomalies"""
        session_data = None
        
        # Get session from Redis
        if self.redis_client:
            try:
                session_key = f"session:{session_id}"
                session_json = self.redis_client.get(session_key)
                if session_json:
                    session_data = json.loads(session_json)
            except RedisError:
                pass
        
        # Fallback to memory
        if not session_data and session_id in self.session_tokens:
            session_data = self.session_tokens[session_id]
        
        if not session_data:
            return {"valid": False, "reason": "Session not found"}
        
        # Check session expiration
        created_at = datetime.fromisoformat(session_data["created_at"])
        if datetime.utcnow() - created_at > timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES):
            return {"valid": False, "reason": "Session expired"}
        
        # Check device fingerprint
        current_fingerprint = self._generate_device_fingerprint(device_info)
        stored_fingerprint = session_data.get("device_fingerprint")
        
        if current_fingerprint != stored_fingerprint:
            # Log potential session hijacking
            log_security_event("potential_session_hijacking", {
                "user_id": session_data["user_id"],
                "session_id": session_id,
                "stored_fingerprint": stored_fingerprint,
                "current_fingerprint": current_fingerprint
            })
            return {"valid": False, "reason": "Device fingerprint mismatch"}
        
        # Update last activity
        session_data["last_activity"] = datetime.utcnow().isoformat()
        
        if self.redis_client:
            try:
                session_key = f"session:{session_id}"
                self.redis_client.setex(
                    session_key,
                    settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                    json.dumps(session_data)
                )
            except RedisError:
                pass
        
        return {"valid": True, "session_data": session_data}
    
    def create_enhanced_token(self, user_id: int, device_info: Dict[str, Any]) -> Dict[str, str]:
        """Create enhanced JWT token with additional security features"""
        # Create secure session first
        session_data = self.create_secure_session(user_id, device_info)
        
        # Create access token
        access_token_data = {
            "sub": user_id,
            "session_id": session_data["session_id"],
            "device_fingerprint": session_data["device_fingerprint"],
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        access_token = jwt.encode(
            access_token_data,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        # Create refresh token
        refresh_token_data = {
            "sub": user_id,
            "session_id": session_data["session_id"],
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        refresh_token = jwt.encode(
            refresh_token_data,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "session_id": session_data["session_id"],
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    def verify_enhanced_token(self, token: str, device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Verify enhanced JWT token with session validation"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            
            # Check token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Validate session
            session_id = payload.get("session_id")
            if not session_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="No session ID in token"
                )
            
            session_validation = self.validate_session(session_id, device_info)
            if not session_validation["valid"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Session validation failed: {session_validation['reason']}"
                )
            
            return {
                "valid": True,
                "user_id": payload.get("sub"),
                "session_data": session_validation["session_data"]
            }
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def detect_security_threats(self, request: Request, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Comprehensive security threat detection"""
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        
        threat_analysis = {
            "threat_level": "low",
            "threats_detected": [],
            "recommended_action": "allow"
        }
        
        # Check for blocked IP
        if self.threat_detector.is_ip_blocked(client_ip):
            threat_analysis["threat_level"] = "critical"
            threat_analysis["threats_detected"].append("Blocked IP address")
            threat_analysis["recommended_action"] = "block"
            return threat_analysis
        
        # Check for DDoS attack
        if self.threat_detector.detect_ddos_attack(client_ip):
            threat_analysis["threat_level"] = "high"
            threat_analysis["threats_detected"].append("DDoS attack detected")
            threat_analysis["recommended_action"] = "block"
            return threat_analysis
        
        # Check for brute force
        if user_id and self.threat_detector.detect_brute_force(f"user:{user_id}"):
            threat_analysis["threat_level"] = "high"
            threat_analysis["threats_detected"].append("Brute force attack detected")
            threat_analysis["recommended_action"] = "block"
            return threat_analysis
        
        # Check for anomalous behavior
        if user_id:
            request_data = {
                "endpoint": request.url.path,
                "method": request.method,
                "user_agent": user_agent,
                "client_ip": client_ip
            }
            
            anomaly_result = self.threat_detector.detect_anomalous_behavior(user_id, request_data)
            if anomaly_result["is_suspicious"]:
                threat_analysis["threat_level"] = "medium"
                threat_analysis["threats_detected"].extend(anomaly_result["anomalies"])
                threat_analysis["recommended_action"] = "monitor"
        
        return threat_analysis


# Global instances
advanced_security_manager = AdvancedSecurityManager()
threat_detector = ThreatDetector()


def enhanced_auth_required():
    """Enhanced authentication decorator with threat detection"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Perform threat detection
            threat_analysis = advanced_security_manager.detect_security_threats(request)
            
            if threat_analysis["recommended_action"] == "block":
                log_security_event("request_blocked", {
                    "ip": request.client.host,
                    "endpoint": request.url.path,
                    "threats": threat_analysis["threats_detected"]
                })
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Request blocked due to security threats"
                )
            
            # Continue with normal authentication
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def get_device_info(request: Request) -> Dict[str, Any]:
    """Extract device information from request"""
    return {
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent", ""),
        "accept_language": request.headers.get("accept-language", ""),
        "accept_encoding": request.headers.get("accept-encoding", ""),
        "platform": request.headers.get("sec-ch-ua-platform", ""),
        "timezone": request.headers.get("timezone", "")
    }


# Export functions
__all__ = [
    "AdvancedSecurityManager",
    "ThreatDetector",
    "advanced_security_manager",
    "threat_detector",
    "enhanced_auth_required",
    "get_device_info"
]