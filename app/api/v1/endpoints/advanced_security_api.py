"""
Advanced Security API Endpoints
API endpoints for the Advanced Security V2 system
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.core.advanced_security_v2 import (
    advanced_security_v2,
    SecurityThreat,
    SecurityPolicy,
    ThreatLevel,
    SecurityEvent
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class SecurityThreatResponse(BaseModel):
    """Security threat response model"""
    threat_id: str
    threat_type: str
    threat_level: str
    source_ip: str
    user_id: Optional[str]
    description: str
    details: Dict[str, Any]
    detected_at: datetime
    resolved: bool
    resolved_at: Optional[datetime]
    mitigation_action: Optional[str]


class SecurityPolicyRequest(BaseModel):
    """Security policy request model"""
    policy_id: str
    name: str
    description: str
    enabled: bool = True
    conditions: Dict[str, Any]
    actions: List[str]
    severity: str
    cooldown_period: int = 300


class SecurityPolicyResponse(BaseModel):
    """Security policy response model"""
    policy_id: str
    name: str
    description: str
    enabled: bool
    conditions: Dict[str, Any]
    actions: List[str]
    severity: str
    cooldown_period: int
    created_at: datetime


class RequestAnalysisRequest(BaseModel):
    """Request analysis request model"""
    source_ip: str
    user_id: Optional[str] = None
    method: str
    path: str
    headers: Dict[str, str] = {}
    body: str = ""
    query_params: Dict[str, Any] = {}


class RequestAnalysisResponse(BaseModel):
    """Request analysis response model"""
    threat_detected: bool
    threat_level: str
    threat_type: Optional[str]
    action: str
    message: str
    analysis_time: float


class SecuritySummaryResponse(BaseModel):
    """Security summary response model"""
    timestamp: str
    security_active: bool
    recent_threats: int
    threats_by_level: Dict[str, int]
    blocked_ips: int
    suspicious_ips: int
    active_policies: int
    total_policies: int
    stats: Dict[str, Any]
    config: Dict[str, Any]


class IPBlockRequest(BaseModel):
    """IP block request model"""
    ip_address: str
    reason: str
    duration: Optional[int] = 3600


class IPUnblockRequest(BaseModel):
    """IP unblock request model"""
    ip_address: str


# Security middleware dependency
async def get_security_analysis(request: Request) -> Dict[str, Any]:
    """Get security analysis for the current request"""
    try:
        # Extract request information
        request_data = {
            "source_ip": request.client.host if request.client else "unknown",
            "method": request.method,
            "path": request.url.path,
            "headers": dict(request.headers),
            "query_params": dict(request.query_params)
        }
        
        # Analyze request
        analysis = await advanced_security_v2.analyze_request(request_data)
        
        # Log security events
        if analysis.get("threat_detected"):
            logger.warning(f"Security threat detected: {analysis}")
            
        return analysis
        
    except Exception as e:
        logger.error(f"Error in security analysis: {e}")
        return {
            "threat_detected": False,
            "threat_level": "low",
            "action": "allow",
            "message": "Analysis error"
        }


# API Endpoints
@router.get("/status", response_model=SecuritySummaryResponse)
async def get_security_status():
    """Get current security system status"""
    try:
        summary = advanced_security_v2.get_security_summary()
        return SecuritySummaryResponse(**summary)
    except Exception as e:
        logger.error(f"Error getting security status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-request", response_model=RequestAnalysisResponse)
async def analyze_request(
    request_data: RequestAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Analyze a request for security threats"""
    try:
        import time
        start_time = time.time()
        
        # Convert to dict
        request_dict = request_data.dict()
        
        # Analyze request
        analysis = await advanced_security_v2.analyze_request(request_dict)
        
        analysis_time = time.time() - start_time
        
        return RequestAnalysisResponse(
            threat_detected=analysis.get("threat_detected", False),
            threat_level=analysis.get("threat_level", "low"),
            threat_type=analysis.get("threat_type"),
            action=analysis.get("action", "allow"),
            message=analysis.get("message", ""),
            analysis_time=analysis_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threats", response_model=List[SecurityThreatResponse])
async def get_security_threats(
    limit: int = 100,
    threat_level: Optional[str] = None,
    resolved: Optional[bool] = None
):
    """Get security threats"""
    try:
        threats = advanced_security_v2.threats
        
        # Apply filters
        if threat_level:
            threats = [t for t in threats if t.threat_level.value == threat_level]
        if resolved is not None:
            threats = [t for t in threats if t.resolved == resolved]
            
        # Limit results
        threats = threats[-limit:] if limit > 0 else threats
        
        # Convert to response format
        threat_responses = []
        for threat in threats:
            threat_responses.append(SecurityThreatResponse(
                threat_id=threat.threat_id,
                threat_type=threat.threat_type.value,
                threat_level=threat.threat_level.value,
                source_ip=threat.source_ip,
                user_id=threat.user_id,
                description=threat.description,
                details=threat.details,
                detected_at=threat.detected_at,
                resolved=threat.resolved,
                resolved_at=threat.resolved_at,
                mitigation_action=threat.mitigation_action
            ))
            
        return threat_responses
        
    except Exception as e:
        logger.error(f"Error getting security threats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threats/{threat_id}", response_model=SecurityThreatResponse)
async def get_security_threat(threat_id: str):
    """Get a specific security threat"""
    try:
        threat = next(
            (t for t in advanced_security_v2.threats if t.threat_id == threat_id),
            None
        )
        
        if not threat:
            raise HTTPException(status_code=404, detail="Threat not found")
            
        return SecurityThreatResponse(
            threat_id=threat.threat_id,
            threat_type=threat.threat_type.value,
            threat_level=threat.threat_level.value,
            source_ip=threat.source_ip,
            user_id=threat.user_id,
            description=threat.description,
            details=threat.details,
            detected_at=threat.detected_at,
            resolved=threat.resolved,
            resolved_at=threat.resolved_at,
            mitigation_action=threat.mitigation_action
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting security threat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threats/{threat_id}/resolve")
async def resolve_threat(threat_id: str):
    """Mark a threat as resolved"""
    try:
        threat = next(
            (t for t in advanced_security_v2.threats if t.threat_id == threat_id),
            None
        )
        
        if not threat:
            raise HTTPException(status_code=404, detail="Threat not found")
            
        threat.resolved = True
        threat.resolved_at = datetime.now()
        
        return {"message": "Threat resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving threat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/policies", response_model=List[SecurityPolicyResponse])
async def get_security_policies():
    """Get all security policies"""
    try:
        policies = []
        for policy in advanced_security_v2.security_policies.values():
            policies.append(SecurityPolicyResponse(
                policy_id=policy.policy_id,
                name=policy.name,
                description=policy.description,
                enabled=policy.enabled,
                conditions=policy.conditions,
                actions=policy.actions,
                severity=policy.severity.value,
                cooldown_period=policy.cooldown_period,
                created_at=policy.created_at
            ))
            
        return policies
        
    except Exception as e:
        logger.error(f"Error getting security policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/policies", response_model=SecurityPolicyResponse)
async def create_security_policy(policy_request: SecurityPolicyRequest):
    """Create a new security policy"""
    try:
        # Validate threat level
        try:
            threat_level = ThreatLevel(policy_request.severity)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid threat level")
            
        # Create policy
        policy = SecurityPolicy(
            policy_id=policy_request.policy_id,
            name=policy_request.name,
            description=policy_request.description,
            enabled=policy_request.enabled,
            conditions=policy_request.conditions,
            actions=policy_request.actions,
            severity=threat_level,
            cooldown_period=policy_request.cooldown_period
        )
        
        # Add policy
        advanced_security_v2.add_security_policy(policy)
        
        return SecurityPolicyResponse(
            policy_id=policy.policy_id,
            name=policy.name,
            description=policy.description,
            enabled=policy.enabled,
            conditions=policy.conditions,
            actions=policy.actions,
            severity=policy.severity.value,
            cooldown_period=policy.cooldown_period,
            created_at=policy.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating security policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/policies/{policy_id}", response_model=SecurityPolicyResponse)
async def update_security_policy(
    policy_id: str,
    policy_request: SecurityPolicyRequest
):
    """Update an existing security policy"""
    try:
        if policy_id not in advanced_security_v2.security_policies:
            raise HTTPException(status_code=404, detail="Policy not found")
            
        # Validate threat level
        try:
            threat_level = ThreatLevel(policy_request.severity)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid threat level")
            
        # Update policy
        advanced_security_v2.update_security_policy(
            policy_id,
            name=policy_request.name,
            description=policy_request.description,
            enabled=policy_request.enabled,
            conditions=policy_request.conditions,
            actions=policy_request.actions,
            severity=threat_level,
            cooldown_period=policy_request.cooldown_period
        )
        
        policy = advanced_security_v2.security_policies[policy_id]
        
        return SecurityPolicyResponse(
            policy_id=policy.policy_id,
            name=policy.name,
            description=policy.description,
            enabled=policy.enabled,
            conditions=policy.conditions,
            actions=policy.actions,
            severity=policy.severity.value,
            cooldown_period=policy.cooldown_period,
            created_at=policy.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating security policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/policies/{policy_id}")
async def delete_security_policy(policy_id: str):
    """Delete a security policy"""
    try:
        if policy_id not in advanced_security_v2.security_policies:
            raise HTTPException(status_code=404, detail="Policy not found")
            
        advanced_security_v2.remove_security_policy(policy_id)
        
        return {"message": "Policy deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting security policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blocked-ips")
async def get_blocked_ips():
    """Get list of blocked IP addresses"""
    try:
        blocked_ips = list(advanced_security_v2.blocked_ips)
        return {
            "blocked_ips": blocked_ips,
            "count": len(blocked_ips)
        }
        
    except Exception as e:
        logger.error(f"Error getting blocked IPs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/block-ip")
async def block_ip(block_request: IPBlockRequest):
    """Block an IP address"""
    try:
        await advanced_security_v2._block_ip(block_request.ip_address, block_request.reason)
        
        return {
            "message": f"IP {block_request.ip_address} blocked successfully",
            "reason": block_request.reason,
            "duration": block_request.duration
        }
        
    except Exception as e:
        logger.error(f"Error blocking IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unblock-ip")
async def unblock_ip(unblock_request: IPUnblockRequest):
    """Unblock an IP address"""
    try:
        advanced_security_v2.unblock_ip(unblock_request.ip_address)
        
        return {
            "message": f"IP {unblock_request.ip_address} unblocked successfully"
        }
        
    except Exception as e:
        logger.error(f"Error unblocking IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suspicious-ips")
async def get_suspicious_ips():
    """Get list of suspicious IP addresses"""
    try:
        suspicious_ips = dict(advanced_security_v2.suspicious_ips)
        return {
            "suspicious_ips": suspicious_ips,
            "count": len(suspicious_ips)
        }
        
    except Exception as e:
        logger.error(f"Error getting suspicious IPs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_security_stats():
    """Get security statistics"""
    try:
        stats = advanced_security_v2.security_stats
        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting security stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-monitoring")
async def start_security_monitoring():
    """Start security monitoring"""
    try:
        await advanced_security_v2.start_security_monitoring()
        return {"message": "Security monitoring started"}
        
    except Exception as e:
        logger.error(f"Error starting security monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop-monitoring")
async def stop_security_monitoring():
    """Stop security monitoring"""
    try:
        await advanced_security_v2.stop_security_monitoring()
        return {"message": "Security monitoring stopped"}
        
    except Exception as e:
        logger.error(f"Error stopping security monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_security_config():
    """Get security configuration"""
    try:
        return {
            "config": advanced_security_v2.config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting security config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config")
async def update_security_config(config: Dict[str, Any]):
    """Update security configuration"""
    try:
        # Validate configuration
        valid_keys = {
            "max_login_attempts", "login_window", "rate_limit_window",
            "max_requests_per_minute", "suspicious_threshold", "block_duration",
            "ai_detection_enabled", "auto_block_enabled"
        }
        
        for key in config.keys():
            if key not in valid_keys:
                raise HTTPException(status_code=400, detail=f"Invalid config key: {key}")
                
        # Update configuration
        advanced_security_v2.config.update(config)
        
        return {
            "message": "Security configuration updated",
            "config": advanced_security_v2.config
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating security config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Security middleware endpoint
@router.get("/middleware/analyze")
async def analyze_current_request(
    security_analysis: Dict[str, Any] = Depends(get_security_analysis)
):
    """Analyze the current request for security threats"""
    return security_analysis
