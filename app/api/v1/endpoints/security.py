"""
Security API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, List
import time

from app.core.auth import get_current_user
from app.models.user import User
from app.core.security_manager import security_manager

router = APIRouter()


@router.get("/metrics")
async def get_security_metrics(current_user: User = Depends(get_current_user)):
    """Get security metrics"""
    try:
        metrics = security_manager.get_metrics()
        return {
            "success": True,
            "data": metrics,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security metrics: {str(e)}")


@router.get("/events")
async def get_security_events(
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Get recent security events"""
    try:
        events = list(security_manager.security_events)[-limit:]
        return {
            "success": True,
            "data": events,
            "count": len(events),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security events: {str(e)}")


@router.post("/check-threats")
async def check_threats(
    request_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Check for security threats in data"""
    try:
        threats = await security_manager.detect_threats(request_data)
        return {
            "success": True,
            "threats_detected": len(threats),
            "threats": threats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check threats: {str(e)}")


@router.get("/blocked-ips")
async def get_blocked_ips(current_user: User = Depends(get_current_user)):
    """Get list of blocked IP addresses"""
    try:
        blocked_ips = list(security_manager.blocked_ips)
        return {
            "success": True,
            "data": blocked_ips,
            "count": len(blocked_ips),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get blocked IPs: {str(e)}")


@router.post("/block-ip")
async def block_ip(
    ip_address: str,
    current_user: User = Depends(get_current_user)
):
    """Block an IP address"""
    try:
        security_manager.blocked_ips.add(ip_address)
        return {
            "success": True,
            "message": f"IP {ip_address} blocked successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to block IP: {str(e)}")


@router.delete("/unblock-ip/{ip_address}")
async def unblock_ip(
    ip_address: str,
    current_user: User = Depends(get_current_user)
):
    """Unblock an IP address"""
    try:
        security_manager.blocked_ips.discard(ip_address)
        return {
            "success": True,
            "message": f"IP {ip_address} unblocked successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unblock IP: {str(e)}")


@router.get("/rate-limit-status")
async def get_rate_limit_status(
    ip_address: str,
    limit_type: str = "api",
    current_user: User = Depends(get_current_user)
):
    """Get rate limit status for an IP"""
    try:
        allowed = await security_manager.check_rate_limit(ip_address, limit_type)
        return {
            "success": True,
            "ip_address": ip_address,
            "limit_type": limit_type,
            "allowed": allowed,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check rate limit: {str(e)}")


@router.get("/security-status")
async def get_security_status(current_user: User = Depends(get_current_user)):
    """Get overall security status"""
    try:
        metrics = security_manager.get_metrics()
        recent_events = list(security_manager.security_events)[-10:]
        
        # Calculate security score
        security_score = 100
        if metrics["events_count"] > 100:
            security_score -= 20
        if metrics["blocked_ips"] > 10:
            security_score -= 10
        
        status = {
            "security_score": max(security_score, 0),
            "status": "healthy" if security_score > 80 else "warning" if security_score > 60 else "critical",
            "metrics": metrics,
            "recent_events": recent_events,
            "blocked_ips_count": len(security_manager.blocked_ips)
        }
        
        return {
            "success": True,
            "data": status,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get security status: {str(e)}")