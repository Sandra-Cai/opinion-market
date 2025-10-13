"""
Security Monitoring API Endpoints
Provides comprehensive security monitoring and threat detection capabilities
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.core.database import get_db
from app.core.security import get_current_user, get_current_active_user
from app.core.advanced_security import advanced_security_manager, threat_detector
from app.core.logging import log_security_event
from app.models.user import User

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/threats/analysis")
async def get_threat_analysis(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive threat analysis for the current request"""
    try:
        # Perform threat detection
        threat_analysis = advanced_security_manager.detect_security_threats(
            request, current_user.id
        )
        
        # Log the analysis
        log_security_event("threat_analysis_requested", {
            "user_id": current_user.id,
            "threat_level": threat_analysis["threat_level"],
            "threats_detected": threat_analysis["threats_detected"]
        })
        
        return {
            "threat_analysis": threat_analysis,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": current_user.id
        }
        
    except Exception as e:
        logger.error(f"Error in threat analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform threat analysis"
        )


@router.get("/sessions/active")
async def get_active_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get active sessions for the current user"""
    try:
        active_sessions = []
        
        # Get sessions from Redis
        if advanced_security_manager.redis_client:
            try:
                # Search for user sessions
                pattern = f"session:*"
                session_keys = advanced_security_manager.redis_client.keys(pattern)
                
                for key in session_keys:
                    session_data = advanced_security_manager.redis_client.get(key)
                    if session_data:
                        import json
                        session = json.loads(session_data)
                        if session.get("user_id") == current_user.id:
                            # Remove sensitive data
                            session_info = {
                                "session_id": session.get("session_id"),
                                "created_at": session.get("created_at"),
                                "last_activity": session.get("last_activity"),
                                "ip_address": session.get("ip_address"),
                                "user_agent": session.get("user_agent", "")[:100]  # Truncate
                            }
                            active_sessions.append(session_info)
            except Exception as e:
                logger.warning(f"Error retrieving sessions from Redis: {e}")
        
        return {
            "active_sessions": active_sessions,
            "total_sessions": len(active_sessions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting active sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active sessions"
        )


@router.post("/sessions/revoke")
async def revoke_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Revoke a specific session"""
    try:
        # Verify session belongs to user
        if advanced_security_manager.redis_client:
            try:
                session_key = f"session:{session_id}"
                session_data = advanced_security_manager.redis_client.get(session_key)
                
                if session_data:
                    import json
                    session = json.loads(session_data)
                    if session.get("user_id") != current_user.id:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Session does not belong to current user"
                        )
                    
                    # Revoke session
                    advanced_security_manager.redis_client.delete(session_key)
                    
                    # Log the revocation
                    log_security_event("session_revoked", {
                        "user_id": current_user.id,
                        "session_id": session_id,
                        "revoked_by": "user"
                    })
                    
                    return {"message": "Session revoked successfully"}
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Session not found"
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error revoking session: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to revoke session"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session management not available"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in session revocation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke session"
        )


@router.get("/security/events")
async def get_security_events(
    limit: int = Query(50, ge=1, le=1000),
    event_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get security events (admin only)"""
    try:
        # Check if user has admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Get security events from Redis
        security_events = []
        
        if advanced_security_manager.redis_client:
            try:
                # Get recent security events
                events_key = "security_events"
                events = advanced_security_manager.redis_client.lrange(events_key, 0, limit - 1)
                
                for event_json in events:
                    try:
                        import json
                        event = json.loads(event_json)
                        
                        # Apply filters
                        if event_type and event.get("type") != event_type:
                            continue
                        if severity and event.get("severity") != severity:
                            continue
                        
                        security_events.append(event)
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logger.warning(f"Error retrieving security events: {e}")
        
        return {
            "security_events": security_events,
            "total_events": len(security_events),
            "filters": {
                "event_type": event_type,
                "severity": severity,
                "limit": limit
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting security events: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security events"
        )


@router.get("/security/stats")
async def get_security_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get security statistics (admin only)"""
    try:
        # Check if user has admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        stats = {
            "threat_detection": {
                "blocked_ips": len(threat_detector.known_bad_ips),
                "suspicious_patterns": len(threat_detector.suspicious_patterns),
                "attack_attempts": len(threat_detector.attack_attempts)
            },
            "sessions": {
                "active_sessions": len(advanced_security_manager.session_tokens),
                "device_fingerprints": len(advanced_security_manager.device_fingerprints)
            },
            "security_events": {
                "total_events": len(advanced_security_manager.security_events),
                "recent_events": len([
                    event for event in advanced_security_manager.security_events
                    if datetime.fromisoformat(event.get("timestamp", "1970-01-01")) > 
                       datetime.utcnow() - timedelta(hours=24)
                ])
            },
            "redis_status": {
                "connected": advanced_security_manager.redis_client is not None,
                "available": advanced_security_manager.redis_client.ping() if advanced_security_manager.redis_client else False
            }
        }
        
        return {
            "security_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting security stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security statistics"
        )


@router.post("/security/block-ip")
async def block_ip_address(
    ip_address: str,
    reason: str,
    duration_hours: int = Query(24, ge=1, le=168),  # Max 1 week
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Block an IP address (admin only)"""
    try:
        # Check if user has admin permissions
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Validate IP address
        import ipaddress
        try:
            ipaddress.ip_address(ip_address)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid IP address format"
            )
        
        # Block the IP
        duration_seconds = duration_hours * 3600
        threat_detector.known_bad_ips.add(ip_address)
        
        if advanced_security_manager.redis_client:
            try:
                advanced_security_manager.redis_client.setex(
                    f"blocked_ip:{ip_address}",
                    duration_seconds,
                    reason
                )
            except Exception as e:
                logger.warning(f"Error blocking IP in Redis: {e}")
        
        # Log the action
        log_security_event("ip_blocked_admin", {
            "ip_address": ip_address,
            "reason": reason,
            "duration_hours": duration_hours,
            "blocked_by": current_user.id
        })
        
        return {
            "message": f"IP address {ip_address} blocked successfully",
            "duration_hours": duration_hours,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error blocking IP address: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to block IP address"
        )


@router.get("/security/health")
async def get_security_health():
    """Get security system health status"""
    try:
        health_status = {
            "status": "healthy",
            "components": {
                "threat_detector": "operational",
                "security_manager": "operational",
                "redis_connection": "unknown",
                "session_management": "unknown"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check Redis connection
        if advanced_security_manager.redis_client:
            try:
                advanced_security_manager.redis_client.ping()
                health_status["components"]["redis_connection"] = "healthy"
                health_status["components"]["session_management"] = "healthy"
            except Exception:
                health_status["components"]["redis_connection"] = "unhealthy"
                health_status["components"]["session_management"] = "degraded"
                health_status["status"] = "degraded"
        else:
            health_status["components"]["redis_connection"] = "disabled"
            health_status["components"]["session_management"] = "disabled"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error checking security health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
