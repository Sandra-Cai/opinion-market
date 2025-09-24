"""
Security Audit API Endpoints
Provides comprehensive security scanning and vulnerability management endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from app.core.security_audit import security_auditor, SecurityLevel, VulnerabilityType
from app.core.auth import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/scan/start")
async def start_security_scan(
    target_paths: Optional[List[str]] = None,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Start comprehensive security scan"""
    try:
        # Run scan in background
        if background_tasks:
            background_tasks.add_task(security_auditor.run_comprehensive_scan, target_paths)
            return {
                "status": "started",
                "message": "Security scan started in background",
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Run scan synchronously
            result = await security_auditor.run_comprehensive_scan(target_paths)
            return {
                "status": "completed",
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error starting security scan: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start security scan"
        )


@router.get("/scan/status")
async def get_scan_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current scan status"""
    try:
        return {
            "status": "success",
            "data": {
                "scan_in_progress": security_auditor.scan_in_progress,
                "last_scan": security_auditor.metrics.last_scan.isoformat() if security_auditor.metrics.last_scan else None,
                "scan_duration": security_auditor.metrics.scan_duration
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting scan status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get scan status"
        )


@router.get("/vulnerabilities")
async def get_vulnerabilities(
    severity: Optional[str] = None,
    status: Optional[str] = None,
    vuln_type: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get security vulnerabilities with optional filtering"""
    try:
        # Parse severity filter
        severity_filter = None
        if severity:
            try:
                severity_filter = SecurityLevel(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid severity level: {severity}. Valid options: low, medium, high, critical"
                )
        
        # Parse vulnerability type filter
        vuln_type_filter = None
        if vuln_type:
            try:
                vuln_type_filter = VulnerabilityType(vuln_type.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid vulnerability type: {vuln_type}"
                )
        
        vulnerabilities = security_auditor.get_vulnerabilities(severity_filter, status)
        
        # Apply vulnerability type filter
        if vuln_type_filter:
            vulnerabilities = [v for v in vulnerabilities if v["type"] == vuln_type_filter.value]
        
        return {
            "status": "success",
            "data": {
                "vulnerabilities": vulnerabilities,
                "count": len(vulnerabilities),
                "filters": {
                    "severity": severity,
                    "status": status,
                    "vuln_type": vuln_type
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting vulnerabilities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve vulnerabilities"
        )


@router.get("/vulnerabilities/{vuln_id}")
async def get_vulnerability_details(
    vuln_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed information about a specific vulnerability"""
    try:
        if vuln_id not in security_auditor.vulnerabilities:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Vulnerability not found"
            )
        
        vulnerability = security_auditor.vulnerabilities[vuln_id]
        
        return {
            "status": "success",
            "data": {
                "id": vulnerability.id,
                "type": vulnerability.type.value,
                "severity": vulnerability.severity.value,
                "title": vulnerability.title,
                "description": vulnerability.description,
                "file_path": vulnerability.file_path,
                "line_number": vulnerability.line_number,
                "code_snippet": vulnerability.code_snippet,
                "recommendation": vulnerability.recommendation,
                "cwe_id": vulnerability.cwe_id,
                "owasp_category": vulnerability.owasp_category,
                "detected_at": vulnerability.detected_at.isoformat(),
                "status": vulnerability.status,
                "false_positive": vulnerability.false_positive
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting vulnerability details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve vulnerability details"
        )


@router.put("/vulnerabilities/{vuln_id}/status")
async def update_vulnerability_status(
    vuln_id: str,
    new_status: str,
    false_positive: Optional[bool] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Update vulnerability status"""
    try:
        if vuln_id not in security_auditor.vulnerabilities:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Vulnerability not found"
            )
        
        valid_statuses = ["open", "fixed", "false_positive", "accepted_risk"]
        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {new_status}. Valid options: {', '.join(valid_statuses)}"
            )
        
        vulnerability = security_auditor.vulnerabilities[vuln_id]
        vulnerability.status = new_status
        
        if false_positive is not None:
            vulnerability.false_positive = false_positive
        
        # Recalculate metrics
        security_auditor._calculate_metrics()
        
        logger.info(f"Vulnerability {vuln_id} status updated to {new_status} by user {current_user.id}")
        
        return {
            "status": "success",
            "message": f"Vulnerability status updated to {new_status}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating vulnerability status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update vulnerability status"
        )


@router.get("/metrics")
async def get_security_metrics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get security metrics and statistics"""
    try:
        metrics = security_auditor._get_metrics_dict()
        
        # Add additional calculated metrics
        total_vulns = metrics["total_vulnerabilities"]
        if total_vulns > 0:
            metrics["risk_score"] = round(
                (metrics["critical_vulnerabilities"] * 4 + 
                 metrics["high_vulnerabilities"] * 3 + 
                 metrics["medium_vulnerabilities"] * 2 + 
                 metrics["low_vulnerabilities"] * 1) / total_vulns, 2
            )
        else:
            metrics["risk_score"] = 0.0
        
        return {
            "status": "success",
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting security metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security metrics"
        )


@router.get("/compliance")
async def get_compliance_report(
    standard: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get security compliance report"""
    try:
        compliance_report = security_auditor.get_compliance_report()
        
        if standard:
            if standard not in compliance_report:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Compliance standard '{standard}' not found"
                )
            compliance_report = {standard: compliance_report[standard]}
        
        return {
            "status": "success",
            "data": compliance_report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting compliance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve compliance report"
        )


@router.get("/dashboard")
async def get_security_dashboard(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get comprehensive security dashboard data"""
    try:
        # Get recent vulnerabilities
        recent_vulnerabilities = security_auditor.get_vulnerabilities()[:10]
        
        # Get metrics
        metrics = security_auditor._get_metrics_dict()
        
        # Get compliance summary
        compliance_report = security_auditor.get_compliance_report()
        compliance_summary = {}
        for standard, data in compliance_report.items():
            compliance_summary[standard] = {
                "total_requirements": data["total_requirements"],
                "compliance_percentage": round(
                    (data["compliant"] / data["total_requirements"]) * 100, 2
                ) if data["total_requirements"] > 0 else 0
            }
        
        # Calculate overall security score
        risk_score = metrics.get("risk_score", 0)
        compliance_score = metrics.get("compliance_score", 0)
        overall_score = max(0, 100 - (risk_score * 10) + (compliance_score * 0.1))
        
        return {
            "status": "success",
            "data": {
                "overall_security_score": round(overall_score, 2),
                "risk_score": risk_score,
                "compliance_score": compliance_score,
                "metrics": metrics,
                "recent_vulnerabilities": recent_vulnerabilities,
                "compliance_summary": compliance_summary,
                "scan_status": {
                    "in_progress": security_auditor.scan_in_progress,
                    "last_scan": security_auditor.metrics.last_scan.isoformat() if security_auditor.metrics.last_scan else None
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting security dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security dashboard"
        )


@router.get("/standards")
async def get_security_standards(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get available security standards and requirements"""
    try:
        return {
            "status": "success",
            "data": {
                "standards": security_auditor.compliance_standards,
                "vulnerability_types": [vtype.value for vtype in VulnerabilityType],
                "severity_levels": [level.value for level in SecurityLevel]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting security standards: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve security standards"
        )


@router.post("/vulnerabilities/{vuln_id}/comment")
async def add_vulnerability_comment(
    vuln_id: str,
    comment: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Add comment to vulnerability (placeholder for future implementation)"""
    try:
        if vuln_id not in security_auditor.vulnerabilities:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Vulnerability not found"
            )
        
        # This would typically store comments in a database
        # For now, we'll just log the comment
        logger.info(f"Comment added to vulnerability {vuln_id} by user {current_user.id}: {comment}")
        
        return {
            "status": "success",
            "message": "Comment added successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error adding vulnerability comment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add comment"
        )
