"""
Data Governance API
API endpoints for data governance and GDPR compliance
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.services.data_governance_engine import data_governance_engine, DataClassification, DataRetentionPolicy, ConsentStatus, DataSubjectRights

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class DataAssetRequest(BaseModel):
    """Data asset request model"""
    name: str
    description: str
    owner: str
    size_bytes: int
    location: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class DataSubjectRequest(BaseModel):
    """Data subject request model"""
    email: str
    name: str


class ConsentRequest(BaseModel):
    """Consent request model"""
    subject_id: str
    consent_granted: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class RightsRequest(BaseModel):
    """Data subject rights request model"""
    subject_id: str
    right_type: str
    request_data: Optional[Dict[str, Any]] = None


class DataBreachRequest(BaseModel):
    """Data breach request model"""
    description: str
    affected_subjects: int
    data_categories: List[str]
    consequences: str
    measures: List[str]
    reported_to_authority: bool = False
    reported_to_subjects: bool = False


# API Endpoints
@router.get("/status")
async def get_governance_status():
    """Get data governance system status"""
    try:
        summary = data_governance_engine.get_governance_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting governance status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assets")
async def register_data_asset(asset_request: DataAssetRequest):
    """Register a new data asset"""
    try:
        asset_data = {
            "name": asset_request.name,
            "description": asset_request.description,
            "owner": asset_request.owner,
            "size_bytes": asset_request.size_bytes,
            "location": asset_request.location,
            "tags": asset_request.tags or [],
            "metadata": asset_request.metadata or {}
        }
        
        asset = await data_governance_engine.register_data_asset(asset_data)
        
        return JSONResponse(content={
            "message": "Data asset registered successfully",
            "asset_id": asset.asset_id,
            "classification": asset.classification.value,
            "retention_policy": asset.retention_policy.value
        })
        
    except Exception as e:
        logger.error(f"Error registering data asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets")
async def get_data_assets():
    """Get all data assets"""
    try:
        assets = []
        for asset_id, asset in data_governance_engine.data_assets.items():
            assets.append({
                "asset_id": asset.asset_id,
                "name": asset.name,
                "description": asset.description,
                "classification": asset.classification.value,
                "retention_policy": asset.retention_policy.value,
                "owner": asset.owner,
                "created_at": asset.created_at.isoformat(),
                "last_accessed": asset.last_accessed.isoformat(),
                "size_bytes": asset.size_bytes,
                "location": asset.location,
                "tags": asset.tags,
                "metadata": asset.metadata
            })
            
        return JSONResponse(content={"assets": assets})
        
    except Exception as e:
        logger.error(f"Error getting data assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subjects")
async def register_data_subject(subject_request: DataSubjectRequest):
    """Register a new data subject"""
    try:
        subject_data = {
            "email": subject_request.email,
            "name": subject_request.name
        }
        
        subject = await data_governance_engine.register_data_subject(subject_data)
        
        return JSONResponse(content={
            "message": "Data subject registered successfully",
            "subject_id": subject.subject_id,
            "consent_status": subject.consent_status.value
        })
        
    except Exception as e:
        logger.error(f"Error registering data subject: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subjects")
async def get_data_subjects():
    """Get all data subjects"""
    try:
        subjects = []
        for subject_id, subject in data_governance_engine.data_subjects.items():
            subjects.append({
                "subject_id": subject.subject_id,
                "email": subject.email,
                "name": subject.name,
                "consent_status": subject.consent_status.value,
                "consent_granted_at": subject.consent_granted_at.isoformat() if subject.consent_granted_at else None,
                "consent_withdrawn_at": subject.consent_withdrawn_at.isoformat() if subject.consent_withdrawn_at else None,
                "data_retention_until": subject.data_retention_until.isoformat() if subject.data_retention_until else None,
                "rights_requests": subject.rights_requests,
                "created_at": subject.created_at.isoformat()
            })
            
        return JSONResponse(content={"subjects": subjects})
        
    except Exception as e:
        logger.error(f"Error getting data subjects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consent")
async def process_consent_request(consent_request: ConsentRequest):
    """Process a consent request"""
    try:
        consent_data = {
            "consent_granted": consent_request.consent_granted,
            "ip_address": consent_request.ip_address,
            "user_agent": consent_request.user_agent
        }
        
        consent_granted = await data_governance_engine.process_consent_request(
            consent_request.subject_id,
            consent_data
        )
        
        return JSONResponse(content={
            "message": "Consent request processed successfully",
            "consent_granted": consent_granted
        })
        
    except Exception as e:
        logger.error(f"Error processing consent request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rights")
async def process_rights_request(rights_request: RightsRequest):
    """Process a data subject rights request"""
    try:
        right_type = DataSubjectRights(rights_request.right_type)
        
        result = await data_governance_engine.process_data_subject_rights_request(
            rights_request.subject_id,
            right_type,
            rights_request.request_data or {}
        )
        
        return JSONResponse(content={
            "message": "Rights request processed successfully",
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error processing rights request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/breaches")
async def report_data_breach(breach_request: DataBreachRequest):
    """Report a data breach"""
    try:
        breach_data = {
            "description": breach_request.description,
            "affected_subjects": breach_request.affected_subjects,
            "data_categories": breach_request.data_categories,
            "consequences": breach_request.consequences,
            "measures": breach_request.measures,
            "reported_to_authority": breach_request.reported_to_authority,
            "reported_to_subjects": breach_request.reported_to_subjects
        }
        
        breach = await data_governance_engine.report_data_breach(breach_data)
        
        return JSONResponse(content={
            "message": "Data breach reported successfully",
            "breach_id": breach.breach_id,
            "discovered_at": breach.discovered_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error reporting data breach: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/breaches")
async def get_data_breaches():
    """Get all data breaches"""
    try:
        breaches = []
        for breach in data_governance_engine.data_breaches:
            breaches.append({
                "breach_id": breach.breach_id,
                "description": breach.description,
                "affected_data_subjects": breach.affected_data_subjects,
                "data_categories": breach.data_categories,
                "likely_consequences": breach.likely_consequences,
                "measures_taken": breach.measures_taken,
                "reported_to_authority": breach.reported_to_authority,
                "reported_to_subjects": breach.reported_to_subjects,
                "discovered_at": breach.discovered_at.isoformat(),
                "reported_at": breach.reported_at.isoformat() if breach.reported_at else None,
                "resolved_at": breach.resolved_at.isoformat() if breach.resolved_at else None
            })
            
        return JSONResponse(content={"breaches": breaches})
        
    except Exception as e:
        logger.error(f"Error getting data breaches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compliance")
async def get_compliance_status():
    """Get GDPR compliance status"""
    try:
        summary = data_governance_engine.get_governance_summary()
        
        # Calculate compliance metrics
        total_subjects = len(data_governance_engine.data_subjects)
        subjects_with_consent = len([
            s for s in data_governance_engine.data_subjects.values()
            if s.consent_status == ConsentStatus.GRANTED
        ])
        consent_rate = (subjects_with_consent / total_subjects * 100) if total_subjects > 0 else 0
        
        active_breaches = len([
            b for b in data_governance_engine.data_breaches
            if not b.resolved_at
        ])
        
        compliance_status = {
            "gdpr_compliant": active_breaches == 0 and consent_rate >= 80,
            "consent_rate": consent_rate,
            "active_breaches": active_breaches,
            "total_data_assets": len(data_governance_engine.data_assets),
            "total_data_subjects": total_subjects,
            "data_retention_compliant": True,  # Simplified
            "privacy_by_design_enabled": data_governance_engine.config["privacy_by_design_enabled"],
            "audit_logging_enabled": data_governance_engine.config["audit_logging_enabled"]
        }
        
        return JSONResponse(content=compliance_status)
        
    except Exception as e:
        logger.error(f"Error getting compliance status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_governance_dashboard():
    """Get data governance dashboard data"""
    try:
        summary = data_governance_engine.get_governance_summary()
        
        # Get recent activities
        recent_assets = list(data_governance_engine.data_assets.values())[-5:]
        recent_subjects = list(data_governance_engine.data_subjects.values())[-5:]
        recent_breaches = data_governance_engine.data_breaches[-5:]
        
        dashboard_data = {
            "summary": summary,
            "recent_assets": [
                {
                    "name": asset.name,
                    "classification": asset.classification.value,
                    "created_at": asset.created_at.isoformat()
                }
                for asset in recent_assets
            ],
            "recent_subjects": [
                {
                    "name": subject.name,
                    "email": subject.email,
                    "consent_status": subject.consent_status.value,
                    "created_at": subject.created_at.isoformat()
                }
                for subject in recent_subjects
            ],
            "recent_breaches": [
                {
                    "breach_id": breach.breach_id,
                    "description": breach.description,
                    "affected_subjects": breach.affected_data_subjects,
                    "discovered_at": breach.discovered_at.isoformat()
                }
                for breach in recent_breaches
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting governance dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_governance():
    """Start data governance engine"""
    try:
        await data_governance_engine.start_governance_engine()
        return JSONResponse(content={"message": "Data governance engine started"})
        
    except Exception as e:
        logger.error(f"Error starting governance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_governance():
    """Stop data governance engine"""
    try:
        await data_governance_engine.stop_governance_engine()
        return JSONResponse(content={"message": "Data governance engine stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping governance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
