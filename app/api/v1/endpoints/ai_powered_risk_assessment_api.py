"""
API endpoints for AI-Powered Risk Assessment Engine
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

from app.services.ai_powered_risk_assessment_engine import ai_powered_risk_assessment_engine, RiskType, RiskLevel, RiskCategory

logger = logging.getLogger(__name__)

router = APIRouter()

class RiskAssessmentRequestModel(BaseModel):
    entity_id: str
    entity_type: str

class RiskAssessmentModel(BaseModel):
    assessment_id: str
    entity_id: str
    entity_type: str
    overall_risk_score: float
    risk_level: str
    risk_category: str
    confidence_score: float
    assessment_date: str
    valid_until: str
    recommendations: List[str]
    mitigation_actions: List[str]
    risk_factors: List[Dict[str, Any]]

class RiskAlertModel(BaseModel):
    alert_id: str
    assessment_id: str
    risk_type: str
    severity: str
    message: str
    triggered_at: str
    acknowledged: bool
    resolved: bool
    actions_taken: List[str]

class RiskFactorModel(BaseModel):
    factor_id: str
    risk_type: str
    name: str
    description: str
    weight: float
    current_value: float
    threshold: float
    impact: float
    probability: float
    trend: str
    last_updated: str

@router.post("/assess")
async def perform_risk_assessment(request: RiskAssessmentRequestModel):
    """Perform risk assessment for an entity"""
    try:
        assessment_id = await ai_powered_risk_assessment_engine.perform_risk_assessment(
            request.entity_id,
            request.entity_type
        )
        
        if assessment_id:
            return {"assessment_id": assessment_id, "status": "completed"}
        else:
            raise HTTPException(status_code=500, detail="Failed to perform risk assessment")
            
    except Exception as e:
        logger.error(f"Error performing risk assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assessment/{assessment_id}")
async def get_risk_assessment(assessment_id: str):
    """Get risk assessment details"""
    try:
        assessment = await ai_powered_risk_assessment_engine.get_risk_assessment(assessment_id)
        
        if assessment:
            return RiskAssessmentModel(**assessment)
        else:
            raise HTTPException(status_code=404, detail="Risk assessment not found")
            
    except Exception as e:
        logger.error(f"Error getting risk assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_risk_alerts():
    """Get all risk alerts"""
    try:
        alerts = await ai_powered_risk_assessment_engine.get_risk_alerts()
        return {"alerts": alerts, "total": len(alerts)}
        
    except Exception as e:
        logger.error(f"Error getting risk alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/active")
async def get_active_risk_alerts():
    """Get active (unresolved) risk alerts"""
    try:
        all_alerts = await ai_powered_risk_assessment_engine.get_risk_alerts()
        active_alerts = [alert for alert in all_alerts if not alert["resolved"]]
        
        return {"alerts": active_alerts, "total": len(active_alerts)}
        
    except Exception as e:
        logger.error(f"Error getting active risk alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_risk_alert(alert_id: str):
    """Acknowledge risk alert"""
    try:
        success = await ai_powered_risk_assessment_engine.acknowledge_risk_alert(alert_id)
        
        if success:
            return {"alert_id": alert_id, "status": "acknowledged"}
        else:
            raise HTTPException(status_code=404, detail="Risk alert not found")
            
    except Exception as e:
        logger.error(f"Error acknowledging risk alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/resolve")
async def resolve_risk_alert(alert_id: str, actions_taken: List[str]):
    """Resolve risk alert"""
    try:
        success = await ai_powered_risk_assessment_engine.resolve_risk_alert(alert_id, actions_taken)
        
        if success:
            return {"alert_id": alert_id, "status": "resolved", "actions_taken": actions_taken}
        else:
            raise HTTPException(status_code=404, detail="Risk alert not found")
            
    except Exception as e:
        logger.error(f"Error resolving risk alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_risk_performance():
    """Get risk assessment performance metrics"""
    try:
        metrics = await ai_powered_risk_assessment_engine.get_risk_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting risk performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-types")
async def get_risk_types():
    """Get available risk types"""
    try:
        risk_types = [
            {"name": rt.value, "description": f"Risk type: {rt.value}"}
            for rt in RiskType
        ]
        
        return {"risk_types": risk_types, "total": len(risk_types)}
        
    except Exception as e:
        logger.error(f"Error getting risk types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-levels")
async def get_risk_levels():
    """Get available risk levels"""
    try:
        risk_levels = [
            {"name": rl.name, "value": rl.value, "description": f"Risk level: {rl.name}"}
            for rl in RiskLevel
        ]
        
        return {"risk_levels": risk_levels, "total": len(risk_levels)}
        
    except Exception as e:
        logger.error(f"Error getting risk levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-categories")
async def get_risk_categories():
    """Get available risk categories"""
    try:
        risk_categories = [
            {"name": rc.value, "description": f"Risk category: {rc.value}"}
            for rc in RiskCategory
        ]
        
        return {"risk_categories": risk_categories, "total": len(risk_categories)}
        
    except Exception as e:
        logger.error(f"Error getting risk categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_risk_health():
    """Get risk assessment engine health"""
    try:
        metrics = await ai_powered_risk_assessment_engine.get_risk_performance_metrics()
        
        health_status = "healthy"
        if metrics["total_models"] == 0:
            health_status = "unhealthy"
        elif metrics["active_models"] < metrics["total_models"] * 0.8:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "total_models": metrics["total_models"],
            "active_models": metrics["active_models"],
            "assessments_performed": metrics["performance_metrics"]["assessments_performed"],
            "alerts_generated": metrics["performance_metrics"]["alerts_generated"],
            "alerts_resolved": metrics["performance_metrics"]["alerts_resolved"],
            "average_accuracy": metrics["performance_metrics"]["average_accuracy"]
        }
        
    except Exception as e:
        logger.error(f"Error getting risk health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_risk_stats():
    """Get comprehensive risk assessment statistics"""
    try:
        metrics = await ai_powered_risk_assessment_engine.get_risk_performance_metrics()
        
        return {
            "performance_metrics": metrics["performance_metrics"],
            "model_statistics": {
                "total_models": metrics["total_models"],
                "active_models": metrics["active_models"]
            },
            "assessment_statistics": {
                "total_assessments": metrics["total_assessments"]
            },
            "alert_statistics": {
                "total_alerts": metrics["total_alerts"],
                "active_alerts": metrics["active_alerts"]
            },
            "risk_factor_statistics": {
                "total_risk_factors": metrics["total_risk_factors"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting risk stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
