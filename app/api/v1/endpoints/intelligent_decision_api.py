"""
API endpoints for Intelligent Decision Engine
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from app.services.intelligent_decision_engine import intelligent_decision_engine, DecisionContext, DecisionComplexity, DecisionOutcome

logger = logging.getLogger(__name__)

router = APIRouter()

class DecisionRequestModel(BaseModel):
    context: str
    description: str
    options: List[Dict[str, Any]]
    criteria: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[Dict[str, Any]] = None
    deadline: Optional[str] = None
    priority: int = 1

class DecisionResultModel(BaseModel):
    result_id: str
    request_id: str
    recommended_option: str
    confidence_score: float
    reasoning: str
    alternative_options: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    implementation_plan: Dict[str, Any]
    monitoring_metrics: List[str]
    created_at: str

class DecisionCriteriaModel(BaseModel):
    criteria_id: str
    name: str
    weight: float
    threshold: float
    importance: str
    description: str
    created_at: str

@router.post("/submit-request")
async def submit_decision_request(request: DecisionRequestModel):
    """Submit decision request"""
    try:
        context = DecisionContext(request.context)
        
        # Parse deadline if provided
        deadline = None
        if request.deadline:
            deadline = datetime.fromisoformat(request.deadline)
        
        request_id = await intelligent_decision_engine.submit_decision_request(
            context,
            request.description,
            request.options,
            request.criteria,
            request.constraints,
            deadline,
            request.priority
        )
        
        if request_id:
            return {"request_id": request_id, "status": "submitted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to submit decision request")
            
    except Exception as e:
        logger.error(f"Error submitting decision request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/result/{request_id}")
async def get_decision_result(request_id: str):
    """Get decision result"""
    try:
        result = await intelligent_decision_engine.get_decision_result(request_id)
        
        if result:
            return DecisionResultModel(**result)
        else:
            raise HTTPException(status_code=404, detail="Decision result not found")
            
    except Exception as e:
        logger.error(f"Error getting decision result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_decision_performance():
    """Get decision engine performance metrics"""
    try:
        metrics = await intelligent_decision_engine.get_decision_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting decision performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/contexts")
async def get_decision_contexts():
    """Get available decision contexts"""
    try:
        contexts = [
            {"name": context.value, "description": f"Decision context: {context.value}"}
            for context in DecisionContext
        ]
        
        return {"contexts": contexts, "total": len(contexts)}
        
    except Exception as e:
        logger.error(f"Error getting decision contexts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/complexities")
async def get_decision_complexities():
    """Get available decision complexities"""
    try:
        complexities = [
            {"name": complexity.value, "description": f"Decision complexity: {complexity.value}"}
            for complexity in DecisionComplexity
        ]
        
        return {"complexities": complexities, "total": len(complexities)}
        
    except Exception as e:
        logger.error(f"Error getting decision complexities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/outcomes")
async def get_decision_outcomes():
    """Get available decision outcomes"""
    try:
        outcomes = [
            {"name": outcome.value, "description": f"Decision outcome: {outcome.value}"}
            for outcome in DecisionOutcome
        ]
        
        return {"outcomes": outcomes, "total": len(outcomes)}
        
    except Exception as e:
        logger.error(f"Error getting decision outcomes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_decision_health():
    """Get decision engine health"""
    try:
        metrics = await intelligent_decision_engine.get_decision_performance_metrics()
        
        health_status = "healthy"
        if metrics["total_requests"] == 0:
            health_status = "unhealthy"
        elif metrics["completed_requests"] < metrics["total_requests"] * 0.8:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "total_requests": metrics["total_requests"],
            "completed_requests": metrics["completed_requests"],
            "decisions_made": metrics["performance_metrics"]["decisions_made"],
            "successful_decisions": metrics["performance_metrics"]["successful_decisions"],
            "average_confidence": metrics["performance_metrics"]["average_confidence"],
            "decision_accuracy": metrics["performance_metrics"]["decision_accuracy"]
        }
        
    except Exception as e:
        logger.error(f"Error getting decision health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_decision_stats():
    """Get comprehensive decision engine statistics"""
    try:
        metrics = await intelligent_decision_engine.get_decision_performance_metrics()
        
        return {
            "performance_metrics": metrics["performance_metrics"],
            "request_statistics": {
                "total_requests": metrics["total_requests"],
                "completed_requests": metrics["completed_requests"]
            },
            "result_statistics": {
                "total_results": metrics["total_results"]
            },
            "history_statistics": {
                "total_history": metrics["total_history"]
            },
            "template_statistics": {
                "decision_templates": metrics["decision_templates"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting decision stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
