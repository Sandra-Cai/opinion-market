"""
API endpoints for Advanced AI Orchestration Engine
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

from app.services.advanced_ai_orchestration_engine import advanced_ai_orchestration_engine, DecisionType, AIModelType, ConfidenceLevel

logger = logging.getLogger(__name__)

router = APIRouter()

class AIRequestModel(BaseModel):
    decision_type: str
    input_data: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0

class AIResponseModel(BaseModel):
    response_id: str
    request_id: str
    decision: Dict[str, Any]
    confidence: float
    confidence_level: str
    reasoning: str
    model_results: Dict[str, Any]
    processing_time: float
    created_at: str

class AIModelModel(BaseModel):
    model_id: str
    model_type: str
    name: str
    version: str
    accuracy: float
    latency: float
    cost: float
    status: str
    last_updated: str
    metadata: Dict[str, Any]

class AIWorkflowModel(BaseModel):
    workflow_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    triggers: List[str]
    status: str
    created_at: str
    last_executed: Optional[str]

@router.post("/submit-request")
async def submit_ai_request(request: AIRequestModel):
    """Submit AI request for processing"""
    try:
        decision_type = DecisionType(request.decision_type)
        
        request_id = await advanced_ai_orchestration_engine.submit_ai_request(
            decision_type,
            request.input_data,
            request.priority,
            request.timeout
        )
        
        if request_id:
            return {"request_id": request_id, "status": "submitted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to submit AI request")
            
    except Exception as e:
        logger.error(f"Error submitting AI request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/response/{request_id}")
async def get_ai_response(request_id: str):
    """Get AI response for request"""
    try:
        response = await advanced_ai_orchestration_engine.get_ai_response(request_id)
        
        if response:
            return AIResponseModel(**response)
        else:
            raise HTTPException(status_code=404, detail="Response not found")
            
    except Exception as e:
        logger.error(f"Error getting AI response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_ai_models():
    """Get all AI models"""
    try:
        models = await advanced_ai_orchestration_engine.get_ai_models()
        return {"models": models, "total": len(models)}
        
    except Exception as e:
        logger.error(f"Error getting AI models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_type}")
async def get_ai_models_by_type(model_type: str):
    """Get AI models by type"""
    try:
        all_models = await advanced_ai_orchestration_engine.get_ai_models()
        filtered_models = [model for model in all_models if model["model_type"] == model_type]
        
        return {"models": filtered_models, "total": len(filtered_models)}
        
    except Exception as e:
        logger.error(f"Error getting AI models by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows")
async def get_ai_workflows():
    """Get all AI workflows"""
    try:
        workflows = await advanced_ai_orchestration_engine.get_ai_workflows()
        return {"workflows": workflows, "total": len(workflows)}
        
    except Exception as e:
        logger.error(f"Error getting AI workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}")
async def get_ai_workflow(workflow_id: str):
    """Get AI workflow by ID"""
    try:
        workflows = await advanced_ai_orchestration_engine.get_ai_workflows()
        
        for workflow in workflows:
            if workflow["workflow_id"] == workflow_id:
                return AIWorkflowModel(**workflow)
        
        raise HTTPException(status_code=404, detail="Workflow not found")
        
    except Exception as e:
        logger.error(f"Error getting AI workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_ai_performance():
    """Get AI orchestration performance metrics"""
    try:
        metrics = await advanced_ai_orchestration_engine.get_ai_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting AI performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/decision-types")
async def get_decision_types():
    """Get available decision types"""
    try:
        decision_types = [
            {"name": dt.value, "description": f"Decision type: {dt.value}"}
            for dt in DecisionType
        ]
        
        return {"decision_types": decision_types, "total": len(decision_types)}
        
    except Exception as e:
        logger.error(f"Error getting decision types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-types")
async def get_model_types():
    """Get available AI model types"""
    try:
        model_types = [
            {"name": mt.value, "description": f"AI model type: {mt.value}"}
            for mt in AIModelType
        ]
        
        return {"model_types": model_types, "total": len(model_types)}
        
    except Exception as e:
        logger.error(f"Error getting model types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/confidence-levels")
async def get_confidence_levels():
    """Get available confidence levels"""
    try:
        confidence_levels = [
            {"name": cl.value, "description": f"Confidence level: {cl.value}"}
            for cl in ConfidenceLevel
        ]
        
        return {"confidence_levels": confidence_levels, "total": len(confidence_levels)}
        
    except Exception as e:
        logger.error(f"Error getting confidence levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_ai_health():
    """Get AI orchestration engine health"""
    try:
        metrics = await advanced_ai_orchestration_engine.get_ai_performance_metrics()
        
        health_status = "healthy"
        if metrics["total_models"] == 0:
            health_status = "unhealthy"
        elif metrics["active_models"] < metrics["total_models"] * 0.8:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "total_models": metrics["total_models"],
            "active_models": metrics["active_models"],
            "requests_processed": metrics["performance_metrics"]["requests_processed"],
            "decisions_made": metrics["performance_metrics"]["decisions_made"],
            "average_confidence": metrics["performance_metrics"]["confidence_score"]
        }
        
    except Exception as e:
        logger.error(f"Error getting AI health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_ai_stats():
    """Get comprehensive AI orchestration statistics"""
    try:
        metrics = await advanced_ai_orchestration_engine.get_ai_performance_metrics()
        models = await advanced_ai_orchestration_engine.get_ai_models()
        workflows = await advanced_ai_orchestration_engine.get_ai_workflows()
        
        # Calculate additional statistics
        model_type_stats = {}
        for model in models:
            model_type = model["model_type"]
            model_type_stats[model_type] = model_type_stats.get(model_type, 0) + 1
        
        return {
            "performance_metrics": metrics["performance_metrics"],
            "model_statistics": {
                "total_models": metrics["total_models"],
                "active_models": metrics["active_models"],
                "model_type_distribution": model_type_stats
            },
            "workflow_statistics": {
                "total_workflows": metrics["total_workflows"],
                "active_workflows": metrics["active_workflows"]
            },
            "request_statistics": {
                "total_requests": metrics["total_requests"],
                "completed_requests": metrics["completed_requests"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting AI stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
