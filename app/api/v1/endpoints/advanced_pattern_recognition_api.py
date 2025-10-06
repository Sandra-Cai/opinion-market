"""
API endpoints for Advanced Pattern Recognition Engine
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

from app.services.advanced_pattern_recognition_engine import advanced_pattern_recognition_engine, PatternType, PatternComplexity, PatternConfidence

logger = logging.getLogger(__name__)

router = APIRouter()

class PatternRequestModel(BaseModel):
    data_type: str
    time_range_start: str
    time_range_end: str
    pattern_types: List[str]
    complexity_threshold: int = 1
    confidence_threshold: int = 1

class PatternResultModel(BaseModel):
    result_id: str
    request_id: str
    patterns_found: List[Dict[str, Any]]
    summary: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    processing_time: float
    created_at: str

class PatternModel(BaseModel):
    model_id: str
    model_type: str
    pattern_types: List[str]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_data_size: int
    last_trained: str
    status: str
    metadata: Dict[str, Any]

@router.post("/submit-request")
async def submit_pattern_request(request: PatternRequestModel):
    """Submit pattern recognition request"""
    try:
        # Parse time range
        start_time = datetime.fromisoformat(request.time_range_start)
        end_time = datetime.fromisoformat(request.time_range_end)
        time_range = (start_time, end_time)
        
        # Convert pattern types
        pattern_types = [PatternType(pt) for pt in request.pattern_types]
        complexity_threshold = PatternComplexity(request.complexity_threshold)
        confidence_threshold = PatternConfidence(request.confidence_threshold)
        
        request_id = await advanced_pattern_recognition_engine.submit_pattern_recognition_request(
            request.data_type,
            time_range,
            pattern_types,
            complexity_threshold,
            confidence_threshold
        )
        
        if request_id:
            return {"request_id": request_id, "status": "submitted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to submit pattern request")
            
    except Exception as e:
        logger.error(f"Error submitting pattern request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/result/{request_id}")
async def get_pattern_result(request_id: str):
    """Get pattern recognition result"""
    try:
        result = await advanced_pattern_recognition_engine.get_pattern_result(request_id)
        
        if result:
            return PatternResultModel(**result)
        else:
            raise HTTPException(status_code=404, detail="Pattern result not found")
            
    except Exception as e:
        logger.error(f"Error getting pattern result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_pattern_models():
    """Get all pattern recognition models"""
    try:
        models = await advanced_pattern_recognition_engine.get_pattern_models()
        return {"models": models, "total": len(models)}
        
    except Exception as e:
        logger.error(f"Error getting pattern models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_type}")
async def get_pattern_models_by_type(model_type: str):
    """Get pattern models by type"""
    try:
        all_models = await advanced_pattern_recognition_engine.get_pattern_models()
        filtered_models = [model for model in all_models if model["model_type"] == model_type]
        
        return {"models": filtered_models, "total": len(filtered_models)}
        
    except Exception as e:
        logger.error(f"Error getting pattern models by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_pattern_performance():
    """Get pattern recognition performance metrics"""
    try:
        metrics = await advanced_pattern_recognition_engine.get_pattern_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting pattern performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pattern-types")
async def get_pattern_types():
    """Get available pattern types"""
    try:
        pattern_types = [
            {"name": pt.value, "description": f"Pattern type: {pt.value}"}
            for pt in PatternType
        ]
        
        return {"pattern_types": pattern_types, "total": len(pattern_types)}
        
    except Exception as e:
        logger.error(f"Error getting pattern types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/complexities")
async def get_pattern_complexities():
    """Get available pattern complexities"""
    try:
        complexities = [
            {"name": complexity.value, "description": f"Pattern complexity: {complexity.value}"}
            for complexity in PatternComplexity
        ]
        
        return {"complexities": complexities, "total": len(complexities)}
        
    except Exception as e:
        logger.error(f"Error getting pattern complexities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/confidence-levels")
async def get_pattern_confidence_levels():
    """Get available pattern confidence levels"""
    try:
        confidence_levels = [
            {"name": confidence.value, "description": f"Pattern confidence: {confidence.value}"}
            for confidence in PatternConfidence
        ]
        
        return {"confidence_levels": confidence_levels, "total": len(confidence_levels)}
        
    except Exception as e:
        logger.error(f"Error getting pattern confidence levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_pattern_health():
    """Get pattern recognition engine health"""
    try:
        metrics = await advanced_pattern_recognition_engine.get_pattern_performance_metrics()
        
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
            "patterns_detected": metrics["performance_metrics"]["patterns_detected"],
            "average_accuracy": metrics["performance_metrics"]["average_accuracy"],
            "model_performance": metrics["performance_metrics"]["model_performance"]
        }
        
    except Exception as e:
        logger.error(f"Error getting pattern health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_pattern_stats():
    """Get comprehensive pattern recognition statistics"""
    try:
        metrics = await advanced_pattern_recognition_engine.get_pattern_performance_metrics()
        models = await advanced_pattern_recognition_engine.get_pattern_models()
        
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
            "request_statistics": {
                "total_requests": metrics["total_requests"],
                "completed_requests": metrics["completed_requests"]
            },
            "pattern_statistics": {
                "total_patterns": metrics["total_patterns"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting pattern stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
