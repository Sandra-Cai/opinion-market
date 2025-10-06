"""
IoT Analytics API Endpoints
REST API for the IoT Analytics Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.iot_analytics_engine import (
    iot_analytics_engine,
    AnalyticsType,
    InsightType,
    AlertSeverity
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class AnalyticsJobResponse(BaseModel):
    job_id: str
    job_type: str
    device_ids: List[str]
    sensor_types: List[str]
    parameters: Dict[str, Any]
    status: str
    progress: float
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    results: Optional[Dict[str, Any]]
    error_message: Optional[str]
    metadata: Dict[str, Any]

class DataInsightResponse(BaseModel):
    insight_id: str
    device_id: str
    sensor_type: str
    insight_type: str
    title: str
    description: str
    confidence: float
    severity: str
    timestamp: str
    data_points: List[float]
    trend: str
    anomaly_score: Optional[float]
    correlation_coefficient: Optional[float]
    prediction_accuracy: Optional[float]
    recommendations: List[str]
    metadata: Dict[str, Any]

class PredictiveModelResponse(BaseModel):
    model_id: str
    model_type: str
    device_id: str
    sensor_type: str
    algorithm: str
    accuracy: float
    training_data_size: int
    last_trained: str
    is_active: bool
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

class DataPatternResponse(BaseModel):
    pattern_id: str
    device_id: str
    sensor_type: str
    pattern_type: str
    frequency: int
    duration: float
    confidence: float
    first_detected: str
    last_detected: str
    pattern_data: List[float]
    metadata: Dict[str, Any]

class CreateAnalyticsJobRequest(BaseModel):
    job_type: str = Field(..., description="Analytics job type")
    device_ids: List[str] = Field(..., description="List of device IDs")
    sensor_types: List[str] = Field(..., description="List of sensor types")
    parameters: Dict[str, Any] = Field(..., description="Job parameters")

class UpdateModelStatusRequest(BaseModel):
    model_id: str = Field(..., description="Model ID")
    is_active: bool = Field(..., description="Active status")

@router.get("/jobs", response_model=List[AnalyticsJobResponse])
async def get_analytics_jobs(
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Get analytics jobs"""
    try:
        # Validate job type
        job_type_enum = None
        if job_type:
            try:
                job_type_enum = AnalyticsType(job_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid job type: {job_type}")
        
        jobs = await iot_analytics_engine.get_analytics_jobs(
            job_type=job_type_enum,
            status=status,
            limit=limit
        )
        return jobs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights", response_model=List[DataInsightResponse])
async def get_data_insights(
    device_id: Optional[str] = None,
    sensor_type: Optional[str] = None,
    insight_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100
):
    """Get data insights"""
    try:
        # Validate insight type
        insight_type_enum = None
        if insight_type:
            try:
                insight_type_enum = InsightType(insight_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid insight type: {insight_type}")
        
        # Validate severity
        severity_enum = None
        if severity:
            try:
                severity_enum = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        insights = await iot_analytics_engine.get_data_insights(
            device_id=device_id,
            sensor_type=sensor_type,
            insight_type=insight_type_enum,
            severity=severity_enum,
            limit=limit
        )
        return insights
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[PredictiveModelResponse])
async def get_predictive_models(
    device_id: Optional[str] = None,
    sensor_type: Optional[str] = None,
    active_only: bool = False
):
    """Get predictive models"""
    try:
        models = await iot_analytics_engine.get_predictive_models(
            device_id=device_id,
            sensor_type=sensor_type,
            active_only=active_only
        )
        return models
        
    except Exception as e:
        logger.error(f"Error getting predictive models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patterns", response_model=List[DataPatternResponse])
async def get_data_patterns(
    device_id: Optional[str] = None,
    sensor_type: Optional[str] = None,
    pattern_type: Optional[str] = None,
    limit: int = 100
):
    """Get data patterns"""
    try:
        patterns = await iot_analytics_engine.get_data_patterns(
            device_id=device_id,
            sensor_type=sensor_type,
            pattern_type=pattern_type,
            limit=limit
        )
        return patterns
        
    except Exception as e:
        logger.error(f"Error getting data patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/jobs", response_model=Dict[str, str])
async def create_analytics_job(job_request: CreateAnalyticsJobRequest):
    """Create an analytics job"""
    try:
        # Validate job type
        try:
            job_type_enum = AnalyticsType(job_request.job_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid job type: {job_request.job_type}")
        
        job_id = await iot_analytics_engine.create_analytics_job(
            job_type=job_type_enum,
            device_ids=job_request.device_ids,
            sensor_types=job_request.sensor_types,
            parameters=job_request.parameters
        )
        
        return {
            "job_id": job_id,
            "status": "created",
            "message": f"Analytics job '{job_request.job_type}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating analytics job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/models/status", response_model=Dict[str, str])
async def update_model_status(status_request: UpdateModelStatusRequest):
    """Update model status"""
    try:
        success = await iot_analytics_engine.update_model_status(
            model_id=status_request.model_id,
            is_active=status_request.is_active
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "model_id": status_request.model_id,
            "status": "updated",
            "message": f"Model status updated to {'active' if status_request.is_active else 'inactive'}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/engine-metrics")
async def get_engine_metrics():
    """Get engine metrics"""
    try:
        metrics = await iot_analytics_engine.get_engine_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-analytics-types")
async def get_available_analytics_types():
    """Get available analytics types and insight types"""
    try:
        return {
            "analytics_types": [
                {
                    "name": analytics_type.value,
                    "display_name": analytics_type.value.replace("_", " ").title(),
                    "description": f"{analytics_type.value.replace('_', ' ').title()} analytics"
                }
                for analytics_type in AnalyticsType
            ],
            "insight_types": [
                {
                    "name": insight_type.value,
                    "display_name": insight_type.value.replace("_", " ").title(),
                    "description": f"{insight_type.value.replace('_', ' ').title()} insight"
                }
                for insight_type in InsightType
            ],
            "alert_severities": [
                {
                    "name": severity.value,
                    "display_name": severity.value.replace("_", " ").title(),
                    "description": f"{severity.value.replace('_', ' ').title()} severity"
                }
                for severity in AlertSeverity
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting available analytics types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_iot_analytics_health():
    """Get IoT analytics engine health status"""
    try:
        return {
            "engine_id": iot_analytics_engine.engine_id,
            "is_running": iot_analytics_engine.is_running,
            "total_jobs": len(iot_analytics_engine.analytics_jobs),
            "total_insights": len(iot_analytics_engine.data_insights),
            "total_models": len(iot_analytics_engine.predictive_models),
            "total_patterns": len(iot_analytics_engine.data_patterns),
            "completed_jobs": len([j for j in iot_analytics_engine.analytics_jobs if j.status == "completed"]),
            "active_models": len([m for m in iot_analytics_engine.predictive_models.values() if m.is_active]),
            "high_severity_insights": len([i for i in iot_analytics_engine.data_insights if i.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]]),
            "supported_analytics_types": [at.value for at in AnalyticsType],
            "uptime": "active" if iot_analytics_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting IoT analytics health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
