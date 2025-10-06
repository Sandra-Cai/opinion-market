"""
Advanced Predictive Analytics API Endpoints
REST API for the Advanced Predictive Analytics Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.advanced_predictive_analytics_engine import (
    advanced_predictive_analytics_engine,
    PredictionType,
    ModelType,
    ForecastHorizon
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    prediction_type: str = Field(..., description="Type of prediction")
    asset: str = Field(..., description="Asset symbol")
    model_type: str = Field(..., description="Model type to use")
    forecast_horizon: str = Field(..., description="Forecast horizon")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for prediction")
    confidence_level: float = Field(0.95, description="Confidence level (0.0 to 1.0)")

class PredictionResultResponse(BaseModel):
    result_id: str
    request_id: str
    prediction_type: str
    asset: str
    model_type: str
    forecast_horizon: str
    predictions: List[float]
    confidence_intervals: List[List[float]]
    accuracy_metrics: Dict[str, float]
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    created_at: str

class AnomalyDetectionResponse(BaseModel):
    anomaly_id: str
    asset: str
    anomaly_type: str
    severity: float
    confidence: float
    detected_at: str
    value: float
    expected_value: float
    deviation: float
    metadata: Dict[str, Any]

@router.post("/submit-prediction", response_model=Dict[str, str])
async def submit_prediction_request(prediction_request: PredictionRequest):
    """Submit a prediction request"""
    try:
        # Validate prediction type
        try:
            prediction_type = PredictionType(prediction_request.prediction_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid prediction type: {prediction_request.prediction_type}")
        
        # Validate model type
        try:
            model_type = ModelType(prediction_request.model_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {prediction_request.model_type}")
        
        # Validate forecast horizon
        try:
            forecast_horizon = ForecastHorizon(prediction_request.forecast_horizon.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid forecast horizon: {prediction_request.forecast_horizon}")
        
        # Submit prediction request
        request_id = await advanced_predictive_analytics_engine.submit_prediction_request(
            prediction_type=prediction_type,
            asset=prediction_request.asset,
            model_type=model_type,
            forecast_horizon=forecast_horizon,
            input_data=prediction_request.input_data,
            confidence_level=prediction_request.confidence_level
        )
        
        return {
            "request_id": request_id,
            "status": "submitted",
            "message": f"Prediction request submitted for {prediction_request.asset}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting prediction request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prediction-result/{request_id}", response_model=PredictionResultResponse)
async def get_prediction_result(request_id: str):
    """Get prediction result"""
    try:
        result = await advanced_predictive_analytics_engine.get_prediction_result(request_id)
        if not result:
            raise HTTPException(status_code=404, detail="Prediction result not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction result {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anomaly-detections", response_model=List[AnomalyDetectionResponse])
async def get_anomaly_detections(asset: Optional[str] = None, limit: int = 50):
    """Get anomaly detections"""
    try:
        anomalies = await advanced_predictive_analytics_engine.get_anomaly_detections(asset=asset, limit=limit)
        return anomalies
        
    except Exception as e:
        logger.error(f"Error getting anomaly detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        performance = await advanced_predictive_analytics_engine.get_model_performance()
        return performance
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-models")
async def get_available_models():
    """Get available prediction models"""
    try:
        models = await advanced_predictive_analytics_engine.get_available_models()
        return {
            "models": models,
            "total_models": len(models),
            "prediction_types": [pt.value for pt in PredictionType],
            "model_types": [mt.value for mt in ModelType],
            "forecast_horizons": [fh.value for fh in ForecastHorizon]
        }
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_predictive_analytics_health():
    """Get predictive analytics engine health status"""
    try:
        return {
            "engine_id": advanced_predictive_analytics_engine.engine_id,
            "is_running": advanced_predictive_analytics_engine.is_running,
            "total_requests": len(advanced_predictive_analytics_engine.prediction_requests),
            "total_results": len(advanced_predictive_analytics_engine.prediction_results),
            "total_anomalies": len(advanced_predictive_analytics_engine.anomaly_detections),
            "trained_models": len(advanced_predictive_analytics_engine.trained_models),
            "uptime": "active" if advanced_predictive_analytics_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting predictive analytics health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
