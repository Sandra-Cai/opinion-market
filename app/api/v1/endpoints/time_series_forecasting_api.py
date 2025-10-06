"""
Time Series Forecasting API Endpoints
REST API for the Time Series Forecasting Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.time_series_forecasting_engine import (
    time_series_forecasting_engine,
    TimeSeriesType,
    SeasonalityType,
    TrendType
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class TimeSeriesResponse(BaseModel):
    series_id: str
    asset: str
    series_type: str
    timestamps: List[str]
    values: List[float]
    metadata: Dict[str, Any]
    created_at: str

class TimeSeriesAnalysisResponse(BaseModel):
    analysis_id: str
    series_id: str
    stationarity_test: Dict[str, Any]
    seasonality_analysis: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    autocorrelation: Dict[str, Any]
    decomposition: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str

class ForecastResultResponse(BaseModel):
    forecast_id: str
    series_id: str
    model_type: str
    forecast_periods: int
    forecast_values: List[float]
    confidence_intervals: List[List[float]]
    accuracy_metrics: Dict[str, float]
    model_parameters: Dict[str, Any]
    created_at: str

@router.get("/time-series/{series_id}", response_model=TimeSeriesResponse)
async def get_time_series(series_id: str):
    """Get time series data"""
    try:
        time_series = await time_series_forecasting_engine.get_time_series(series_id)
        if not time_series:
            raise HTTPException(status_code=404, detail="Time series not found")
        
        return time_series
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting time series {series_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/time-series-analysis/{series_id}", response_model=TimeSeriesAnalysisResponse)
async def get_time_series_analysis(series_id: str):
    """Get time series analysis"""
    try:
        analysis = await time_series_forecasting_engine.get_time_series_analysis(series_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Time series analysis not found")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting time series analysis {series_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast-results/{series_id}", response_model=List[ForecastResultResponse])
async def get_forecast_results(series_id: str):
    """Get forecast results for a time series"""
    try:
        results = await time_series_forecasting_engine.get_forecast_results(series_id)
        return results
        
    except Exception as e:
        logger.error(f"Error getting forecast results for {series_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-time-series")
async def get_available_time_series():
    """Get available time series"""
    try:
        time_series_list = await time_series_forecasting_engine.get_available_time_series()
        return {
            "time_series": time_series_list,
            "total_series": len(time_series_list),
            "series_types": [st.value for st in TimeSeriesType],
            "seasonality_types": [st.value for st in SeasonalityType],
            "trend_types": [tt.value for tt in TrendType]
        }
        
    except Exception as e:
        logger.error(f"Error getting available time series: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/engine-metrics")
async def get_engine_metrics():
    """Get engine metrics"""
    try:
        metrics = await time_series_forecasting_engine.get_engine_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_time_series_forecasting_health():
    """Get time series forecasting engine health status"""
    try:
        return {
            "engine_id": time_series_forecasting_engine.engine_id,
            "is_running": time_series_forecasting_engine.is_running,
            "total_time_series": len(time_series_forecasting_engine.time_series_data),
            "total_analyses": len(time_series_forecasting_engine.time_series_analyses),
            "total_forecasts": len(time_series_forecasting_engine.forecast_results),
            "model_configs": time_series_forecasting_engine.model_configs,
            "uptime": "active" if time_series_forecasting_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting time series forecasting health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
