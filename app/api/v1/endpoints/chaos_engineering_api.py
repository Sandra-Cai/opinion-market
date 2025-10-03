"""
Chaos Engineering API
API endpoints for chaos engineering and resilience testing
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.services.chaos_engineering_engine import chaos_engineering_engine, ChaosExperimentType, ExperimentStatus, FailureMode

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class ChaosExperimentRequest(BaseModel):
    """Chaos experiment request model"""
    name: str
    description: str
    experiment_type: str
    target_services: List[str]
    duration: int = 300
    intensity: float = 0.5
    failure_mode: str = "graceful"


class ExperimentResult(BaseModel):
    """Experiment result model"""
    experiment_id: str
    status: str
    results: Dict[str, Any]
    metrics: Dict[str, Any]


# API Endpoints
@router.get("/status")
async def get_chaos_engineering_status():
    """Get chaos engineering system status"""
    try:
        summary = chaos_engineering_engine.get_chaos_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting chaos engineering status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments")
async def create_chaos_experiment(experiment_request: ChaosExperimentRequest):
    """Create a new chaos experiment"""
    try:
        experiment_data = {
            "name": experiment_request.name,
            "description": experiment_request.description,
            "type": experiment_request.experiment_type,
            "target_services": experiment_request.target_services,
            "duration": experiment_request.duration,
            "intensity": experiment_request.intensity,
            "failure_mode": experiment_request.failure_mode
        }
        
        experiment = await chaos_engineering_engine.create_experiment(experiment_data)
        
        return JSONResponse(content={
            "message": "Chaos experiment created successfully",
            "experiment_id": experiment.experiment_id,
            "status": experiment.status.value
        })
        
    except Exception as e:
        logger.error(f"Error creating chaos experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def get_chaos_experiments():
    """Get all chaos experiments"""
    try:
        experiments = []
        for experiment_id, experiment in chaos_engineering_engine.experiments.items():
            experiments.append({
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "description": experiment.description,
                "experiment_type": experiment.experiment_type.value,
                "target_services": experiment.target_services,
                "duration": experiment.duration,
                "intensity": experiment.intensity,
                "failure_mode": experiment.failure_mode.value,
                "status": experiment.status.value,
                "created_at": experiment.created_at.isoformat(),
                "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
                "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None,
                "results": experiment.results,
                "metrics": experiment.metrics
            })
            
        return JSONResponse(content={"experiments": experiments})
        
    except Exception as e:
        logger.error(f"Error getting chaos experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/run")
async def run_chaos_experiment(experiment_id: str):
    """Run a chaos experiment"""
    try:
        success = await chaos_engineering_engine.run_experiment(experiment_id)
        
        return JSONResponse(content={
            "message": "Chaos experiment executed",
            "experiment_id": experiment_id,
            "success": success
        })
        
    except Exception as e:
        logger.error(f"Error running chaos experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}")
async def get_chaos_experiment(experiment_id: str):
    """Get a specific chaos experiment"""
    try:
        experiment = chaos_engineering_engine.experiments.get(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
            
        return JSONResponse(content={
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "experiment_type": experiment.experiment_type.value,
            "target_services": experiment.target_services,
            "duration": experiment.duration,
            "intensity": experiment.intensity,
            "failure_mode": experiment.failure_mode.value,
            "status": experiment.status.value,
            "created_at": experiment.created_at.isoformat(),
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None,
            "results": experiment.results,
            "metrics": experiment.metrics,
            "recovery_time": experiment.recovery_time
        })
        
    except Exception as e:
        logger.error(f"Error getting chaos experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resilience-metrics")
async def get_resilience_metrics():
    """Get resilience metrics"""
    try:
        metrics = []
        for metric in chaos_engineering_engine.resilience_metrics[-50:]:  # Last 50 metrics
            metrics.append({
                "service_name": metric.service_name,
                "availability": metric.availability,
                "response_time": metric.response_time,
                "error_rate": metric.error_rate,
                "recovery_time": metric.recovery_time,
                "throughput": metric.throughput,
                "timestamp": metric.timestamp.isoformat()
            })
            
        return JSONResponse(content={"metrics": metrics})
        
    except Exception as e:
        logger.error(f"Error getting resilience metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/baseline-metrics")
async def get_baseline_metrics():
    """Get baseline metrics"""
    try:
        return JSONResponse(content={"baseline_metrics": chaos_engineering_engine.baseline_metrics})
        
    except Exception as e:
        logger.error(f"Error getting baseline metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_chaos_engineering_dashboard():
    """Get chaos engineering dashboard data"""
    try:
        summary = chaos_engineering_engine.get_chaos_summary()
        
        # Get recent experiments
        recent_experiments = list(chaos_engineering_engine.experiments.values())[-5:]
        
        # Get recent resilience metrics
        recent_metrics = chaos_engineering_engine.resilience_metrics[-10:]
        
        dashboard_data = {
            "summary": summary,
            "recent_experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "experiment_type": exp.experiment_type.value,
                    "status": exp.status.value,
                    "created_at": exp.created_at.isoformat()
                }
                for exp in recent_experiments
            ],
            "recent_metrics": [
                {
                    "service_name": metric.service_name,
                    "availability": metric.availability,
                    "response_time": metric.response_time,
                    "timestamp": metric.timestamp.isoformat()
                }
                for metric in recent_metrics
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting chaos engineering dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_chaos_engineering():
    """Start chaos engineering engine"""
    try:
        await chaos_engineering_engine.start_chaos_engine()
        return JSONResponse(content={"message": "Chaos engineering engine started"})
        
    except Exception as e:
        logger.error(f"Error starting chaos engineering: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_chaos_engineering():
    """Stop chaos engineering engine"""
    try:
        await chaos_engineering_engine.stop_chaos_engine()
        return JSONResponse(content={"message": "Chaos engineering engine stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping chaos engineering: {e}")
        raise HTTPException(status_code=500, detail=str(e))
