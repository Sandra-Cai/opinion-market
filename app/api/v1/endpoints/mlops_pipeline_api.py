"""
MLOps Pipeline API
API endpoints for MLOps pipeline management
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.services.mlops_pipeline_engine import mlops_pipeline_engine, PipelineStage, PipelineStatus, ModelVersion

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class MLPipelineRequest(BaseModel):
    """ML pipeline request model"""
    name: str
    description: str
    template: str = "market_prediction"
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelDeploymentRequest(BaseModel):
    """Model deployment request model"""
    model_name: str
    version: str
    environment: str = "staging"
    traffic_percentage: float = 100.0


# API Endpoints
@router.get("/status")
async def get_mlops_status():
    """Get MLOps pipeline system status"""
    try:
        summary = mlops_pipeline_engine.get_mlops_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting MLOps status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipelines")
async def create_ml_pipeline(pipeline_request: MLPipelineRequest):
    """Create a new ML pipeline"""
    try:
        pipeline_data = {
            "name": pipeline_request.name,
            "description": pipeline_request.description,
            "template": pipeline_request.template,
            "config": pipeline_request.config or {},
            "metadata": pipeline_request.metadata or {}
        }
        
        pipeline = await mlops_pipeline_engine.create_pipeline(pipeline_data)
        
        return JSONResponse(content={
            "message": "ML pipeline created successfully",
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "status": pipeline.status.value
        })
        
    except Exception as e:
        logger.error(f"Error creating ML pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines")
async def get_ml_pipelines():
    """Get all ML pipelines"""
    try:
        pipelines = []
        for pipeline_id, pipeline in mlops_pipeline_engine.pipelines.items():
            pipelines.append({
                "pipeline_id": pipeline.pipeline_id,
                "name": pipeline.name,
                "description": pipeline.description,
                "stages": [stage.value for stage in pipeline.stages],
                "current_stage": pipeline.current_stage.value,
                "status": pipeline.status.value,
                "created_at": pipeline.created_at.isoformat(),
                "started_at": pipeline.started_at.isoformat() if pipeline.started_at else None,
                "completed_at": pipeline.completed_at.isoformat() if pipeline.completed_at else None,
                "config": pipeline.config,
                "metadata": pipeline.metadata,
                "results": pipeline.results,
                "error_message": pipeline.error_message
            })
            
        return JSONResponse(content={"pipelines": pipelines})
        
    except Exception as e:
        logger.error(f"Error getting ML pipelines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipelines/{pipeline_id}/run")
async def run_ml_pipeline(pipeline_id: str):
    """Run an ML pipeline"""
    try:
        success = await mlops_pipeline_engine.run_pipeline(pipeline_id)
        
        return JSONResponse(content={
            "message": "ML pipeline executed",
            "pipeline_id": pipeline_id,
            "success": success
        })
        
    except Exception as e:
        logger.error(f"Error running ML pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines/{pipeline_id}")
async def get_ml_pipeline(pipeline_id: str):
    """Get a specific ML pipeline"""
    try:
        pipeline = mlops_pipeline_engine.pipelines.get(pipeline_id)
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
            
        return JSONResponse(content={
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "description": pipeline.description,
            "stages": [stage.value for stage in pipeline.stages],
            "current_stage": pipeline.current_stage.value,
            "status": pipeline.status.value,
            "created_at": pipeline.created_at.isoformat(),
            "started_at": pipeline.started_at.isoformat() if pipeline.started_at else None,
            "completed_at": pipeline.completed_at.isoformat() if pipeline.completed_at else None,
            "config": pipeline.config,
            "metadata": pipeline.metadata,
            "results": pipeline.results,
            "error_message": pipeline.error_message
        })
        
    except Exception as e:
        logger.error(f"Error getting ML pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/artifacts")
async def get_model_artifacts():
    """Get all model artifacts"""
    try:
        artifacts = []
        for artifact_id, artifact in mlops_pipeline_engine.model_artifacts.items():
            artifacts.append({
                "artifact_id": artifact.artifact_id,
                "model_name": artifact.model_name,
                "version": artifact.version,
                "model_version": artifact.model_version.value,
                "pipeline_id": artifact.pipeline_id,
                "stage": artifact.stage.value,
                "artifact_type": artifact.artifact_type,
                "size_bytes": artifact.size_bytes,
                "checksum": artifact.checksum,
                "created_at": artifact.created_at.isoformat(),
                "metadata": artifact.metadata
            })
            
        return JSONResponse(content={"artifacts": artifacts})
        
    except Exception as e:
        logger.error(f"Error getting model artifacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments")
async def get_model_deployments():
    """Get all model deployments"""
    try:
        deployments = []
        for deployment_id, deployment in mlops_pipeline_engine.model_deployments.items():
            deployments.append({
                "deployment_id": deployment.deployment_id,
                "model_name": deployment.model_name,
                "version": deployment.version,
                "environment": deployment.environment,
                "endpoint_url": deployment.endpoint_url,
                "status": deployment.status,
                "created_at": deployment.created_at.isoformat(),
                "updated_at": deployment.updated_at.isoformat(),
                "traffic_percentage": deployment.traffic_percentage,
                "health_checks": deployment.health_checks
            })
            
        return JSONResponse(content={"deployments": deployments})
        
    except Exception as e:
        logger.error(f"Error getting model deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployments")
async def create_model_deployment(deployment_request: ModelDeploymentRequest):
    """Create a new model deployment"""
    try:
        # This would create an actual deployment
        # For now, we'll simulate it
        deployment_id = f"deployment_{int(time.time())}_{secrets.token_hex(4)}"
        
        deployment = {
            "deployment_id": deployment_id,
            "model_name": deployment_request.model_name,
            "version": deployment_request.version,
            "environment": deployment_request.environment,
            "endpoint_url": f"https://api.example.com/models/{deployment_id}",
            "status": "deployed",
            "traffic_percentage": deployment_request.traffic_percentage,
            "created_at": datetime.now().isoformat()
        }
        
        return JSONResponse(content={
            "message": "Model deployment created successfully",
            "deployment": deployment
        })
        
    except Exception as e:
        logger.error(f"Error creating model deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates")
async def get_pipeline_templates():
    """Get available pipeline templates"""
    try:
        templates = []
        for template_name, template_config in mlops_pipeline_engine.pipeline_templates.items():
            templates.append({
                "name": template_name,
                "display_name": template_config["name"],
                "description": template_config["description"],
                "stages": [stage.value for stage in template_config["stages"]],
                "config": template_config["config"]
            })
            
        return JSONResponse(content={"templates": templates})
        
    except Exception as e:
        logger.error(f"Error getting pipeline templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_mlops_dashboard():
    """Get MLOps dashboard data"""
    try:
        summary = mlops_pipeline_engine.get_mlops_summary()
        
        # Get recent pipelines
        recent_pipelines = list(mlops_pipeline_engine.pipelines.values())[-5:]
        
        # Get recent deployments
        recent_deployments = list(mlops_pipeline_engine.model_deployments.values())[-5:]
        
        dashboard_data = {
            "summary": summary,
            "recent_pipelines": [
                {
                    "pipeline_id": pipeline.pipeline_id,
                    "name": pipeline.name,
                    "current_stage": pipeline.current_stage.value,
                    "status": pipeline.status.value,
                    "created_at": pipeline.created_at.isoformat()
                }
                for pipeline in recent_pipelines
            ],
            "recent_deployments": [
                {
                    "deployment_id": deployment.deployment_id,
                    "model_name": deployment.model_name,
                    "version": deployment.version,
                    "environment": deployment.environment,
                    "status": deployment.status,
                    "created_at": deployment.created_at.isoformat()
                }
                for deployment in recent_deployments
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting MLOps dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_mlops():
    """Start MLOps pipeline engine"""
    try:
        await mlops_pipeline_engine.start_mlops_engine()
        return JSONResponse(content={"message": "MLOps pipeline engine started"})
        
    except Exception as e:
        logger.error(f"Error starting MLOps: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_mlops():
    """Stop MLOps pipeline engine"""
    try:
        await mlops_pipeline_engine.stop_mlops_engine()
        return JSONResponse(content={"message": "MLOps pipeline engine stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping MLOps: {e}")
        raise HTTPException(status_code=500, detail=str(e))
