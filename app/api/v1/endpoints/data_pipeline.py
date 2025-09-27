"""
Data Pipeline Management API Endpoints
Provides management and monitoring for data processing pipelines
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import time

from app.core.auth import get_current_user
from app.models.user import User
from app.data_pipeline.pipeline_manager import (
    pipeline_manager, PipelineStep, PipelineStatus
)

router = APIRouter()


@router.post("/pipelines/register")
async def register_pipeline(
    pipeline_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Register a new data processing pipeline"""
    try:
        pipeline_name = pipeline_data["name"]
        steps_data = pipeline_data["steps"]
        
        # Convert step data to PipelineStep objects
        steps = []
        for step_data in steps_data:
            step = PipelineStep(
                name=step_data["name"],
                step_type=step_data["type"],
                config=step_data.get("config", {}),
                dependencies=step_data.get("dependencies", []),
                retry_count=step_data.get("retry_count", 3),
                timeout=step_data.get("timeout", 300),
                enabled=step_data.get("enabled", True)
            )
            steps.append(step)
        
        # Register pipeline
        success = pipeline_manager.register_pipeline(pipeline_name, steps)
        
        if success:
            return {
                "success": True,
                "message": f"Pipeline '{pipeline_name}' registered successfully",
                "pipeline_name": pipeline_name,
                "steps_count": len(steps),
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to register pipeline")
            
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register pipeline: {str(e)}")


@router.post("/pipelines/{pipeline_name}/execute")
async def execute_pipeline(
    pipeline_name: str,
    config: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user)
):
    """Execute a data processing pipeline"""
    try:
        # Execute pipeline
        pipeline_id = await pipeline_manager.execute_pipeline(pipeline_name, config)
        
        return {
            "success": True,
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline_name,
            "status": "started",
            "message": f"Pipeline '{pipeline_name}' execution started",
            "timestamp": time.time()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute pipeline: {str(e)}")


@router.get("/pipelines")
async def list_pipelines(current_user: User = Depends(get_current_user)):
    """List all registered pipelines"""
    try:
        pipelines = {}
        for name, steps in pipeline_manager.pipelines.items():
            pipelines[name] = {
                "name": name,
                "steps_count": len(steps),
                "steps": [
                    {
                        "name": step.name,
                        "type": step.step_type,
                        "dependencies": step.dependencies,
                        "retry_count": step.retry_count,
                        "timeout": step.timeout,
                        "enabled": step.enabled
                    }
                    for step in steps
                ]
            }
        
        return {
            "success": True,
            "data": pipelines,
            "total_pipelines": len(pipelines),
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list pipelines: {str(e)}")


@router.get("/pipelines/{pipeline_name}")
async def get_pipeline_details(
    pipeline_name: str,
    current_user: User = Depends(get_current_user)
):
    """Get detailed information about a specific pipeline"""
    try:
        if pipeline_name not in pipeline_manager.pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
        
        steps = pipeline_manager.pipelines[pipeline_name]
        
        return {
            "success": True,
            "data": {
                "name": pipeline_name,
                "steps_count": len(steps),
                "steps": [
                    {
                        "name": step.name,
                        "type": step.step_type,
                        "config": step.config,
                        "dependencies": step.dependencies,
                        "retry_count": step.retry_count,
                        "timeout": step.timeout,
                        "enabled": step.enabled
                    }
                    for step in steps
                ]
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline details: {str(e)}")


@router.get("/executions/{pipeline_id}")
async def get_execution_status(
    pipeline_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get status of a pipeline execution"""
    try:
        execution = await pipeline_manager.get_pipeline_status(pipeline_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail=f"Pipeline execution '{pipeline_id}' not found")
        
        return {
            "success": True,
            "data": {
                "pipeline_id": execution.pipeline_id,
                "pipeline_name": execution.pipeline_name,
                "status": execution.status.value,
                "start_time": execution.start_time,
                "end_time": execution.end_time,
                "duration": execution.end_time - execution.start_time if execution.end_time else None,
                "steps_completed": execution.steps_completed,
                "steps_failed": execution.steps_failed,
                "error_message": execution.error_message,
                "metrics": execution.metrics
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get execution status: {str(e)}")


@router.get("/executions")
async def list_executions(
    status: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """List pipeline executions with optional filtering"""
    try:
        executions = []
        
        for pipeline_id, execution in pipeline_manager.executions.items():
            # Filter by status if specified
            if status and execution.status.value != status:
                continue
            
            executions.append({
                "pipeline_id": execution.pipeline_id,
                "pipeline_name": execution.pipeline_name,
                "status": execution.status.value,
                "start_time": execution.start_time,
                "end_time": execution.end_time,
                "duration": execution.end_time - execution.start_time if execution.end_time else None,
                "steps_completed": len(execution.steps_completed),
                "steps_failed": len(execution.steps_failed),
                "error_message": execution.error_message
            })
        
        # Sort by start time (newest first)
        executions.sort(key=lambda x: x["start_time"], reverse=True)
        
        # Apply limit
        executions = executions[:limit]
        
        return {
            "success": True,
            "data": executions,
            "total_executions": len(executions),
            "filters": {
                "status": status,
                "limit": limit
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list executions: {str(e)}")


@router.post("/executions/{pipeline_id}/cancel")
async def cancel_execution(
    pipeline_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a running pipeline execution"""
    try:
        success = await pipeline_manager.cancel_pipeline(pipeline_id)
        
        if success:
            return {
                "success": True,
                "message": f"Pipeline execution '{pipeline_id}' cancelled successfully",
                "pipeline_id": pipeline_id,
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Pipeline execution '{pipeline_id}' not found or not running")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel execution: {str(e)}")


@router.get("/metrics")
async def get_pipeline_metrics(current_user: User = Depends(get_current_user)):
    """Get pipeline execution metrics"""
    try:
        metrics = await pipeline_manager.get_pipeline_metrics()
        
        return {
            "success": True,
            "data": metrics,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/cleanup")
async def cleanup_old_executions(
    max_age_hours: int = 24,
    current_user: User = Depends(get_current_user)
):
    """Clean up old pipeline executions"""
    try:
        await pipeline_manager.cleanup_old_executions(max_age_hours)
        
        return {
            "success": True,
            "message": f"Cleaned up executions older than {max_age_hours} hours",
            "max_age_hours": max_age_hours,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup executions: {str(e)}")


@router.post("/pipelines/{pipeline_name}/steps/{step_name}/test")
async def test_pipeline_step(
    pipeline_name: str,
    step_name: str,
    test_data: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
):
    """Test a specific pipeline step"""
    try:
        if pipeline_name not in pipeline_manager.pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_name}' not found")
        
        steps = pipeline_manager.pipelines[pipeline_name]
        step = next((s for s in steps if s.name == step_name), None)
        
        if not step:
            raise HTTPException(status_code=404, detail=f"Step '{step_name}' not found in pipeline '{pipeline_name}'")
        
        # Test the step
        start_time = time.time()
        success = await pipeline_manager._execute_step_with_retry(step, test_data)
        end_time = time.time()
        
        return {
            "success": success,
            "step_name": step_name,
            "pipeline_name": pipeline_name,
            "execution_time": end_time - start_time,
            "message": f"Step test {'passed' if success else 'failed'}",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to test step: {str(e)}")


@router.get("/step-handlers")
async def list_step_handlers(current_user: User = Depends(get_current_user)):
    """List available step handlers"""
    try:
        handlers = list(pipeline_manager.step_handlers.keys())
        
        return {
            "success": True,
            "data": {
                "available_handlers": handlers,
                "handler_count": len(handlers)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list step handlers: {str(e)}")


@router.post("/step-handlers/{handler_type}")
async def register_step_handler(
    handler_type: str,
    handler_info: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Register a custom step handler"""
    try:
        # In a real implementation, you would register the actual handler function
        # For now, we'll just acknowledge the registration
        
        return {
            "success": True,
            "message": f"Step handler '{handler_type}' registration acknowledged",
            "handler_type": handler_type,
            "handler_info": handler_info,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register step handler: {str(e)}")


@router.get("/data/{step_name}")
async def get_pipeline_data(
    step_name: str,
    current_user: User = Depends(get_current_user)
):
    """Get data from a specific pipeline step"""
    try:
        from app.core.enhanced_cache import enhanced_cache
        
        # Try to get data from various possible cache keys
        possible_keys = [
            f"pipeline_data_{step_name}",
            f"pipeline_data_{step_name}_transformed",
            f"pipeline_data_{step_name}_aggregated",
            f"pipeline_data_{step_name}_enriched"
        ]
        
        data = None
        data_key = None
        
        for key in possible_keys:
            cached_data = await enhanced_cache.get(key)
            if cached_data:
                data = cached_data
                data_key = key
                break
        
        if not data:
            raise HTTPException(status_code=404, detail=f"No data found for step '{step_name}'")
        
        return {
            "success": True,
            "data": data,
            "data_key": data_key,
            "step_name": step_name,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline data: {str(e)}")


@router.delete("/data/{step_name}")
async def clear_pipeline_data(
    step_name: str,
    current_user: User = Depends(get_current_user)
):
    """Clear data from a specific pipeline step"""
    try:
        from app.core.enhanced_cache import enhanced_cache
        
        # Clear data from various possible cache keys
        possible_keys = [
            f"pipeline_data_{step_name}",
            f"pipeline_data_{step_name}_transformed",
            f"pipeline_data_{step_name}_aggregated",
            f"pipeline_data_{step_name}_enriched"
        ]
        
        cleared_keys = []
        for key in possible_keys:
            if await enhanced_cache.get(key):
                await enhanced_cache.delete(key)
                cleared_keys.append(key)
        
        return {
            "success": True,
            "message": f"Cleared data for step '{step_name}'",
            "step_name": step_name,
            "cleared_keys": cleared_keys,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear pipeline data: {str(e)}")


@router.get("/health")
async def pipeline_health_check(current_user: User = Depends(get_current_user)):
    """Get pipeline system health status"""
    try:
        metrics = await pipeline_manager.get_pipeline_metrics()
        
        # Determine health status
        if metrics["running_executions"] > pipeline_manager.max_concurrent_pipelines:
            health_status = "degraded"
        elif metrics["success_rate"] < 0.8:
            health_status = "degraded"
        else:
            health_status = "healthy"
        
        return {
            "success": True,
            "data": {
                "status": health_status,
                "metrics": metrics,
                "max_concurrent_pipelines": pipeline_manager.max_concurrent_pipelines,
                "registered_pipelines": len(pipeline_manager.pipelines),
                "available_handlers": len(pipeline_manager.step_handlers)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")
