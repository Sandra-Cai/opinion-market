"""
API endpoints for Edge Computing Engine
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

from app.services.edge_computing_engine import edge_computing_engine, TaskPriority, EdgeNodeType

logger = logging.getLogger(__name__)

router = APIRouter()

class EdgeTaskRequest(BaseModel):
    task_type: str
    data: Dict[str, Any]
    priority: int = 2

class EdgeWorkloadRequest(BaseModel):
    tasks: List[Dict[str, Any]]
    distribution_strategy: str = "round_robin"

class EdgeNodeResponse(BaseModel):
    node_id: str
    node_type: str
    location: str
    capacity: Dict[str, float]
    status: str
    last_heartbeat: str
    metadata: Dict[str, Any]

class EdgeTaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]]
    created_at: str

class EdgeWorkloadResponse(BaseModel):
    workload_id: str
    status: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    pending_tasks: int
    distribution_strategy: str
    created_at: str

class EdgePerformanceResponse(BaseModel):
    performance_metrics: Dict[str, Any]
    total_nodes: int
    active_nodes: int
    total_tasks: int
    pending_tasks: int
    completed_tasks: int
    failed_tasks: int

@router.post("/submit-task")
async def submit_edge_task(request: EdgeTaskRequest):
    """Submit task to edge computing engine"""
    try:
        priority = TaskPriority(request.priority)
        task_id = await edge_computing_engine.submit_edge_task(
            request.task_type,
            request.data,
            priority
        )
        
        if task_id:
            return {"task_id": task_id, "status": "submitted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to submit task")
            
    except Exception as e:
        logger.error(f"Error submitting edge task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/task/{task_id}")
async def get_edge_task_status(task_id: str):
    """Get edge task status"""
    try:
        status = await edge_computing_engine.get_edge_task_status(task_id)
        
        if status:
            return EdgeTaskResponse(**status)
        else:
            raise HTTPException(status_code=404, detail="Task not found")
            
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nodes")
async def get_edge_nodes():
    """Get all edge nodes"""
    try:
        nodes = await edge_computing_engine.get_edge_nodes()
        return {"nodes": nodes, "total": len(nodes)}
        
    except Exception as e:
        logger.error(f"Error getting edge nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nodes/{node_type}")
async def get_edge_nodes_by_type(node_type: str):
    """Get edge nodes by type"""
    try:
        all_nodes = await edge_computing_engine.get_edge_nodes()
        filtered_nodes = [node for node in all_nodes if node["node_type"] == node_type]
        
        return {"nodes": filtered_nodes, "total": len(filtered_nodes)}
        
    except Exception as e:
        logger.error(f"Error getting nodes by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_edge_performance():
    """Get edge computing performance metrics"""
    try:
        metrics = await edge_computing_engine.get_edge_performance_metrics()
        return EdgePerformanceResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workload")
async def create_edge_workload(request: EdgeWorkloadRequest):
    """Create edge workload"""
    try:
        workload_id = await edge_computing_engine.create_edge_workload(
            request.tasks,
            request.distribution_strategy
        )
        
        if workload_id:
            return {"workload_id": workload_id, "status": "created"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create workload")
            
    except Exception as e:
        logger.error(f"Error creating edge workload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workload/{workload_id}")
async def get_edge_workload_status(workload_id: str):
    """Get edge workload status"""
    try:
        status = await edge_computing_engine.get_edge_workload_status(workload_id)
        
        if status:
            return EdgeWorkloadResponse(**status)
        else:
            raise HTTPException(status_code=404, detail="Workload not found")
            
    except Exception as e:
        logger.error(f"Error getting workload status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workloads")
async def get_all_edge_workloads():
    """Get all edge workloads"""
    try:
        workloads = []
        for workload_id in edge_computing_engine.edge_workloads:
            status = await edge_computing_engine.get_edge_workload_status(workload_id)
            if status:
                workloads.append(status)
        
        return {"workloads": workloads, "total": len(workloads)}
        
    except Exception as e:
        logger.error(f"Error getting all workloads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks")
async def get_all_edge_tasks():
    """Get all edge tasks"""
    try:
        tasks = []
        for task_id in edge_computing_engine.edge_tasks:
            status = await edge_computing_engine.get_edge_task_status(task_id)
            if status:
                tasks.append(status)
        
        return {"tasks": tasks, "total": len(tasks)}
        
    except Exception as e:
        logger.error(f"Error getting all tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{status}")
async def get_edge_tasks_by_status(status: str):
    """Get edge tasks by status"""
    try:
        all_tasks = []
        for task_id in edge_computing_engine.edge_tasks:
            task_status = await edge_computing_engine.get_edge_task_status(task_id)
            if task_status and task_status["status"] == status:
                all_tasks.append(task_status)
        
        return {"tasks": all_tasks, "total": len(all_tasks)}
        
    except Exception as e:
        logger.error(f"Error getting tasks by status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_edge_health():
    """Get edge computing engine health"""
    try:
        metrics = await edge_computing_engine.get_edge_performance_metrics()
        
        health_status = "healthy"
        if metrics["active_nodes"] == 0:
            health_status = "unhealthy"
        elif metrics["active_nodes"] < metrics["total_nodes"] * 0.5:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "active_nodes": metrics["active_nodes"],
            "total_nodes": metrics["total_nodes"],
            "edge_utilization": metrics["performance_metrics"]["edge_utilization"],
            "network_latency": metrics["performance_metrics"]["network_latency"]
        }
        
    except Exception as e:
        logger.error(f"Error getting edge health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nodes/{node_id}/maintenance")
async def set_node_maintenance(node_id: str, maintenance: bool = True):
    """Set node maintenance mode"""
    try:
        if node_id in edge_computing_engine.edge_nodes:
            node = edge_computing_engine.edge_nodes[node_id]
            if maintenance:
                node.status = edge_computing_engine.EdgeNodeStatus.MAINTENANCE
            else:
                node.status = edge_computing_engine.EdgeNodeStatus.ACTIVE
            
            return {"node_id": node_id, "maintenance": maintenance, "status": node.status.value}
        else:
            raise HTTPException(status_code=404, detail="Node not found")
            
    except Exception as e:
        logger.error(f"Error setting node maintenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_edge_stats():
    """Get comprehensive edge computing statistics"""
    try:
        metrics = await edge_computing_engine.get_edge_performance_metrics()
        nodes = await edge_computing_engine.get_edge_nodes()
        
        # Calculate additional statistics
        node_types = {}
        for node in nodes:
            node_type = node["node_type"]
            if node_type not in node_types:
                node_types[node_type] = {"total": 0, "active": 0}
            node_types[node_type]["total"] += 1
            if node["status"] == "active":
                node_types[node_type]["active"] += 1
        
        return {
            "performance_metrics": metrics["performance_metrics"],
            "node_statistics": {
                "total_nodes": metrics["total_nodes"],
                "active_nodes": metrics["active_nodes"],
                "node_types": node_types
            },
            "task_statistics": {
                "total_tasks": metrics["total_tasks"],
                "pending_tasks": metrics["pending_tasks"],
                "completed_tasks": metrics["completed_tasks"],
                "failed_tasks": metrics["failed_tasks"]
            },
            "workload_statistics": {
                "total_workloads": len(edge_computing_engine.edge_workloads)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting edge stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
