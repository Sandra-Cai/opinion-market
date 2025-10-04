"""
API endpoints for Autonomous Systems Engine
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

from app.services.autonomous_systems_engine import autonomous_systems_engine, SystemComponent, HealthStatus, RecoveryAction, AlertLevel

logger = logging.getLogger(__name__)

router = APIRouter()

class SystemNodeResponse(BaseModel):
    node_id: str
    component: str
    status: str
    health_score: float
    last_health_check: str
    dependencies: List[str]
    recovery_actions: List[str]
    metadata: Dict[str, Any]

class SystemAlertResponse(BaseModel):
    alert_id: str
    node_id: str
    alert_level: str
    message: str
    timestamp: str
    acknowledged: bool
    resolved: bool

class RecoveryPlanResponse(BaseModel):
    plan_id: str
    node_id: str
    issue_description: str
    recovery_actions: List[str]
    estimated_downtime: float
    success_probability: float
    created_at: str
    executed: bool

class AutonomousPerformanceResponse(BaseModel):
    performance_metrics: Dict[str, Any]
    total_nodes: int
    healthy_nodes: int
    total_alerts: int
    active_alerts: int
    total_recovery_plans: int
    executed_plans: int

@router.get("/nodes")
async def get_system_nodes():
    """Get all system nodes"""
    try:
        nodes = await autonomous_systems_engine.get_system_nodes()
        return {"nodes": nodes, "total": len(nodes)}
        
    except Exception as e:
        logger.error(f"Error getting system nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nodes/{node_id}")
async def get_system_node(node_id: str):
    """Get system node by ID"""
    try:
        nodes = await autonomous_systems_engine.get_system_nodes()
        
        for node in nodes:
            if node["node_id"] == node_id:
                return SystemNodeResponse(**node)
        
        raise HTTPException(status_code=404, detail="Node not found")
        
    except Exception as e:
        logger.error(f"Error getting system node: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nodes/component/{component}")
async def get_nodes_by_component(component: str):
    """Get system nodes by component"""
    try:
        all_nodes = await autonomous_systems_engine.get_system_nodes()
        filtered_nodes = [node for node in all_nodes if node["component"] == component]
        
        return {"nodes": filtered_nodes, "total": len(filtered_nodes)}
        
    except Exception as e:
        logger.error(f"Error getting nodes by component: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nodes/status/{status}")
async def get_nodes_by_status(status: str):
    """Get system nodes by status"""
    try:
        all_nodes = await autonomous_systems_engine.get_system_nodes()
        filtered_nodes = [node for node in all_nodes if node["status"] == status]
        
        return {"nodes": filtered_nodes, "total": len(filtered_nodes)}
        
    except Exception as e:
        logger.error(f"Error getting nodes by status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_system_alerts():
    """Get all system alerts"""
    try:
        alerts = await autonomous_systems_engine.get_system_alerts()
        return {"alerts": alerts, "total": len(alerts)}
        
    except Exception as e:
        logger.error(f"Error getting system alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/{alert_id}")
async def get_system_alert(alert_id: str):
    """Get system alert by ID"""
    try:
        alerts = await autonomous_systems_engine.get_system_alerts()
        
        for alert in alerts:
            if alert["alert_id"] == alert_id:
                return SystemAlertResponse(**alert)
        
        raise HTTPException(status_code=404, detail="Alert not found")
        
    except Exception as e:
        logger.error(f"Error getting system alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/level/{level}")
async def get_alerts_by_level(level: str):
    """Get system alerts by level"""
    try:
        all_alerts = await autonomous_systems_engine.get_system_alerts()
        filtered_alerts = [alert for alert in all_alerts if alert["alert_level"] == level]
        
        return {"alerts": filtered_alerts, "total": len(filtered_alerts)}
        
    except Exception as e:
        logger.error(f"Error getting alerts by level: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/active")
async def get_active_alerts():
    """Get active (unresolved) system alerts"""
    try:
        all_alerts = await autonomous_systems_engine.get_system_alerts()
        active_alerts = [alert for alert in all_alerts if not alert["resolved"]]
        
        return {"alerts": active_alerts, "total": len(active_alerts)}
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge system alert"""
    try:
        success = await autonomous_systems_engine.acknowledge_alert(alert_id)
        
        if success:
            return {"alert_id": alert_id, "status": "acknowledged"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve system alert"""
    try:
        success = await autonomous_systems_engine.resolve_alert(alert_id)
        
        if success:
            return {"alert_id": alert_id, "status": "resolved"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recovery-plans")
async def get_recovery_plans():
    """Get all recovery plans"""
    try:
        plans = await autonomous_systems_engine.get_recovery_plans()
        return {"recovery_plans": plans, "total": len(plans)}
        
    except Exception as e:
        logger.error(f"Error getting recovery plans: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recovery-plans/{plan_id}")
async def get_recovery_plan(plan_id: str):
    """Get recovery plan by ID"""
    try:
        plans = await autonomous_systems_engine.get_recovery_plans()
        
        for plan in plans:
            if plan["plan_id"] == plan_id:
                return RecoveryPlanResponse(**plan)
        
        raise HTTPException(status_code=404, detail="Recovery plan not found")
        
    except Exception as e:
        logger.error(f"Error getting recovery plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recovery-plans/executed")
async def get_executed_recovery_plans():
    """Get executed recovery plans"""
    try:
        all_plans = await autonomous_systems_engine.get_recovery_plans()
        executed_plans = [plan for plan in all_plans if plan["executed"]]
        
        return {"recovery_plans": executed_plans, "total": len(executed_plans)}
        
    except Exception as e:
        logger.error(f"Error getting executed recovery plans: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recovery-plans/pending")
async def get_pending_recovery_plans():
    """Get pending (unexecuted) recovery plans"""
    try:
        all_plans = await autonomous_systems_engine.get_recovery_plans()
        pending_plans = [plan for plan in all_plans if not plan["executed"]]
        
        return {"recovery_plans": pending_plans, "total": len(pending_plans)}
        
    except Exception as e:
        logger.error(f"Error getting pending recovery plans: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_autonomous_performance():
    """Get autonomous systems performance metrics"""
    try:
        metrics = await autonomous_systems_engine.get_autonomous_performance_metrics()
        return AutonomousPerformanceResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_autonomous_health():
    """Get autonomous systems engine health"""
    try:
        metrics = await autonomous_systems_engine.get_autonomous_performance_metrics()
        
        health_status = "healthy"
        if metrics["healthy_nodes"] == 0:
            health_status = "unhealthy"
        elif metrics["healthy_nodes"] < metrics["total_nodes"] * 0.8:
            health_status = "degraded"
        
        return {
            "status": health_status,
            "total_nodes": metrics["total_nodes"],
            "healthy_nodes": metrics["healthy_nodes"],
            "system_uptime": metrics["performance_metrics"]["system_uptime"],
            "active_alerts": metrics["active_alerts"],
            "self_healing_success_rate": metrics["performance_metrics"]["self_healing_success_rate"]
        }
        
    except Exception as e:
        logger.error(f"Error getting autonomous health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/components")
async def get_system_components():
    """Get available system components"""
    try:
        components = [
            {"name": component.value, "description": f"System component: {component.value}"}
            for component in SystemComponent
        ]
        
        return {"components": components, "total": len(components)}
        
    except Exception as e:
        logger.error(f"Error getting system components: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health-statuses")
async def get_health_statuses():
    """Get available health statuses"""
    try:
        statuses = [
            {"name": status.value, "description": f"Health status: {status.value}"}
            for status in HealthStatus
        ]
        
        return {"health_statuses": statuses, "total": len(statuses)}
        
    except Exception as e:
        logger.error(f"Error getting health statuses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recovery-actions")
async def get_recovery_actions():
    """Get available recovery actions"""
    try:
        actions = [
            {"name": action.value, "description": f"Recovery action: {action.value}"}
            for action in RecoveryAction
        ]
        
        return {"recovery_actions": actions, "total": len(actions)}
        
    except Exception as e:
        logger.error(f"Error getting recovery actions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alert-levels")
async def get_alert_levels():
    """Get available alert levels"""
    try:
        levels = [
            {"name": level.value, "description": f"Alert level: {level.value}"}
            for level in AlertLevel
        ]
        
        return {"alert_levels": levels, "total": len(levels)}
        
    except Exception as e:
        logger.error(f"Error getting alert levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_autonomous_stats():
    """Get comprehensive autonomous systems statistics"""
    try:
        metrics = await autonomous_systems_engine.get_autonomous_performance_metrics()
        nodes = await autonomous_systems_engine.get_system_nodes()
        alerts = await autonomous_systems_engine.get_system_alerts()
        plans = await autonomous_systems_engine.get_recovery_plans()
        
        # Calculate additional statistics
        component_stats = {}
        status_stats = {}
        alert_level_stats = {}
        
        for node in nodes:
            component = node["component"]
            status = node["status"]
            
            component_stats[component] = component_stats.get(component, 0) + 1
            status_stats[status] = status_stats.get(status, 0) + 1
        
        for alert in alerts:
            level = alert["alert_level"]
            alert_level_stats[level] = alert_level_stats.get(level, 0) + 1
        
        return {
            "performance_metrics": metrics["performance_metrics"],
            "node_statistics": {
                "total_nodes": metrics["total_nodes"],
                "healthy_nodes": metrics["healthy_nodes"],
                "component_distribution": component_stats,
                "status_distribution": status_stats
            },
            "alert_statistics": {
                "total_alerts": metrics["total_alerts"],
                "active_alerts": metrics["active_alerts"],
                "level_distribution": alert_level_stats
            },
            "recovery_statistics": {
                "total_recovery_plans": metrics["total_recovery_plans"],
                "executed_plans": metrics["executed_plans"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting autonomous stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
