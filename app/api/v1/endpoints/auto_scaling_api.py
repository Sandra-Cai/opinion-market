"""
Auto Scaling API
Manage intelligent auto-scaling based on performance metrics
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime

from app.core.auth import get_current_user
from app.models.user import User
from app.services.auto_scaling_manager import auto_scaling_manager, ScalingPolicy

router = APIRouter()


@router.post("/start-scaling")
async def start_auto_scaling(
    current_user: User = Depends(get_current_user)
):
    """Start the auto-scaling manager"""
    try:
        await auto_scaling_manager.start_scaling()
        return {
            "success": True,
            "message": "Auto-scaling manager started",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start auto-scaling: {str(e)}")


@router.post("/stop-scaling")
async def stop_auto_scaling(
    current_user: User = Depends(get_current_user)
):
    """Stop the auto-scaling manager"""
    try:
        await auto_scaling_manager.stop_scaling()
        return {
            "success": True,
            "message": "Auto-scaling manager stopped",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop auto-scaling: {str(e)}")


@router.get("/summary")
async def get_scaling_summary(
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive auto-scaling summary"""
    try:
        summary = auto_scaling_manager.get_scaling_summary()
        
        return {
            "success": True,
            "data": summary,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scaling summary: {str(e)}")


@router.get("/policies")
async def get_scaling_policies(
    current_user: User = Depends(get_current_user)
):
    """Get all scaling policies"""
    try:
        policies = {}
        for name, policy in auto_scaling_manager.scaling_policies.items():
            policies[name] = {
                "name": policy.name,
                "metric_name": policy.metric_name,
                "scale_up_threshold": policy.scale_up_threshold,
                "scale_down_threshold": policy.scale_down_threshold,
                "min_instances": policy.min_instances,
                "max_instances": policy.max_instances,
                "cooldown_period": policy.cooldown_period,
                "enabled": policy.enabled,
                "last_action_time": policy.last_action_time.isoformat() if policy.last_action_time else None
            }
            
        return {
            "success": True,
            "data": {
                "policies": policies,
                "total_policies": len(policies)
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scaling policies: {str(e)}")


@router.post("/policies")
async def create_scaling_policy(
    policy_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Create a new scaling policy"""
    try:
        # Validate required fields
        required_fields = ["name", "metric_name", "scale_up_threshold", "scale_down_threshold"]
        for field in required_fields:
            if field not in policy_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
                
        # Create scaling policy
        policy = ScalingPolicy(
            name=policy_data["name"],
            metric_name=policy_data["metric_name"],
            scale_up_threshold=policy_data["scale_up_threshold"],
            scale_down_threshold=policy_data["scale_down_threshold"],
            min_instances=policy_data.get("min_instances", 1),
            max_instances=policy_data.get("max_instances", 10),
            cooldown_period=policy_data.get("cooldown_period", 300),
            enabled=policy_data.get("enabled", True)
        )
        
        auto_scaling_manager.add_scaling_policy(policy)
        
        return {
            "success": True,
            "message": f"Scaling policy '{policy.name}' created successfully",
            "data": {
                "name": policy.name,
                "metric_name": policy.metric_name,
                "scale_up_threshold": policy.scale_up_threshold,
                "scale_down_threshold": policy.scale_down_threshold,
                "min_instances": policy.min_instances,
                "max_instances": policy.max_instances,
                "cooldown_period": policy.cooldown_period,
                "enabled": policy.enabled
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create scaling policy: {str(e)}")


@router.put("/policies/{policy_name}")
async def update_scaling_policy(
    policy_name: str,
    policy_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Update an existing scaling policy"""
    try:
        if policy_name not in auto_scaling_manager.scaling_policies:
            raise HTTPException(status_code=404, detail=f"Scaling policy '{policy_name}' not found")
            
        # Update policy
        auto_scaling_manager.update_scaling_policy(policy_name, **policy_data)
        
        # Get updated policy
        updated_policy = auto_scaling_manager.scaling_policies[policy_name]
        
        return {
            "success": True,
            "message": f"Scaling policy '{policy_name}' updated successfully",
            "data": {
                "name": updated_policy.name,
                "metric_name": updated_policy.metric_name,
                "scale_up_threshold": updated_policy.scale_up_threshold,
                "scale_down_threshold": updated_policy.scale_down_threshold,
                "min_instances": updated_policy.min_instances,
                "max_instances": updated_policy.max_instances,
                "cooldown_period": updated_policy.cooldown_period,
                "enabled": updated_policy.enabled
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update scaling policy: {str(e)}")


@router.delete("/policies/{policy_name}")
async def delete_scaling_policy(
    policy_name: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a scaling policy"""
    try:
        if policy_name not in auto_scaling_manager.scaling_policies:
            raise HTTPException(status_code=404, detail=f"Scaling policy '{policy_name}' not found")
            
        auto_scaling_manager.remove_scaling_policy(policy_name)
        
        return {
            "success": True,
            "message": f"Scaling policy '{policy_name}' deleted successfully",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete scaling policy: {str(e)}")


@router.get("/decisions")
async def get_scaling_decisions(
    limit: int = 50,
    action: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get scaling decisions"""
    try:
        decisions = auto_scaling_manager.scaling_decisions
        
        # Filter by action if specified
        if action:
            decisions = [d for d in decisions if d.action == action]
            
        # Limit results
        decisions = decisions[-limit:]
        
        return {
            "success": True,
            "data": [
                {
                    "action": d.action,
                    "target_instances": d.target_instances,
                    "current_instances": d.current_instances,
                    "reason": d.reason,
                    "confidence": d.confidence,
                    "metrics": d.metrics,
                    "predicted_impact": d.predicted_impact,
                    "created_at": d.created_at.isoformat()
                }
                for d in decisions
            ],
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scaling decisions: {str(e)}")


@router.post("/scale-manual")
async def manual_scale(
    target_instances: int,
    reason: str = "Manual scaling request",
    current_user: User = Depends(get_current_user)
):
    """Manually trigger scaling to a specific number of instances"""
    try:
        # Validate target instances
        if target_instances < 1:
            raise HTTPException(status_code=400, detail="Target instances must be at least 1")
            
        # Create manual scaling decision
        from app.services.auto_scaling_manager import ScalingDecision
        
        decision = ScalingDecision(
            action="scale_up" if target_instances > auto_scaling_manager.current_instances else "scale_down",
            target_instances=target_instances,
            current_instances=auto_scaling_manager.current_instances,
            reason=reason,
            confidence=1.0,
            metrics={},
            predicted_impact={}
        )
        
        # Execute scaling action
        success = await auto_scaling_manager._execute_scaling_action(decision)
        
        if success:
            auto_scaling_manager.current_instances = target_instances
            auto_scaling_manager.target_instances = target_instances
            auto_scaling_manager.scaling_decisions.append(decision)
            
            return {
                "success": True,
                "message": f"Successfully scaled to {target_instances} instances",
                "data": {
                    "target_instances": target_instances,
                    "previous_instances": decision.current_instances,
                    "reason": reason
                },
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to execute scaling action")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform manual scaling: {str(e)}")


@router.get("/metrics")
async def get_scaling_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get auto-scaling performance metrics"""
    try:
        metrics = auto_scaling_manager.scaling_metrics
        
        return {
            "success": True,
            "data": {
                "scaling_actions_taken": metrics["scaling_actions_taken"],
                "successful_scales": metrics["successful_scales"],
                "failed_scales": metrics["failed_scales"],
                "average_scale_time": metrics["average_scale_time"],
                "cost_savings": metrics["cost_savings"],
                "success_rate": (
                    metrics["successful_scales"] / max(1, metrics["scaling_actions_taken"]) * 100
                )
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scaling metrics: {str(e)}")


@router.get("/health")
async def get_scaling_health(
    current_user: User = Depends(get_current_user)
):
    """Get auto-scaling health status"""
    try:
        summary = auto_scaling_manager.get_scaling_summary()
        
        health_status = {
            "scaling_active": summary.get("scaling_active", False),
            "current_instances": summary.get("current_instances", 1),
            "target_instances": summary.get("target_instances", 1),
            "active_policies": len([p for p in summary.get("policies", {}).values() if p.get("enabled", False)]),
            "total_policies": len(summary.get("policies", {})),
            "recent_decisions": len(summary.get("recent_decisions", [])),
            "status": "healthy" if summary.get("scaling_active", False) else "inactive"
        }
        
        return {
            "success": True,
            "data": health_status,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")


@router.post("/evaluate-scaling")
async def evaluate_scaling_manually(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Manually trigger scaling evaluation"""
    try:
        # Run scaling evaluation in background
        background_tasks.add_task(auto_scaling_manager._evaluate_scaling_decisions)
        
        return {
            "success": True,
            "message": "Scaling evaluation triggered",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate scaling: {str(e)}")


@router.get("/recommendations")
async def get_scaling_recommendations(
    current_user: User = Depends(get_current_user)
):
    """Get intelligent scaling recommendations"""
    try:
        recommendations = []
        
        # Get current performance metrics
        from app.core.advanced_performance_optimizer import advanced_performance_optimizer
        perf_summary = advanced_performance_optimizer.get_performance_summary()
        current_metrics = perf_summary.get("metrics", {})
        
        # Analyze each policy
        for policy_name, policy in auto_scaling_manager.scaling_policies.items():
            if not policy.enabled:
                continue
                
            current_value = current_metrics.get(policy.metric_name, {}).get("current", 0)
            
            if current_value > policy.scale_up_threshold:
                recommendations.append({
                    "type": "scale_up",
                    "policy": policy_name,
                    "metric": policy.metric_name,
                    "current_value": current_value,
                    "threshold": policy.scale_up_threshold,
                    "recommended_instances": min(
                        policy.max_instances,
                        int(auto_scaling_manager.current_instances * 1.5)
                    ),
                    "reason": f"{policy.metric_name} exceeds scale-up threshold",
                    "priority": "high" if current_value > policy.scale_up_threshold * 1.2 else "medium"
                })
                
            elif current_value < policy.scale_down_threshold:
                recommendations.append({
                    "type": "scale_down",
                    "policy": policy_name,
                    "metric": policy.metric_name,
                    "current_value": current_value,
                    "threshold": policy.scale_down_threshold,
                    "recommended_instances": max(
                        policy.min_instances,
                        int(auto_scaling_manager.current_instances * 0.7)
                    ),
                    "reason": f"{policy.metric_name} below scale-down threshold",
                    "priority": "low"
                })
                
        return {
            "success": True,
            "data": {
                "recommendations": recommendations,
                "total_recommendations": len(recommendations),
                "current_instances": auto_scaling_manager.current_instances
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scaling recommendations: {str(e)}")


