"""
AI Optimization API Endpoints
Provides AI-powered system optimization and recommendations
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import time

from app.core.auth import get_current_user
from app.models.user import User
from app.core.ai_optimizer import ai_optimizer

router = APIRouter()


@router.get("/recommendations")
async def get_optimization_recommendations(current_user: User = Depends(get_current_user)):
    """Get current AI optimization recommendations"""
    try:
        recommendations = ai_optimizer.get_current_recommendations()
        
        return {
            "success": True,
            "data": recommendations,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.get("/predictions")
async def get_performance_predictions(current_user: User = Depends(get_current_user)):
    """Get AI performance predictions"""
    try:
        predictions = ai_optimizer.get_performance_predictions()
        
        return {
            "success": True,
            "data": predictions,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")


@router.get("/history")
async def get_optimization_history(current_user: User = Depends(get_current_user)):
    """Get optimization history"""
    try:
        history = ai_optimizer.get_optimization_history()
        
        return {
            "success": True,
            "data": history,
            "count": len(history),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization history: {str(e)}")


@router.post("/start-monitoring")
async def start_ai_monitoring(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Start AI optimization monitoring"""
    try:
        await ai_optimizer.start_optimization_monitoring()
        
        return {
            "success": True,
            "message": "AI optimization monitoring started",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start AI monitoring: {str(e)}")


@router.post("/stop-monitoring")
async def stop_ai_monitoring(current_user: User = Depends(get_current_user)):
    """Stop AI optimization monitoring"""
    try:
        await ai_optimizer.stop_optimization_monitoring()
        
        return {
            "success": True,
            "message": "AI optimization monitoring stopped",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop AI monitoring: {str(e)}")


@router.get("/status")
async def get_ai_optimization_status(current_user: User = Depends(get_current_user)):
    """Get AI optimization system status"""
    try:
        recommendations = ai_optimizer.get_current_recommendations()
        history = ai_optimizer.get_optimization_history()
        
        status = {
            "monitoring_active": True,  # Would check actual status
            "recommendations_count": recommendations.get("count", 0),
            "optimizations_applied": len(history),
            "last_optimization": history[-1]["applied_at"] if history else None,
            "system_health": "healthy"  # Would calculate actual health
        }
        
        return {
            "success": True,
            "data": status,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AI optimization status: {str(e)}")