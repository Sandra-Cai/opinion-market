"""
Database Optimization API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
import time

from app.core.auth import get_current_user
from app.models.user import User
from app.core.database_optimizer import database_optimizer

router = APIRouter()


@router.get("/metrics")
async def get_database_metrics(current_user: User = Depends(get_current_user)):
    """Get database performance metrics"""
    try:
        metrics = await database_optimizer.get_performance_metrics()
        return {
            "success": True,
            "data": metrics,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database metrics: {str(e)}")


@router.get("/slow-queries")
async def get_slow_queries(
    limit: int = 10,
    current_user: User = Depends(get_current_user)
):
    """Get slow queries"""
    try:
        slow_queries = await database_optimizer.get_slow_queries(limit)
        return {
            "success": True,
            "data": slow_queries,
            "count": len(slow_queries),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get slow queries: {str(e)}")


@router.get("/optimization-suggestions")
async def get_optimization_suggestions(current_user: User = Depends(get_current_user)):
    """Get query optimization suggestions"""
    try:
        suggestions = await database_optimizer.optimize_queries()
        return {
            "success": True,
            "data": suggestions,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization suggestions: {str(e)}")


@router.get("/index-recommendations")
async def get_index_recommendations(current_user: User = Depends(get_current_user)):
    """Get index optimization recommendations"""
    try:
        recommendations = await database_optimizer.get_index_recommendations()
        return {
            "success": True,
            "data": recommendations,
            "count": len(recommendations),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get index recommendations: {str(e)}")


@router.post("/start-monitoring")
async def start_database_monitoring(current_user: User = Depends(get_current_user)):
    """Start database performance monitoring"""
    try:
        await database_optimizer.start_monitoring()
        return {
            "success": True,
            "message": "Database monitoring started",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.get("/performance-summary")
async def get_performance_summary(current_user: User = Depends(get_current_user)):
    """Get database performance summary"""
    try:
        metrics = await database_optimizer.get_performance_metrics()
        slow_queries = await database_optimizer.get_slow_queries(5)
        suggestions = await database_optimizer.optimize_queries()
        
        # Calculate performance score
        performance_score = 100
        if metrics["metrics"]["cache_hit_ratio"] < 0.95:
            performance_score -= 20
        if len(slow_queries) > 5:
            performance_score -= 30
        if metrics["metrics"]["connection_count"] > 80:
            performance_score -= 25
        
        summary = {
            "performance_score": max(performance_score, 0),
            "status": "excellent" if performance_score > 90 else "good" if performance_score > 70 else "needs_attention",
            "metrics": metrics,
            "recent_slow_queries": slow_queries,
            "optimization_suggestions": suggestions["optimization_suggestions"][:5]
        }
        
        return {
            "success": True,
            "data": summary,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")
