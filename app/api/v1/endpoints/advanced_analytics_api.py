"""
Advanced Analytics API
Expose machine learning insights and predictive analytics
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import time
from datetime import datetime, timedelta

from app.core.auth import get_current_user
from app.models.user import User
from app.services.advanced_analytics_engine import advanced_analytics_engine
from app.core.advanced_performance_optimizer import advanced_performance_optimizer

router = APIRouter()


@router.post("/start-analytics")
async def start_advanced_analytics(
    current_user: User = Depends(get_current_user)
):
    """Start the advanced analytics engine"""
    try:
        await advanced_analytics_engine.start_analytics()
        return {
            "success": True,
            "message": "Advanced analytics engine started",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analytics: {str(e)}")


@router.post("/stop-analytics")
async def stop_advanced_analytics(
    current_user: User = Depends(get_current_user)
):
    """Stop the advanced analytics engine"""
    try:
        await advanced_analytics_engine.stop_analytics()
        return {
            "success": True,
            "message": "Advanced analytics engine stopped",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop analytics: {str(e)}")


@router.get("/summary")
async def get_analytics_summary(
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive analytics summary"""
    try:
        summary = advanced_analytics_engine.get_analytics_summary()
        
        return {
            "success": True,
            "data": summary,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics summary: {str(e)}")


@router.get("/predictions")
async def get_predictions(
    metric_name: Optional[str] = None,
    horizon: Optional[int] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get predictions from the analytics engine"""
    try:
        predictions = advanced_analytics_engine.predictions
        
        # Filter by metric name if specified
        if metric_name:
            predictions = [p for p in predictions if p.metric_name == metric_name]
            
        # Filter by horizon if specified
        if horizon:
            predictions = [p for p in predictions if p.time_horizon == horizon]
            
        # Limit results
        predictions = predictions[-limit:]
        
        return {
            "success": True,
            "data": [
                {
                    "metric_name": p.metric_name,
                    "predicted_value": p.predicted_value,
                    "confidence": p.confidence,
                    "time_horizon": p.time_horizon,
                    "trend": p.trend,
                    "factors": p.factors,
                    "created_at": p.created_at.isoformat()
                }
                for p in predictions
            ],
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")


@router.get("/insights")
async def get_insights(
    insight_type: Optional[str] = None,
    impact: Optional[str] = None,
    limit: int = 20,
    current_user: User = Depends(get_current_user)
):
    """Get analytical insights"""
    try:
        insights = advanced_analytics_engine.insights
        
        # Filter by insight type if specified
        if insight_type:
            insights = [i for i in insights if i.insight_type == insight_type]
            
        # Filter by impact if specified
        if impact:
            insights = [i for i in insights if i.impact == impact]
            
        # Limit results
        insights = insights[-limit:]
        
        return {
            "success": True,
            "data": [
                {
                    "insight_type": i.insight_type,
                    "title": i.title,
                    "description": i.description,
                    "confidence": i.confidence,
                    "impact": i.impact,
                    "recommendations": i.recommendations,
                    "data_points": i.data_points,
                    "created_at": i.created_at.isoformat()
                }
                for i in insights
            ],
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")


@router.get("/anomalies")
async def get_anomalies(
    metric_name: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get detected anomalies"""
    try:
        anomalies = advanced_analytics_engine.anomalies
        
        # Filter by metric name if specified
        if metric_name:
            anomalies = [a for a in anomalies if a.metric_name == metric_name]
            
        # Filter by severity if specified
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
            
        # Limit results
        anomalies = anomalies[-limit:]
        
        return {
            "success": True,
            "data": [
                {
                    "metric_name": a.metric_name,
                    "anomaly_score": a.anomaly_score,
                    "is_anomaly": a.is_anomaly,
                    "expected_value": a.expected_value,
                    "actual_value": a.actual_value,
                    "severity": a.severity,
                    "description": a.description,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in anomalies
            ],
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get anomalies: {str(e)}")


@router.post("/add-data-point")
async def add_data_point(
    metric_name: str,
    value: float,
    metadata: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
):
    """Add a data point for analysis"""
    try:
        advanced_analytics_engine.add_data_point(
            metric_name=metric_name,
            value=value,
            metadata=metadata
        )
        
        return {
            "success": True,
            "message": f"Data point added for {metric_name}",
            "data": {
                "metric_name": metric_name,
                "value": value,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add data point: {str(e)}")


@router.get("/models")
async def get_model_info(
    current_user: User = Depends(get_current_user)
):
    """Get information about trained models"""
    try:
        models_info = {}
        
        for metric_name, model in advanced_analytics_engine.models.items():
            model_info = {
                "metric_name": metric_name,
                "model_type": type(model).__name__,
                "accuracy": advanced_analytics_engine.performance_metrics["model_accuracy"].get(metric_name, 0),
                "last_trained": getattr(model, '_last_trained', None)
            }
            
            # Add feature importance if available
            if hasattr(model, 'feature_importances_'):
                model_info["feature_importance"] = model.feature_importances_.tolist()
                
            models_info[metric_name] = model_info
            
        return {
            "success": True,
            "data": {
                "total_models": len(advanced_analytics_engine.models),
                "models": models_info,
                "ml_available": advanced_analytics_engine.get_analytics_summary().get("ml_available", False)
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.post("/generate-insights")
async def generate_insights_manually(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Manually trigger insight generation"""
    try:
        # Run insight generation in background
        background_tasks.add_task(advanced_analytics_engine._generate_insights)
        
        return {
            "success": True,
            "message": "Insight generation triggered",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")


@router.post("/detect-anomalies")
async def detect_anomalies_manually(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Manually trigger anomaly detection"""
    try:
        # Run anomaly detection in background
        background_tasks.add_task(advanced_analytics_engine._detect_anomalies)
        
        return {
            "success": True,
            "message": "Anomaly detection triggered",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect anomalies: {str(e)}")


@router.get("/performance-metrics")
async def get_analytics_performance_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get analytics engine performance metrics"""
    try:
        metrics = advanced_analytics_engine.performance_metrics
        
        return {
            "success": True,
            "data": {
                "predictions_generated": metrics["predictions_generated"],
                "insights_generated": metrics["insights_generated"],
                "anomalies_detected": metrics["anomalies_detected"],
                "model_accuracy": metrics["model_accuracy"],
                "data_points_by_metric": {
                    metric: len(history) 
                    for metric, history in advanced_analytics_engine.data_history.items()
                }
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.get("/health")
async def get_analytics_health(
    current_user: User = Depends(get_current_user)
):
    """Get analytics engine health status"""
    try:
        summary = advanced_analytics_engine.get_analytics_summary()
        
        health_status = {
            "analytics_active": summary.get("analytics_active", False),
            "data_points_collected": summary.get("data_points_collected", 0),
            "models_trained": summary.get("models_trained", 0),
            "ml_available": summary.get("ml_available", False),
            "uptime": "active" if summary.get("analytics_active", False) else "inactive",
            "last_activity": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "data": health_status,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")


@router.post("/sync-performance-data")
async def sync_performance_data(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Sync data from performance optimizer to analytics engine"""
    try:
        # Get performance summary
        perf_summary = advanced_performance_optimizer.get_performance_summary()
        
        # Add metrics to analytics engine
        for metric_name, metric_data in perf_summary.get("metrics", {}).items():
            if "current" in metric_data:
                advanced_analytics_engine.add_data_point(
                    metric_name=metric_name,
                    value=metric_data["current"],
                    metadata={"source": "performance_optimizer"}
                )
        
        return {
            "success": True,
            "message": "Performance data synced to analytics engine",
            "data": {
                "metrics_synced": len(perf_summary.get("metrics", {})),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync performance data: {str(e)}")


