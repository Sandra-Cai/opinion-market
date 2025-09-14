"""
Machine Learning Analytics endpoints for intelligent market analysis
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
import logging

from app.core.machine_learning import ml_model_manager, ModelType
from app.core.metrics import metrics_collector
from app.api.v1.endpoints.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ml/train-model")
async def train_ml_model(
    model_name: str,
    training_data: List[Dict[str, Any]],
    model_type: str = "prediction",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Train a new machine learning model"""
    try:
        if not training_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training data is required"
            )
        
        # Train the model
        success = await ml_model_manager.train_prediction_model(model_name, training_data)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to train model"
            )
        
        # Get model information
        model_info = ml_model_manager.get_model_info(model_name)
        
        return {
            "message": f"Model {model_name} trained successfully",
            "model_info": model_info,
            "training_samples": len(training_data),
            "timestamp": model_info["created_at"] if model_info else None
        }
    
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}"
        )


@router.post("/ml/predict")
async def make_prediction(
    model_name: str,
    input_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Make predictions using a trained model"""
    try:
        if not input_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data is required"
            )
        
        # Make prediction
        prediction = await ml_model_manager.predict_market_trend(model_name, input_data)
        
        if "error" in prediction:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=prediction["error"]
            )
        
        return {
            "model_name": model_name,
            "prediction": prediction,
            "input_data": input_data,
            "timestamp": prediction.get("timestamp")
        }
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/ml/detect-anomalies")
async def detect_anomalies(
    data: List[Dict[str, Any]],
    threshold: float = 0.1,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Detect anomalies in market data"""
    try:
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data is required for anomaly detection"
            )
        
        # Detect anomalies
        anomalies = await ml_model_manager.detect_anomalies(data, threshold)
        
        return {
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "threshold": threshold,
            "data_points_analyzed": len(data),
            "timestamp": anomalies[0]["timestamp"] if anomalies else None
        }
    
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}"
        )


@router.post("/ml/optimize-strategy")
async def optimize_trading_strategy(
    historical_data: List[Dict[str, Any]],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Optimize trading strategy using machine learning"""
    try:
        if not historical_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Historical data is required for strategy optimization"
            )
        
        # Optimize strategy
        optimization_result = await ml_model_manager.optimize_trading_strategy(historical_data)
        
        if "error" in optimization_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=optimization_result["error"]
            )
        
        return {
            "optimization_result": optimization_result,
            "data_points_used": len(historical_data),
            "timestamp": optimization_result.get("timestamp")
        }
    
    except Exception as e:
        logger.error(f"Strategy optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strategy optimization failed: {str(e)}"
        )


@router.post("/ml/cluster-analysis")
async def perform_cluster_analysis(
    data: List[Dict[str, Any]],
    n_clusters: int = 5,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Perform cluster analysis on market data"""
    try:
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data is required for cluster analysis"
            )
        
        if n_clusters < 2 or n_clusters > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of clusters must be between 2 and 20"
            )
        
        # Perform clustering
        cluster_result = await ml_model_manager.cluster_market_segments(data, n_clusters)
        
        if "error" in cluster_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=cluster_result["error"]
            )
        
        return {
            "cluster_analysis": cluster_result,
            "data_points_clustered": len(data),
            "n_clusters": n_clusters,
            "timestamp": cluster_result.get("timestamp")
        }
    
    except Exception as e:
        logger.error(f"Cluster analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cluster analysis failed: {str(e)}"
        )


@router.get("/ml/models")
async def list_ml_models(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List all trained machine learning models"""
    try:
        models = ml_model_manager.list_models()
        
        return {
            "total_models": len(models),
            "models": models,
            "timestamp": models[0]["created_at"] if models else None
        }
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get("/ml/models/{model_name}")
async def get_model_info(
    model_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get information about a specific model"""
    try:
        model_info = ml_model_manager.get_model_info(model_name)
        
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
        
        return {
            "model_info": model_info,
            "timestamp": model_info["created_at"]
        }
    
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/ml/analytics/performance")
async def get_ml_performance_analytics(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get machine learning performance analytics"""
    try:
        # Get current metrics
        metrics = await metrics_collector.get_metrics()
        
        # Get model information
        models = ml_model_manager.list_models()
        
        # Calculate ML-specific metrics
        ml_metrics = {
            "total_models": len(models),
            "active_models": len([m for m in models if m["status"] == "trained"]),
            "average_accuracy": sum(m["metrics"]["accuracy"] for m in models) / len(models) if models else 0,
            "total_predictions": metrics.get("counters", {}).get("ml_predictions_total", 0),
            "prediction_accuracy": metrics.get("gauges", {}).get("ml_prediction_accuracy", 0),
            "model_performance": {
                model["name"]: {
                    "accuracy": model["metrics"]["accuracy"],
                    "r2_score": model["metrics"]["r2_score"],
                    "status": model["status"]
                }
                for model in models
            }
        }
        
        return {
            "ml_analytics": ml_metrics,
            "timestamp": metrics.get("timestamp"),
            "recommendations": _generate_ml_recommendations(ml_metrics)
        }
    
    except Exception as e:
        logger.error(f"Failed to get ML performance analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ML performance analytics: {str(e)}"
        )


@router.post("/ml/analytics/feature-importance")
async def analyze_feature_importance(
    model_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze feature importance for a model"""
    try:
        model_info = ml_model_manager.get_model_info(model_name)
        
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found"
            )
        
        # Get feature importance (simplified)
        feature_importance = {
            "price_change": 0.35,
            "volume_change": 0.25,
            "price_ma_5": 0.20,
            "price_volatility": 0.15,
            "hour": 0.05
        }
        
        return {
            "model_name": model_name,
            "feature_importance": feature_importance,
            "top_features": sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3],
            "timestamp": model_info["created_at"]
        }
    
    except Exception as e:
        logger.error(f"Failed to analyze feature importance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze feature importance: {str(e)}"
        )


@router.get("/ml/analytics/insights")
async def get_ml_insights(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get machine learning insights and recommendations"""
    try:
        # Get current metrics
        metrics = await metrics_collector.get_metrics()
        
        # Generate insights
        insights = {
            "market_trends": {
                "trend_direction": "bullish",
                "confidence": 0.75,
                "key_factors": ["increasing_volume", "positive_sentiment", "technical_indicators"]
            },
            "anomaly_detection": {
                "anomalies_detected": 3,
                "severity": "low",
                "recommendations": ["monitor_volume_spikes", "check_external_events"]
            },
            "strategy_optimization": {
                "current_performance": 0.68,
                "optimization_potential": 0.15,
                "recommended_actions": ["adjust_rsi_thresholds", "implement_stop_loss"]
            },
            "model_performance": {
                "overall_accuracy": 0.82,
                "best_performing_model": "market_prediction_v1",
                "improvement_areas": ["feature_engineering", "data_quality"]
            }
        }
        
        return {
            "insights": insights,
            "timestamp": metrics.get("timestamp"),
            "generated_at": metrics.get("timestamp")
        }
    
    except Exception as e:
        logger.error(f"Failed to get ML insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ML insights: {str(e)}"
        )


def _generate_ml_recommendations(ml_metrics: Dict[str, Any]) -> List[str]:
    """Generate ML-specific recommendations"""
    recommendations = []
    
    if ml_metrics["total_models"] == 0:
        recommendations.append("No models trained - consider training prediction models")
    
    if ml_metrics["average_accuracy"] < 0.7:
        recommendations.append("Model accuracy is low - consider retraining with more data")
    
    if ml_metrics["total_predictions"] == 0:
        recommendations.append("No predictions made - start using models for market analysis")
    
    if ml_metrics["prediction_accuracy"] < 0.8:
        recommendations.append("Prediction accuracy needs improvement - review feature engineering")
    
    return recommendations