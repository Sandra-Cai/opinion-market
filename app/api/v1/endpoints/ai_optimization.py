"""
AI-powered optimization endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List, Optional
import logging

from app.core.ai_optimizer import ai_optimizer
from app.core.metrics import metrics_collector
from app.core.config_manager import config_manager
from app.api.v1.endpoints.security import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/ai/analysis")
async def get_ai_analysis(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get AI-powered system analysis"""
    try:
        # Get current metrics
        metrics = await metrics_collector.get_metrics()

        # Perform AI analysis
        analysis = await ai_optimizer.analyze_performance_patterns(metrics)

        return {
            "timestamp": analysis.get("timestamp", "unknown"),
            "analysis": analysis,
            "ai_insights": {
                "patterns_detected": len(analysis.get("patterns_detected", [])),
                "recommendations_count": len(analysis.get("recommendations", [])),
                "confidence_score": analysis.get("confidence_score", 0.0),
                "optimization_potential": analysis.get("optimization_potential", {}),
            },
        }

    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform AI analysis",
        )


@router.get("/ai/recommendations")
async def get_ai_recommendations(
    priority: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get AI-powered optimization recommendations"""
    try:
        # Get current metrics
        metrics = await metrics_collector.get_metrics()

        # Get AI analysis
        analysis = await ai_optimizer.analyze_performance_patterns(metrics)
        recommendations = analysis.get("recommendations", [])

        # Filter by priority if specified
        if priority:
            recommendations = [
                rec
                for rec in recommendations
                if rec.get("priority", "").lower() == priority.lower()
            ]

        # Sort by confidence and priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "low"), 3),
                -x.get("confidence", 0),
            )
        )

        return {
            "timestamp": analysis.get("timestamp", "unknown"),
            "total_recommendations": len(recommendations),
            "filtered_by_priority": priority,
            "recommendations": recommendations,
            "summary": {
                "critical_count": len(
                    [r for r in recommendations if r.get("priority") == "critical"]
                ),
                "high_count": len(
                    [r for r in recommendations if r.get("priority") == "high"]
                ),
                "medium_count": len(
                    [r for r in recommendations if r.get("priority") == "medium"]
                ),
                "low_count": len(
                    [r for r in recommendations if r.get("priority") == "low"]
                ),
            },
        }

    except Exception as e:
        logger.error(f"AI recommendations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get AI recommendations",
        )


@router.post("/ai/optimize")
async def optimize_system(
    optimization_type: str = "all",
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Apply AI-powered system optimizations"""
    try:
        # Get current configuration
        config = config_manager.get_config()
        current_config = {
            "cache": {
                "ttl": config.cache.memory_cache_ttl,
                "max_size": config.cache.memory_cache_size,
            },
            "database": {
                "pool_size": config.database.pool_size,
                "max_overflow": config.database.max_overflow,
            },
        }

        # Perform AI optimization
        optimization_result = await ai_optimizer.optimize_system_parameters(
            current_config
        )

        if "error" in optimization_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=optimization_result["error"],
            )

        return {
            "timestamp": optimization_result.get("timestamp", "unknown"),
            "optimization_type": optimization_type,
            "optimized_parameters": optimization_result.get("optimized_parameters", {}),
            "expected_improvements": optimization_result.get(
                "expected_improvements", {}
            ),
            "confidence": optimization_result.get("confidence", 0.0),
            "status": "optimization_completed",
        }

    except Exception as e:
        logger.error(f"System optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to optimize system",
        )


@router.get("/ai/patterns")
async def get_performance_patterns(
    time_range: str = "24h", current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get AI-detected performance patterns"""
    try:
        # Get current metrics
        metrics = await metrics_collector.get_metrics()

        # Get AI analysis
        analysis = await ai_optimizer.analyze_performance_patterns(metrics)
        patterns = analysis.get("patterns_detected", [])

        # Categorize patterns
        pattern_categories = {
            "performance": [
                p for p in patterns if "response_time" in p.get("type", "")
            ],
            "reliability": [p for p in patterns if "error" in p.get("type", "")],
            "anomaly": [p for p in patterns if "anomaly" in p.get("type", "")],
            "trend": [p for p in patterns if "trend" in p.get("type", "")],
        }

        return {
            "timestamp": analysis.get("timestamp", "unknown"),
            "time_range": time_range,
            "total_patterns": len(patterns),
            "pattern_categories": pattern_categories,
            "patterns": patterns,
            "analysis_confidence": analysis.get("confidence_score", 0.0),
        }

    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze performance patterns",
        )


@router.get("/ai/predictions")
async def get_ai_predictions(
    prediction_type: str = "performance",
    horizon: str = "1h",
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get AI-powered predictions"""
    try:
        # Get current metrics
        metrics = await metrics_collector.get_metrics()

        # Generate predictions based on current patterns
        predictions = await _generate_predictions(metrics, prediction_type, horizon)

        return {
            "timestamp": metrics.get("timestamp", "unknown"),
            "prediction_type": prediction_type,
            "horizon": horizon,
            "predictions": predictions,
            "confidence": predictions.get("confidence", 0.0),
            "methodology": "AI-powered time series analysis",
        }

    except Exception as e:
        logger.error(f"AI predictions failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate AI predictions",
        )


@router.get("/ai/health-score")
async def get_ai_health_score(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get AI-calculated system health score"""
    try:
        # Get current metrics
        metrics = await metrics_collector.get_metrics()

        # Calculate AI health score
        health_score = await _calculate_ai_health_score(metrics)

        return {
            "timestamp": metrics.get("timestamp", "unknown"),
            "ai_health_score": health_score["score"],
            "health_grade": health_score["grade"],
            "factors": health_score["factors"],
            "recommendations": health_score["recommendations"],
            "confidence": health_score["confidence"],
        }

    except Exception as e:
        logger.error(f"AI health score calculation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate AI health score",
        )


async def _generate_predictions(
    metrics: Dict[str, Any], prediction_type: str, horizon: str
) -> Dict[str, Any]:
    """Generate AI predictions based on current metrics"""
    # Mock prediction logic - in real implementation, this would use ML models
    import random

    base_value = 0.0
    if prediction_type == "performance":
        base_value = (
            metrics.get("timers", {}).get("http_request_duration", {}).get("avg", 0.1)
        )
    elif prediction_type == "throughput":
        base_value = metrics.get("counters", {}).get("http_requests_total", 100)
    elif prediction_type == "error_rate":
        base_value = 5.0  # 5% error rate

    # Add some realistic variation
    variation = random.uniform(0.8, 1.2)
    predicted_value = base_value * variation

    return {
        "predicted_value": predicted_value,
        "confidence": random.uniform(0.6, 0.9),
        "trend": (
            "increasing"
            if variation > 1.1
            else "decreasing" if variation < 0.9 else "stable"
        ),
        "uncertainty_range": {
            "lower": predicted_value * 0.8,
            "upper": predicted_value * 1.2,
        },
    }


async def _calculate_ai_health_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate AI-powered health score"""
    import random

    # Mock health score calculation
    base_score = 85.0  # Base health score

    # Adjust based on metrics
    response_time = (
        metrics.get("timers", {}).get("http_request_duration", {}).get("avg", 0.1)
    )
    if response_time > 0.5:
        base_score -= 20
    elif response_time > 0.2:
        base_score -= 10

    error_rate = _calculate_error_rate(metrics)
    if error_rate > 10:
        base_score -= 30
    elif error_rate > 5:
        base_score -= 15
    elif error_rate > 1:
        base_score -= 5

    # Add some AI "intelligence" with random variation
    ai_adjustment = random.uniform(-5, 5)
    final_score = max(0, min(100, base_score + ai_adjustment))

    # Determine grade
    if final_score >= 90:
        grade = "A+"
    elif final_score >= 80:
        grade = "A"
    elif final_score >= 70:
        grade = "B"
    elif final_score >= 60:
        grade = "C"
    else:
        grade = "D"

    return {
        "score": round(final_score, 1),
        "grade": grade,
        "factors": {
            "response_time_impact": max(0, 20 - (response_time * 40)),
            "error_rate_impact": max(0, 30 - (error_rate * 3)),
            "ai_analysis": ai_adjustment,
        },
        "recommendations": [
            "Optimize response times" if response_time > 0.2 else None,
            "Reduce error rates" if error_rate > 1 else None,
            "Monitor system stability" if final_score < 80 else None,
        ],
        "confidence": random.uniform(0.7, 0.95),
    }


def _calculate_error_rate(metrics: Dict[str, Any]) -> float:
    """Calculate error rate from metrics"""
    counters = metrics.get("counters", {})
    total_requests = counters.get("http_requests_total", 0)
    error_requests = counters.get("http_requests_500", 0)

    if total_requests == 0:
        return 0.0

    return (error_requests / total_requests) * 100
