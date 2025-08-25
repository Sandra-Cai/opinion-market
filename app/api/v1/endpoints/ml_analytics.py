from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
import asyncio

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.services.machine_learning import get_ml_service

router = APIRouter()

@router.get("/market/{market_id}/prediction")
def get_market_prediction(
    market_id: int,
    horizon: str = Query("24h", description="Prediction horizon (1h, 24h, 7d)"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get ML-powered market price prediction"""
    # Validate horizon
    valid_horizons = ["1h", "24h", "7d"]
    if horizon not in valid_horizons:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid horizon. Must be one of: {valid_horizons}"
        )
    
    ml_service = get_ml_service()
    
    # Get prediction
    prediction = asyncio.run(ml_service.predict_market_price(market_id, horizon))
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not available for this market"
        )
    
    return {
        "market_id": prediction.market_id,
        "prediction_horizon": prediction.prediction_horizon,
        "predicted_price_a": prediction.predicted_price_a,
        "predicted_price_b": prediction.predicted_price_b,
        "confidence": prediction.confidence,
        "model_type": prediction.model_type,
        "features_used": prediction.features_used,
        "timestamp": prediction.timestamp.isoformat(),
        "metadata": prediction.metadata
    }

@router.get("/user/behavior-analysis")
def get_user_behavior_analysis(
    current_user: User = Depends(get_current_user)
):
    """Get ML-powered user behavior analysis"""
    ml_service = get_ml_service()
    
    # Get user behavior profile
    profile = asyncio.run(ml_service.analyze_user_behavior(current_user.id))
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User behavior analysis not available"
        )
    
    return {
        "user_id": profile.user_id,
        "risk_tolerance": profile.risk_tolerance,
        "trading_frequency": profile.trading_frequency,
        "preferred_market_categories": profile.preferred_market_categories,
        "average_trade_size": profile.average_trade_size,
        "win_rate": profile.win_rate,
        "holding_period": profile.holding_period,
        "sentiment_bias": profile.sentiment_bias,
        "volatility_preference": profile.volatility_preference,
        "metadata": profile.metadata
    }

@router.get("/trading/recommendation")
def get_trading_recommendation(
    market_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get personalized trading recommendation"""
    ml_service = get_ml_service()
    
    # Generate recommendation
    recommendation = asyncio.run(ml_service.generate_trading_recommendation(
        current_user.id, market_id
    ))
    
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trading recommendation not available"
        )
    
    return {
        "user_id": recommendation.user_id,
        "market_id": recommendation.market_id,
        "recommendation_type": recommendation.recommendation_type,
        "confidence": recommendation.confidence,
        "reasoning": recommendation.reasoning,
        "expected_return": recommendation.expected_return,
        "risk_level": recommendation.risk_level,
        "time_horizon": recommendation.time_horizon,
        "timestamp": recommendation.timestamp.isoformat(),
        "metadata": recommendation.metadata
    }

@router.get("/markets/bulk-predictions")
def get_bulk_market_predictions(
    market_ids: str = Query(..., description="Comma-separated market IDs"),
    horizon: str = Query("24h", description="Prediction horizon"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get predictions for multiple markets"""
    try:
        market_id_list = [int(id.strip()) for id in market_ids.split(",")]
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid market IDs format"
        )
    
    if len(market_id_list) > 20:  # Limit to 20 markets at once
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 20 markets allowed per request"
        )
    
    ml_service = get_ml_service()
    
    predictions = []
    for market_id in market_id_list:
        prediction = asyncio.run(ml_service.predict_market_price(market_id, horizon))
        if prediction:
            predictions.append({
                "market_id": prediction.market_id,
                "predicted_price_a": prediction.predicted_price_a,
                "predicted_price_b": prediction.predicted_price_b,
                "confidence": prediction.confidence,
                "model_type": prediction.model_type
            })
    
    return {
        "predictions": predictions,
        "total": len(predictions),
        "horizon": horizon,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/models/performance")
def get_model_performance(
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get ML model performance metrics"""
    ml_service = get_ml_service()
    
    return {
        "models": ml_service.model_performance,
        "feature_importance": ml_service._get_feature_importance(),
        "last_updated": datetime.utcnow().isoformat()
    }

@router.post("/models/retrain")
def retrain_models(
    model_type: str = Query(..., description="Model type to retrain (price_prediction, volume_prediction, user_behavior)"),
    current_user: User = Depends(get_current_user)
):
    """Manually trigger model retraining"""
    # Validate model type
    valid_models = ["price_prediction", "volume_prediction", "user_behavior"]
    if model_type not in valid_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model type. Must be one of: {valid_models}"
        )
    
    ml_service = get_ml_service()
    
    # Trigger retraining
    if model_type == "price_prediction":
        asyncio.run(ml_service._train_price_prediction_model())
    elif model_type == "volume_prediction":
        asyncio.run(ml_service._train_volume_prediction_model())
    elif model_type == "user_behavior":
        asyncio.run(ml_service._train_user_behavior_model())
    
    return {
        "message": f"Model {model_type} retraining initiated",
        "model_type": model_type,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/analytics/user-insights")
def get_user_insights(
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive user insights and analytics"""
    ml_service = get_ml_service()
    
    # Get user behavior profile
    profile = asyncio.run(ml_service.analyze_user_behavior(current_user.id))
    
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User insights not available"
        )
    
    # Generate insights based on profile
    insights = {
        "trading_style": _analyze_trading_style(profile),
        "risk_assessment": _assess_risk_profile(profile),
        "improvement_suggestions": _generate_improvement_suggestions(profile),
        "market_recommendations": _get_market_recommendations(profile),
        "performance_metrics": {
            "win_rate": profile.win_rate,
            "avg_trade_size": profile.average_trade_size,
            "trading_frequency": profile.trading_frequency,
            "holding_period": profile.holding_period
        }
    }
    
    return {
        "user_id": current_user.id,
        "profile": {
            "risk_tolerance": profile.risk_tolerance,
            "preferred_categories": profile.preferred_market_categories,
            "sentiment_bias": profile.sentiment_bias,
            "volatility_preference": profile.volatility_preference
        },
        "insights": insights,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/analytics/market-insights/{market_id}")
def get_market_insights(
    market_id: int,
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get ML-powered market insights"""
    ml_service = get_ml_service()
    
    # Get market prediction
    prediction = asyncio.run(ml_service.predict_market_price(market_id, "24h"))
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market insights not available"
        )
    
    # Generate market insights
    insights = {
        "price_trend": _analyze_price_trend(prediction),
        "confidence_analysis": _analyze_confidence(prediction),
        "risk_assessment": _assess_market_risk(prediction),
        "trading_opportunities": _identify_trading_opportunities(prediction)
    }
    
    return {
        "market_id": market_id,
        "prediction": {
            "predicted_price_a": prediction.predicted_price_a,
            "predicted_price_b": prediction.predicted_price_b,
            "confidence": prediction.confidence,
            "horizon": prediction.prediction_horizon
        },
        "insights": insights,
        "model_info": {
            "type": prediction.model_type,
            "features_used": prediction.features_used,
            "performance": prediction.metadata.get('model_performance', {})
        },
        "timestamp": datetime.utcnow().isoformat()
    }

def _analyze_trading_style(profile) -> Dict:
    """Analyze user's trading style"""
    if profile.trading_frequency > 5:
        style = "High Frequency"
    elif profile.trading_frequency > 1:
        style = "Active"
    else:
        style = "Conservative"
    
    return {
        "style": style,
        "frequency_level": "high" if profile.trading_frequency > 3 else "medium" if profile.trading_frequency > 1 else "low",
        "holding_style": "short_term" if profile.holding_period < 1 else "medium_term" if profile.holding_period < 7 else "long_term"
    }

def _assess_risk_profile(profile) -> Dict:
    """Assess user's risk profile"""
    if profile.risk_tolerance > 0.7:
        risk_level = "High"
        risk_description = "You prefer high-risk, high-reward opportunities"
    elif profile.risk_tolerance > 0.4:
        risk_level = "Medium"
        risk_description = "You balance risk and reward moderately"
    else:
        risk_level = "Low"
        risk_description = "You prefer stable, low-risk investments"
    
    return {
        "risk_level": risk_level,
        "description": risk_description,
        "risk_tolerance_score": profile.risk_tolerance,
        "volatility_preference": profile.volatility_preference
    }

def _generate_improvement_suggestions(profile) -> List[str]:
    """Generate improvement suggestions based on profile"""
    suggestions = []
    
    if profile.win_rate < 0.5:
        suggestions.append("Consider improving your market research before trading")
    
    if profile.trading_frequency > 10:
        suggestions.append("High trading frequency may lead to increased fees - consider longer holding periods")
    
    if profile.risk_tolerance > 0.8:
        suggestions.append("Consider diversifying your portfolio to reduce risk")
    
    if profile.holding_period < 0.5:
        suggestions.append("Very short holding periods may miss longer-term trends")
    
    if not suggestions:
        suggestions.append("Your trading profile looks well-balanced")
    
    return suggestions

def _get_market_recommendations(profile) -> Dict:
    """Get market recommendations based on user profile"""
    return {
        "preferred_categories": profile.preferred_market_categories,
        "suggested_categories": _suggest_new_categories(profile.preferred_market_categories),
        "risk_appropriate_markets": _get_risk_appropriate_markets(profile.risk_tolerance)
    }

def _suggest_new_categories(preferred_categories: List[str]) -> List[str]:
    """Suggest new market categories to explore"""
    all_categories = ["politics", "sports", "crypto", "finance", "technology", "entertainment", "weather", "health"]
    return [cat for cat in all_categories if cat not in preferred_categories]

def _get_risk_appropriate_markets(risk_tolerance: float) -> List[str]:
    """Get market types appropriate for user's risk tolerance"""
    if risk_tolerance > 0.7:
        return ["crypto", "finance", "technology"]
    elif risk_tolerance > 0.4:
        return ["politics", "sports", "entertainment"]
    else:
        return ["weather", "health", "politics"]

def _analyze_price_trend(prediction) -> Dict:
    """Analyze predicted price trend"""
    price_a = prediction.predicted_price_a
    price_b = prediction.predicted_price_b
    
    if price_a > 0.6:
        trend = "Strongly Bullish"
        strength = "high"
    elif price_a > 0.55:
        trend = "Moderately Bullish"
        strength = "medium"
    elif price_a < 0.4:
        trend = "Strongly Bearish"
        strength = "high"
    elif price_a < 0.45:
        trend = "Moderately Bearish"
        strength = "medium"
    else:
        trend = "Neutral"
        strength = "low"
    
    return {
        "trend": trend,
        "strength": strength,
        "price_a_probability": price_a,
        "price_b_probability": price_b
    }

def _analyze_confidence(prediction) -> Dict:
    """Analyze prediction confidence"""
    confidence = prediction.confidence
    
    if confidence > 0.8:
        confidence_level = "Very High"
        reliability = "excellent"
    elif confidence > 0.6:
        confidence_level = "High"
        reliability = "good"
    elif confidence > 0.4:
        confidence_level = "Medium"
        reliability = "fair"
    else:
        confidence_level = "Low"
        reliability = "poor"
    
    return {
        "confidence_level": confidence_level,
        "reliability": reliability,
        "confidence_score": confidence,
        "recommendation": "Use with caution" if confidence < 0.5 else "Reliable for decision making"
    }

def _assess_market_risk(prediction) -> Dict:
    """Assess market risk based on prediction"""
    confidence = prediction.confidence
    price_a = prediction.predicted_price_a
    
    # Risk increases with uncertainty and extreme predictions
    risk_score = (1 - confidence) + abs(price_a - 0.5) * 2
    
    if risk_score > 0.8:
        risk_level = "High"
        risk_description = "High uncertainty or extreme price prediction"
    elif risk_score > 0.5:
        risk_level = "Medium"
        risk_description = "Moderate uncertainty in prediction"
    else:
        risk_level = "Low"
        risk_description = "Relatively stable prediction"
    
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "description": risk_description,
        "factors": {
            "prediction_confidence": confidence,
            "price_extremity": abs(price_a - 0.5)
        }
    }

def _identify_trading_opportunities(prediction) -> Dict:
    """Identify trading opportunities based on prediction"""
    price_a = prediction.predicted_price_a
    confidence = prediction.confidence
    
    opportunities = []
    
    if price_a > 0.6 and confidence > 0.6:
        opportunities.append({
            "type": "buy_outcome_a",
            "confidence": confidence,
            "expected_return": (price_a - 0.5) * 2 * 100,
            "reasoning": "Strong bullish prediction for outcome A"
        })
    
    if price_a < 0.4 and confidence > 0.6:
        opportunities.append({
            "type": "buy_outcome_b",
            "confidence": confidence,
            "expected_return": (0.5 - price_a) * 2 * 100,
            "reasoning": "Strong bearish prediction for outcome A"
        })
    
    if confidence < 0.4:
        opportunities.append({
            "type": "wait",
            "confidence": confidence,
            "expected_return": 0,
            "reasoning": "Low confidence prediction - wait for clearer signals"
        })
    
    return {
        "opportunities": opportunities,
        "best_opportunity": max(opportunities, key=lambda x: x["expected_return"]) if opportunities else None
    }
