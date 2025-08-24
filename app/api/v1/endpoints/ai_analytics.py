from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.market import Market
from app.services.ai_analytics import get_ai_analytics_service

router = APIRouter()

@router.get("/market/{market_id}/prediction")
def get_market_prediction(
    market_id: int,
    db: Session = Depends(get_db)
):
    """Get AI-powered market prediction"""
    ai_service = get_ai_analytics_service()
    prediction = ai_service.get_market_prediction(market_id)
    
    if "error" in prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=prediction["error"]
        )
    
    return prediction

@router.get("/user/insights")
def get_user_insights(
    current_user: User = Depends(get_current_user)
):
    """Get AI-powered insights for current user"""
    ai_service = get_ai_analytics_service()
    insights = ai_service.get_user_insights(current_user.id)
    
    if "error" in insights:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=insights["error"]
        )
    
    return insights

@router.get("/user/{user_id}/insights")
def get_user_insights_public(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get public AI insights for a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    ai_service = get_ai_analytics_service()
    insights = ai_service.get_user_insights(user_id)
    
    if "error" in insights:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=insights["error"]
        )
    
    # Return only public data
    return {
        "user_id": user_id,
        "username": user.username,
        "performance_metrics": insights["performance_metrics"],
        "trading_patterns": {
            "frequency": insights["trading_patterns"]["frequency"],
            "diversification": insights["trading_patterns"]["diversification"],
            "preferred_markets": insights["trading_patterns"]["preferred_markets"]
        },
        "risk_assessment": insights["risk_assessment"]
    }

@router.get("/market/{market_id}/sentiment")
def get_market_sentiment(
    market_id: int,
    db: Session = Depends(get_db)
):
    """Get market sentiment analysis"""
    ai_service = get_ai_analytics_service()
    prediction = ai_service.get_market_prediction(market_id)
    
    if "error" in prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=prediction["error"]
        )
    
    return {
        "market_id": market_id,
        "sentiment": prediction["market_sentiment"],
        "technical_indicators": prediction["technical_indicators"],
        "risk_assessment": prediction["risk_assessment"]
    }

@router.get("/market/{market_id}/recommendations")
def get_market_recommendations(
    market_id: int,
    db: Session = Depends(get_db)
):
    """Get AI-powered trading recommendations for a market"""
    ai_service = get_ai_analytics_service()
    prediction = ai_service.get_market_prediction(market_id)
    
    if "error" in prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=prediction["error"]
        )
    
    return {
        "market_id": market_id,
        "recommendations": prediction["recommendations"],
        "confidence": prediction["confidence"],
        "prediction": prediction["prediction"]
    }

@router.get("/user/recommendations")
def get_user_recommendations(
    current_user: User = Depends(get_current_user)
):
    """Get personalized AI recommendations for current user"""
    ai_service = get_ai_analytics_service()
    insights = ai_service.get_user_insights(current_user.id)
    
    if "error" in insights:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=insights["error"]
        )
    
    return {
        "user_id": current_user.id,
        "recommendations": insights["recommendations"],
        "performance_metrics": insights["performance_metrics"],
        "risk_assessment": insights["risk_assessment"]
    }

@router.get("/market/{market_id}/risk-analysis")
def get_market_risk_analysis(
    market_id: int,
    db: Session = Depends(get_db)
):
    """Get comprehensive risk analysis for a market"""
    ai_service = get_ai_analytics_service()
    prediction = ai_service.get_market_prediction(market_id)
    
    if "error" in prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=prediction["error"]
        )
    
    return {
        "market_id": market_id,
        "risk_assessment": prediction["risk_assessment"],
        "technical_indicators": prediction["technical_indicators"],
        "market_sentiment": prediction["market_sentiment"]
    }

@router.get("/platform/ai-insights")
def get_platform_ai_insights(
    db: Session = Depends(get_db)
):
    """Get AI-powered platform-wide insights"""
    # Get top markets by volume
    top_markets = db.query(Market).filter(
        Market.status == 'open'
    ).order_by(Market.volume_24h.desc()).limit(10).all()
    
    ai_service = get_ai_analytics_service()
    
    market_insights = []
    for market in top_markets:
        try:
            prediction = ai_service.get_market_prediction(market.id)
            if "error" not in prediction:
                market_insights.append({
                    "market_id": market.id,
                    "title": market.title,
                    "category": market.category,
                    "volume_24h": market.volume_24h,
                    "sentiment": prediction["market_sentiment"]["sentiment"],
                    "confidence": prediction["confidence"],
                    "risk_level": prediction["risk_assessment"]["risk_level"]
                })
        except:
            continue
    
    return {
        "total_markets_analyzed": len(market_insights),
        "market_insights": market_insights,
        "platform_sentiment": "bullish" if len([m for m in market_insights if m["sentiment"] == "bullish"]) > len(market_insights) / 2 else "bearish",
        "average_confidence": sum(m["confidence"] for m in market_insights) / len(market_insights) if market_insights else 0
    }

@router.get("/market/{market_id}/price-forecast")
def get_price_forecast(
    market_id: int,
    timeframe: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    db: Session = Depends(get_db)
):
    """Get AI-powered price forecast for different timeframes"""
    ai_service = get_ai_analytics_service()
    prediction = ai_service.get_market_prediction(market_id)
    
    if "error" in prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=prediction["error"]
        )
    
    # Map timeframe to prediction
    timeframe_map = {
        "1h": "short_term",
        "24h": "short_term", 
        "7d": "medium_term",
        "30d": "long_term"
    }
    
    forecast_period = timeframe_map.get(timeframe, "short_term")
    
    return {
        "market_id": market_id,
        "timeframe": timeframe,
        "forecast": prediction["prediction"][forecast_period],
        "confidence": prediction["confidence"],
        "technical_indicators": prediction["technical_indicators"],
        "sentiment": prediction["market_sentiment"]["sentiment"]
    }

@router.get("/user/performance-analysis")
def get_user_performance_analysis(
    current_user: User = Depends(get_current_user)
):
    """Get detailed AI analysis of user performance"""
    ai_service = get_ai_analytics_service()
    insights = ai_service.get_user_insights(current_user.id)
    
    if "error" in insights:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=insights["error"]
        )
    
    return {
        "user_id": current_user.id,
        "performance_metrics": insights["performance_metrics"],
        "trading_patterns": insights["trading_patterns"],
        "risk_assessment": insights["risk_assessment"],
        "recommendations": insights["recommendations"],
        "ai_analysis": {
            "trading_style": "conservative" if insights["risk_assessment"]["risk_score"] < 0.3 else "aggressive" if insights["risk_assessment"]["risk_score"] > 0.7 else "moderate",
            "strengths": _identify_strengths(insights),
            "areas_for_improvement": _identify_improvements(insights)
        }
    }

def _identify_strengths(insights: dict) -> List[str]:
    """Identify user's trading strengths"""
    strengths = []
    
    if insights["performance_metrics"]["win_rate"] > 0.6:
        strengths.append("High win rate")
    
    if insights["trading_patterns"]["diversification"] > 0.5:
        strengths.append("Good market diversification")
    
    if insights["risk_assessment"]["risk_score"] < 0.5:
        strengths.append("Conservative risk management")
    
    if insights["performance_metrics"]["risk_adjusted_return"] > 0.5:
        strengths.append("Strong risk-adjusted returns")
    
    return strengths

def _identify_improvements(insights: dict) -> List[str]:
    """Identify areas for improvement"""
    improvements = []
    
    if insights["performance_metrics"]["win_rate"] < 0.4:
        improvements.append("Improve win rate through better market selection")
    
    if insights["trading_patterns"]["diversification"] < 0.3:
        improvements.append("Increase market diversification to reduce risk")
    
    if insights["risk_assessment"]["risk_score"] > 0.7:
        improvements.append("Consider reducing position sizes for better risk management")
    
    if insights["performance_metrics"]["risk_adjusted_return"] < 0.2:
        improvements.append("Focus on risk-adjusted returns rather than absolute returns")
    
    return improvements
