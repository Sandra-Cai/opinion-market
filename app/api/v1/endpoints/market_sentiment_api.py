"""
Market Sentiment API Endpoints
REST API for the Market Sentiment Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.market_sentiment_engine import (
    market_sentiment_engine,
    SentimentSource,
    SentimentType
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class SentimentResponse(BaseModel):
    asset: str
    overall_sentiment: str
    sentiment_score: float
    confidence: float
    source_breakdown: Dict[str, float]
    sentiment_distribution: Dict[str, int]
    volume: int
    trend: str
    last_updated: str

class SentimentAlertResponse(BaseModel):
    alert_id: str
    asset: str
    alert_type: str
    severity: str
    message: str
    sentiment_score: float
    threshold: float
    created_at: str

class AddSentimentDataRequest(BaseModel):
    source: str = Field(..., description="Sentiment source")
    asset: str = Field(..., description="Asset symbol")
    text: str = Field(..., description="Text to analyze")
    sentiment_score: Optional[float] = Field(None, description="Manual sentiment score (-1.0 to 1.0)")
    confidence: float = Field(0.5, description="Confidence level (0.0 to 1.0)")

@router.get("/sentiment/{asset}", response_model=SentimentResponse)
async def get_sentiment(asset: str):
    """Get sentiment for a specific asset"""
    try:
        sentiment = await market_sentiment_engine.get_sentiment(asset)
        if not sentiment:
            raise HTTPException(status_code=404, detail=f"Sentiment data not found for {asset}")
        
        return sentiment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentiment for {asset}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment", response_model=List[SentimentResponse])
async def get_all_sentiments():
    """Get sentiment for all assets"""
    try:
        sentiments = await market_sentiment_engine.get_all_sentiments()
        return sentiments
        
    except Exception as e:
        logger.error(f"Error getting all sentiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", response_model=List[SentimentAlertResponse])
async def get_sentiment_alerts(limit: int = 20):
    """Get recent sentiment alerts"""
    try:
        alerts = await market_sentiment_engine.get_sentiment_alerts(limit=limit)
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting sentiment alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{asset}")
async def get_sentiment_history(asset: str, hours: int = 24):
    """Get sentiment history for an asset"""
    try:
        history = await market_sentiment_engine.get_sentiment_history(asset, hours=hours)
        return {
            "asset": asset,
            "hours": hours,
            "sentiment_history": history,
            "data_points": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting sentiment history for {asset}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-sentiment", response_model=Dict[str, str])
async def add_sentiment_data(sentiment_request: AddSentimentDataRequest):
    """Add sentiment data manually"""
    try:
        # Validate source
        try:
            source = SentimentSource(sentiment_request.source.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid sentiment source: {sentiment_request.source}")
        
        # Validate sentiment score if provided
        if sentiment_request.sentiment_score is not None:
            if not -1.0 <= sentiment_request.sentiment_score <= 1.0:
                raise HTTPException(status_code=400, detail="Sentiment score must be between -1.0 and 1.0")
        
        # Add sentiment data
        data_id = await market_sentiment_engine.add_sentiment_data(
            source=source,
            asset=sentiment_request.asset,
            text=sentiment_request.text,
            sentiment_score=sentiment_request.sentiment_score,
            confidence=sentiment_request.confidence
        )
        
        return {
            "data_id": data_id,
            "status": "added",
            "message": f"Sentiment data added for {sentiment_request.asset}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding sentiment data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sources")
async def get_sentiment_sources():
    """Get available sentiment sources"""
    try:
        sources = [
            {
                "name": source.value,
                "description": f"{source.value.replace('_', ' ').title()} sentiment source",
                "enabled": True
            }
            for source in SentimentSource
        ]
        
        return {
            "sources": sources,
            "total_sources": len(sources)
        }
        
    except Exception as e:
        logger.error(f"Error getting sentiment sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def get_sentiment_types():
    """Get available sentiment types"""
    try:
        types = [
            {
                "name": sentiment_type.value,
                "description": f"{sentiment_type.value.replace('_', ' ').title()} sentiment",
                "threshold": market_sentiment_engine.sentiment_thresholds.get(sentiment_type, 0.0)
            }
            for sentiment_type in SentimentType
        ]
        
        return {
            "sentiment_types": types,
            "total_types": len(types)
        }
        
    except Exception as e:
        logger.error(f"Error getting sentiment types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_sentiment_metrics():
    """Get sentiment engine metrics"""
    try:
        metrics = await market_sentiment_engine.get_sentiment_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting sentiment metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_sentiment_engine_health():
    """Get sentiment engine health status"""
    try:
        return {
            "engine_id": market_sentiment_engine.engine_id,
            "is_running": market_sentiment_engine.is_running,
            "total_assets": len(market_sentiment_engine.sentiment_aggregates),
            "total_data_points": sum(len(data_list) for data_list in market_sentiment_engine.sentiment_data.values()),
            "total_alerts": len(market_sentiment_engine.sentiment_alerts),
            "uptime": "active" if market_sentiment_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting sentiment engine health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard")
async def get_sentiment_dashboard():
    """Get sentiment dashboard data"""
    try:
        # Get all sentiments
        sentiments = await market_sentiment_engine.get_all_sentiments()
        
        # Get recent alerts
        alerts = await market_sentiment_engine.get_sentiment_alerts(limit=10)
        
        # Calculate summary statistics
        total_assets = len(sentiments)
        bullish_assets = len([s for s in sentiments if s["overall_sentiment"] in ["bullish", "very_bullish"]])
        bearish_assets = len([s for s in sentiments if s["overall_sentiment"] in ["bearish", "very_bearish"]])
        neutral_assets = len([s for s in sentiments if s["overall_sentiment"] == "neutral"])
        
        # Calculate average sentiment score
        avg_sentiment_score = sum(s["sentiment_score"] for s in sentiments) / len(sentiments) if sentiments else 0
        
        # Calculate average confidence
        avg_confidence = sum(s["confidence"] for s in sentiments) / len(sentiments) if sentiments else 0
        
        return {
            "summary": {
                "total_assets": total_assets,
                "bullish_assets": bullish_assets,
                "bearish_assets": bearish_assets,
                "neutral_assets": neutral_assets,
                "average_sentiment_score": avg_sentiment_score,
                "average_confidence": avg_confidence
            },
            "recent_alerts": alerts,
            "sentiments": sentiments,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting sentiment dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))
