"""
Pydantic schemas for AI Analytics API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta


class PredictionRequest(BaseModel):
    """Request model for batch predictions"""

    market_ids: List[int] = Field(..., description="List of market IDs to predict")

    class Config:
        schema_extra = {"example": {"market_ids": [1, 2, 3, 4, 5]}}


class PredictionResponse(BaseModel):
    """Response model for market predictions"""

    market_id: int = Field(..., description="Market ID")
    predicted_outcome: str = Field(..., description="Predicted outcome (A or B)")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction confidence (0-1)"
    )
    probability_a: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of outcome A"
    )
    probability_b: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of outcome B"
    )
    recommendation: str = Field(..., description="Trading recommendation")
    risk_level: str = Field(..., description="Risk level (LOW, MEDIUM, HIGH)")
    model_version: str = Field(..., description="AI model version used")
    prediction_time: datetime = Field(..., description="When prediction was made")
    validity_duration: timedelta = Field(
        ..., description="How long prediction is valid"
    )
    features_used: List[str] = Field(..., description="Features used in prediction")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "predicted_outcome": "A",
                "confidence": 0.85,
                "probability_a": 0.75,
                "probability_b": 0.25,
                "recommendation": "High confidence - recommended position in A",
                "risk_level": "MEDIUM",
                "model_version": "v1.0.0",
                "prediction_time": "2024-01-15T10:30:00Z",
                "validity_duration": 86400,
                "features_used": [
                    "total_volume",
                    "participant_count",
                    "price_volatility",
                ],
            }
        }


class MarketAnalyticsResponse(BaseModel):
    """Response model for market analytics"""

    market_id: int = Field(..., description="Market ID")
    current_price: float = Field(..., description="Current market price")
    price_change_24h: float = Field(..., description="24-hour price change percentage")
    volume_24h: float = Field(..., description="24-hour trading volume")
    volume_change_24h: float = Field(
        ..., description="24-hour volume change percentage"
    )
    participant_count: int = Field(..., description="Total number of participants")
    active_traders: int = Field(..., description="Number of active traders")
    volatility_score: float = Field(..., ge=0.0, description="Price volatility score")
    momentum_score: float = Field(..., description="Price momentum score")
    liquidity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Liquidity score (0-1)"
    )
    social_activity: int = Field(..., description="Social media activity count")
    news_mentions: int = Field(..., description="News mentions count")
    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Sentiment score (-1 to 1)"
    )
    prediction_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Historical prediction accuracy"
    )
    last_updated: datetime = Field(..., description="When metrics were last updated")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "current_price": 0.55,
                "price_change_24h": 5.2,
                "volume_24h": 50000.0,
                "volume_change_24h": 12.5,
                "participant_count": 150,
                "active_traders": 45,
                "volatility_score": 0.25,
                "momentum_score": 0.15,
                "liquidity_score": 0.8,
                "social_activity": 25,
                "news_mentions": 10,
                "sentiment_score": 0.6,
                "prediction_accuracy": 0.75,
                "last_updated": "2024-01-15T10:30:00Z",
            }
        }


class UserAnalyticsResponse(BaseModel):
    """Response model for user analytics"""

    user_id: int = Field(..., description="User ID")
    total_trades: int = Field(..., description="Total number of trades")
    successful_trades: int = Field(..., description="Number of successful trades")
    total_volume: float = Field(..., description="Total trading volume")
    avg_trade_size: float = Field(..., description="Average trade size")
    win_rate: float = Field(..., ge=0.0, le=100.0, description="Win rate percentage")
    profit_loss: float = Field(..., description="Total profit/loss")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="User risk score (0-1)")
    trading_frequency: float = Field(..., description="Trades per day")
    preferred_categories: List[str] = Field(
        ..., description="Preferred trading categories"
    )
    last_active: datetime = Field(..., description="Last active timestamp")

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "total_trades": 50,
                "successful_trades": 35,
                "total_volume": 25000.0,
                "avg_trade_size": 500.0,
                "win_rate": 70.0,
                "profit_loss": 1500.0,
                "risk_score": 0.5,
                "trading_frequency": 2.5,
                "preferred_categories": ["technology", "finance"],
                "last_active": "2024-01-15T10:30:00Z",
            }
        }


class SystemAnalyticsResponse(BaseModel):
    """Response model for system analytics"""

    total_users: int = Field(..., description="Total number of users")
    active_users: int = Field(..., description="Number of active users")
    total_markets: int = Field(..., description="Total number of markets")
    active_markets: int = Field(..., description="Number of active markets")
    total_volume_24h: float = Field(..., description="Total 24-hour volume")
    total_trades_24h: int = Field(..., description="Total 24-hour trades")
    avg_response_time: float = Field(..., description="Average API response time")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate (0-1)")
    cpu_usage: float = Field(..., ge=0.0, le=1.0, description="CPU usage (0-1)")
    memory_usage: float = Field(..., ge=0.0, le=1.0, description="Memory usage (0-1)")
    database_connections: int = Field(..., description="Active database connections")
    cache_hit_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Cache hit rate (0-1)"
    )
    last_updated: datetime = Field(..., description="When metrics were last updated")

    class Config:
        schema_extra = {
            "example": {
                "total_users": 1000,
                "active_users": 150,
                "total_markets": 50,
                "active_markets": 25,
                "total_volume_24h": 500000.0,
                "total_trades_24h": 1000,
                "avg_response_time": 0.15,
                "error_rate": 0.02,
                "cpu_usage": 0.45,
                "memory_usage": 0.60,
                "database_connections": 25,
                "cache_hit_rate": 0.85,
                "last_updated": "2024-01-15T10:30:00Z",
            }
        }


class TrendAnalysisResponse(BaseModel):
    """Response model for trend analysis"""

    market_id: int = Field(..., description="Market ID")
    time_period_hours: int = Field(..., description="Analysis time period in hours")
    prices: List[float] = Field(..., description="Historical prices")
    volumes: List[float] = Field(..., description="Historical volumes")
    timestamps: List[str] = Field(..., description="Timestamps for data points")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "time_period_hours": 24,
                "prices": [0.5, 0.52, 0.48, 0.55, 0.53],
                "volumes": [1000.0, 1200.0, 800.0, 1500.0, 1100.0],
                "timestamps": [
                    "2024-01-14T10:30:00Z",
                    "2024-01-14T12:30:00Z",
                    "2024-01-14T14:30:00Z",
                    "2024-01-14T16:30:00Z",
                    "2024-01-14T18:30:00Z",
                ],
            }
        }


class MarketInsightResponse(BaseModel):
    """Response model for comprehensive market insights"""

    market_id: int = Field(..., description="Market ID")
    timestamp: datetime = Field(..., description="Insight generation timestamp")
    prediction: Optional[Dict[str, Any]] = Field(None, description="AI prediction data")
    analytics: Optional[Dict[str, Any]] = Field(
        None, description="Market analytics data"
    )
    trends: Dict[str, Any] = Field(..., description="Market trends data")
    summary: Dict[str, str] = Field(..., description="Market summary")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "timestamp": "2024-01-15T10:30:00Z",
                "prediction": {
                    "predicted_outcome": "A",
                    "confidence": 0.85,
                    "recommendation": "High confidence - recommended position in A",
                    "risk_level": "MEDIUM",
                },
                "analytics": {
                    "current_price": 0.55,
                    "volume_24h": 50000.0,
                    "volatility_score": 0.25,
                    "momentum_score": 0.15,
                    "sentiment_score": 0.6,
                },
                "trends": {
                    "prices": [0.5, 0.52, 0.48, 0.55, 0.53],
                    "volumes": [1000.0, 1200.0, 800.0, 1500.0, 1100.0],
                },
                "summary": {
                    "market_health": "healthy",
                    "trading_activity": "high",
                    "prediction_reliability": "high",
                },
            }
        }


class PortfolioInsightResponse(BaseModel):
    """Response model for portfolio insights"""

    user_id: int = Field(..., description="User ID")
    timestamp: datetime = Field(..., description="Insight generation timestamp")
    portfolio_summary: Dict[str, Any] = Field(..., description="Portfolio summary")
    performance_analysis: Dict[str, str] = Field(
        ..., description="Performance analysis"
    )
    recommendations: List[str] = Field(..., description="Trading recommendations")

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "timestamp": "2024-01-15T10:30:00Z",
                "portfolio_summary": {
                    "total_trades": 50,
                    "win_rate": 70.0,
                    "total_volume": 25000.0,
                    "profit_loss": 1500.0,
                    "risk_score": 0.5,
                },
                "performance_analysis": {
                    "performance_rating": "excellent",
                    "risk_assessment": "moderate",
                    "trading_style": "conservative",
                },
                "recommendations": [
                    "Consider diversifying your portfolio across different market categories",
                    "Monitor your risk exposure and adjust position sizes accordingly",
                    "Review your trading strategy based on your win rate performance",
                ],
            }
        }


class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages"""

    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(..., description="Message timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")


class WebSocketSubscriptionRequest(BaseModel):
    """Request model for WebSocket subscriptions"""

    type: str = Field("subscribe_market", description="Message type")
    market_id: int = Field(..., description="Market ID to subscribe to")


class WebSocketAnalyticsRequest(BaseModel):
    """Request model for WebSocket analytics requests"""

    type: str = Field("get_market_analytics", description="Message type")
    market_id: int = Field(..., description="Market ID for analytics")


class HealthCheckResponse(BaseModel):
    """Response model for health checks"""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(..., description="Health check timestamp")
    features: List[str] = Field(..., description="Available features")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "ai_analytics",
                "timestamp": "2024-01-15T10:30:00Z",
                "features": [
                    "market_predictions",
                    "real_time_analytics",
                    "trend_analysis",
                    "portfolio_insights",
                    "websocket_updates",
                ],
            }
        }
