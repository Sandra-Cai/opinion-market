"""
AI Analytics API Endpoints
Provides AI predictions, real-time analytics, and market insights
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.redis_client import get_redis_client
from app.services.ai_predictions import get_ai_prediction_service, PredictionResult
from app.services.real_time_analytics import (
    get_real_time_analytics_service,
    MarketMetrics,
    UserMetrics,
    SystemMetrics,
)
from app.schemas.ai_analytics import (
    PredictionRequest,
    PredictionResponse,
    MarketAnalyticsResponse,
    UserAnalyticsResponse,
    SystemAnalyticsResponse,
    TrendAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connections for real-time updates
websocket_connections: Dict[str, WebSocket] = {}

# Redis client is now imported from core.redis_client


@router.get("/predictions/{market_id}", response_model=PredictionResponse)
async def get_market_prediction(
    market_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get AI prediction for a specific market
    """
    try:
        # Get AI prediction service
        redis_client = await get_redis_client()
        ai_service = await get_ai_prediction_service(redis_client, db)

        # Check for cached prediction first
        cached_prediction = await ai_service.get_cached_prediction(market_id)
        if cached_prediction:
            return PredictionResponse(
                market_id=market_id,
                predicted_outcome=cached_prediction.predicted_outcome,
                confidence=cached_prediction.confidence,
                probability_a=cached_prediction.probability_a,
                probability_b=cached_prediction.probability_b,
                recommendation=cached_prediction.recommendation,
                risk_level=cached_prediction.risk_level,
                model_version=cached_prediction.model_version,
                prediction_time=cached_prediction.prediction_time,
                validity_duration=cached_prediction.validity_duration,
                features_used=cached_prediction.features_used,
            )

        # Generate new prediction
        prediction = await ai_service.predict_market_outcome(market_id)
        if not prediction:
            raise HTTPException(
                status_code=404, detail="Market not found or insufficient data"
            )

        return PredictionResponse(
            market_id=market_id,
            predicted_outcome=prediction.predicted_outcome,
            confidence=prediction.confidence,
            probability_a=prediction.probability_a,
            probability_b=prediction.probability_b,
            recommendation=prediction.recommendation,
            risk_level=prediction.risk_level,
            model_version=prediction.model_version,
            prediction_time=prediction.prediction_time,
            validity_duration=prediction.validity_duration,
            features_used=prediction.features_used,
        )

    except Exception as e:
        logger.error(f"Error getting market prediction: {e}")
        raise HTTPException(status_code=500, detail="Error generating prediction")


@router.post("/predictions/batch", response_model=List[PredictionResponse])
async def get_batch_predictions(
    request: PredictionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get AI predictions for multiple markets
    """
    try:
        redis_client = await get_redis_client()
        ai_service = await get_ai_prediction_service(redis_client, db)

        predictions = []
        for market_id in request.market_ids:
            prediction = await ai_service.predict_market_outcome(market_id)
            if prediction:
                predictions.append(
                    PredictionResponse(
                        market_id=market_id,
                        predicted_outcome=prediction.predicted_outcome,
                        confidence=prediction.confidence,
                        probability_a=prediction.probability_a,
                        probability_b=prediction.probability_b,
                        recommendation=prediction.recommendation,
                        risk_level=prediction.risk_level,
                        model_version=prediction.model_version,
                        prediction_time=prediction.prediction_time,
                        validity_duration=prediction.validity_duration,
                        features_used=prediction.features_used,
                    )
                )

        return predictions

    except Exception as e:
        logger.error(f"Error getting batch predictions: {e}")
        raise HTTPException(status_code=500, detail="Error generating predictions")


@router.get("/analytics/market/{market_id}", response_model=MarketAnalyticsResponse)
async def get_market_analytics(
    market_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get real-time analytics for a specific market
    """
    try:
        redis_client = await get_redis_client()
        analytics_service = await get_real_time_analytics_service(redis_client, db)

        metrics = await analytics_service.get_market_metrics(market_id)
        if not metrics:
            raise HTTPException(status_code=404, detail="Market not found")

        return MarketAnalyticsResponse(
            market_id=market_id,
            current_price=metrics.current_price,
            price_change_24h=metrics.price_change_24h,
            volume_24h=metrics.volume_24h,
            volume_change_24h=metrics.volume_change_24h,
            participant_count=metrics.participant_count,
            active_traders=metrics.active_traders,
            volatility_score=metrics.volatility_score,
            momentum_score=metrics.momentum_score,
            liquidity_score=metrics.liquidity_score,
            social_activity=metrics.social_activity,
            news_mentions=metrics.news_mentions,
            sentiment_score=metrics.sentiment_score,
            prediction_accuracy=metrics.prediction_accuracy,
            last_updated=metrics.last_updated,
        )

    except Exception as e:
        logger.error(f"Error getting market analytics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving market analytics")


@router.get("/analytics/user/{user_id}", response_model=UserAnalyticsResponse)
async def get_user_analytics(
    user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get analytics for a specific user
    """
    try:
        redis_client = await get_redis_client()
        analytics_service = await get_real_time_analytics_service(redis_client, db)

        metrics = await analytics_service.get_user_metrics(user_id)
        if not metrics:
            raise HTTPException(status_code=404, detail="User not found")

        return UserAnalyticsResponse(
            user_id=user_id,
            total_trades=metrics.total_trades,
            successful_trades=metrics.successful_trades,
            total_volume=metrics.total_volume,
            avg_trade_size=metrics.avg_trade_size,
            win_rate=metrics.win_rate,
            profit_loss=metrics.profit_loss,
            risk_score=metrics.risk_score,
            trading_frequency=metrics.trading_frequency,
            preferred_categories=metrics.preferred_categories,
            last_active=metrics.last_active,
        )

    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving user analytics")


@router.get("/analytics/system", response_model=SystemAnalyticsResponse)
async def get_system_analytics(
    db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get system-wide analytics
    """
    try:
        redis_client = await get_redis_client()
        analytics_service = await get_real_time_analytics_service(redis_client, db)

        metrics = await analytics_service.get_system_metrics()

        return SystemAnalyticsResponse(
            total_users=metrics.total_users,
            active_users=metrics.active_users,
            total_markets=metrics.total_markets,
            active_markets=metrics.active_markets,
            total_volume_24h=metrics.total_volume_24h,
            total_trades_24h=metrics.total_trades_24h,
            avg_response_time=metrics.avg_response_time,
            error_rate=metrics.error_rate,
            cpu_usage=metrics.cpu_usage,
            memory_usage=metrics.memory_usage,
            database_connections=metrics.database_connections,
            cache_hit_rate=metrics.cache_hit_rate,
            last_updated=metrics.last_updated,
        )

    except Exception as e:
        logger.error(f"Error getting system analytics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving system analytics")


@router.get("/analytics/top-markets", response_model=List[MarketAnalyticsResponse])
async def get_top_markets(
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get top markets by volume
    """
    try:
        redis_client = await get_redis_client()
        analytics_service = await get_real_time_analytics_service(redis_client, db)

        top_markets = await analytics_service.get_top_markets(limit)

        return [
            MarketAnalyticsResponse(
                market_id=market.market_id,
                current_price=market.current_price,
                price_change_24h=market.price_change_24h,
                volume_24h=market.volume_24h,
                volume_change_24h=market.volume_change_24h,
                participant_count=market.participant_count,
                active_traders=market.active_traders,
                volatility_score=market.volatility_score,
                momentum_score=market.momentum_score,
                liquidity_score=market.liquidity_score,
                social_activity=market.social_activity,
                news_mentions=market.news_mentions,
                sentiment_score=market.sentiment_score,
                prediction_accuracy=market.prediction_accuracy,
                last_updated=market.last_updated,
            )
            for market in top_markets
        ]

    except Exception as e:
        logger.error(f"Error getting top markets: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving top markets")


@router.get("/analytics/top-traders", response_model=List[UserAnalyticsResponse])
async def get_top_traders(
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get top traders by volume
    """
    try:
        redis_client = await get_redis_client()
        analytics_service = await get_real_time_analytics_service(redis_client, db)

        top_traders = await analytics_service.get_top_traders(limit)

        return [
            UserAnalyticsResponse(
                user_id=user.user_id,
                total_trades=user.total_trades,
                successful_trades=user.successful_trades,
                total_volume=user.total_volume,
                avg_trade_size=user.avg_trade_size,
                win_rate=user.win_rate,
                profit_loss=user.profit_loss,
                risk_score=user.risk_score,
                trading_frequency=user.trading_frequency,
                preferred_categories=user.preferred_categories,
                last_active=user.last_active,
            )
            for user in top_traders
        ]

    except Exception as e:
        logger.error(f"Error getting top traders: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving top traders")


@router.get("/analytics/trends/{market_id}", response_model=TrendAnalysisResponse)
async def get_market_trends(
    market_id: int,
    hours: int = 24,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get market trends over time
    """
    try:
        redis_client = await get_redis_client()
        analytics_service = await get_real_time_analytics_service(redis_client, db)

        trends = await analytics_service.get_market_trends(market_id, hours)

        return TrendAnalysisResponse(
            market_id=market_id,
            time_period_hours=hours,
            prices=trends["prices"],
            volumes=trends["volumes"],
            timestamps=trends["timestamps"],
        )

    except Exception as e:
        logger.error(f"Error getting market trends: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving market trends")


@router.websocket("/ws/analytics/{client_id}")
async def websocket_analytics(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time analytics updates
    """
    await websocket.accept()
    websocket_connections[client_id] = websocket

    try:
        # Send initial connection confirmation
        await websocket.send_text(
            json.dumps(
                {
                    "type": "connection_established",
                    "client_id": client_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        )

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "subscribe_market":
                    market_id = message.get("market_id")
                    # Subscribe to specific market updates
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "subscription_confirmed",
                                "market_id": market_id,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )

                elif message.get("type") == "get_market_analytics":
                    market_id = message.get("market_id")
                    # Get current market analytics
                    # This would integrate with your analytics service
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "market_analytics",
                                "market_id": market_id,
                                "data": {
                                    "current_price": 0.55,
                                    "volume_24h": 50000.0,
                                    "participant_count": 150,
                                },
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                )

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    finally:
        if client_id in websocket_connections:
            del websocket_connections[client_id]


@router.get("/insights/market/{market_id}")
async def get_market_insights(
    market_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get comprehensive market insights including AI predictions and analytics
    """
    try:
        redis_client = await get_redis_client()

        # Get AI prediction
        ai_service = await get_ai_prediction_service(redis_client, db)
        prediction = await ai_service.predict_market_outcome(market_id)

        # Get market analytics
        analytics_service = await get_real_time_analytics_service(redis_client, db)
        metrics = await analytics_service.get_market_metrics(market_id)

        # Get market trends
        trends = await analytics_service.get_market_trends(market_id, 24)

        insights = {
            "market_id": market_id,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": (
                {
                    "predicted_outcome": (
                        prediction.predicted_outcome if prediction else None
                    ),
                    "confidence": prediction.confidence if prediction else None,
                    "recommendation": prediction.recommendation if prediction else None,
                    "risk_level": prediction.risk_level if prediction else None,
                }
                if prediction
                else None
            ),
            "analytics": (
                {
                    "current_price": metrics.current_price if metrics else None,
                    "volume_24h": metrics.volume_24h if metrics else None,
                    "volatility_score": metrics.volatility_score if metrics else None,
                    "momentum_score": metrics.momentum_score if metrics else None,
                    "sentiment_score": metrics.sentiment_score if metrics else None,
                }
                if metrics
                else None
            ),
            "trends": trends,
            "summary": {
                "market_health": (
                    "healthy"
                    if metrics and metrics.volatility_score < 0.3
                    else "volatile"
                ),
                "trading_activity": (
                    "high" if metrics and metrics.volume_24h > 100000 else "moderate"
                ),
                "prediction_reliability": (
                    "high" if prediction and prediction.confidence > 0.7 else "moderate"
                ),
            },
        }

        return JSONResponse(content=insights)

    except Exception as e:
        logger.error(f"Error getting market insights: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving market insights")


@router.get("/insights/portfolio/{user_id}")
async def get_portfolio_insights(
    user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get portfolio insights for a user
    """
    try:
        redis_client = await get_redis_client()

        # Get user analytics
        analytics_service = await get_real_time_analytics_service(redis_client, db)
        user_metrics = await analytics_service.get_user_metrics(user_id)

        # Get user's active markets (this would depend on your data model)
        active_markets = []  # You would query this from your database

        insights = {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "portfolio_summary": (
                {
                    "total_trades": user_metrics.total_trades if user_metrics else 0,
                    "win_rate": user_metrics.win_rate if user_metrics else 0.0,
                    "total_volume": user_metrics.total_volume if user_metrics else 0.0,
                    "profit_loss": user_metrics.profit_loss if user_metrics else 0.0,
                    "risk_score": user_metrics.risk_score if user_metrics else 0.0,
                }
                if user_metrics
                else {}
            ),
            "performance_analysis": (
                {
                    "performance_rating": (
                        "excellent"
                        if user_metrics and user_metrics.win_rate > 0.7
                        else (
                            "good"
                            if user_metrics and user_metrics.win_rate > 0.5
                            else "needs_improvement"
                        )
                    ),
                    "risk_assessment": (
                        "low"
                        if user_metrics and user_metrics.risk_score < 0.3
                        else (
                            "moderate"
                            if user_metrics and user_metrics.risk_score < 0.6
                            else "high"
                        )
                    ),
                    "trading_style": (
                        "conservative"
                        if user_metrics and user_metrics.avg_trade_size < 1000
                        else "aggressive"
                    ),
                }
                if user_metrics
                else {}
            ),
            "recommendations": [
                "Consider diversifying your portfolio across different market categories",
                "Monitor your risk exposure and adjust position sizes accordingly",
                "Review your trading strategy based on your win rate performance",
            ],
        }

        return JSONResponse(content=insights)

    except Exception as e:
        logger.error(f"Error getting portfolio insights: {e}")
        raise HTTPException(
            status_code=500, detail="Error retrieving portfolio insights"
        )


@router.get("/health")
async def ai_analytics_health():
    """
    Health check for AI analytics services
    """
    return {
        "status": "healthy",
        "service": "ai_analytics",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "market_predictions",
            "real_time_analytics",
            "trend_analysis",
            "portfolio_insights",
            "websocket_updates",
        ],
    }
