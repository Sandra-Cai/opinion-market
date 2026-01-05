import logging
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import json
import asyncio

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.services.market_data_feed import (
    get_market_data_feed,
    MarketDataPoint,
    MarketAlert,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/market/{market_id}/data")
def get_market_data(
    market_id: int, current_user: Optional[User] = Depends(get_current_user)
):
    """Get real-time market data for a specific market"""
    market_data_feed = get_market_data_feed()

    # Get market data from cache
    data_point = asyncio.run(market_data_feed.get_market_data(market_id))

    if not data_point:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market data not available"
        )

    return {
        "market_id": data_point.market_id,
        "timestamp": data_point.timestamp.isoformat(),
        "price_a": data_point.price_a,
        "price_b": data_point.price_b,
        "volume_24h": data_point.volume_24h,
        "volume_total": data_point.volume_total,
        "unique_traders": data_point.unique_traders,
        "price_change_24h": data_point.price_change_24h,
        "price_change_1h": data_point.price_change_1h,
        "liquidity_a": data_point.liquidity_a,
        "liquidity_b": data_point.liquidity_b,
        "spread": data_point.spread,
        "volatility": data_point.volatility,
    }


@router.get("/markets/data")
def get_all_markets_data(current_user: Optional[User] = Depends(get_current_user)):
    """Get real-time data for all active markets"""
    market_data_feed = get_market_data_feed()

    # Get all market data from cache
    all_data = asyncio.run(market_data_feed.get_all_market_data())

    markets_data = []
    for market_id, data_point in all_data.items():
        markets_data.append(
            {
                "market_id": data_point.market_id,
                "timestamp": data_point.timestamp.isoformat(),
                "price_a": data_point.price_a,
                "price_b": data_point.price_b,
                "volume_24h": data_point.volume_24h,
                "volume_total": data_point.volume_total,
                "unique_traders": data_point.unique_traders,
                "price_change_24h": data_point.price_change_24h,
                "price_change_1h": data_point.price_change_1h,
                "liquidity_a": data_point.liquidity_a,
                "liquidity_b": data_point.liquidity_b,
                "spread": data_point.spread,
                "volatility": data_point.volatility,
            }
        )

    return {
        "markets": markets_data,
        "total": len(markets_data),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/alerts")
def get_market_alerts(
    market_id: Optional[int] = Query(None, description="Filter by market ID"),
    severity: Optional[str] = Query(
        None, description="Filter by severity (low, medium, high, critical)"
    ),
    limit: int = Query(50, description="Number of alerts to return"),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get market alerts with optional filtering"""
    market_data_feed = get_market_data_feed()

    # Validate severity filter
    if severity and severity not in ["low", "medium", "high", "critical"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid severity level"
        )

    # Get alerts
    alerts = asyncio.run(market_data_feed.get_market_alerts(market_id, severity))

    # Limit results
    alerts = alerts[-limit:] if limit > 0 else alerts

    alerts_data = []
    for alert in alerts:
        alerts_data.append(
            {
                "market_id": alert.market_id,
                "alert_type": alert.alert_type,
                "message": alert.message,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat(),
                "data": alert.data,
            }
        )

    return {
        "alerts": alerts_data,
        "total": len(alerts_data),
        "filters": {"market_id": market_id, "severity": severity},
    }


@router.get("/market/{market_id}/statistics")
def get_market_statistics(
    market_id: int,
    period: str = Query("24h", description="Time period (1h, 24h, 7d, 30d)"),
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get detailed market statistics"""
    from app.models.market import Market
    from app.models.trade import Trade

    # Validate period
    valid_periods = ["1h", "24h", "7d", "30d"]
    if period not in valid_periods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period. Must be one of: {valid_periods}",
        )

    # Calculate time range
    now = datetime.utcnow()
    if period == "1h":
        start_time = now - timedelta(hours=1)
    elif period == "24h":
        start_time = now - timedelta(days=1)
    elif period == "7d":
        start_time = now - timedelta(days=7)
    else:  # 30d
        start_time = now - timedelta(days=30)

    # Get market
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    # Get trades in period
    trades = (
        db.query(Trade)
        .filter(Trade.market_id == market_id, Trade.created_at >= start_time)
        .all()
    )

    if not trades:
        return {
            "market_id": market_id,
            "period": period,
            "total_volume": 0.0,
            "total_trades": 0,
            "unique_traders": 0,
            "average_trade_size": 0.0,
            "price_volatility": 0.0,
            "volume_distribution": {"outcome_a": 0.0, "outcome_b": 0.0},
            "price_range": {
                "min": market.current_price_a,
                "max": market.current_price_a,
                "current": market.current_price_a,
            },
        }

    # Calculate statistics
    total_volume = sum(trade.total_value for trade in trades)
    total_trades = len(trades)
    unique_traders = len(set(trade.user_id for trade in trades))
    average_trade_size = total_volume / total_trades if total_trades > 0 else 0.0

    # Calculate volume distribution
    volume_a = sum(
        trade.total_value for trade in trades if trade.outcome == "outcome_a"
    )
    volume_b = sum(
        trade.total_value for trade in trades if trade.outcome == "outcome_b"
    )

    # Calculate price volatility
    prices = [trade.price_per_share for trade in trades]
    if len(prices) > 1:
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        volatility = variance**0.5
    else:
        volatility = 0.0

    # Calculate price range
    min_price = min(prices) if prices else market.current_price_a
    max_price = max(prices) if prices else market.current_price_a

    return {
        "market_id": market_id,
        "period": period,
        "total_volume": total_volume,
        "total_trades": total_trades,
        "unique_traders": unique_traders,
        "average_trade_size": average_trade_size,
        "price_volatility": volatility,
        "volume_distribution": {"outcome_a": volume_a, "outcome_b": volume_b},
        "price_range": {
            "min": min_price,
            "max": max_price,
            "current": market.current_price_a,
        },
    }


@router.get("/markets/trending")
def get_trending_markets(
    limit: int = Query(10, description="Number of trending markets to return"),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get trending markets based on volume and price changes"""
    market_data_feed = get_market_data_feed()

    # Get all market data
    all_data = asyncio.run(market_data_feed.get_all_market_data())

    # Calculate trending scores
    trending_markets = []
    for market_id, data_point in all_data.items():
        # Simple trending algorithm based on volume and price change
        volume_score = data_point.volume_24h / 1000  # Normalize volume
        price_change_score = (
            abs(data_point.price_change_24h) * 10
        )  # Amplify price changes
        trending_score = volume_score + price_change_score

        trending_markets.append(
            {
                "market_id": market_id,
                "trending_score": trending_score,
                "volume_24h": data_point.volume_24h,
                "price_change_24h": data_point.price_change_24h,
                "unique_traders": data_point.unique_traders,
                "volatility": data_point.volatility,
            }
        )

    # Sort by trending score and limit results
    trending_markets.sort(key=lambda x: x["trending_score"], reverse=True)
    trending_markets = trending_markets[:limit]

    return {
        "trending_markets": trending_markets,
        "total": len(trending_markets),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/markets/volatile")
def get_volatile_markets(
    limit: int = Query(10, description="Number of volatile markets to return"),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get markets with highest volatility"""
    market_data_feed = get_market_data_feed()

    # Get all market data
    all_data = asyncio.run(market_data_feed.get_all_market_data())

    # Sort by volatility
    volatile_markets = []
    for market_id, data_point in all_data.items():
        volatile_markets.append(
            {
                "market_id": market_id,
                "volatility": data_point.volatility,
                "price_change_1h": data_point.price_change_1h,
                "price_change_24h": data_point.price_change_24h,
                "volume_24h": data_point.volume_24h,
                "spread": data_point.spread,
            }
        )

    # Sort by volatility and limit results
    volatile_markets.sort(key=lambda x: x["volatility"], reverse=True)
    volatile_markets = volatile_markets[:limit]

    return {
        "volatile_markets": volatile_markets,
        "total": len(volatile_markets),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/markets/high-volume")
def get_high_volume_markets(
    limit: int = Query(10, description="Number of high volume markets to return"),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Get markets with highest trading volume"""
    market_data_feed = get_market_data_feed()

    # Get all market data
    all_data = asyncio.run(market_data_feed.get_all_market_data())

    # Sort by volume
    high_volume_markets = []
    for market_id, data_point in all_data.items():
        high_volume_markets.append(
            {
                "market_id": market_id,
                "volume_24h": data_point.volume_24h,
                "volume_total": data_point.volume_total,
                "unique_traders": data_point.unique_traders,
                "price_change_24h": data_point.price_change_24h,
                "volatility": data_point.volatility,
            }
        )

    # Sort by volume and limit results
    high_volume_markets.sort(key=lambda x: x["volume_24h"], reverse=True)
    high_volume_markets = high_volume_markets[:limit]

    return {
        "high_volume_markets": high_volume_markets,
        "total": len(high_volume_markets),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await websocket.accept()

    try:
        market_data_feed = get_market_data_feed()

        # Send initial data
        all_data = await market_data_feed.get_all_market_data()
        await websocket.send_text(
            json.dumps(
                {
                    "type": "initial_data",
                    "data": {
                        market_id: {
                            "price_a": data_point.price_a,
                            "price_b": data_point.price_b,
                            "volume_24h": data_point.volume_24h,
                            "price_change_24h": data_point.price_change_24h,
                        }
                        for market_id, data_point in all_data.items()
                    },
                }
            )
        )

        # Keep connection alive and send updates
        while True:
            await asyncio.sleep(5)  # Update every 5 seconds

            # Get latest data
            all_data = await market_data_feed.get_all_market_data()

            # Send update
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            market_id: {
                                "price_a": data_point.price_a,
                                "price_b": data_point.price_b,
                                "volume_24h": data_point.volume_24h,
                                "price_change_24h": data_point.price_change_24h,
                            }
                            for market_id, data_point in all_data.items()
                        },
                    }
                )
            )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close()


@router.websocket("/ws/market-alerts")
async def websocket_market_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time market alerts"""
    await websocket.accept()

    try:
        market_data_feed = get_market_data_feed()

        # Send recent alerts
        recent_alerts = await market_data_feed.get_market_alerts()
        recent_alerts = recent_alerts[-10:]  # Last 10 alerts

        await websocket.send_text(
            json.dumps(
                {
                    "type": "initial_alerts",
                    "alerts": [
                        {
                            "market_id": alert.market_id,
                            "alert_type": alert.alert_type,
                            "message": alert.message,
                            "severity": alert.severity,
                            "timestamp": alert.timestamp.isoformat(),
                        }
                        for alert in recent_alerts
                    ],
                }
            )
        )

        # Keep connection alive
        while True:
            await asyncio.sleep(30)  # Check for new alerts every 30 seconds

            # Send heartbeat
            await websocket.send_text(
                json.dumps(
                    {"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()}
                )
            )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close()


@router.get("/health")
def get_market_data_health():
    """Get health status of market data feed"""
    market_data_feed = get_market_data_feed()

    # Get basic stats
    all_data = asyncio.run(market_data_feed.get_all_market_data())
    alerts = asyncio.run(market_data_feed.get_market_alerts())

    # Count alerts by severity
    alert_counts = {}
    for alert in alerts:
        severity = alert.severity
        alert_counts[severity] = alert_counts.get(severity, 0) + 1

    return {
        "status": "healthy",
        "active_markets": len(all_data),
        "total_alerts": len(alerts),
        "alert_distribution": alert_counts,
        "last_update": datetime.utcnow().isoformat(),
    }
