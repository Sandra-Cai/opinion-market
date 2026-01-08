import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.decorators import handle_errors, log_execution_time
from app.core.query_helpers import get_or_404
from app.models.user import User
from app.models.market import Market, MarketCategory
from app.services.analytics_service import analytics_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/market/{market_id}")
@handle_errors(default_message="Failed to retrieve market analytics")
@log_execution_time
async def get_market_analytics(
    market_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get comprehensive analytics for a specific market.
    
    Args:
        market_id: Market ID to get analytics for
        db: Database session
        
    Returns:
        Market analytics data
        
    Raises:
        HTTPException: If market not found
    """
    # Verify market exists
    get_or_404(
        db.query(Market).filter(Market.id == market_id),
        error_message="Market not found"
    )
    
    analytics = analytics_service.get_market_analytics(market_id)
    if not analytics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analytics not available for this market"
        )
    return analytics


@router.get("/user/me")
@handle_errors(default_message="Failed to retrieve user analytics")
@log_execution_time
async def get_my_analytics(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get analytics for the current authenticated user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User analytics data
    """
    return analytics_service.get_user_analytics(current_user.id)


@router.get("/user/{user_id}")
@handle_errors(default_message="Failed to retrieve user analytics")
@log_execution_time
async def get_user_analytics(
    user_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get analytics for a specific user (public data only).
    
    Args:
        user_id: User ID to get analytics for
        db: Database session
        
    Returns:
        Public user analytics data
        
    Raises:
        HTTPException: If user not found
    """
    user = get_or_404(
        db.query(User).filter(User.id == user_id),
        error_message="User not found"
    )

    # Return public analytics only
    return {
        "user_id": user.id,
        "username": user.username,
        "total_trades": user.total_trades,
        "win_rate": user.win_rate,
        "reputation_score": user.reputation_score,
        "total_volume": user.total_volume,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.get("/platform")
@handle_errors(default_message="Failed to retrieve platform analytics")
@log_execution_time
async def get_platform_analytics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get platform-wide analytics and statistics.
    
    Args:
        db: Database session
        
    Returns:
        Platform analytics data
    """
    return analytics_service.get_platform_analytics()


@router.get("/market/{market_id}/predictions")
@handle_errors(default_message="Failed to retrieve market predictions")
@log_execution_time
async def get_market_predictions(
    market_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get market prediction analytics.
    
    Args:
        market_id: Market ID to get predictions for
        db: Database session
        
    Returns:
        Market prediction data
        
    Raises:
        HTTPException: If market not found or predictions unavailable
    """
    # Verify market exists
    get_or_404(
        db.query(Market).filter(Market.id == market_id),
        error_message="Market not found"
    )
    
    predictions = analytics_service.get_market_predictions(market_id)
    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Predictions not available for this market"
        )
    return predictions


@router.get("/trending")
@handle_errors(default_message="Failed to retrieve trending analytics")
@log_execution_time
async def get_trending_analytics(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of trending items to return"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get trending markets and analytics.
    
    Args:
        limit: Maximum number of trending items to return
        db: Database session
        
    Returns:
        Trending markets and analytics data
    """
    # Get trending markets
    trending_markets = (
        db.query(Market)
        .filter(Market.status == "open")
        .order_by(Market.trending_score.desc())
        .limit(limit)
        .all()
    )

    trending_data = []
    for market in trending_markets:
        analytics = analytics_service.get_market_analytics(market.id)
        trending_data.append(
            {
                "market": {
                    "id": market.id,
                    "title": market.title,
                    "category": market.category,
                    "trending_score": market.trending_score,
                    "volume_24h": market.volume_24h,
                    "current_price_a": market.current_price_a,
                    "current_price_b": market.current_price_b,
                },
                "analytics": analytics,
            }
        )

    return {"trending_markets": trending_data, "total": len(trending_data)}


@router.get("/volume-by-category")
def get_volume_by_category(
    period: str = Query("all_time", pattern="^(24h|7d|30d|all_time)$"),
    db: Session = Depends(get_db),
):
    """Get trading volume by market category"""
    from app.models.trade import Trade
    from sqlalchemy import func

    # Calculate time filter
    if period == "24h":
        since = datetime.utcnow() - timedelta(days=1)
    elif period == "7d":
        since = datetime.utcnow() - timedelta(days=7)
    elif period == "30d":
        since = datetime.utcnow() - timedelta(days=30)
    else:
        since = None

    # Build query
    query = db.query(
        Market.category,
        func.sum(Trade.total_value).label("volume"),
        func.count(Trade.id).label("trade_count"),
    ).join(Trade, Market.id == Trade.market_id)

    if since:
        query = query.filter(Trade.created_at >= since)

    result = query.group_by(Market.category).order_by(func.desc("volume")).all()

    return {
        "period": period,
        "categories": [
            {
                "category": str(r.category),
                "volume": float(r.volume),
                "trade_count": r.trade_count,
            }
            for r in result
        ],
    }


@router.get("/price-movements")
def get_price_movements(
    market_id: Optional[int] = None,
    hours: int = Query(24, ge=1, le=168),
    db: Session = Depends(get_db),
):
    """Get price movement data for markets"""
    from app.models.trade import Trade
    from sqlalchemy import func

    since = datetime.utcnow() - timedelta(hours=hours)

    query = db.query(
        Trade.market_id,
        func.date_trunc("hour", Trade.created_at).label("time"),
        func.avg(Trade.price_per_share).label("avg_price"),
        func.sum(Trade.total_value).label("volume"),
    ).filter(Trade.created_at >= since)

    if market_id:
        query = query.filter(Trade.market_id == market_id)

    result = (
        query.group_by(Trade.market_id, func.date_trunc("hour", Trade.created_at))
        .order_by("time")
        .all()
    )

    # Group by market
    movements = {}
    for r in result:
        if r.market_id not in movements:
            movements[r.market_id] = []

        movements[r.market_id].append(
            {
                "time": str(r.time),
                "avg_price": float(r.avg_price),
                "volume": float(r.volume),
            }
        )

    return {"hours": hours, "price_movements": movements}


@router.get("/sentiment-analysis")
def get_sentiment_analysis(
    market_id: Optional[int] = None, db: Session = Depends(get_db)
):
    """Get trading sentiment analysis"""
    from app.models.trade import Trade
    from sqlalchemy import func

    query = db.query(
        Trade.market_id,
        func.sum(
            func.case([(Trade.trade_type == "buy", Trade.total_value)], else_=0)
        ).label("buy_volume"),
        func.sum(
            func.case([(Trade.trade_type == "sell", Trade.total_value)], else_=0)
        ).label("sell_volume"),
        func.count(func.case([(Trade.trade_type == "buy", 1)], else_=None)).label(
            "buy_trades"
        ),
        func.count(func.case([(Trade.trade_type == "sell", 1)], else_=None)).label(
            "sell_trades"
        ),
    )

    if market_id:
        query = query.filter(Trade.market_id == market_id)

    result = query.group_by(Trade.market_id).all()

    sentiment_data = []
    for r in result:
        total_volume = float(r.buy_volume + r.sell_volume)
        buy_ratio = float(r.buy_volume) / total_volume if total_volume > 0 else 0.5

        sentiment_data.append(
            {
                "market_id": r.market_id,
                "buy_volume": float(r.buy_volume),
                "sell_volume": float(r.sell_volume),
                "buy_trades": r.buy_trades,
                "sell_trades": r.sell_trades,
                "buy_ratio": buy_ratio,
                "sentiment": (
                    "bullish"
                    if buy_ratio > 0.6
                    else "bearish" if buy_ratio < 0.4 else "neutral"
                ),
            }
        )

    return {"sentiment_analysis": sentiment_data, "total_markets": len(sentiment_data)}
