from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy import desc

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.market import Market
from app.models.trade import Trade
from app.models.notification import Notification
from app.services.mobile_api import get_mobile_api_service

router = APIRouter()


@router.post("/device/register")
def register_device(
    push_token: str, device_info: Dict, current_user: User = Depends(get_current_user)
):
    """Register mobile device for push notifications"""
    mobile_service = get_mobile_api_service()
    result = mobile_service.register_device(current_user.id, push_token, device_info)

    return result


@router.post("/device/unregister")
def unregister_device(current_user: User = Depends(get_current_user)):
    """Unregister mobile device"""
    mobile_service = get_mobile_api_service()
    result = mobile_service.unregister_device(current_user.id)

    return result


@router.get("/dashboard")
def get_mobile_dashboard(current_user: User = Depends(get_current_user)):
    """Get mobile-optimized dashboard"""
    mobile_service = get_mobile_api_service()
    dashboard = mobile_service.get_mobile_dashboard(current_user.id)

    if "error" in dashboard:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=dashboard["error"]
        )

    return dashboard


@router.get("/market/{market_id}")
def get_mobile_market_details(
    market_id: int, current_user: Optional[User] = Depends(get_current_user)
):
    """Get mobile-optimized market details"""
    mobile_service = get_mobile_api_service()
    user_id = current_user.id if current_user else None
    market_details = mobile_service.get_mobile_market_details(market_id, user_id)

    if "error" in market_details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=market_details["error"]
        )

    return market_details


@router.get("/portfolio")
def get_mobile_portfolio(current_user: User = Depends(get_current_user)):
    """Get mobile-optimized portfolio view"""
    mobile_service = get_mobile_api_service()
    portfolio = mobile_service.get_mobile_portfolio(current_user.id)

    if "error" in portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=portfolio["error"]
        )

    return portfolio


@router.get("/leaderboard")
def get_mobile_leaderboard(
    category: str = Query("traders", pattern="^(traders|volume|win_rate)$"),
    period: str = Query("7d", pattern="^(24h|7d|30d|all_time)$"),
    limit: int = Query(20, ge=1, le=100),
):
    """Get mobile-optimized leaderboard"""
    mobile_service = get_mobile_api_service()
    leaderboard = mobile_service.get_mobile_leaderboard(category, period, limit)

    if "error" in leaderboard:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=leaderboard["error"]
        )

    return leaderboard


@router.post("/push-notification")
def send_push_notification(
    title: str,
    body: str,
    data: Optional[Dict] = None,
    current_user: User = Depends(get_current_user),
):
    """Send push notification to user's device"""
    mobile_service = get_mobile_api_service()
    result = mobile_service.send_push_notification(current_user.id, title, body, data)

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
        )

    return result


@router.get("/markets/recent")
def get_recent_markets(
    limit: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)
):
    """Get recent markets for mobile"""
    markets = (
        db.query(Market)
        .filter(Market.status == "open")
        .order_by(desc(Market.created_at))
        .limit(limit)
        .all()
    )

    return {
        "markets": [
            {
                "id": market.id,
                "title": market.title,
                "category": market.category,
                "current_price_a": market.current_price_a,
                "current_price_b": market.current_price_b,
                "volume_24h": market.volume_24h,
                "trending_score": market.trending_score,
                "created_at": market.created_at,
            }
            for market in markets
        ],
        "total": len(markets),
    }


@router.get("/markets/trending")
def get_trending_markets(
    limit: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)
):
    """Get trending markets for mobile"""
    markets = (
        db.query(Market)
        .filter(Market.status == "open")
        .order_by(desc(Market.trending_score))
        .limit(limit)
        .all()
    )

    return {
        "markets": [
            {
                "id": market.id,
                "title": market.title,
                "category": market.category,
                "current_price_a": market.current_price_a,
                "current_price_b": market.current_price_b,
                "volume_24h": market.volume_24h,
                "trending_score": market.trending_score,
            }
            for market in markets
        ],
        "total": len(markets),
    }


@router.get("/markets/category/{category}")
def get_markets_by_category(
    category: str, limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)
):
    """Get markets by category for mobile"""
    markets = (
        db.query(Market)
        .filter(Market.status == "open", Market.category == category)
        .order_by(desc(Market.volume_24h))
        .limit(limit)
        .all()
    )

    return {
        "category": category,
        "markets": [
            {
                "id": market.id,
                "title": market.title,
                "current_price_a": market.current_price_a,
                "current_price_b": market.current_price_b,
                "volume_24h": market.volume_24h,
                "trending_score": market.trending_score,
            }
            for market in markets
        ],
        "total": len(markets),
    }


@router.get("/trades/recent")
def get_recent_trades(
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's recent trades for mobile"""
    trades = (
        db.query(Trade)
        .filter(Trade.user_id == current_user.id)
        .order_by(desc(Trade.created_at))
        .limit(limit)
        .all()
    )

    return {
        "trades": [
            {
                "id": trade.id,
                "market_id": trade.market_id,
                "market_title": trade.market.title,
                "trade_type": trade.trade_type,
                "outcome": trade.outcome,
                "amount": trade.total_value,
                "profit_loss": trade.profit_loss,
                "created_at": trade.created_at,
            }
            for trade in trades
        ],
        "total": len(trades),
    }


@router.get("/notifications/unread-count")
def get_unread_notifications_count(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get unread notifications count for mobile"""
    unread_count = (
        db.query(Notification)
        .filter(Notification.user_id == current_user.id, Notification.is_read == False)
        .count()
    )

    return {"user_id": current_user.id, "unread_count": unread_count}


@router.get("/user/profile")
def get_mobile_user_profile(current_user: User = Depends(get_current_user)):
    """Get mobile-optimized user profile"""
    return {
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "bio": current_user.bio,
            "reputation_score": current_user.reputation_score,
            "total_trades": current_user.total_trades,
            "win_rate": current_user.win_rate,
            "total_profit": current_user.total_profit,
            "total_volume": current_user.total_volume,
            "available_balance": current_user.available_balance,
            "created_at": current_user.created_at,
        }
    }


@router.get("/search/markets")
def search_markets_mobile(
    query: str, limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)
):
    """Search markets for mobile"""
    markets = (
        db.query(Market)
        .filter(Market.status == "open", Market.title.ilike(f"%{query}%"))
        .order_by(desc(Market.volume_24h))
        .limit(limit)
        .all()
    )

    return {
        "query": query,
        "markets": [
            {
                "id": market.id,
                "title": market.title,
                "category": market.category,
                "current_price_a": market.current_price_a,
                "current_price_b": market.current_price_b,
                "volume_24h": market.volume_24h,
            }
            for market in markets
        ],
        "total": len(markets),
    }
