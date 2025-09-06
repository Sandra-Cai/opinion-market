from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.user import User
from app.models.market import Market
from app.models.trade import Trade
from app.schemas.user import UserResponse

router = APIRouter()


@router.get("/traders")
def get_top_traders(
    period: str = Query("all_time", regex="^(24h|7d|30d|all_time)$"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get top traders by profit/loss"""

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
    query = db.query(User).filter(User.is_active == True)

    if since:
        # Filter by recent activity
        recent_traders = (
            db.query(Trade.user_id)
            .filter(Trade.created_at >= since)
            .distinct()
            .subquery()
        )
        query = query.filter(User.id.in_(recent_traders))

    # Order by profit
    top_traders = query.order_by(desc(User.total_profit)).limit(limit).all()

    return {
        "period": period,
        "traders": [
            {
                "rank": i + 1,
                "user": {
                    "id": trader.id,
                    "username": trader.username,
                    "total_profit": trader.total_profit,
                    "total_volume": trader.total_volume,
                    "win_rate": trader.win_rate,
                    "reputation_score": trader.reputation_score,
                    "total_trades": trader.total_trades,
                },
            }
            for i, trader in enumerate(top_traders)
        ],
    }


@router.get("/volume")
def get_top_volume_traders(
    period: str = Query("all_time", regex="^(24h|7d|30d|all_time)$"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get top traders by volume"""

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
    query = db.query(User).filter(User.is_active == True)

    if since:
        # Filter by recent activity
        recent_traders = (
            db.query(Trade.user_id)
            .filter(Trade.created_at >= since)
            .distinct()
            .subquery()
        )
        query = query.filter(User.id.in_(recent_traders))

    # Order by volume
    top_traders = query.order_by(desc(User.total_volume)).limit(limit).all()

    return {
        "period": period,
        "traders": [
            {
                "rank": i + 1,
                "user": {
                    "id": trader.id,
                    "username": trader.username,
                    "total_volume": trader.total_volume,
                    "total_profit": trader.total_profit,
                    "avg_trade_size": trader.avg_trade_size,
                    "total_trades": trader.total_trades,
                },
            }
            for i, trader in enumerate(top_traders)
        ],
    }


@router.get("/markets")
def get_top_market_creators(
    period: str = Query("all_time", regex="^(24h|7d|30d|all_time)$"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get top market creators by volume"""

    # Calculate time filter
    if period == "24h":
        since = datetime.utcnow() - timedelta(days=1)
    elif period == "7d":
        since = datetime.utcnow() - timedelta(days=7)
    elif period == "30d":
        since = datetime.utcnow() - timedelta(days=30)
    else:
        since = None

    # Build query for market creators
    query = (
        db.query(
            User.id,
            User.username,
            func.count(Market.id).label("markets_created"),
            func.sum(Market.volume_total).label("total_volume"),
            func.avg(Market.volume_total).label("avg_volume"),
        )
        .join(Market, User.id == Market.creator_id)
        .filter(User.is_active == True)
        .group_by(User.id, User.username)
    )

    if since:
        query = query.filter(Market.created_at >= since)

    # Order by total volume
    top_creators = query.order_by(desc("total_volume")).limit(limit).all()

    return {
        "period": period,
        "creators": [
            {
                "rank": i + 1,
                "user": {
                    "id": creator.id,
                    "username": creator.username,
                    "markets_created": creator.markets_created,
                    "total_volume": creator.total_volume or 0,
                    "avg_volume": creator.avg_volume or 0,
                },
            }
            for i, creator in enumerate(top_creators)
        ],
    }


@router.get("/win-rate")
def get_top_win_rate_traders(
    min_trades: int = Query(10, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get top traders by win rate (minimum trades required)"""

    # Get traders with minimum trades
    top_traders = (
        db.query(User)
        .filter(User.is_active == True, User.total_trades >= min_trades)
        .order_by(desc(User.win_rate))
        .limit(limit)
        .all()
    )

    return {
        "min_trades": min_trades,
        "traders": [
            {
                "rank": i + 1,
                "user": {
                    "id": trader.id,
                    "username": trader.username,
                    "win_rate": trader.win_rate,
                    "total_trades": trader.total_trades,
                    "successful_trades": trader.successful_trades,
                    "total_profit": trader.total_profit,
                },
            }
            for i, trader in enumerate(top_traders)
        ],
    }


@router.get("/reputation")
def get_top_reputation_traders(
    limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)
):
    """Get top traders by reputation score"""

    top_traders = (
        db.query(User)
        .filter(User.is_active == True, User.reputation_score > 0)
        .order_by(desc(User.reputation_score))
        .limit(limit)
        .all()
    )

    return {
        "traders": [
            {
                "rank": i + 1,
                "user": {
                    "id": trader.id,
                    "username": trader.username,
                    "reputation_score": trader.reputation_score,
                    "win_rate": trader.win_rate,
                    "total_trades": trader.total_trades,
                    "total_profit": trader.total_profit,
                },
            }
            for i, trader in enumerate(top_traders)
        ]
    }
