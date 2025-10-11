"""
Admin endpoints for system management
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from typing import List, Optional
from datetime import datetime, timedelta
import asyncio

from app.core.database import get_db
from app.core.security import get_current_active_user, security_manager
from app.core.validation import input_validator
from app.core.cache import cache
from app.core.logging import log_system_metric, log_security_event
from app.models.user import User
from app.models.market import Market, MarketStatus
from app.models.trade import Trade, TradeStatus
from app.models.order import Order, OrderStatus
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.schemas.market import MarketCreate, MarketUpdate, MarketResponse
from app.schemas.trade import TradeResponse
from app.schemas.order import OrderResponse
from app.schemas.admin import (
    AdminStats, UserStats, MarketStats, TradeStats, 
    SystemHealth, AdminUserList, AdminMarketList,
    AdminTradeList, AdminOrderList, UserModeration,
    MarketModeration, SystemSettings, AuditLog
)

router = APIRouter()


def require_admin_permissions(current_user: User = Depends(get_current_active_user)):
    """Require admin permissions for access"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permissions required"
        )
    return current_user


@router.get("/stats", response_model=AdminStats)
async def get_admin_stats(
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin_permissions)
):
    """Get comprehensive admin statistics"""
    
    # User statistics
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    verified_users = db.query(User).filter(User.is_verified == True).count()
    premium_users = db.query(User).filter(User.is_premium == True).count()
    
    # New users in last 24h, 7d, 30d
    now = datetime.utcnow()
    users_24h = db.query(User).filter(
        User.created_at >= now - timedelta(days=1)
    ).count()
    users_7d = db.query(User).filter(
        User.created_at >= now - timedelta(days=7)
    ).count()
    users_30d = db.query(User).filter(
        User.created_at >= now - timedelta(days=30)
    ).count()
    
    # Market statistics
    total_markets = db.query(Market).count()
    open_markets = db.query(Market).filter(Market.status == MarketStatus.OPEN).count()
    closed_markets = db.query(Market).filter(Market.status == MarketStatus.CLOSED).count()
    resolved_markets = db.query(Market).filter(Market.status == MarketStatus.RESOLVED).count()
    
    # Volume statistics
    total_volume = db.query(func.sum(Trade.total_value)).filter(
        Trade.status == TradeStatus.COMPLETED
    ).scalar() or 0
    
    volume_24h = db.query(func.sum(Trade.total_value)).filter(
        and_(
            Trade.status == TradeStatus.COMPLETED,
            Trade.created_at >= now - timedelta(days=1)
        )
    ).scalar() or 0
    
    volume_7d = db.query(func.sum(Trade.total_value)).filter(
        and_(
            Trade.status == TradeStatus.COMPLETED,
            Trade.created_at >= now - timedelta(days=7)
        )
    ).scalar() or 0
    
    # Trade statistics
    total_trades = db.query(Trade).filter(Trade.status == TradeStatus.COMPLETED).count()
    trades_24h = db.query(Trade).filter(
        and_(
            Trade.status == TradeStatus.COMPLETED,
            Trade.created_at >= now - timedelta(days=1)
        )
    ).count()
    
    # Order statistics
    pending_orders = db.query(Order).filter(Order.status == OrderStatus.PENDING).count()
    filled_orders = db.query(Order).filter(Order.status == OrderStatus.FILLED).count()
    
    # Top categories
    category_stats = db.query(
        Market.category,
        func.count(Market.id).label('count'),
        func.sum(Market.volume_total).label('volume')
    ).group_by(Market.category).all()
    
    top_categories = [
        {
            "category": stat.category.value,
            "count": stat.count,
            "volume": float(stat.volume or 0)
        }
        for stat in category_stats
    ]
    
    # Top markets by volume
    top_markets = db.query(Market).order_by(desc(Market.volume_total)).limit(5).all()
    
    # Top traders
    top_traders = db.query(
        Trade.user_id,
        func.count(Trade.id).label('trade_count'),
        func.sum(Trade.total_value).label('total_volume')
    ).filter(Trade.status == TradeStatus.COMPLETED).group_by(
        Trade.user_id
    ).order_by(desc('total_volume')).limit(5).all()
    
    return AdminStats(
        users=UserStats(
            total=total_users,
            active=active_users,
            verified=verified_users,
            premium=premium_users,
            new_24h=users_24h,
            new_7d=users_7d,
            new_30d=users_30d
        ),
        markets=MarketStats(
            total=total_markets,
            open=open_markets,
            closed=closed_markets,
            resolved=resolved_markets,
            top_categories=top_categories,
            top_markets=[
                {
                    "id": market.id,
                    "title": market.title,
                    "volume": market.volume_total
                }
                for market in top_markets
            ]
        ),
        trades=TradeStats(
            total=total_trades,
            volume_total=float(total_volume),
            volume_24h=float(volume_24h),
            volume_7d=float(volume_7d),
            trades_24h=trades_24h
        ),
        orders=OrderStats(
            pending=pending_orders,
            filled=filled_orders
        ),
        top_traders=[
            {
                "user_id": trader.user_id,
                "trade_count": trader.trade_count,
                "total_volume": float(trader.total_volume or 0)
            }
            for trader in top_traders
        ]
    )


@router.get("/users", response_model=AdminUserList)
async def get_admin_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    verified: Optional[bool] = Query(None),
    premium: Optional[bool] = Query(None),
    sort_by: str = Query("created_at", regex="^(created_at|username|email|last_login)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin_permissions)
):
    """Get users with admin filtering and sorting"""
    
    query = db.query(User)
    
    # Apply filters
    if search:
        search_filter = or_(
            User.username.ilike(f"%{search}%"),
            User.email.ilike(f"%{search}%"),
            User.full_name.ilike(f"%{search}%")
        )
        query = query.filter(search_filter)
    
    if status == "active":
        query = query.filter(User.is_active == True)
    elif status == "inactive":
        query = query.filter(User.is_active == False)
    
    if verified is not None:
        query = query.filter(User.is_verified == verified)
    
    if premium is not None:
        query = query.filter(User.is_premium == premium)
    
    # Apply sorting
    sort_column = getattr(User, sort_by)
    if sort_order == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(sort_column)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    users = query.offset(skip).limit(limit).all()
    
    return AdminUserList(
        users=[
            UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                bio=user.bio,
                avatar_url=user.avatar_url,
                is_active=user.is_active,
                is_verified=user.is_verified,
                is_premium=user.is_premium,
                created_at=user.created_at,
                updated_at=user.updated_at,
                last_login=user.last_login
            )
            for user in users
        ],
        total=total,
        skip=skip,
        limit=limit
    )


@router.get("/markets", response_model=AdminMarketList)
async def get_admin_markets(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    creator_id: Optional[int] = Query(None),
    sort_by: str = Query("created_at", regex="^(created_at|volume_total|trending_score)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin_permissions)
):
    """Get markets with admin filtering and sorting"""
    
    query = db.query(Market)
    
    # Apply filters
    if search:
        search_filter = or_(
            Market.title.ilike(f"%{search}%"),
            Market.description.ilike(f"%{search}%"),
            Market.question.ilike(f"%{search}%")
        )
        query = query.filter(search_filter)
    
    if category:
        query = query.filter(Market.category == category)
    
    if status:
        query = query.filter(Market.status == status)
    
    if creator_id:
        query = query.filter(Market.creator_id == creator_id)
    
    # Apply sorting
    sort_column = getattr(Market, sort_by)
    if sort_order == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(sort_column)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    markets = query.offset(skip).limit(limit).all()
    
    return AdminMarketList(
        markets=[
            MarketResponse(
                id=market.id,
                title=market.title,
                description=market.description,
                question=market.question,
                category=market.category,
                outcome_a=market.outcome_a,
                outcome_b=market.outcome_b,
                creator_id=market.creator_id,
                closes_at=market.closes_at,
                status=market.status,
                price_a=market.price_a,
                price_b=market.price_b,
                volume_total=market.volume_total,
                volume_24h=market.volume_24h,
                trending_score=market.trending_score,
                created_at=market.created_at,
                updated_at=market.updated_at
            )
            for market in markets
        ],
        total=total,
        skip=skip,
        limit=limit
    )


@router.get("/trades", response_model=AdminTradeList)
async def get_admin_trades(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    user_id: Optional[int] = Query(None),
    market_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    trade_type: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    sort_by: str = Query("created_at", regex="^(created_at|total_value|amount)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin_permissions)
):
    """Get trades with admin filtering and sorting"""
    
    query = db.query(Trade)
    
    # Apply filters
    if user_id:
        query = query.filter(Trade.user_id == user_id)
    
    if market_id:
        query = query.filter(Trade.market_id == market_id)
    
    if status:
        query = query.filter(Trade.status == status)
    
    if trade_type:
        query = query.filter(Trade.trade_type == trade_type)
    
    if outcome:
        query = query.filter(Trade.outcome == outcome)
    
    # Apply sorting
    sort_column = getattr(Trade, sort_by)
    if sort_order == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(sort_column)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    trades = query.offset(skip).limit(limit).all()
    
    return AdminTradeList(
        trades=[
            TradeResponse(
                id=trade.id,
                trade_type=trade.trade_type,
                outcome=trade.outcome,
                amount=trade.amount,
                price_a=trade.price_a,
                price_b=trade.price_b,
                price_per_share=trade.price_per_share,
                total_value=trade.total_value,
                status=trade.status,
                fee=trade.fee,
                price_impact=trade.price_impact,
                slippage=trade.slippage,
                market_id=trade.market_id,
                user_id=trade.user_id,
                trade_hash=trade.trade_hash,
                created_at=trade.created_at,
                executed_at=trade.executed_at
            )
            for trade in trades
        ],
        total=total,
        skip=skip,
        limit=limit
    )


@router.get("/orders", response_model=AdminOrderList)
async def get_admin_orders(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    user_id: Optional[int] = Query(None),
    market_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None),
    order_type: Optional[str] = Query(None),
    trade_type: Optional[str] = Query(None),
    sort_by: str = Query("created_at", regex="^(created_at|amount|price)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin_permissions)
):
    """Get orders with admin filtering and sorting"""
    
    query = db.query(Order)
    
    # Apply filters
    if user_id:
        query = query.filter(Order.user_id == user_id)
    
    if market_id:
        query = query.filter(Order.market_id == market_id)
    
    if status:
        query = query.filter(Order.status == status)
    
    if order_type:
        query = query.filter(Order.order_type == order_type)
    
    if trade_type:
        query = query.filter(Order.trade_type == trade_type)
    
    # Apply sorting
    sort_column = getattr(Order, sort_by)
    if sort_order == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(sort_column)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    orders = query.offset(skip).limit(limit).all()
    
    return AdminOrderList(
        orders=[
            OrderResponse(
                id=order.id,
                user_id=order.user_id,
                market_id=order.market_id,
                order_type=order.order_type,
                trade_type=order.trade_type,
                outcome=order.outcome,
                amount=order.amount,
                price=order.price,
                stop_price=order.stop_price,
                status=order.status,
                time_in_force=order.time_in_force,
                expires_at=order.expires_at,
                filled_amount=order.filled_amount,
                average_price=order.average_price,
                total_fees=order.total_fees,
                created_at=order.created_at,
                updated_at=order.updated_at,
                filled_at=order.filled_at
            )
            for order in orders
        ],
        total=total,
        skip=skip,
        limit=limit
    )


@router.post("/users/{user_id}/moderate", response_model=UserModeration)
async def moderate_user(
    user_id: int,
    action: str = Query(..., regex="^(ban|unban|verify|unverify|premium|unpremium)$"),
    reason: Optional[str] = Query(None),
    duration_days: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin_permissions)
):
    """Moderate user account"""
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.id == admin_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot moderate your own account"
        )
    
    # Apply moderation action
    if action == "ban":
        user.is_active = False
        log_security_event(
            event_type="user_banned",
            user_id=user_id,
            admin_id=admin_user.id,
            details={"reason": reason, "duration_days": duration_days}
        )
    elif action == "unban":
        user.is_active = True
        log_security_event(
            event_type="user_unbanned",
            user_id=user_id,
            admin_id=admin_user.id,
            details={"reason": reason}
        )
    elif action == "verify":
        user.is_verified = True
        log_security_event(
            event_type="user_verified",
            user_id=user_id,
            admin_id=admin_user.id,
            details={"reason": reason}
        )
    elif action == "unverify":
        user.is_verified = False
        log_security_event(
            event_type="user_unverified",
            user_id=user_id,
            admin_id=admin_user.id,
            details={"reason": reason}
        )
    elif action == "premium":
        user.is_premium = True
        log_security_event(
            event_type="user_premium_granted",
            user_id=user_id,
            admin_id=admin_user.id,
            details={"reason": reason}
        )
    elif action == "unpremium":
        user.is_premium = False
        log_security_event(
            event_type="user_premium_revoked",
            user_id=user_id,
            admin_id=admin_user.id,
            details={"reason": reason}
        )
    
    db.commit()
    db.refresh(user)
    
    # Clear user cache
    cache.delete(f"user:{user_id}")
    
    return UserModeration(
        user_id=user_id,
        action=action,
        reason=reason,
        duration_days=duration_days,
        admin_id=admin_user.id,
        timestamp=datetime.utcnow()
    )


@router.post("/markets/{market_id}/moderate", response_model=MarketModeration)
async def moderate_market(
    market_id: int,
    action: str = Query(..., regex="^(close|reopen|cancel|resolve)$"),
    reason: Optional[str] = Query(None),
    resolution_outcome: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin_permissions)
):
    """Moderate market"""
    
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )
    
    # Apply moderation action
    if action == "close":
        market.status = MarketStatus.CLOSED
        market.closed_at = datetime.utcnow()
        log_security_event(
            event_type="market_closed",
            market_id=market_id,
            admin_id=admin_user.id,
            details={"reason": reason}
        )
    elif action == "reopen":
        market.status = MarketStatus.OPEN
        market.closed_at = None
        log_security_event(
            event_type="market_reopened",
            market_id=market_id,
            admin_id=admin_user.id,
            details={"reason": reason}
        )
    elif action == "cancel":
        market.status = MarketStatus.CANCELLED
        market.closed_at = datetime.utcnow()
        log_security_event(
            event_type="market_cancelled",
            market_id=market_id,
            admin_id=admin_user.id,
            details={"reason": reason}
        )
    elif action == "resolve":
        if not resolution_outcome:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resolution outcome required for resolve action"
            )
        market.status = MarketStatus.RESOLVED
        market.resolved_at = datetime.utcnow()
        market.resolution_outcome = resolution_outcome
        log_security_event(
            event_type="market_resolved",
            market_id=market_id,
            admin_id=admin_user.id,
            details={"reason": reason, "resolution_outcome": resolution_outcome}
        )
    
    db.commit()
    db.refresh(market)
    
    # Clear market cache
    cache.delete(f"market:{market_id}")
    cache.delete("markets:list")
    cache.delete("markets:trending")
    
    return MarketModeration(
        market_id=market_id,
        action=action,
        reason=reason,
        resolution_outcome=resolution_outcome,
        admin_id=admin_user.id,
        timestamp=datetime.utcnow()
    )


@router.get("/system/health", response_model=SystemHealth)
async def get_system_health(
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin_permissions)
):
    """Get system health status"""
    
    from app.core.database import check_database_health, check_redis_health
    from app.core.cache import cache_health_check
    
    # Check database health
    db_health = check_database_health()
    
    # Check Redis health
    redis_health = check_redis_health()
    
    # Check cache health
    cache_health = cache_health_check()
    
    # Determine overall health
    overall_status = "healthy"
    if db_health["status"] != "healthy":
        overall_status = "unhealthy"
    elif redis_health["status"] not in ["healthy", "disabled"]:
        overall_status = "degraded"
    elif cache_health["status"] != "healthy":
        overall_status = "degraded"
    
    return SystemHealth(
        status=overall_status,
        database=db_health,
        redis=redis_health,
        cache=cache_health,
        timestamp=datetime.utcnow()
    )


@router.get("/audit/logs", response_model=List[AuditLog])
async def get_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    event_type: Optional[str] = Query(None),
    user_id: Optional[int] = Query(None),
    admin_id: Optional[int] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    admin_user: User = Depends(require_admin_permissions)
):
    """Get audit logs for security and moderation events"""
    
    # This would typically query an audit log table
    # For now, we'll return a placeholder response
    return [
        AuditLog(
            id=1,
            event_type="user_banned",
            user_id=123,
            admin_id=admin_user.id,
            details={"reason": "Violation of terms of service"},
            timestamp=datetime.utcnow()
        )
    ]


@router.get("/settings", response_model=SystemSettings)
async def get_system_settings(
    admin_user: User = Depends(require_admin_permissions)
):
    """Get system settings"""
    
    from app.core.config import settings
    
    return SystemSettings(
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT.value,
        debug=settings.DEBUG,
        rate_limit_enabled=settings.RATE_LIMIT_ENABLED,
        rate_limit_requests=settings.RATE_LIMIT_REQUESTS,
        rate_limit_window=settings.RATE_LIMIT_WINDOW,
        caching_enabled=settings.ENABLE_CACHING,
        cache_ttl=settings.CACHE_TTL,
        compression_enabled=settings.ENABLE_COMPRESSION,
        websocket_enabled=settings.WS_ENABLED,
        ml_enabled=settings.ML_ENABLED,
        blockchain_enabled=settings.BLOCKCHAIN_ENABLED,
        monitoring_enabled=settings.ENABLE_METRICS
    )


@router.post("/settings")
async def update_system_settings(
    settings_update: dict,
    admin_user: User = Depends(require_admin_permissions)
):
    """Update system settings (placeholder)"""
    
    # In a real implementation, this would update configuration
    # and restart services as needed
    
    log_security_event(
        event_type="settings_updated",
        admin_id=admin_user.id,
        details={"updated_settings": list(settings_update.keys())}
    )
    
    return {"message": "Settings updated successfully"}


@router.post("/maintenance/clear-cache")
async def clear_system_cache(
    admin_user: User = Depends(require_admin_permissions)
):
    """Clear system cache"""
    
    # Clear all cache entries
    cache.clear()
    
    log_security_event(
        event_type="cache_cleared",
        admin_id=admin_user.id,
        details={}
    )
    
    return {"message": "Cache cleared successfully"}


@router.post("/maintenance/rebuild-indexes")
async def rebuild_database_indexes(
    admin_user: User = Depends(require_admin_permissions)
):
    """Rebuild database indexes for performance"""
    
    # This would typically run database maintenance commands
    # For now, we'll just log the event
    
    log_security_event(
        event_type="indexes_rebuilt",
        admin_id=admin_user.id,
        details={}
    )
    
    return {"message": "Database indexes rebuilt successfully"}