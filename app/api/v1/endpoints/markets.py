import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, and_, or_
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.config import settings
from app.core.security import (
    get_current_user, 
    get_current_active_user,
    rate_limit,
    validate_request_data,
    log_security_event,
    get_client_ip
)
from app.core.cache import cache, cached
from app.core.decorators import handle_errors, log_execution_time
from app.core.logging import log_trading_event, log_system_metric
from app.models.user import User
from app.models.market import Market, MarketStatus, MarketCategory
from app.models.trade import Trade
from app.schemas.market import (
    MarketCreate,
    MarketUpdate,
    MarketResponse,
    MarketListResponse,
    MarketStats,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=MarketListResponse)
@cached(ttl=60)  # Cache for 1 minute
async def list_markets(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[MarketCategory] = None,
    status: Optional[MarketStatus] = None,
    trending: bool = Query(False),
    search: Optional[str] = Query(None, max_length=100),
    db: Session = Depends(get_db)
):
    """List markets with filtering, pagination, and caching"""
    query = db.query(Market)
    
    # Apply filters
    if category:
        query = query.filter(Market.category == category)
    
    if status:
        query = query.filter(Market.status == status)
    
    if trending:
        query = query.filter(Market.trending_score >= settings.MARKET_TRENDING_THRESHOLD)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                Market.title.ilike(search_term),
                Market.description.ilike(search_term),
                Market.question.ilike(search_term)
            )
        )
    
    # Get total count
    total = query.count()
    
    # Apply pagination and ordering
    markets = query.order_by(desc(Market.created_at)).offset(skip).limit(limit).all()
    
    return MarketListResponse(
        markets=markets,
        total=total,
        skip=skip,
        limit=limit
    )


@router.get("/trending", response_model=List[MarketResponse])
@cached(ttl=300)  # Cache for 5 minutes
async def get_trending_markets(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get trending markets based on activity and volume"""
    markets = (
        db.query(Market)
        .filter(Market.status == MarketStatus.OPEN)
        .filter(Market.trending_score >= settings.MARKET_TRENDING_THRESHOLD)
        .order_by(desc(Market.trending_score))
        .limit(limit)
        .all()
    )
    
    return markets


@router.get("/stats", response_model=MarketStats)
@cached(ttl=300)  # Cache for 5 minutes
async def get_market_stats(db: Session = Depends(get_db)):
    """Get comprehensive market statistics"""
    # Total markets
    total_markets = db.query(Market).count()
    
    # Active markets
    active_markets = db.query(Market).filter(Market.status == MarketStatus.OPEN).count()
    
    # Total volume
    total_volume = db.query(func.sum(Market.volume_total)).scalar() or 0
    
    # 24h volume
    yesterday = datetime.utcnow() - timedelta(days=1)
    volume_24h = db.query(func.sum(Market.volume_24h)).scalar() or 0
    
    # Top categories
    category_stats = (
        db.query(Market.category, func.count(Market.id), func.sum(Market.volume_total))
        .group_by(Market.category)
        .order_by(desc(func.count(Market.id)))
        .limit(5)
        .all()
    )
    
    return MarketStats(
        total_markets=total_markets,
        active_markets=active_markets,
        total_volume=total_volume,
        volume_24h=volume_24h,
        top_categories=[
            {"category": cat.value, "count": count, "volume": volume or 0}
            for cat, count, volume in category_stats
        ]
    )


@router.post("/", response_model=MarketResponse)
@rate_limit(requests=10, window=3600)  # 10 markets per hour
@validate_request_data()
@handle_errors(default_message="Failed to create market")
@log_execution_time
async def create_market(
    market_data: MarketCreate,
    current_user: User = Depends(get_current_active_user),
    request: Request = None,
    db: Session = Depends(get_db),
):
    """
    Create a new prediction market with comprehensive validation.
    
    Args:
        market_data: Market creation data
        current_user: Current authenticated user
        request: FastAPI request object
        db: Database session
        
    Returns:
        Created market response
        
    Raises:
        HTTPException: If validation fails or rate limit exceeded
    """
    try:
        # Validate closing date is in the future
        if market_data.closes_at <= datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Market closing date must be in the future",
            )

        # Validate minimum liquidity
        if market_data.total_liquidity < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Minimum liquidity is $100"
            )

        # Validate maximum liquidity
        if market_data.total_liquidity > 1000000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum liquidity is $1,000,000"
            )

        # Check if user can create markets (rate limiting)
        recent_markets = db.query(Market).filter(
            Market.creator_id == current_user.id,
            Market.created_at >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        
        if recent_markets >= 10:  # Max 10 markets per day
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded: Maximum 10 markets per day"
            )

        # Create market with proper validation
        db_market = Market(
            **market_data.dict(),
            creator_id=current_user.id,
            liquidity_pool_a=market_data.total_liquidity / 2,
            liquidity_pool_b=market_data.total_liquidity / 2,
        )

        # Generate trade hash for the market
        db_market.generate_trade_hash() if hasattr(db_market, 'generate_trade_hash') else None

        db.add(db_market)
        db.commit()
        db.refresh(db_market)

        # Log market creation
        logging.info(f"Market created: {db_market.id} by user {current_user.id}")

        return db_market

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logging.error(f"Error creating market: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create market"
        )


# Removed duplicate endpoints - using the enhanced versions above


@router.get("/{market_id}", response_model=MarketResponse)
def get_market(market_id: int, db: Session = Depends(get_db)):
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )
    return market


@router.put("/{market_id}", response_model=MarketResponse)
def update_market(
    market_id: int,
    market_update: MarketUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    # Only creator can update market
    if market.creator_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only market creator can update market",
        )

    # Update market fields
    for field, value in market_update.dict(exclude_unset=True).items():
        setattr(market, field, value)

    db.commit()
    db.refresh(market)

    return market


@router.post("/{market_id}/resolve")
def resolve_market(
    market_id: int,
    outcome: str,
    resolution_source: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    # Only creator can resolve market
    if market.creator_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only market creator can resolve market",
        )

    # Validate outcome
    if outcome not in [market.outcome_a, market.outcome_b]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid outcome"
        )

    market.status = MarketStatus.RESOLVED
    market.resolved_outcome = outcome
    market.resolution_source = resolution_source
    market.resolved_at = datetime.utcnow()

    db.commit()

    return {"message": "Market resolved successfully"}


@router.get("/{market_id}/price-history")
def get_price_history(
    market_id: int,
    hours: int = Query(24, ge=1, le=168),  # Max 7 days
    db: Session = Depends(get_db),
):
    """Get price history for a market"""
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    # Get trades in the last N hours
    since = datetime.utcnow() - timedelta(hours=hours)
    trades = (
        db.query(Trade)
        .filter(Trade.market_id == market_id, Trade.created_at >= since)
        .order_by(Trade.created_at)
        .all()
    )

    # Create price history points
    price_history = []
    for trade in trades:
        price_history.append(
            {
                "timestamp": trade.created_at,
                "price_a": (
                    trade.price_per_share
                    if trade.outcome.value == "outcome_a"
                    else 1 - trade.price_per_share
                ),
                "price_b": (
                    trade.price_per_share
                    if trade.outcome.value == "outcome_b"
                    else 1 - trade.price_per_share
                ),
                "volume": trade.total_value,
            }
        )

    return price_history
