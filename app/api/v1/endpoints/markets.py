from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.market import Market, MarketStatus, MarketCategory
from app.models.trade import Trade
from app.schemas.market import MarketCreate, MarketUpdate, MarketResponse, MarketListResponse, MarketStats

router = APIRouter()

@router.post("/", response_model=MarketResponse)
def create_market(
    market_data: MarketCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Validate closing date is in the future
    if market_data.closes_at <= datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Market closing date must be in the future"
        )
    
    # Validate minimum liquidity
    if market_data.total_liquidity < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Minimum liquidity is $100"
        )
    
    db_market = Market(
        **market_data.dict(),
        creator_id=current_user.id,
        liquidity_pool_a=market_data.total_liquidity / 2,
        liquidity_pool_b=market_data.total_liquidity / 2
    )
    
    db.add(db_market)
    db.commit()
    db.refresh(db_market)
    
    return db_market

@router.get("/", response_model=MarketListResponse)
def get_markets(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[MarketCategory] = None,
    status: Optional[MarketStatus] = None,
    search: Optional[str] = None,
    sort_by: str = Query("created_at", regex="^(created_at|volume|price|closes_at)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db)
):
    query = db.query(Market)
    
    # Apply filters
    if category:
        query = query.filter(Market.category == category)
    
    if status:
        query = query.filter(Market.status == status)
    
    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            (Market.title.ilike(search_filter)) |
            (Market.description.ilike(search_filter)) |
            (Market.question.ilike(search_filter))
        )
    
    # Apply sorting
    if sort_by == "volume":
        sort_column = Market.volume_total
    elif sort_by == "price":
        sort_column = Market.current_price_a
    elif sort_by == "closes_at":
        sort_column = Market.closes_at
    else:
        sort_column = Market.created_at
    
    if sort_order == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(sort_column)
    
    total = query.count()
    markets = query.offset(skip).limit(limit).all()
    
    return MarketListResponse(
        markets=markets,
        total=total,
        page=skip // limit + 1,
        per_page=limit
    )

@router.get("/trending", response_model=List[MarketResponse])
def get_trending_markets(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get trending markets based on 24h volume"""
    yesterday = datetime.utcnow() - timedelta(days=1)
    
    # Get markets with highest 24h volume
    trending_markets = db.query(Market).filter(
        Market.status == MarketStatus.OPEN,
        Market.created_at >= yesterday
    ).order_by(desc(Market.volume_24h)).limit(limit).all()
    
    return trending_markets

@router.get("/stats", response_model=MarketStats)
def get_market_stats(db: Session = Depends(get_db)):
    """Get overall market statistics"""
    total_markets = db.query(Market).count()
    active_markets = db.query(Market).filter(Market.status == MarketStatus.OPEN).count()
    
    # Calculate total volumes
    total_volume_24h = db.query(func.sum(Market.volume_24h)).scalar() or 0
    total_volume_all_time = db.query(func.sum(Market.volume_total)).scalar() or 0
    
    # Get most active category
    most_active_category = db.query(
        Market.category,
        func.count(Market.id).label('count')
    ).group_by(Market.category).order_by(desc('count')).first()
    
    # Get trending markets
    trending_markets = get_trending_markets(limit=5, db=db)
    
    return MarketStats(
        total_markets=total_markets,
        active_markets=active_markets,
        total_volume_24h=total_volume_24h,
        total_volume_all_time=total_volume_all_time,
        most_active_category=most_active_category[0] if most_active_category else "other",
        trending_markets=trending_markets
    )

@router.get("/{market_id}", response_model=MarketResponse)
def get_market(market_id: int, db: Session = Depends(get_db)):
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )
    return market

@router.put("/{market_id}", response_model=MarketResponse)
def update_market(
    market_id: int,
    market_update: MarketUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )
    
    # Only creator can update market
    if market.creator_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only market creator can update market"
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
    db: Session = Depends(get_db)
):
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )
    
    # Only creator can resolve market
    if market.creator_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only market creator can resolve market"
        )
    
    # Validate outcome
    if outcome not in [market.outcome_a, market.outcome_b]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid outcome"
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
    db: Session = Depends(get_db)
):
    """Get price history for a market"""
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )
    
    # Get trades in the last N hours
    since = datetime.utcnow() - timedelta(hours=hours)
    trades = db.query(Trade).filter(
        Trade.market_id == market_id,
        Trade.created_at >= since
    ).order_by(Trade.created_at).all()
    
    # Create price history points
    price_history = []
    for trade in trades:
        price_history.append({
            "timestamp": trade.created_at,
            "price_a": trade.price_per_share if trade.outcome.value == "outcome_a" else 1 - trade.price_per_share,
            "price_b": trade.price_per_share if trade.outcome.value == "outcome_b" else 1 - trade.price_per_share,
            "volume": trade.total_value
        })
    
    return price_history
