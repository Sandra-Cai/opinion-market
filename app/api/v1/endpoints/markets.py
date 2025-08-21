from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.market import Market, MarketStatus, MarketCategory
from app.schemas.market import MarketCreate, MarketUpdate, MarketResponse, MarketListResponse

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
    
    db_market = Market(
        **market_data.dict(),
        creator_id=current_user.id
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
    db: Session = Depends(get_db)
):
    query = db.query(Market)
    
    if category:
        query = query.filter(Market.category == category)
    
    if status:
        query = query.filter(Market.status == status)
    
    total = query.count()
    markets = query.offset(skip).limit(limit).all()
    
    return MarketListResponse(
        markets=markets,
        total=total,
        page=skip // limit + 1,
        per_page=limit
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
    market.resolved_at = datetime.utcnow()
    
    db.commit()
    
    return {"message": "Market resolved successfully"}
