from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.market import Market, MarketStatus
from app.models.trade import Trade, TradeType, TradeOutcome
from app.schemas.trade import TradeCreate, TradeResponse, TradeListResponse

router = APIRouter()

@router.post("/", response_model=TradeResponse)
def create_trade(
    trade_data: TradeCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get market
    market = db.query(Market).filter(Market.id == trade_data.market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Market not found"
        )
    
    # Check if market is active
    if not market.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Market is not active for trading"
        )
    
    # Calculate total value
    total_value = trade_data.amount * trade_data.price_per_share
    
    # Create trade
    db_trade = Trade(
        trade_type=trade_data.trade_type,
        outcome=trade_data.outcome,
        amount=trade_data.amount,
        price_per_share=trade_data.price_per_share,
        total_value=total_value,
        market_id=trade_data.market_id,
        user_id=current_user.id
    )
    
    db.add(db_trade)
    
    # Update market prices based on trade
    if trade_data.outcome == TradeOutcome.OUTCOME_A:
        if trade_data.trade_type == TradeType.BUY:
            market.current_price_a = min(1.0, market.current_price_a + 0.01)
            market.current_price_b = max(0.0, market.current_price_b - 0.01)
        else:  # SELL
            market.current_price_a = max(0.0, market.current_price_a - 0.01)
            market.current_price_b = min(1.0, market.current_price_b + 0.01)
    else:  # OUTCOME_B
        if trade_data.trade_type == TradeType.BUY:
            market.current_price_b = min(1.0, market.current_price_b + 0.01)
            market.current_price_a = max(0.0, market.current_price_a - 0.01)
        else:  # SELL
            market.current_price_b = max(0.0, market.current_price_b - 0.01)
            market.current_price_a = min(1.0, market.current_price_a + 0.01)
    
    # Update user stats
    current_user.total_trades += 1
    
    db.commit()
    db.refresh(db_trade)
    
    return db_trade

@router.get("/", response_model=TradeListResponse)
def get_trades(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    market_id: int = None,
    user_id: int = None,
    db: Session = Depends(get_db)
):
    query = db.query(Trade)
    
    if market_id:
        query = query.filter(Trade.market_id == market_id)
    
    if user_id:
        query = query.filter(Trade.user_id == user_id)
    
    total = query.count()
    trades = query.offset(skip).limit(limit).all()
    
    return TradeListResponse(
        trades=trades,
        total=total,
        page=skip // limit + 1,
        per_page=limit
    )

@router.get("/{trade_id}", response_model=TradeResponse)
def get_trade(trade_id: int, db: Session = Depends(get_db)):
    trade = db.query(Trade).filter(Trade.id == trade_id).first()
    if not trade:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trade not found"
        )
    return trade

@router.get("/user/me", response_model=TradeListResponse)
def get_my_trades(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Trade).filter(Trade.user_id == current_user.id)
    total = query.count()
    trades = query.offset(skip).limit(limit).all()
    
    return TradeListResponse(
        trades=trades,
        total=total,
        page=skip // limit + 1,
        per_page=limit
    )
