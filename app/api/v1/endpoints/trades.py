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
    
    # Validate trade amount
    if trade_data.amount < market.min_trade_amount:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Minimum trade amount is ${market.min_trade_amount}"
        )
    
    if trade_data.amount > market.max_trade_amount:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum trade amount is ${market.max_trade_amount}"
        )
    
    # Calculate trade details
    total_value = trade_data.amount * trade_data.price_per_share
    fee_amount = total_value * market.fee_rate
    net_value = total_value + fee_amount
    
    # Check user balance
    if current_user.available_balance < net_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient balance"
        )
    
    # Calculate price impact
    price_impact = market.price_impact * trade_data.amount
    
    # Create trade
    db_trade = Trade(
        trade_type=trade_data.trade_type,
        outcome=trade_data.outcome,
        amount=trade_data.amount,
        price_per_share=trade_data.price_per_share,
        total_value=total_value,
        fee_amount=fee_amount,
        price_impact=price_impact,
        market_id=trade_data.market_id,
        user_id=current_user.id,
        trade_hash=f"trade_{current_user.id}_{market.id}_{datetime.utcnow().timestamp()}"
    )
    
    db.add(db_trade)
    
    # Update market prices using AMM formula
    market.update_prices(trade_data.amount, trade_data.outcome.value, trade_data.trade_type.value)
    
    # Update market volume
    market.volume_total += total_value
    market.volume_24h += total_value
    
    # Update user balance and stats
    current_user.available_balance -= net_value
    current_user.total_trades += 1
    current_user.total_volume += total_value
    current_user.update_stats(total_value)
    
    # Update or create position
    from app.models.position import Position
    position = db.query(Position).filter(
        Position.user_id == current_user.id,
        Position.market_id == trade_data.market_id,
        Position.outcome == trade_data.outcome.value
    ).first()
    
    if not position:
        position = Position(
            user_id=current_user.id,
            market_id=trade_data.market_id,
            outcome=trade_data.outcome.value
        )
        db.add(position)
    
    # Update position
    shares_change = trade_data.amount if trade_data.trade_type == TradeType.BUY else -trade_data.amount
    position.update_position(shares_change, trade_data.price_per_share, total_value)
    
    # Update user portfolio value
    current_user.portfolio_value = sum(
        p.current_value for p in db.query(Position).filter(Position.user_id == current_user.id)
    )
    
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
