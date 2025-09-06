from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.models.position import Position
from app.models.market import Market
from app.schemas.position import (
    PositionResponse,
    PositionListResponse,
    PortfolioSummary,
)

router = APIRouter()


@router.get("/", response_model=PositionListResponse)
def get_positions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    active_only: bool = Query(True),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's positions"""
    query = db.query(Position).filter(Position.user_id == current_user.id)

    if active_only:
        query = query.filter(Position.is_active == True)

    total = query.count()
    positions = (
        query.order_by(desc(Position.last_updated)).offset(skip).limit(limit).all()
    )

    # Update current prices and values
    for position in positions:
        market = db.query(Market).filter(Market.id == position.market_id).first()
        if market:
            if position.outcome == "outcome_a":
                position.current_price = market.current_price_a
            else:
                position.current_price = market.current_price_b

            position.current_value = position.shares_owned * position.current_price
            position.unrealized_pnl = position.current_value - position.total_invested
            position.total_pnl = position.realized_pnl + position.unrealized_pnl

    return PositionListResponse(
        positions=positions, total=total, page=skip // limit + 1, per_page=limit
    )


@router.get("/portfolio", response_model=PortfolioSummary)
def get_portfolio_summary(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get user's portfolio summary"""
    positions = (
        db.query(Position)
        .filter(Position.user_id == current_user.id, Position.is_active == True)
        .all()
    )

    total_positions = len(positions)
    active_positions = sum(1 for p in positions if p.shares_owned > 0)

    total_portfolio_value = 0
    total_unrealized_pnl = 0
    total_realized_pnl = 0
    total_pnl = 0

    # Update current values and calculate totals
    for position in positions:
        market = db.query(Market).filter(Market.id == position.market_id).first()
        if market:
            if position.outcome == "outcome_a":
                position.current_price = market.current_price_a
            else:
                position.current_price = market.current_price_b

            position.current_value = position.shares_owned * position.current_price
            position.unrealized_pnl = position.current_value - position.total_invested
            position.total_pnl = position.realized_pnl + position.unrealized_pnl

            total_portfolio_value += position.current_value
            total_unrealized_pnl += position.unrealized_pnl
            total_realized_pnl += position.realized_pnl
            total_pnl += position.total_pnl

    # Calculate portfolio return percentage
    total_invested = sum(p.total_invested for p in positions)
    portfolio_return_percentage = (
        (total_pnl / total_invested * 100) if total_invested > 0 else 0
    )

    return PortfolioSummary(
        total_positions=total_positions,
        active_positions=active_positions,
        total_portfolio_value=total_portfolio_value,
        total_unrealized_pnl=total_unrealized_pnl,
        total_realized_pnl=total_realized_pnl,
        total_pnl=total_pnl,
        portfolio_return_percentage=portfolio_return_percentage,
        positions=positions,
    )


@router.get("/{position_id}", response_model=PositionResponse)
def get_position(
    position_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get specific position details"""
    position = (
        db.query(Position)
        .filter(Position.id == position_id, Position.user_id == current_user.id)
        .first()
    )

    if not position:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Position not found"
        )

    # Update current values
    market = db.query(Market).filter(Market.id == position.market_id).first()
    if market:
        if position.outcome == "outcome_a":
            position.current_price = market.current_price_a
        else:
            position.current_price = market.current_price_b

        position.current_value = position.shares_owned * position.current_price
        position.unrealized_pnl = position.current_value - position.total_invested
        position.total_pnl = position.realized_pnl + position.unrealized_pnl

    return position


@router.post("/{position_id}/close")
def close_position(
    position_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Close a position (sell all shares)"""
    position = (
        db.query(Position)
        .filter(
            Position.id == position_id,
            Position.user_id == current_user.id,
            Position.is_active == True,
        )
        .first()
    )

    if not position:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Position not found or already closed",
        )

    if position.shares_owned <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No shares to close"
        )

    # Get current market price
    market = db.query(Market).filter(Market.id == position.market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    close_price = (
        market.current_price_a
        if position.outcome == "outcome_a"
        else market.current_price_b
    )

    # Close the position
    position.close_position(close_price)

    # Update user balance
    close_value = position.shares_owned * close_price
    current_user.available_balance += close_value
    current_user.portfolio_value -= position.current_value

    db.commit()

    return {"message": "Position closed successfully", "close_value": close_value}
