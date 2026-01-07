import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from datetime import datetime, timedelta

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.decorators import handle_errors, log_execution_time
from app.models.user import User
from app.models.market import Market, MarketStatus
from app.models.order import Order, OrderFill, OrderType, OrderStatus, OrderSide
from app.models.trade import Trade, TradeType
from app.schemas.order import (
    OrderCreate,
    OrderUpdate,
    OrderResponse,
    OrderListResponse,
    OrderBookResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=OrderResponse)
@handle_errors(default_message="Failed to create order")
@log_execution_time
async def create_order(
    order_data: OrderCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OrderResponse:
    """
    Create a new trading order.
    
    Args:
        order_data: Order creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created order response
        
    Raises:
        HTTPException: If market not found, inactive, or validation fails
    """

    # Check if market exists and is active
    market = db.query(Market).filter(Market.id == order_data.market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    if not market.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Market is not active for trading",
        )

    # Validate order data
    if order_data.order_type == OrderType.LIMIT and not order_data.limit_price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit price is required for limit orders",
        )

    if order_data.order_type == OrderType.STOP and not order_data.stop_price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stop price is required for stop orders",
        )

    # Check user balance for buy orders
    if order_data.side == OrderSide.BUY:
        required_balance = order_data.original_amount * (
            order_data.limit_price or market.current_price_a
        )
        if current_user.available_balance < required_balance:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Insufficient balance"
            )

    # Create order
    db_order = Order(
        user_id=current_user.id,
        market_id=order_data.market_id,
        order_type=order_data.order_type,
        side=order_data.side,
        outcome=order_data.outcome,
        original_amount=order_data.original_amount,
        remaining_amount=order_data.original_amount,
        limit_price=order_data.limit_price,
        stop_price=order_data.stop_price,
        expires_at=order_data.expires_at,
        order_hash=f"order_{current_user.id}_{order_data.market_id}_{datetime.utcnow().timestamp()}",
    )

    db.add(db_order)
    db.commit()
    db.refresh(db_order)

    # Try to match the order immediately
    if order_data.order_type == OrderType.MARKET:
        _match_market_order(db_order, db)

    return db_order


@router.get("/", response_model=OrderListResponse)
def get_orders(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[OrderStatus] = None,
    market_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's orders"""
    query = db.query(Order).filter(Order.user_id == current_user.id)

    if status:
        query = query.filter(Order.status == status)

    if market_id:
        query = query.filter(Order.market_id == market_id)

    total = query.count()
    orders = query.order_by(desc(Order.created_at)).offset(skip).limit(limit).all()

    return OrderListResponse(
        orders=orders, total=total, page=skip // limit + 1, per_page=limit
    )


@router.get("/{order_id}", response_model=OrderResponse)
def get_order(
    order_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get specific order"""
    order = (
        db.query(Order)
        .filter(Order.id == order_id, Order.user_id == current_user.id)
        .first()
    )

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Order not found"
        )

    return order


@router.put("/{order_id}", response_model=OrderResponse)
def update_order(
    order_id: int,
    order_update: OrderUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update an order"""
    order = (
        db.query(Order)
        .filter(Order.id == order_id, Order.user_id == current_user.id)
        .first()
    )

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Order not found"
        )

    if not order.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot update inactive order",
        )

    # Update order fields
    for field, value in order_update.dict(exclude_unset=True).items():
        setattr(order, field, value)

    db.commit()
    db.refresh(order)

    return order


@router.post("/{order_id}/cancel")
def cancel_order(
    order_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cancel an order"""
    order = (
        db.query(Order)
        .filter(Order.id == order_id, Order.user_id == current_user.id)
        .first()
    )

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Order not found"
        )

    if not order.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Order is not active"
        )

    order.cancel()
    db.commit()

    return {"message": "Order cancelled successfully"}


@router.get("/market/{market_id}/orderbook", response_model=OrderBookResponse)
def get_order_book(
    market_id: int, levels: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)
):
    """Get order book for a market"""

    # Check if market exists
    market = db.query(Market).filter(Market.id == market_id).first()
    if not market:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Market not found"
        )

    # Get active buy orders (highest price first)
    buy_orders = (
        db.query(Order)
        .filter(
            Order.market_id == market_id,
            Order.side == OrderSide.BUY,
            Order.status.in_([OrderStatus.PENDING, OrderStatus.PARTIAL]),
        )
        .order_by(desc(Order.limit_price))
        .limit(levels)
        .all()
    )

    # Get active sell orders (lowest price first)
    sell_orders = (
        db.query(Order)
        .filter(
            Order.market_id == market_id,
            Order.side == OrderSide.SELL,
            Order.status.in_([OrderStatus.PENDING, OrderStatus.PARTIAL]),
        )
        .order_by(Order.limit_price)
        .limit(levels)
        .all()
    )

    # Aggregate orders by price level
    bids = _aggregate_orders_by_price(buy_orders)
    asks = _aggregate_orders_by_price(sell_orders)

    # Calculate spread
    best_bid = bids[0]["price"] if bids else None
    best_ask = asks[0]["price"] if asks else None
    spread = best_ask - best_bid if best_bid and best_ask else 0.0

    return OrderBookResponse(
        market_id=market_id,
        bids=bids,
        asks=asks,
        spread=spread,
        best_bid=best_bid,
        best_ask=best_ask,
    )


def _aggregate_orders_by_price(orders: List[Order]) -> List[dict]:
    """Aggregate orders by price level"""
    price_levels = {}

    for order in orders:
        price = order.limit_price
        if price not in price_levels:
            price_levels[price] = {"price": price, "amount": 0.0, "total": 0.0}

        price_levels[price]["amount"] += order.remaining_amount
        price_levels[price]["total"] += order.remaining_amount * price

    return list(price_levels.values())


def _match_market_order(order: Order, db: Session):
    """Match a market order against existing orders"""
    if order.side == OrderSide.BUY:
        # Match against sell orders
        matching_orders = (
            db.query(Order)
            .filter(
                Order.market_id == order.market_id,
                Order.side == OrderSide.SELL,
                Order.status.in_([OrderStatus.PENDING, OrderStatus.PARTIAL]),
                Order.outcome == order.outcome,
            )
            .order_by(Order.limit_price)
            .all()
        )
    else:
        # Match against buy orders
        matching_orders = (
            db.query(Order)
            .filter(
                Order.market_id == order.market_id,
                Order.side == OrderSide.BUY,
                Order.status.in_([OrderStatus.PENDING, OrderStatus.PARTIAL]),
                Order.outcome == order.outcome,
            )
            .order_by(desc(Order.limit_price))
            .all()
        )

    remaining_amount = order.remaining_amount

    for matching_order in matching_orders:
        if remaining_amount <= 0:
            break

        # Check if orders can match
        if order.side == OrderSide.BUY:
            if matching_order.limit_price > order.limit_price:
                continue
        else:
            if matching_order.limit_price < order.limit_price:
                continue

        # Calculate fill amount
        fill_amount = min(remaining_amount, matching_order.remaining_amount)
        fill_price = matching_order.limit_price

        # Create trade
        trade = Trade(
            trade_type=TradeType.BUY if order.side == OrderSide.BUY else TradeType.SELL,
            outcome=order.outcome,
            amount=fill_amount,
            price_per_share=fill_price,
            total_value=fill_amount * fill_price,
            market_id=order.market_id,
            user_id=order.user_id,
        )

        db.add(trade)

        # Update orders
        order.fill(fill_price, fill_amount)
        matching_order.fill(fill_price, fill_amount)

        remaining_amount -= fill_amount

    db.commit()
