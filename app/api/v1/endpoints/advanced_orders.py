from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.user import User
from app.services.advanced_orders import get_advanced_order_manager

router = APIRouter()


@router.post("/stop-loss")
def create_stop_loss_order(
    market_id: int,
    outcome: str,
    shares: float,
    stop_price: float,
    order_type: str = "market",
    current_user: User = Depends(get_current_user),
):
    """Create a stop-loss order to limit losses"""
    if shares <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Shares must be greater than 0",
        )

    if stop_price <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stop price must be greater than 0",
        )

    if outcome not in ["outcome_a", "outcome_b"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Outcome must be 'outcome_a' or 'outcome_b'",
        )

    order_manager = get_advanced_order_manager()
    result = order_manager.create_stop_loss_order(
        current_user.id, market_id, outcome, shares, stop_price, order_type
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
        )

    return result


@router.post("/take-profit")
def create_take_profit_order(
    market_id: int,
    outcome: str,
    shares: float,
    take_profit_price: float,
    order_type: str = "market",
    current_user: User = Depends(get_current_user),
):
    """Create a take-profit order to secure gains"""
    if shares <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Shares must be greater than 0",
        )

    if take_profit_price <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Take profit price must be greater than 0",
        )

    if outcome not in ["outcome_a", "outcome_b"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Outcome must be 'outcome_a' or 'outcome_b'",
        )

    order_manager = get_advanced_order_manager()
    result = order_manager.create_take_profit_order(
        current_user.id, market_id, outcome, shares, take_profit_price, order_type
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
        )

    return result


@router.post("/trailing-stop")
def create_trailing_stop_order(
    market_id: int,
    outcome: str,
    shares: float,
    trailing_percentage: float,
    current_user: User = Depends(get_current_user),
):
    """Create a trailing stop order that follows price movements"""
    if shares <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Shares must be greater than 0",
        )

    if trailing_percentage <= 0 or trailing_percentage > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trailing percentage must be between 0 and 50",
        )

    if outcome not in ["outcome_a", "outcome_b"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Outcome must be 'outcome_a' or 'outcome_b'",
        )

    order_manager = get_advanced_order_manager()
    result = order_manager.create_trailing_stop_order(
        current_user.id, market_id, outcome, shares, trailing_percentage
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
        )

    return result


@router.post("/conditional")
def create_conditional_order(
    market_id: int,
    outcome: str,
    shares: float,
    limit_price: float,
    condition_market_id: int,
    condition_outcome: str,
    condition_price: float,
    condition_type: str,
    current_user: User = Depends(get_current_user),
):
    """Create a conditional order that triggers based on another market's price"""
    if shares <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Shares must be greater than 0",
        )

    if limit_price <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit price must be greater than 0",
        )

    if condition_price <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Condition price must be greater than 0",
        )

    if outcome not in ["outcome_a", "outcome_b"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Outcome must be 'outcome_a' or 'outcome_b'",
        )

    if condition_outcome not in ["outcome_a", "outcome_b"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Condition outcome must be 'outcome_a' or 'outcome_b'",
        )

    if condition_type not in ["above", "below"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Condition type must be 'above' or 'below'",
        )

    order_manager = get_advanced_order_manager()
    result = order_manager.create_conditional_order(
        current_user.id,
        market_id,
        outcome,
        shares,
        limit_price,
        condition_market_id,
        condition_outcome,
        condition_price,
        condition_type,
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
        )

    return result


@router.post("/bracket")
def create_bracket_order(
    market_id: int,
    outcome: str,
    shares: float,
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: float,
    current_user: User = Depends(get_current_user),
):
    """Create a bracket order with stop-loss and take-profit"""
    if shares <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Shares must be greater than 0",
        )

    if entry_price <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Entry price must be greater than 0",
        )

    if stop_loss_price <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stop loss price must be greater than 0",
        )

    if take_profit_price <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Take profit price must be greater than 0",
        )

    if outcome not in ["outcome_a", "outcome_b"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Outcome must be 'outcome_a' or 'outcome_b'",
        )

    # Validate price relationships
    if stop_loss_price >= entry_price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stop loss price must be below entry price",
        )

    if take_profit_price <= entry_price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Take profit price must be above entry price",
        )

    order_manager = get_advanced_order_manager()
    result = order_manager.create_bracket_order(
        current_user.id,
        market_id,
        outcome,
        shares,
        entry_price,
        stop_loss_price,
        take_profit_price,
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"]
        )

    return result


@router.get("/my-orders")
def get_my_advanced_orders(current_user: User = Depends(get_current_user)):
    """Get user's advanced orders"""
    order_manager = get_advanced_order_manager()
    orders = order_manager.get_user_advanced_orders(current_user.id)

    return orders


@router.get("/order-types")
def get_available_order_types():
    """Get list of available advanced order types"""
    return {
        "order_types": [
            {
                "type": "stop_loss",
                "name": "Stop Loss",
                "description": "Automatically sell when price falls below a certain level",
                "parameters": [
                    "market_id",
                    "outcome",
                    "shares",
                    "stop_price",
                    "order_type (optional)",
                ],
            },
            {
                "type": "take_profit",
                "name": "Take Profit",
                "description": "Automatically sell when price rises above a certain level",
                "parameters": [
                    "market_id",
                    "outcome",
                    "shares",
                    "take_profit_price",
                    "order_type (optional)",
                ],
            },
            {
                "type": "trailing_stop",
                "name": "Trailing Stop",
                "description": "Stop loss that follows price movements to protect gains",
                "parameters": ["market_id", "outcome", "shares", "trailing_percentage"],
            },
            {
                "type": "conditional",
                "name": "Conditional Order",
                "description": "Order that triggers based on another market's price",
                "parameters": [
                    "market_id",
                    "outcome",
                    "shares",
                    "limit_price",
                    "condition_market_id",
                    "condition_outcome",
                    "condition_price",
                    "condition_type",
                ],
            },
            {
                "type": "bracket",
                "name": "Bracket Order",
                "description": "Entry order with automatic stop-loss and take-profit",
                "parameters": [
                    "market_id",
                    "outcome",
                    "shares",
                    "entry_price",
                    "stop_loss_price",
                    "take_profit_price",
                ],
            },
        ]
    }


@router.get("/risk-management")
def get_risk_management_tips():
    """Get risk management tips for advanced orders"""
    return {
        "tips": [
            {
                "title": "Set Realistic Stop Losses",
                "description": "Don't set stop losses too tight. Allow for normal price volatility.",
                "recommendation": "Use 5-15% below entry price for most markets",
            },
            {
                "title": "Use Trailing Stops for Trending Markets",
                "description": "Trailing stops help protect profits in strong trending markets.",
                "recommendation": "Start with 10-20% trailing percentage",
            },
            {
                "title": "Don't Over-Leverage",
                "description": "Don't risk too much of your portfolio on a single trade.",
                "recommendation": "Risk no more than 1-2% of portfolio per trade",
            },
            {
                "title": "Test Orders on Small Positions",
                "description": "Test new order types with small positions first.",
                "recommendation": "Start with 10-20% of your normal position size",
            },
            {
                "title": "Monitor Order Performance",
                "description": "Regularly review how your advanced orders are performing.",
                "recommendation": "Track win rate and average profit/loss",
            },
        ],
        "risk_warnings": [
            "Advanced orders may not execute at expected prices during high volatility",
            "Market gaps can cause stop losses to execute at worse prices than expected",
            "Conditional orders depend on other markets and may not trigger as expected",
            "Always monitor your positions and be prepared to adjust orders manually",
        ],
    }


@router.get("/order-statistics")
def get_order_statistics(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """Get statistics about user's advanced orders"""
    from app.models.order import Order, OrderStatus, OrderType

    # Get all advanced orders for user
    advanced_orders = (
        db.query(Order)
        .filter(Order.user_id == current_user.id, Order.order_type == OrderType.STOP)
        .all()
    )

    total_orders = len(advanced_orders)
    active_orders = len([o for o in advanced_orders if o.status == OrderStatus.PENDING])
    filled_orders = len([o for o in advanced_orders if o.status == OrderStatus.FILLED])
    cancelled_orders = len(
        [o for o in advanced_orders if o.status == OrderStatus.CANCELLED]
    )

    # Calculate average execution time for filled orders
    execution_times = []
    for order in advanced_orders:
        if order.status == OrderStatus.FILLED and order.updated_at and order.created_at:
            execution_time = (
                order.updated_at - order.created_at
            ).total_seconds() / 3600  # hours
            execution_times.append(execution_time)

    avg_execution_time = (
        sum(execution_times) / len(execution_times) if execution_times else 0
    )

    # Get order types breakdown
    order_types = {}
    for order in advanced_orders:
        order_type = (
            order.metadata.get("order_type", "unknown") if order.metadata else "unknown"
        )
        order_types[order_type] = order_types.get(order_type, 0) + 1

    return {
        "total_orders": total_orders,
        "active_orders": active_orders,
        "filled_orders": filled_orders,
        "cancelled_orders": cancelled_orders,
        "fill_rate": (filled_orders / total_orders * 100) if total_orders > 0 else 0,
        "average_execution_time_hours": avg_execution_time,
        "order_types_breakdown": order_types,
        "success_rate": (
            (filled_orders / (filled_orders + cancelled_orders) * 100)
            if (filled_orders + cancelled_orders) > 0
            else 0
        ),
    }
