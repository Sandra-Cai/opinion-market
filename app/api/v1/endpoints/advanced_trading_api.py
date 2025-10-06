"""
Advanced Trading API Endpoints
REST API for the Advanced Trading Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.advanced_trading_engine import (
    advanced_trading_engine,
    TradingStrategy,
    OrderType,
    OrderSide,
    OrderStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class TradingOrderRequest(BaseModel):
    asset: str = Field(..., description="Asset symbol (e.g., BTC, ETH, AAPL)")
    side: str = Field(..., description="Order side: buy, sell, short, cover")
    order_type: str = Field(..., description="Order type: market, limit, stop, stop_limit")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, description="Limit price for limit orders")
    stop_price: Optional[float] = Field(None, description="Stop price for stop orders")
    strategy: str = Field("algorithmic", description="Trading strategy to use")

class MarketDataUpdate(BaseModel):
    asset: str = Field(..., description="Asset symbol")
    price: float = Field(..., gt=0, description="Current price")
    volume: float = Field(0, description="Trading volume")

class TradingSignalResponse(BaseModel):
    signal_id: str
    strategy: str
    asset: str
    side: str
    strength: float
    confidence: float
    price_target: float
    stop_loss: float
    take_profit: float
    risk_score: float
    created_at: str

class TradingPositionResponse(BaseModel):
    position_id: str
    asset: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    strategy: Optional[str]
    created_at: str

class TradingPerformanceResponse(BaseModel):
    performance_metrics: Dict[str, Any]
    current_positions: int
    active_orders: int
    daily_pnl: float
    trading_active: bool

@router.post("/submit-order", response_model=Dict[str, str])
async def submit_trading_order(order_request: TradingOrderRequest):
    """Submit a trading order"""
    try:
        # Validate order side
        try:
            side = OrderSide(order_request.side.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid order side: {order_request.side}")
        
        # Validate order type
        try:
            order_type = OrderType(order_request.order_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid order type: {order_request.order_type}")
        
        # Validate strategy
        try:
            strategy = TradingStrategy(order_request.strategy.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid strategy: {order_request.strategy}")
        
        # Submit order
        order_id = await advanced_trading_engine.submit_trading_order(
            asset=order_request.asset,
            side=side,
            order_type=order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            stop_price=order_request.stop_price,
            strategy=strategy
        )
        
        return {
            "order_id": order_id,
            "status": "submitted",
            "message": f"Trading order submitted successfully for {order_request.asset}"
        }
        
    except Exception as e:
        logger.error(f"Error submitting trading order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions", response_model=List[TradingPositionResponse])
async def get_trading_positions():
    """Get all trading positions"""
    try:
        positions = await advanced_trading_engine.get_trading_positions()
        return positions
        
    except Exception as e:
        logger.error(f"Error getting trading positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance", response_model=TradingPerformanceResponse)
async def get_trading_performance():
    """Get trading performance metrics"""
    try:
        performance = await advanced_trading_engine.get_trading_performance()
        return performance
        
    except Exception as e:
        logger.error(f"Error getting trading performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals", response_model=List[TradingSignalResponse])
async def get_trading_signals():
    """Get recent trading signals"""
    try:
        signals = await advanced_trading_engine.get_trading_signals()
        return signals
        
    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/market-data")
async def update_market_data(market_data: MarketDataUpdate):
    """Update market data for an asset"""
    try:
        await advanced_trading_engine.update_market_data(
            asset=market_data.asset,
            price=market_data.price,
            volume=market_data.volume
        )
        
        return {
            "status": "success",
            "message": f"Market data updated for {market_data.asset}"
        }
        
    except Exception as e:
        logger.error(f"Error updating market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies")
async def get_trading_strategies():
    """Get available trading strategies"""
    try:
        strategies = [
            {
                "name": strategy.value,
                "description": f"{strategy.value.replace('_', ' ').title()} trading strategy",
                "enabled": strategy in advanced_trading_engine.strategies
            }
            for strategy in TradingStrategy
        ]
        
        return {
            "strategies": strategies,
            "total_strategies": len(strategies),
            "active_strategies": len(advanced_trading_engine.strategies)
        }
        
    except Exception as e:
        logger.error(f"Error getting trading strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orders")
async def get_trading_orders():
    """Get all trading orders"""
    try:
        orders = []
        for order in advanced_trading_engine.orders.values():
            orders.append({
                "order_id": order.order_id,
                "strategy": order.strategy.value,
                "asset": order.asset,
                "side": order.side.value,
                "order_type": order.order_type.value,
                "quantity": order.quantity,
                "price": order.price,
                "stop_price": order.stop_price,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "average_price": order.average_price,
                "commission": order.commission,
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat()
            })
        
        return {
            "orders": orders,
            "total_orders": len(orders),
            "pending_orders": len([o for o in orders if o["status"] == "pending"]),
            "filled_orders": len([o for o in orders if o["status"] == "filled"])
        }
        
    except Exception as e:
        logger.error(f"Error getting trading orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-trading")
async def start_trading():
    """Start the trading engine"""
    try:
        if not advanced_trading_engine.is_running:
            await advanced_trading_engine.start_trading_engine()
        
        advanced_trading_engine.trading_active = True
        
        return {
            "status": "success",
            "message": "Trading engine started",
            "trading_active": advanced_trading_engine.trading_active
        }
        
    except Exception as e:
        logger.error(f"Error starting trading engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop-trading")
async def stop_trading():
    """Stop the trading engine"""
    try:
        advanced_trading_engine.trading_active = False
        
        return {
            "status": "success",
            "message": "Trading engine stopped",
            "trading_active": advanced_trading_engine.trading_active
        }
        
    except Exception as e:
        logger.error(f"Error stopping trading engine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emergency-stop")
async def emergency_stop():
    """Emergency stop all trading"""
    try:
        await advanced_trading_engine._emergency_stop_trading()
        
        return {
            "status": "success",
            "message": "Emergency stop executed - all positions closed",
            "trading_active": advanced_trading_engine.trading_active
        }
        
    except Exception as e:
        logger.error(f"Error executing emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-limits")
async def get_risk_limits():
    """Get current risk limits"""
    try:
        return {
            "risk_limits": advanced_trading_engine.risk_limits,
            "current_exposure": sum(
                pos.quantity * pos.current_price 
                for pos in advanced_trading_engine.positions.values()
            ),
            "max_position_size": advanced_trading_engine.risk_limits["max_position_size"],
            "max_daily_loss": advanced_trading_engine.risk_limits["max_daily_loss"],
            "max_drawdown": advanced_trading_engine.risk_limits["max_drawdown"]
        }
        
    except Exception as e:
        logger.error(f"Error getting risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-conditions")
async def get_market_conditions():
    """Get current market conditions"""
    try:
        conditions = {}
        for asset, condition in advanced_trading_engine.market_conditions.items():
            conditions[asset] = condition.value
        
        return {
            "market_conditions": conditions,
            "total_assets": len(conditions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting market conditions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_trading_engine_health():
    """Get trading engine health status"""
    try:
        return {
            "engine_id": advanced_trading_engine.engine_id,
            "is_running": advanced_trading_engine.is_running,
            "trading_active": advanced_trading_engine.trading_active,
            "active_strategies": len(advanced_trading_engine.strategies),
            "current_positions": len(advanced_trading_engine.positions),
            "pending_orders": len([
                o for o in advanced_trading_engine.orders.values() 
                if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
            ]),
            "total_signals": len(advanced_trading_engine.signals),
            "uptime": "active" if advanced_trading_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting trading engine health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
