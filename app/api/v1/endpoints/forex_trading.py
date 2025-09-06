"""
Forex Trading API Endpoints
Provides endpoints for spot FX, forwards, swaps, and comprehensive FX analytics
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
import asyncio
from sqlalchemy.orm import Session

from app.services.forex_trading import (
    ForexTradingService,
    CurrencyPair,
    FXPrice,
    FXPosition,
    ForwardContract,
    SwapContract,
    FXOrder,
)
from app.schemas.forex_trading import (
    CurrencyPairCreate,
    CurrencyPairResponse,
    FXPriceCreate,
    FXPriceResponse,
    FXPositionCreate,
    FXPositionResponse,
    ForwardContractCreate,
    ForwardContractResponse,
    SwapContractCreate,
    SwapContractResponse,
    FXOrderCreate,
    FXOrderResponse,
    FXMetricsResponse,
    CrossCurrencyRatesResponse,
    ForwardPointsResponse,
)
from app.core.database import get_db
from app.core.redis import get_redis

logger = logging.getLogger(__name__)

router = APIRouter()
websocket_connections: Dict[str, WebSocket] = {}


@router.post("/currency-pairs", response_model=CurrencyPairResponse)
async def create_currency_pair(
    pair_data: CurrencyPairCreate,
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Create a new currency pair"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        pair = await service.create_currency_pair(
            base_currency=pair_data.base_currency,
            quote_currency=pair_data.quote_currency,
            pip_value=pair_data.pip_value,
            lot_size=pair_data.lot_size,
            min_trade_size=pair_data.min_trade_size,
            max_trade_size=pair_data.max_trade_size,
            margin_requirement=pair_data.margin_requirement,
            swap_long=pair_data.swap_long,
            swap_short=pair_data.swap_short,
        )

        return CurrencyPairResponse(
            pair_id=pair.pair_id,
            base_currency=pair.base_currency,
            quote_currency=pair.quote_currency,
            pair_name=pair.pair_name,
            pip_value=pair.pip_value,
            lot_size=pair.lot_size,
            min_trade_size=pair.min_trade_size,
            max_trade_size=pair.max_trade_size,
            margin_requirement=pair.margin_requirement,
            swap_long=pair.swap_long,
            swap_short=pair.swap_short,
            is_active=pair.is_active,
            trading_hours=pair.trading_hours,
            created_at=pair.created_at,
            last_updated=pair.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating currency pair: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/currency-pairs", response_model=List[CurrencyPairResponse])
async def get_currency_pairs(
    active_only: bool = Query(True, description="Return only active currency pairs"),
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Get all currency pairs"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        pairs = list(service.currency_pairs.values())

        if active_only:
            pairs = [p for p in pairs if p.is_active]

        return [
            CurrencyPairResponse(
                pair_id=pair.pair_id,
                base_currency=bond.base_currency,
                quote_currency=bond.quote_currency,
                pair_name=bond.pair_name,
                pip_value=bond.pip_value,
                lot_size=bond.lot_size,
                min_trade_size=bond.min_trade_size,
                max_trade_size=bond.max_trade_size,
                margin_requirement=bond.margin_requirement,
                swap_long=bond.swap_long,
                swap_short=bond.swap_short,
                is_active=bond.is_active,
                trading_hours=bond.trading_hours,
                created_at=bond.created_at,
                last_updated=bond.last_updated,
            )
            for bond in pairs
        ]

    except Exception as e:
        logger.error(f"Error getting currency pairs: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/currency-pairs/{pair_id}", response_model=CurrencyPairResponse)
async def get_currency_pair(
    pair_id: str, db: Session = Depends(get_db), redis_client=Depends(get_redis)
):
    """Get a specific currency pair"""
    try:
        service = await get_forex_trading_service(redis_client, db)

        if pair_id not in service.currency_pairs:
            raise HTTPException(status_code=404, detail="Currency pair not found")

        pair = service.currency_pairs[pair_id]

        return CurrencyPairResponse(
            pair_id=pair.pair_id,
            base_currency=pair.base_currency,
            quote_currency=pair.quote_currency,
            pair_name=pair.pair_name,
            pip_value=pair.pip_value,
            lot_size=pair.lot_size,
            min_trade_size=pair.min_trade_size,
            max_trade_size=pair.max_trade_size,
            margin_requirement=pair.margin_requirement,
            swap_long=pair.swap_long,
            swap_short=pair.swap_short,
            is_active=pair.is_active,
            trading_hours=pair.trading_hours,
            created_at=pair.created_at,
            last_updated=pair.last_updated,
        )

    except Exception as e:
        logger.error(f"Error getting currency pair: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/prices", response_model=FXPriceResponse)
async def add_fx_price(
    price_data: FXPriceCreate,
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Add a new FX price"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        price = await service.add_fx_price(
            pair_id=price_data.pair_id,
            bid_price=price_data.bid_price,
            ask_price=price_data.ask_price,
            volume_24h=price_data.volume_24h,
            high_24h=price_data.high_24h,
            low_24h=price_data.low_24h,
            change_24h=price_data.change_24h,
            source=price_data.source,
        )

        return FXPriceResponse(
            price_id=price.price_id,
            pair_id=price.pair_id,
            bid_price=price.bid_price,
            ask_price=price.ask_price,
            mid_price=price.mid_price,
            spread=price.spread,
            pip_value=price.pip_value,
            timestamp=price.timestamp,
            source=price.source,
            volume_24h=price.volume_24h,
            high_24h=price.high_24h,
            low_24h=price.low_24h,
            change_24h=price.change_24h,
            change_pct_24h=price.change_pct_24h,
        )

    except Exception as e:
        logger.error(f"Error adding FX price: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/prices/{pair_id}", response_model=List[FXPriceResponse])
async def get_fx_prices(
    pair_id: str,
    limit: int = Query(100, description="Number of prices to return"),
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Get FX prices for a currency pair"""
    try:
        service = await get_forex_trading_service(redis_client, db)

        if pair_id not in service.currency_pairs:
            raise HTTPException(status_code=404, detail="Currency pair not found")

        prices = service.fx_prices.get(pair_id, [])
        prices = prices[-limit:] if prices else []

        return [
            FXPriceResponse(
                price_id=price.price_id,
                pair_id=price.pair_id,
                bid_price=price.bid_price,
                ask_price=price.ask_price,
                mid_price=price.mid_price,
                spread=price.spread,
                pip_value=price.pip_value,
                timestamp=price.timestamp,
                source=price.source,
                volume_24h=price.volume_24h,
                high_24h=price.high_24h,
                low_24h=price.low_24h,
                change_24h=price.change_24h,
                change_pct_24h=price.change_pct_24h,
            )
            for price in prices
        ]

    except Exception as e:
        logger.error(f"Error getting FX prices: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/positions", response_model=FXPositionResponse)
async def create_fx_position(
    position_data: FXPositionCreate,
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Create a new FX position"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        position = await service.create_fx_position(
            user_id=position_data.user_id,
            pair_id=position_data.pair_id,
            position_type=position_data.position_type,
            quantity=position_data.quantity,
            entry_price=position_data.entry_price,
            leverage=position_data.leverage,
            stop_loss=position_data.stop_loss,
            take_profit=position_data.take_profit,
        )

        return FXPositionResponse(
            position_id=position.position_id,
            user_id=position.user_id,
            pair_id=position.pair_id,
            position_type=position.position_type,
            quantity=position.quantity,
            entry_price=position.entry_price,
            current_price=position.current_price,
            pip_value=position.pip_value,
            unrealized_pnl=position.unrealized_pnl,
            realized_pnl=position.realized_pnl,
            swap_charges=position.swap_charges,
            margin_used=position.margin_used,
            leverage=position.leverage,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            created_at=position.created_at,
            last_updated=position.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating FX position: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/positions/{user_id}", response_model=List[FXPositionResponse])
async def get_user_positions(
    user_id: int, db: Session = Depends(get_db), redis_client=Depends(get_redis)
):
    """Get FX positions for a user"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        positions = [p for p in service.fx_positions.values() if p.user_id == user_id]

        return [
            FXPositionResponse(
                position_id=position.position_id,
                user_id=position.user_id,
                pair_id=position.pair_id,
                position_type=position.position_type,
                quantity=position.quantity,
                entry_price=position.entry_price,
                current_price=position.current_price,
                pip_value=position.pip_value,
                unrealized_pnl=position.unrealized_pnl,
                realized_pnl=position.realized_pnl,
                swap_charges=position.swap_charges,
                margin_used=position.margin_used,
                leverage=position.leverage,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                created_at=position.created_at,
                last_updated=position.last_updated,
            )
            for position in positions
        ]

    except Exception as e:
        logger.error(f"Error getting user positions: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/forward-contracts", response_model=ForwardContractResponse)
async def create_forward_contract(
    contract_data: ForwardContractCreate,
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Create a new FX forward contract"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        contract = await service.create_forward_contract(
            user_id=contract_data.user_id,
            pair_id=contract_data.pair_id,
            quantity=contract_data.quantity,
            forward_rate=contract_data.forward_rate,
            spot_rate=contract_data.spot_rate,
            value_date=contract_data.value_date,
            maturity_date=contract_data.maturity_date,
            contract_type=contract_data.contract_type,
            is_deliverable=contract_data.is_deliverable,
        )

        return ForwardContractResponse(
            contract_id=contract.contract_id,
            pair_id=contract.pair_id,
            user_id=contract.user_id,
            quantity=contract.quantity,
            forward_rate=contract.forward_rate,
            spot_rate=contract.spot_rate,
            forward_points=contract.forward_points,
            value_date=contract.value_date,
            maturity_date=contract.maturity_date,
            contract_type=contract.contract_type,
            is_deliverable=contract.is_deliverable,
            created_at=contract.created_at,
            last_updated=contract.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating forward contract: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/forward-contracts/{user_id}", response_model=List[ForwardContractResponse]
)
async def get_user_forward_contracts(
    user_id: int, db: Session = Depends(get_db), redis_client=Depends(get_redis)
):
    """Get forward contracts for a user"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        contracts = [
            c for c in service.forward_contracts.values() if c.user_id == user_id
        ]

        return [
            ForwardContractResponse(
                contract_id=contract.contract_id,
                pair_id=contract.pair_id,
                user_id=contract.user_id,
                quantity=contract.quantity,
                forward_rate=contract.forward_rate,
                spot_rate=contract.spot_rate,
                forward_points=contract.forward_points,
                value_date=contract.value_date,
                maturity_date=contract.maturity_date,
                contract_type=contract.contract_type,
                is_deliverable=contract.is_deliverable,
                created_at=contract.created_at,
                last_updated=contract.last_updated,
            )
            for contract in contracts
        ]

    except Exception as e:
        logger.error(f"Error getting user forward contracts: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/swap-contracts", response_model=SwapContractResponse)
async def create_swap_contract(
    swap_data: SwapContractCreate,
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Create a new FX swap contract"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        swap = await service.create_swap_contract(
            user_id=swap_data.user_id,
            pair_id=swap_data.pair_id,
            near_leg=swap_data.near_leg,
            far_leg=swap_data.far_leg,
            swap_rate=swap_data.swap_rate,
            value_date=swap_data.value_date,
            maturity_date=swap_data.maturity_date,
        )

        return SwapContractResponse(
            swap_id=swap.swap_id,
            pair_id=swap.pair_id,
            user_id=swap.user_id,
            near_leg=swap.near_leg,
            far_leg=swap.far_leg,
            swap_rate=swap.swap_rate,
            swap_points=swap.swap_points,
            value_date=swap.value_date,
            maturity_date=swap.maturity_date,
            created_at=swap.created_at,
            last_updated=swap.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating swap contract: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/swap-contracts/{user_id}", response_model=List[SwapContractResponse])
async def get_user_swap_contracts(
    user_id: int, db: Session = Depends(get_db), redis_client=Depends(get_redis)
):
    """Get swap contracts for a user"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        swaps = [s for s in service.swap_contracts.values() if s.user_id == user_id]

        return [
            SwapContractResponse(
                swap_id=swap.swap_id,
                pair_id=swap.pair_id,
                user_id=swap.user_id,
                near_leg=swap.near_leg,
                far_leg=swap.far_leg,
                swap_rate=swap.swap_rate,
                swap_points=swap.swap_points,
                value_date=swap.value_date,
                maturity_date=swap.maturity_date,
                created_at=swap.created_at,
                last_updated=swap.last_updated,
            )
            for swap in swaps
        ]

    except Exception as e:
        logger.error(f"Error getting user swap contracts: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/orders", response_model=FXOrderResponse)
async def place_fx_order(
    order_data: FXOrderCreate,
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Place a new FX order"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        order = await service.place_fx_order(
            user_id=order_data.user_id,
            pair_id=order_data.pair_id,
            order_type=order_data.order_type,
            side=order_data.side,
            quantity=order_data.quantity,
            price=order_data.price,
            stop_price=order_data.stop_price,
            limit_price=order_data.limit_price,
            time_in_force=order_data.time_in_force,
        )

        return FXOrderResponse(
            order_id=order.order_id,
            user_id=order.user_id,
            pair_id=order.pair_id,
            order_type=order.order_type,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            limit_price=order.limit_price,
            time_in_force=order.time_in_force,
            status=order.status,
            filled_quantity=order.filled_quantity,
            filled_price=order.filled_price,
            created_at=order.created_at,
            last_updated=order.last_updated,
        )

    except Exception as e:
        logger.error(f"Error placing FX order: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/orders/{user_id}", response_model=List[FXOrderResponse])
async def get_user_orders(
    user_id: int,
    status: Optional[str] = Query(None, description="Filter by order status"),
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Get FX orders for a user"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        orders = [o for o in service.fx_orders.values() if o.user_id == user_id]

        if status:
            orders = [o for o in orders if o.status == status]

        return [
            FXOrderResponse(
                order_id=order.order_id,
                user_id=order.user_id,
                pair_id=order.pair_id,
                order_type=order.order_type,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                limit_price=order.limit_price,
                time_in_force=order.time_in_force,
                status=order.status,
                filled_quantity=order.filled_quantity,
                filled_price=order.filled_price,
                created_at=order.created_at,
                last_updated=order.last_updated,
            )
            for order in orders
        ]

    except Exception as e:
        logger.error(f"Error getting user orders: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/metrics/{pair_id}", response_model=FXMetricsResponse)
async def get_fx_metrics(
    pair_id: str, db: Session = Depends(get_db), redis_client=Depends(get_redis)
):
    """Get comprehensive metrics for a currency pair"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        metrics = await service.calculate_fx_metrics(pair_id)

        return FXMetricsResponse(**metrics)

    except Exception as e:
        logger.error(f"Error getting FX metrics: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/cross-rates/{base_currency}", response_model=CrossCurrencyRatesResponse)
async def get_cross_currency_rates(
    base_currency: str, db: Session = Depends(get_db), redis_client=Depends(get_redis)
):
    """Get cross currency rates for a base currency"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        cross_rates = await service.get_cross_currency_rates(base_currency)

        return CrossCurrencyRatesResponse(**cross_rates)

    except Exception as e:
        logger.error(f"Error getting cross currency rates: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/forward-points", response_model=ForwardPointsResponse)
async def calculate_forward_points(
    pair_id: str,
    spot_rate: float,
    interest_rate_base: float,
    interest_rate_quote: float,
    days_to_maturity: int,
    db: Session = Depends(get_db),
    redis_client=Depends(get_redis),
):
    """Calculate forward points using interest rate parity"""
    try:
        service = await get_forex_trading_service(redis_client, db)
        forward_points = await service.calculate_forward_points(
            pair_id=pair_id,
            spot_rate=spot_rate,
            interest_rate_base=interest_rate_base,
            interest_rate_quote=interest_rate_quote,
            days_to_maturity=days_to_maturity,
        )

        return ForwardPointsResponse(**forward_points)

    except Exception as e:
        logger.error(f"Error calculating forward points: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/trading-sessions")
async def get_trading_sessions(
    db: Session = Depends(get_db), redis_client=Depends(get_redis)
):
    """Get current trading sessions status"""
    try:
        service = await get_forex_trading_service(redis_client, db)

        # Get current time in UTC
        current_time = datetime.utcnow()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_time_str = f"{current_hour:02d}:{current_minute:02d}"

        # Check which sessions are currently active
        active_sessions = []
        for session_name, session_times in service.trading_sessions.items():
            start_time = session_times["start"]
            end_time = session_times["end"]

            # Simple time comparison (in practice, this would be more sophisticated)
            if start_time <= current_time_str <= end_time:
                active_sessions.append(session_name)

        return {
            "current_time_utc": current_time_str,
            "trading_sessions": service.trading_sessions,
            "active_sessions": active_sessions,
            "next_session": "london" if "london" not in active_sessions else "new_york",
        }

    except Exception as e:
        logger.error(f"Error getting trading sessions: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.websocket("/ws/fx-updates/{pair_id}")
async def websocket_fx_updates(websocket: WebSocket, pair_id: str):
    """WebSocket endpoint for real-time FX updates"""
    await websocket.accept()
    connection_id = f"fx_{pair_id}_{id(websocket)}"
    websocket_connections[connection_id] = websocket

    try:
        while True:
            # Send periodic updates (in practice, this would be real-time data)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "fx_update",
                        "pair_id": pair_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "message": "FX price update",
                    }
                )
            )

            await asyncio.sleep(5)  # Send update every 5 seconds

    except WebSocketDisconnect:
        if connection_id in websocket_connections:
            del websocket_connections[connection_id]


# Import the service function
from app.services.forex_trading import get_forex_trading_service
