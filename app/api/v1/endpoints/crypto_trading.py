"""
Cryptocurrency Trading API Endpoints
Provides spot trading, DeFi integration, and crypto-specific analytics
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.services.crypto_trading import (
    get_crypto_trading_service,
    Cryptocurrency, CryptoPrice, CryptoPosition, DeFiProtocol, CryptoOrder
)
from app.schemas.crypto_trading import (
    CryptocurrencyRequest, CryptocurrencyResponse,
    CryptoPriceRequest, CryptoPriceResponse,
    CryptoPositionRequest, CryptoPositionResponse,
    DeFiProtocolRequest, DeFiProtocolResponse,
    CryptoOrderRequest, CryptoOrderResponse,
    CryptoMetricsResponse, DeFiAnalyticsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connections for real-time crypto updates
websocket_connections: Dict[str, WebSocket] = {}


@router.post("/cryptocurrencies/create", response_model=CryptocurrencyResponse)
async def create_cryptocurrency(
    request: CryptocurrencyRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Create a new cryptocurrency
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        crypto = await crypto_service.create_cryptocurrency(
            symbol=request.symbol,
            name=request.name,
            blockchain=request.blockchain,
            contract_address=request.contract_address,
            decimals=request.decimals,
            total_supply=request.total_supply
        )

        return CryptocurrencyResponse(
            crypto_id=crypto.crypto_id,
            symbol=crypto.symbol,
            name=crypto.name,
            blockchain=crypto.blockchain,
            contract_address=crypto.contract_address,
            decimals=crypto.decimals,
            total_supply=crypto.total_supply,
            circulating_supply=crypto.circulating_supply,
            market_cap=crypto.market_cap,
            is_active=crypto.is_active,
            created_at=crypto.created_at,
            last_updated=crypto.last_updated
        )

    except Exception as e:
        logger.error(f"Error creating cryptocurrency: {e}")
        raise HTTPException(status_code=500, detail="Error creating cryptocurrency")


@router.get("/cryptocurrencies", response_model=List[CryptocurrencyResponse])
async def get_cryptocurrencies(
    blockchain: Optional[str] = Query(None, description="Filter by blockchain"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get cryptocurrencies with optional filtering
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        cryptocurrencies = []
        for crypto in crypto_service.cryptocurrencies.values():
            # Apply filters
            if blockchain and crypto.blockchain != blockchain:
                continue
            if is_active is not None and crypto.is_active != is_active:
                continue

            cryptocurrencies.append(CryptocurrencyResponse(
                crypto_id=crypto.crypto_id,
                symbol=crypto.symbol,
                name=crypto.name,
                blockchain=crypto.blockchain,
                contract_address=crypto.contract_address,
                decimals=crypto.decimals,
                total_supply=crypto.total_supply,
                circulating_supply=crypto.circulating_supply,
                market_cap=crypto.market_cap,
                is_active=crypto.is_active,
                created_at=crypto.created_at,
                last_updated=crypto.last_updated
            ))

        return cryptocurrencies

    except Exception as e:
        logger.error(f"Error getting cryptocurrencies: {e}")
        raise HTTPException(status_code=500, detail="Error getting cryptocurrencies")


@router.get("/cryptocurrencies/{crypto_id}", response_model=CryptocurrencyResponse)
async def get_cryptocurrency(
    crypto_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get cryptocurrency by ID
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        crypto = crypto_service.cryptocurrencies.get(crypto_id)
        if not crypto:
            raise HTTPException(status_code=404, detail="Cryptocurrency not found")

        return CryptocurrencyResponse(
            crypto_id=crypto.crypto_id,
            symbol=crypto.symbol,
            name=crypto.name,
            blockchain=crypto.blockchain,
            contract_address=crypto.contract_address,
            decimals=crypto.decimals,
            total_supply=crypto.total_supply,
            circulating_supply=crypto.circulating_supply,
            market_cap=crypto.market_cap,
            is_active=crypto.is_active,
            created_at=crypto.created_at,
            last_updated=crypto.last_updated
        )

    except Exception as e:
        logger.error(f"Error getting cryptocurrency: {e}")
        raise HTTPException(status_code=500, detail="Error getting cryptocurrency")


@router.post("/cryptocurrencies/{crypto_id}/prices", response_model=CryptoPriceResponse)
async def add_crypto_price(
    crypto_id: str,
    request: CryptoPriceRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Add a new cryptocurrency price
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        price = await crypto_service.add_crypto_price(
            crypto_id=crypto_id,
            price_usd=request.price_usd,
            price_btc=request.price_btc,
            price_eth=request.price_eth,
            volume_24h=request.volume_24h,
            market_cap=request.market_cap,
            price_change_24h=request.price_change_24h,
            price_change_percent_24h=request.price_change_percent_24h,
            high_24h=request.high_24h,
            low_24h=request.low_24h,
            source=request.source
        )

        return CryptoPriceResponse(
            price_id=price.price_id,
            crypto_id=price.crypto_id,
            price_usd=price.price_usd,
            price_btc=price.price_btc,
            price_eth=price.price_eth,
            volume_24h=price.volume_24h,
            market_cap=price.market_cap,
            price_change_24h=price.price_change_24h,
            price_change_percent_24h=price.price_change_percent_24h,
            high_24h=price.high_24h,
            low_24h=price.low_24h,
            timestamp=price.timestamp,
            source=price.source
        )

    except Exception as e:
        logger.error(f"Error adding crypto price: {e}")
        raise HTTPException(status_code=500, detail="Error adding crypto price")


@router.get("/cryptocurrencies/{crypto_id}/prices")
async def get_crypto_prices(
    crypto_id: str,
    limit: int = Query(100, description="Number of prices to return"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get cryptocurrency prices
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        prices = crypto_service.crypto_prices.get(crypto_id, [])
        
        # Sort by timestamp and limit
        prices.sort(key=lambda x: x.timestamp, reverse=True)
        prices = prices[:limit]

        return JSONResponse(content={
            'crypto_id': crypto_id,
            'prices': [
                {
                    'price_id': price.price_id,
                    'price_usd': price.price_usd,
                    'price_btc': price.price_btc,
                    'price_eth': price.price_eth,
                    'volume_24h': price.volume_24h,
                    'market_cap': price.market_cap,
                    'price_change_24h': price.price_change_24h,
                    'price_change_percent_24h': price.price_change_percent_24h,
                    'high_24h': price.high_24h,
                    'low_24h': price.low_24h,
                    'timestamp': price.timestamp.isoformat(),
                    'source': price.source
                }
                for price in prices
            ],
            'count': len(prices)
        })

    except Exception as e:
        logger.error(f"Error getting crypto prices: {e}")
        raise HTTPException(status_code=500, detail="Error getting crypto prices")


@router.post("/positions/create", response_model=CryptoPositionResponse)
async def create_crypto_position(
    request: CryptoPositionRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Create a new cryptocurrency position
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        position = await crypto_service.create_crypto_position(
            user_id=current_user.id,
            crypto_id=request.crypto_id,
            position_type=request.position_type,
            size=request.size,
            entry_price=request.entry_price,
            leverage=request.leverage,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit
        )

        return CryptoPositionResponse(
            position_id=position.position_id,
            user_id=position.user_id,
            crypto_id=position.crypto_id,
            position_type=position.position_type,
            size=position.size,
            entry_price=position.entry_price,
            current_price=position.current_price,
            unrealized_pnl=position.unrealized_pnl,
            realized_pnl=position.realized_pnl,
            margin_used=position.margin_used,
            leverage=position.leverage,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            created_at=position.created_at,
            last_updated=position.last_updated
        )

    except Exception as e:
        logger.error(f"Error creating crypto position: {e}")
        raise HTTPException(status_code=500, detail="Error creating crypto position")


@router.get("/positions")
async def get_crypto_positions(
    crypto_id: Optional[str] = Query(None, description="Filter by cryptocurrency ID"),
    position_type: Optional[str] = Query(None, description="Filter by position type"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get cryptocurrency positions with optional filtering
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        positions = []
        for position in crypto_service.crypto_positions.values():
            if position.user_id != current_user.id:
                continue

            # Apply filters
            if crypto_id and position.crypto_id != crypto_id:
                continue
            if position_type and position.position_type != position_type:
                continue

            positions.append({
                'position_id': position.position_id,
                'crypto_id': position.crypto_id,
                'position_type': position.position_type,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'margin_used': position.margin_used,
                'leverage': position.leverage,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'created_at': position.created_at.isoformat(),
                'last_updated': position.last_updated.isoformat()
            })

        return JSONResponse(content={
            'positions': positions,
            'count': len(positions),
            'filters': {
                'crypto_id': crypto_id,
                'position_type': position_type
            }
        })

    except Exception as e:
        logger.error(f"Error getting crypto positions: {e}")
        raise HTTPException(status_code=500, detail="Error getting crypto positions")


@router.post("/defi/protocols/create", response_model=DeFiProtocolResponse)
async def create_defi_protocol(
    request: DeFiProtocolRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Create a new DeFi protocol
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        protocol = await crypto_service.create_defi_protocol(
            name=request.name,
            protocol_type=request.protocol_type,
            blockchain=request.blockchain,
            tvl=request.tvl,
            apy=request.apy,
            risk_score=request.risk_score
        )

        return DeFiProtocolResponse(
            protocol_id=protocol.protocol_id,
            name=protocol.name,
            protocol_type=protocol.protocol_type,
            blockchain=protocol.blockchain,
            tvl=protocol.tvl,
            apy=protocol.apy,
            risk_score=protocol.risk_score,
            is_active=protocol.is_active,
            created_at=protocol.created_at,
            last_updated=protocol.last_updated
        )

    except Exception as e:
        logger.error(f"Error creating DeFi protocol: {e}")
        raise HTTPException(status_code=500, detail="Error creating DeFi protocol")


@router.get("/defi/protocols", response_model=List[DeFiProtocolResponse])
async def get_defi_protocols(
    protocol_type: Optional[str] = Query(None, description="Filter by protocol type"),
    blockchain: Optional[str] = Query(None, description="Filter by blockchain"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get DeFi protocols with optional filtering
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        protocols = []
        for protocol in crypto_service.defi_protocols.values():
            # Apply filters
            if protocol_type and protocol.protocol_type != protocol_type:
                continue
            if blockchain and protocol.blockchain != blockchain:
                continue
            if is_active is not None and protocol.is_active != is_active:
                continue

            protocols.append(DeFiProtocolResponse(
                protocol_id=protocol.protocol_id,
                name=protocol.name,
                protocol_type=protocol.protocol_type,
                blockchain=protocol.blockchain,
                tvl=protocol.tvl,
                apy=protocol.apy,
                risk_score=protocol.risk_score,
                is_active=protocol.is_active,
                created_at=protocol.created_at,
                last_updated=protocol.last_updated
            ))

        return protocols

    except Exception as e:
        logger.error(f"Error getting DeFi protocols: {e}")
        raise HTTPException(status_code=500, detail="Error getting DeFi protocols")


@router.post("/orders/create", response_model=CryptoOrderResponse)
async def create_crypto_order(
    request: CryptoOrderRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Create a new cryptocurrency order
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        order = await crypto_service.create_crypto_order(
            user_id=current_user.id,
            crypto_id=request.crypto_id,
            order_type=request.order_type,
            side=request.side,
            size=request.size,
            price=request.price,
            stop_price=request.stop_price,
            limit_price=request.limit_price,
            time_in_force=request.time_in_force
        )

        return CryptoOrderResponse(
            order_id=order.order_id,
            user_id=order.user_id,
            crypto_id=order.crypto_id,
            order_type=order.order_type,
            side=order.side,
            size=order.size,
            price=order.price,
            stop_price=order.stop_price,
            limit_price=order.limit_price,
            time_in_force=order.time_in_force,
            status=order.status,
            filled_size=order.filled_size,
            filled_price=order.filled_price,
            commission=order.commission,
            created_at=order.created_at,
            last_updated=order.last_updated
        )

    except Exception as e:
        logger.error(f"Error creating crypto order: {e}")
        raise HTTPException(status_code=500, detail="Error creating crypto order")


@router.get("/orders")
async def get_crypto_orders(
    crypto_id: Optional[str] = Query(None, description="Filter by cryptocurrency ID"),
    status: Optional[str] = Query(None, description="Filter by order status"),
    limit: int = Query(50, description="Number of orders to return"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get cryptocurrency orders with optional filtering
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        orders = []
        for order in crypto_service.crypto_orders.values():
            if order.user_id != current_user.id:
                continue

            # Apply filters
            if crypto_id and order.crypto_id != crypto_id:
                continue
            if status and order.status != status:
                continue

            orders.append({
                'order_id': order.order_id,
                'crypto_id': order.crypto_id,
                'order_type': order.order_type,
                'side': order.side,
                'size': order.size,
                'price': order.price,
                'stop_price': order.stop_price,
                'limit_price': order.limit_price,
                'time_in_force': order.time_in_force,
                'status': order.status,
                'filled_size': order.filled_size,
                'filled_price': order.filled_price,
                'commission': order.commission,
                'created_at': order.created_at.isoformat(),
                'last_updated': order.last_updated.isoformat()
            })

        # Sort by creation date and limit
        orders.sort(key=lambda x: x['created_at'], reverse=True)
        orders = orders[:limit]

        return JSONResponse(content={
            'orders': orders,
            'count': len(orders),
            'filters': {
                'crypto_id': crypto_id,
                'status': status
            }
        })

    except Exception as e:
        logger.error(f"Error getting crypto orders: {e}")
        raise HTTPException(status_code=500, detail="Error getting crypto orders")


@router.get("/cryptocurrencies/{crypto_id}/metrics", response_model=CryptoMetricsResponse)
async def get_crypto_metrics(
    crypto_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive metrics for a cryptocurrency
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        metrics = await crypto_service.calculate_crypto_metrics(crypto_id)

        return CryptoMetricsResponse(**metrics)

    except Exception as e:
        logger.error(f"Error getting crypto metrics: {e}")
        raise HTTPException(status_code=500, detail="Error getting crypto metrics")


@router.get("/defi/analytics", response_model=DeFiAnalyticsResponse)
async def get_defi_analytics(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive DeFi analytics
    """
    try:
        redis_client = None  # You'll need to get this from your Redis connection
        crypto_service = await get_crypto_trading_service(redis_client, db)

        analytics = await crypto_service.get_defi_analytics()

        return DeFiAnalyticsResponse(**analytics)

    except Exception as e:
        logger.error(f"Error getting DeFi analytics: {e}")
        raise HTTPException(status_code=500, detail="Error getting DeFi analytics")


@router.websocket("/ws/crypto/{crypto_id}")
async def websocket_crypto_updates(websocket: WebSocket, crypto_id: str):
    """
    WebSocket endpoint for real-time cryptocurrency updates
    """
    await websocket.accept()
    connection_id = f"crypto_{crypto_id}_{id(websocket)}"
    websocket_connections[connection_id] = websocket
    
    try:
        while True:
            # Send periodic updates
            await websocket.send_text(json.dumps({
                'type': 'crypto_update',
                'crypto_id': crypto_id,
                'timestamp': datetime.utcnow().isoformat(),
                'message': 'Cryptocurrency data updated'
            }))
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
    except WebSocketDisconnect:
        del websocket_connections[connection_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if connection_id in websocket_connections:
            del websocket_connections[connection_id]
