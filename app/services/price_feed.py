import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from fastapi import WebSocket, WebSocketDisconnect
from app.core.database import SessionLocal
from app.models.market import Market
from app.models.trade import Trade


class PriceFeedManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = (
            {}
        )  # market_id -> connections
        self.price_history: Dict[int, List[Dict]] = {}  # market_id -> price history

    async def connect(self, websocket: WebSocket, market_id: int):
        await websocket.accept()
        if market_id not in self.active_connections:
            self.active_connections[market_id] = []
        self.active_connections[market_id].append(websocket)

        # Send current market data
        await self.send_market_update(market_id, websocket)

    def disconnect(self, websocket: WebSocket, market_id: int):
        if market_id in self.active_connections:
            self.active_connections[market_id].remove(websocket)
            if not self.active_connections[market_id]:
                del self.active_connections[market_id]

    async def send_market_update(
        self, market_id: int, websocket: Optional[WebSocket] = None
    ):
        """Send market update to connected clients"""
        db = SessionLocal()
        try:
            market = db.query(Market).filter(Market.id == market_id).first()
            if not market:
                return

            # Get recent trades for price history
            recent_trades = (
                db.query(Trade)
                .filter(
                    Trade.market_id == market_id,
                    Trade.created_at >= datetime.utcnow() - timedelta(hours=24),
                )
                .order_by(Trade.created_at.desc())
                .limit(100)
                .all()
            )

            # Create price history
            price_history = []
            for trade in reversed(recent_trades):
                price_history.append(
                    {
                        "timestamp": trade.created_at.isoformat(),
                        "price_a": market.current_price_a,
                        "price_b": market.current_price_b,
                        "volume": trade.total_value,
                    }
                )

            # Create market update message
            update_data = {
                "type": "market_update",
                "market_id": market_id,
                "data": {
                    "current_price_a": market.current_price_a,
                    "current_price_b": market.current_price_b,
                    "volume_24h": market.volume_24h,
                    "total_volume": market.volume_total,
                    "total_trades": market.total_trades,
                    "liquidity_pool_a": market.liquidity_pool_a,
                    "liquidity_pool_b": market.liquidity_pool_b,
                    "price_history": price_history,
                    "last_updated": datetime.utcnow().isoformat(),
                },
            }

            # Send to specific websocket or all connected clients
            if websocket:
                await websocket.send_text(json.dumps(update_data))
            else:
                await self.broadcast_to_market(market_id, update_data)

        finally:
            db.close()

    async def broadcast_to_market(self, market_id: int, message: Dict):
        """Broadcast message to all clients connected to a market"""
        if market_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[market_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except WebSocketDisconnect:
                    disconnected.append(connection)
                except Exception as e:
                    print(f"Error sending to websocket: {e}")
                    disconnected.append(connection)

            # Remove disconnected connections
            for connection in disconnected:
                self.disconnect(connection, market_id)

    async def broadcast_trade(self, trade: Trade):
        """Broadcast new trade to all connected clients"""
        message = {
            "type": "new_trade",
            "market_id": trade.market_id,
            "data": {
                "trade_id": trade.id,
                "trade_type": trade.trade_type,
                "outcome": trade.outcome,
                "amount": trade.amount,
                "price_per_share": trade.price_per_share,
                "total_value": trade.total_value,
                "timestamp": trade.created_at.isoformat(),
            },
        }

        # Update market prices
        await self.send_market_update(trade.market_id)

        # Broadcast trade
        await self.broadcast_to_market(trade.market_id, message)

    async def start_price_feed(self):
        """Start background price feed updates"""
        print("🔄 Starting price feed service...")
        error_count = 0
        max_errors = 10
        
        while True:
            try:
                # Update all active markets
                db = SessionLocal()
                try:
                    # Use a simple query without relationships to avoid SQLAlchemy issues
                    active_markets = (
                        db.query(Market.id, Market.status)
                        .filter(Market.status == "open")
                        .all()
                    )

                    for market in active_markets:
                        if market.id in self.active_connections:
                            await self.send_market_update(market.id)
                    
                    # Reset error count on successful operation
                    error_count = 0

                except Exception as db_error:
                    error_count += 1
                    print(f"⚠️  Database error in price feed (attempt {error_count}): {str(db_error)[:100]}...")
                    
                    # If too many errors, wait longer before retrying
                    if error_count >= max_errors:
                        print(f"🛑 Too many database errors ({error_count}), waiting 30 seconds before retry...")
                        await asyncio.sleep(30)
                        error_count = 0
                    
                finally:
                    db.close()

                # Wait 5 seconds before next update
                await asyncio.sleep(5)

            except Exception as e:
                error_count += 1
                print(f"❌ Critical error in price feed (attempt {error_count}): {str(e)[:100]}...")
                
                # If too many critical errors, wait longer
                if error_count >= max_errors:
                    print(f"🛑 Too many critical errors ({error_count}), waiting 60 seconds before retry...")
                    await asyncio.sleep(60)
                    error_count = 0
                else:
                    await asyncio.sleep(5)


# Global price feed manager
price_feed_manager = PriceFeedManager()
