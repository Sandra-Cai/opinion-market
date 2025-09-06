from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from typing import Optional
import json

from app.services.price_feed import price_feed_manager
from app.core.auth import verify_token
from app.core.database import get_db

router = APIRouter()


@router.websocket("/ws/market/{market_id}")
async def websocket_market_feed(websocket: WebSocket, market_id: int):
    """WebSocket endpoint for real-time market price feeds"""
    await price_feed_manager.connect(websocket, market_id)

    try:
        while True:
            # Keep connection alive and handle any client messages
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "subscribe":
                # Already subscribed when connected
                await websocket.send_text(
                    json.dumps({"type": "subscribed", "market_id": market_id})
                )

    except WebSocketDisconnect:
        price_feed_manager.disconnect(websocket, market_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        price_feed_manager.disconnect(websocket, market_id)


@router.websocket("/ws/user")
async def websocket_user_feed(websocket: WebSocket, token: Optional[str] = None):
    """WebSocket endpoint for user-specific updates (portfolio, trades, etc.)"""
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        return

    # Verify token
    username = verify_token(token)
    if not username:
        await websocket.close(code=4001, reason="Invalid token")
        return

    await websocket.accept()

    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "get_portfolio":
                # Send portfolio update
                portfolio_update = {
                    "type": "portfolio_update",
                    "data": {"message": "Portfolio update would be sent here"},
                }
                await websocket.send_text(json.dumps(portfolio_update))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"User WebSocket error: {e}")


@router.websocket("/ws/global")
async def websocket_global_feed(websocket: WebSocket):
    """WebSocket endpoint for global market updates"""
    await websocket.accept()

    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "get_market_stats":
                # Send global market stats
                stats_update = {
                    "type": "market_stats",
                    "data": {"message": "Global market stats would be sent here"},
                }
                await websocket.send_text(json.dumps(stats_update))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Global WebSocket error: {e}")
