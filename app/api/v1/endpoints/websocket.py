"""
WebSocket Endpoints for Opinion Market
Handles real-time WebSocket connections and communication
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.websockets import WebSocketState

from app.core.websocket import websocket_service, WebSocketMessageType
from app.core.security import get_current_user
from app.core.logging import log_system_metric
from app.core.security_audit import security_auditor, SecurityEventType, SecuritySeverity
from app.models.user import User

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication"""
    connection_id = None
    
    try:
        # Get client information
        client_ip = websocket.client.host if websocket.client else "unknown"
        user_agent = websocket.headers.get("user-agent", "unknown")
        
        # Connect to WebSocket manager
        connection_id = await websocket_service.get_manager().connect(websocket, client_ip, user_agent)
        
        # Log connection
        log_system_metric("websocket_connection_established", 1, {
            "connection_id": connection_id,
            "client_ip": client_ip
        })
        
        # Main message loop
        while True:
            try:
                # Receive message
                message_text = await websocket.receive_text()
                message_data = json.loads(message_text)
                
                # Handle message
                await websocket_service.get_manager().handle_message(connection_id, message_data)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                # Send error message for invalid JSON
                await websocket_service.get_manager().send_to_connection(
                    connection_id,
                    WebSocketMessageType.ERROR,
                    {"error": "Invalid JSON format"}
                )
            except Exception as e:
                # Log error and send error message
                log_system_metric("websocket_message_error", 1, {"error": str(e)})
                await websocket_service.get_manager().send_to_connection(
                    connection_id,
                    WebSocketMessageType.ERROR,
                    {"error": "Internal server error"}
                )
    
    except Exception as e:
        log_system_metric("websocket_endpoint_error", 1, {"error": str(e)})
        
        # Log security event for suspicious WebSocket activity
        await security_auditor.log_security_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            SecuritySeverity.MEDIUM,
            client_ip,
            {
                "endpoint": "/ws",
                "error": str(e),
                "connection_id": connection_id
            }
        )
    
    finally:
        # Clean up connection
        if connection_id:
            await websocket_service.get_manager().disconnect(connection_id)


@router.websocket("/ws/authenticated")
async def authenticated_websocket_endpoint(websocket: WebSocket, token: str = None):
    """Authenticated WebSocket endpoint for user-specific real-time features"""
    connection_id = None
    
    try:
        # Get client information
        client_ip = websocket.client.host if websocket.client else "unknown"
        user_agent = websocket.headers.get("user-agent", "unknown")
        
        # Connect to WebSocket manager
        connection_id = await websocket_service.get_manager().connect(websocket, client_ip, user_agent)
        
        # Authenticate user
        if token:
            try:
                # In a real implementation, you'd validate the JWT token here
                # For now, we'll extract user_id from token (simplified)
                user_id = int(token) if token.isdigit() else None
                
                if user_id:
                    await websocket_service.get_manager().authenticate(connection_id, user_id)
                    
                    # Auto-subscribe to user-specific rooms
                    await websocket_service.get_manager().subscribe(connection_id, f"user:{user_id}")
                    await websocket_service.get_manager().subscribe(connection_id, "notifications")
                    await websocket_service.get_manager().subscribe(connection_id, "trading")
                    
                    # Send welcome message
                    await websocket_service.get_manager().send_to_connection(
                        connection_id,
                        WebSocketMessageType.SYSTEM_MESSAGE,
                        {
                            "message": f"Welcome, User {user_id}!",
                            "authenticated": True,
                            "user_id": user_id,
                            "subscribed_rooms": [f"user:{user_id}", "notifications", "trading"]
                        }
                    )
                else:
                    await websocket_service.get_manager().send_to_connection(
                        connection_id,
                        WebSocketMessageType.ERROR,
                        {"error": "Invalid authentication token"}
                    )
                    
            except Exception as e:
                await websocket_service.get_manager().send_to_connection(
                    connection_id,
                    WebSocketMessageType.ERROR,
                    {"error": "Authentication failed"}
                )
        else:
            await websocket_service.get_manager().send_to_connection(
                connection_id,
                WebSocketMessageType.ERROR,
                {"error": "Authentication token required"}
            )
        
        # Main message loop
        while True:
            try:
                # Receive message
                message_text = await websocket.receive_text()
                message_data = json.loads(message_text)
                
                # Handle message
                await websocket_service.get_manager().handle_message(connection_id, message_data)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_service.get_manager().send_to_connection(
                    connection_id,
                    WebSocketMessageType.ERROR,
                    {"error": "Invalid JSON format"}
                )
            except Exception as e:
                log_system_metric("authenticated_websocket_message_error", 1, {"error": str(e)})
                await websocket_service.get_manager().send_to_connection(
                    connection_id,
                    WebSocketMessageType.ERROR,
                    {"error": "Internal server error"}
                )
    
    except Exception as e:
        log_system_metric("authenticated_websocket_endpoint_error", 1, {"error": str(e)})
        
        # Log security event
        await security_auditor.log_security_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            SecuritySeverity.MEDIUM,
            client_ip,
            {
                "endpoint": "/ws/authenticated",
                "error": str(e),
                "connection_id": connection_id
            }
        )
    
    finally:
        if connection_id:
            await websocket_service.get_manager().disconnect(connection_id)


@router.websocket("/ws/market/{market_id}")
async def market_websocket_endpoint(websocket: WebSocket, market_id: int):
    """Market-specific WebSocket endpoint for real-time market data"""
    connection_id = None
    
    try:
        # Get client information
        client_ip = websocket.client.host if websocket.client else "unknown"
        user_agent = websocket.headers.get("user-agent", "unknown")
        
        # Connect to WebSocket manager
        connection_id = await websocket_service.get_manager().connect(websocket, client_ip, user_agent)
        
        # Subscribe to market room
        await websocket_service.get_manager().subscribe(connection_id, f"market:{market_id}")
        
        # Send market subscription confirmation
        await websocket_service.get_manager().send_to_connection(
            connection_id,
            WebSocketMessageType.SUBSCRIPTION,
            {
                "room": f"market:{market_id}",
                "market_id": market_id,
                "message": f"Subscribed to market {market_id} updates"
            }
        )
        
        # Main message loop
        while True:
            try:
                # Receive message
                message_text = await websocket.receive_text()
                message_data = json.loads(message_text)
                
                # Handle message
                await websocket_service.get_manager().handle_message(connection_id, message_data)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_service.get_manager().send_to_connection(
                    connection_id,
                    WebSocketMessageType.ERROR,
                    {"error": "Invalid JSON format"}
                )
            except Exception as e:
                log_system_metric("market_websocket_message_error", 1, {"error": str(e)})
                await websocket_service.get_manager().send_to_connection(
                    connection_id,
                    WebSocketMessageType.ERROR,
                    {"error": "Internal server error"}
                )
    
    except Exception as e:
        log_system_metric("market_websocket_endpoint_error", 1, {"error": str(e)})
        
        # Log security event
        await security_auditor.log_security_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            SecuritySeverity.MEDIUM,
            client_ip,
            {
                "endpoint": f"/ws/market/{market_id}",
                "error": str(e),
                "connection_id": connection_id,
                "market_id": market_id
            }
        )
    
    finally:
        if connection_id:
            await websocket_service.get_manager().disconnect(connection_id)


@router.get("/ws/stats")
async def websocket_stats():
    """Get WebSocket connection statistics"""
    try:
        stats = websocket_service.get_manager().get_connection_stats()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        log_system_metric("websocket_stats_error", 1, {"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get WebSocket statistics"
        )


@router.post("/ws/broadcast")
async def broadcast_message(message: str, message_type: str = "info"):
    """Broadcast system message to all connected WebSocket clients"""
    try:
        await websocket_service.broadcast_system_message(message, message_type)
        return {
            "status": "success",
            "message": "Message broadcasted successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        log_system_metric("websocket_broadcast_error", 1, {"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to broadcast message"
        )


@router.post("/ws/notify/{user_id}")
async def send_user_notification(user_id: int, notification: Dict[str, Any]):
    """Send notification to a specific user via WebSocket"""
    try:
        await websocket_service.send_user_notification(user_id, notification)
        return {
            "status": "success",
            "message": f"Notification sent to user {user_id}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        log_system_metric("websocket_user_notification_error", 1, {"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send user notification"
        )


@router.post("/ws/market/{market_id}/update")
async def broadcast_market_update(market_id: int, update_data: Dict[str, Any]):
    """Broadcast market update to all subscribers"""
    try:
        update_data["market_id"] = market_id
        await websocket_service.broadcast_market_update(update_data)
        return {
            "status": "success",
            "message": f"Market update broadcasted for market {market_id}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        log_system_metric("websocket_market_update_error", 1, {"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to broadcast market update"
        )


@router.post("/ws/trade/update")
async def broadcast_trade_update(trade_data: Dict[str, Any]):
    """Broadcast trade update to relevant subscribers"""
    try:
        await websocket_service.broadcast_trade_update(trade_data)
        return {
            "status": "success",
            "message": "Trade update broadcasted successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        log_system_metric("websocket_trade_update_error", 1, {"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to broadcast trade update"
        )


@router.post("/ws/price/update")
async def broadcast_price_update(market_id: int, price_a: float, price_b: float):
    """Broadcast price update for a specific market"""
    try:
        await websocket_service.broadcast_price_update(market_id, price_a, price_b)
        return {
            "status": "success",
            "message": f"Price update broadcasted for market {market_id}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        log_system_metric("websocket_price_update_error", 1, {"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to broadcast price update"
        )