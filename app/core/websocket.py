"""
WebSocket Support for Opinion Market
Handles real-time communication for live updates, notifications, and market data
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from fastapi import WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.websockets import WebSocketState
from sqlalchemy.orm import Session

from app.core.database import get_db, get_redis_client
from app.core.config import settings
from app.core.logging import log_system_metric
from app.core.security import get_client_ip
from app.core.cache import cache


class WebSocketMessageType(str, Enum):
    """Types of WebSocket messages"""
    PRICE_UPDATE = "price_update"
    TRADE_UPDATE = "trade_update"
    MARKET_UPDATE = "market_update"
    USER_NOTIFICATION = "user_notification"
    SYSTEM_MESSAGE = "system_message"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    AUTHENTICATION = "authentication"
    SUBSCRIPTION = "subscription"
    UNSUBSCRIPTION = "unsubscription"


class WebSocketConnectionState(str, Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """WebSocket message data structure"""
    message_id: str
    message_type: WebSocketMessageType
    timestamp: datetime
    data: Dict[str, Any]
    target_user_id: Optional[int] = None
    target_market_id: Optional[int] = None
    target_room: Optional[str] = None


@dataclass
class WebSocketConnection:
    """WebSocket connection data structure"""
    connection_id: str
    websocket: WebSocket
    user_id: Optional[int]
    state: WebSocketConnectionState
    connected_at: datetime
    last_heartbeat: datetime
    subscriptions: Set[str]
    client_ip: str
    user_agent: str


class WebSocketManager:
    """Manages WebSocket connections and real-time communication"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[int, Set[str]] = {}
        self.room_connections: Dict[str, Set[str]] = {}
        self.redis_client = get_redis_client()
        self.message_handlers: Dict[WebSocketMessageType, List[Callable]] = {}
        self.heartbeat_interval = settings.WS_HEARTBEAT_INTERVAL
        self.max_connections = settings.WS_MAX_CONNECTIONS
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._cleanup_stale_connections())
        
        # Register default message handlers
        self._register_default_handlers()
    
    async def connect(self, websocket: WebSocket, client_ip: str, user_agent: str) -> str:
        """Accept WebSocket connection and return connection ID"""
        try:
            # Check connection limit
            if len(self.active_connections) >= self.max_connections:
                await websocket.close(code=1013, reason="Server overloaded")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Maximum connections reached"
                )
            
            # Accept connection
            await websocket.accept()
            
            # Create connection ID
            connection_id = str(uuid.uuid4())
            
            # Create connection object
            connection = WebSocketConnection(
                connection_id=connection_id,
                websocket=websocket,
                user_id=None,
                state=WebSocketConnectionState.CONNECTED,
                connected_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                subscriptions=set(),
                client_ip=client_ip,
                user_agent=user_agent
            )
            
            # Store connection
            self.active_connections[connection_id] = connection
            
            # Log connection
            log_system_metric("websocket_connected", 1, {
                "connection_id": connection_id,
                "client_ip": client_ip,
                "total_connections": len(self.active_connections)
            })
            
            # Send welcome message
            await self._send_message(connection_id, WebSocketMessageType.SYSTEM_MESSAGE, {
                "message": "Connected to Opinion Market WebSocket",
                "connection_id": connection_id,
                "server_time": datetime.utcnow().isoformat()
            })
            
            return connection_id
            
        except Exception as e:
            log_system_metric("websocket_connection_error", 1, {"error": str(e)})
            raise
    
    async def disconnect(self, connection_id: str):
        """Disconnect WebSocket connection"""
        try:
            if connection_id in self.active_connections:
                connection = self.active_connections[connection_id]
                
                # Remove from user connections
                if connection.user_id and connection.user_id in self.user_connections:
                    self.user_connections[connection.user_id].discard(connection_id)
                    if not self.user_connections[connection.user_id]:
                        del self.user_connections[connection.user_id]
                
                # Remove from room connections
                for room in connection.subscriptions:
                    if room in self.room_connections:
                        self.room_connections[room].discard(connection_id)
                        if not self.room_connections[room]:
                            del self.room_connections[room]
                
                # Close WebSocket if still open
                if connection.websocket.client_state == WebSocketState.CONNECTED:
                    await connection.websocket.close()
                
                # Remove from active connections
                del self.active_connections[connection_id]
                
                # Log disconnection
                log_system_metric("websocket_disconnected", 1, {
                    "connection_id": connection_id,
                    "user_id": connection.user_id,
                    "total_connections": len(self.active_connections)
                })
                
        except Exception as e:
            log_system_metric("websocket_disconnection_error", 1, {"error": str(e)})
    
    async def authenticate(self, connection_id: str, user_id: int) -> bool:
        """Authenticate WebSocket connection"""
        try:
            if connection_id not in self.active_connections:
                return False
            
            connection = self.active_connections[connection_id]
            connection.user_id = user_id
            connection.state = WebSocketConnectionState.AUTHENTICATED
            
            # Add to user connections
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
            
            # Send authentication success message
            await self._send_message(connection_id, WebSocketMessageType.AUTHENTICATION, {
                "authenticated": True,
                "user_id": user_id,
                "message": "Authentication successful"
            })
            
            # Log authentication
            log_system_metric("websocket_authenticated", 1, {
                "connection_id": connection_id,
                "user_id": user_id
            })
            
            return True
            
        except Exception as e:
            log_system_metric("websocket_authentication_error", 1, {"error": str(e)})
            return False
    
    async def subscribe(self, connection_id: str, room: str) -> bool:
        """Subscribe connection to a room"""
        try:
            if connection_id not in self.active_connections:
                return False
            
            connection = self.active_connections[connection_id]
            connection.subscriptions.add(room)
            
            # Add to room connections
            if room not in self.room_connections:
                self.room_connections[room] = set()
            self.room_connections[room].add(connection_id)
            
            # Send subscription confirmation
            await self._send_message(connection_id, WebSocketMessageType.SUBSCRIPTION, {
                "room": room,
                "subscribed": True,
                "message": f"Subscribed to {room}"
            })
            
            # Log subscription
            log_system_metric("websocket_subscribed", 1, {
                "connection_id": connection_id,
                "room": room,
                "user_id": connection.user_id
            })
            
            return True
            
        except Exception as e:
            log_system_metric("websocket_subscription_error", 1, {"error": str(e)})
            return False
    
    async def unsubscribe(self, connection_id: str, room: str) -> bool:
        """Unsubscribe connection from a room"""
        try:
            if connection_id not in self.active_connections:
                return False
            
            connection = self.active_connections[connection_id]
            connection.subscriptions.discard(room)
            
            # Remove from room connections
            if room in self.room_connections:
                self.room_connections[room].discard(connection_id)
                if not self.room_connections[room]:
                    del self.room_connections[room]
            
            # Send unsubscription confirmation
            await self._send_message(connection_id, WebSocketMessageType.UNSUBSCRIPTION, {
                "room": room,
                "unsubscribed": True,
                "message": f"Unsubscribed from {room}"
            })
            
            # Log unsubscription
            log_system_metric("websocket_unsubscribed", 1, {
                "connection_id": connection_id,
                "room": room,
                "user_id": connection.user_id
            })
            
            return True
            
        except Exception as e:
            log_system_metric("websocket_unsubscription_error", 1, {"error": str(e)})
            return False
    
    async def send_to_user(self, user_id: int, message_type: WebSocketMessageType, data: Dict[str, Any]):
        """Send message to all connections of a specific user"""
        try:
            if user_id in self.user_connections:
                for connection_id in self.user_connections[user_id]:
                    await self._send_message(connection_id, message_type, data)
                    
        except Exception as e:
            log_system_metric("websocket_send_to_user_error", 1, {"error": str(e)})
    
    async def send_to_room(self, room: str, message_type: WebSocketMessageType, data: Dict[str, Any]):
        """Send message to all connections in a specific room"""
        try:
            if room in self.room_connections:
                for connection_id in self.room_connections[room]:
                    await self._send_message(connection_id, message_type, data)
                    
        except Exception as e:
            log_system_metric("websocket_send_to_room_error", 1, {"error": str(e)})
    
    async def send_to_connection(self, connection_id: str, message_type: WebSocketMessageType, data: Dict[str, Any]):
        """Send message to a specific connection"""
        try:
            await self._send_message(connection_id, message_type, data)
        except Exception as e:
            log_system_metric("websocket_send_to_connection_error", 1, {"error": str(e)})
    
    async def broadcast(self, message_type: WebSocketMessageType, data: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        try:
            for connection_id in list(self.active_connections.keys()):
                await self._send_message(connection_id, message_type, data)
                
        except Exception as e:
            log_system_metric("websocket_broadcast_error", 1, {"error": str(e)})
    
    async def _send_message(self, connection_id: str, message_type: WebSocketMessageType, data: Dict[str, Any]):
        """Send message to a specific connection"""
        try:
            if connection_id not in self.active_connections:
                return
            
            connection = self.active_connections[connection_id]
            
            # Check if connection is still open
            if connection.websocket.client_state != WebSocketState.CONNECTED:
                await self.disconnect(connection_id)
                return
            
            # Create message
            message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=message_type,
                timestamp=datetime.utcnow(),
                data=data
            )
            
            # Send message
            await connection.websocket.send_text(json.dumps(asdict(message), default=str))
            
            # Update last heartbeat
            connection.last_heartbeat = datetime.utcnow()
            
        except WebSocketDisconnect:
            await self.disconnect(connection_id)
        except Exception as e:
            log_system_metric("websocket_send_message_error", 1, {"error": str(e)})
            await self.disconnect(connection_id)
    
    async def _heartbeat_monitor(self):
        """Monitor WebSocket connections and send heartbeats"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for connection_id, connection in list(self.active_connections.items()):
                    # Send heartbeat if needed
                    if (current_time - connection.last_heartbeat).total_seconds() > self.heartbeat_interval:
                        await self._send_message(connection_id, WebSocketMessageType.HEARTBEAT, {
                            "timestamp": current_time.isoformat(),
                            "server_time": current_time.isoformat()
                        })
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                log_system_metric("websocket_heartbeat_monitor_error", 1, {"error": str(e)})
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _cleanup_stale_connections(self):
        """Clean up stale WebSocket connections"""
        while True:
            try:
                current_time = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection in self.active_connections.items():
                    # Mark as stale if no heartbeat for 5 minutes
                    if (current_time - connection.last_heartbeat).total_seconds() > 300:
                        stale_connections.append(connection_id)
                
                # Disconnect stale connections
                for connection_id in stale_connections:
                    await self.disconnect(connection_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                log_system_metric("websocket_cleanup_error", 1, {"error": str(e)})
                await asyncio.sleep(60)
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers[WebSocketMessageType.PRICE_UPDATE] = []
        self.message_handlers[WebSocketMessageType.TRADE_UPDATE] = []
        self.message_handlers[WebSocketMessageType.MARKET_UPDATE] = []
        self.message_handlers[WebSocketMessageType.USER_NOTIFICATION] = []
    
    def register_message_handler(self, message_type: WebSocketMessageType, handler: Callable):
        """Register a custom message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    async def handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        try:
            message_type = message.get("type")
            data = message.get("data", {})
            
            if message_type == "authenticate":
                user_id = data.get("user_id")
                if user_id:
                    await self.authenticate(connection_id, user_id)
            
            elif message_type == "subscribe":
                room = data.get("room")
                if room:
                    await self.subscribe(connection_id, room)
            
            elif message_type == "unsubscribe":
                room = data.get("room")
                if room:
                    await self.unsubscribe(connection_id, room)
            
            elif message_type == "heartbeat":
                # Update last heartbeat
                if connection_id in self.active_connections:
                    self.active_connections[connection_id].last_heartbeat = datetime.utcnow()
            
            # Call registered handlers
            if message_type in self.message_handlers:
                for handler in self.message_handlers[message_type]:
                    try:
                        await handler(connection_id, data)
                    except Exception as e:
                        log_system_metric("websocket_handler_error", 1, {"error": str(e)})
                        
        except Exception as e:
            log_system_metric("websocket_message_handle_error", 1, {"error": str(e)})
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "authenticated_connections": len([c for c in self.active_connections.values() if c.user_id]),
            "total_rooms": len(self.room_connections),
            "total_users": len(self.user_connections),
            "connections_by_state": {
                state.value: len([c for c in self.active_connections.values() if c.state == state])
                for state in WebSocketConnectionState
            }
        }


class WebSocketService:
    """Service for WebSocket-related operations"""
    
    def __init__(self):
        self.manager = WebSocketManager()
        self.redis_client = get_redis_client()
    
    async def broadcast_price_update(self, market_id: int, price_a: float, price_b: float):
        """Broadcast price update to market subscribers"""
        try:
            room = f"market:{market_id}"
            await self.manager.send_to_room(room, WebSocketMessageType.PRICE_UPDATE, {
                "market_id": market_id,
                "price_a": price_a,
                "price_b": price_b,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Also broadcast to general market room
            await self.manager.send_to_room("markets", WebSocketMessageType.PRICE_UPDATE, {
                "market_id": market_id,
                "price_a": price_a,
                "price_b": price_b,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            log_system_metric("websocket_price_update_error", 1, {"error": str(e)})
    
    async def broadcast_trade_update(self, trade_data: Dict[str, Any]):
        """Broadcast trade update to relevant subscribers"""
        try:
            market_id = trade_data.get("market_id")
            user_id = trade_data.get("user_id")
            
            # Broadcast to market subscribers
            if market_id:
                room = f"market:{market_id}"
                await self.manager.send_to_room(room, WebSocketMessageType.TRADE_UPDATE, trade_data)
            
            # Send to user if authenticated
            if user_id:
                await self.manager.send_to_user(user_id, WebSocketMessageType.TRADE_UPDATE, trade_data)
            
            # Broadcast to general trading room
            await self.manager.send_to_room("trading", WebSocketMessageType.TRADE_UPDATE, trade_data)
            
        except Exception as e:
            log_system_metric("websocket_trade_update_error", 1, {"error": str(e)})
    
    async def broadcast_market_update(self, market_data: Dict[str, Any]):
        """Broadcast market update to relevant subscribers"""
        try:
            market_id = market_data.get("market_id")
            
            # Broadcast to market subscribers
            if market_id:
                room = f"market:{market_id}"
                await self.manager.send_to_room(room, WebSocketMessageType.MARKET_UPDATE, market_data)
            
            # Broadcast to general markets room
            await self.manager.send_to_room("markets", WebSocketMessageType.MARKET_UPDATE, market_data)
            
        except Exception as e:
            log_system_metric("websocket_market_update_error", 1, {"error": str(e)})
    
    async def send_user_notification(self, user_id: int, notification_data: Dict[str, Any]):
        """Send notification to a specific user"""
        try:
            await self.manager.send_to_user(user_id, WebSocketMessageType.USER_NOTIFICATION, notification_data)
        except Exception as e:
            log_system_metric("websocket_user_notification_error", 1, {"error": str(e)})
    
    async def broadcast_system_message(self, message: str, message_type: str = "info"):
        """Broadcast system message to all connected clients"""
        try:
            await self.manager.broadcast(WebSocketMessageType.SYSTEM_MESSAGE, {
                "message": message,
                "type": message_type,
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            log_system_metric("websocket_system_message_error", 1, {"error": str(e)})
    
    def get_manager(self) -> WebSocketManager:
        """Get WebSocket manager instance"""
        return self.manager


# Global WebSocket service instance
websocket_service = WebSocketService()
