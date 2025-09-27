"""
WebSocket Manager
Manages WebSocket connections and real-time communication
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from app.core.enhanced_cache import enhanced_cache
from app.core.security_manager import security_manager

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    PING = "ping"
    PONG = "pong"
    AUTH = "auth"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    MARKET_UPDATE = "market_update"
    TRADE_UPDATE = "trade_update"
    NOTIFICATION = "notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    message_id: str
    message_type: MessageType
    data: Dict[str, Any]
    timestamp: float
    sender_id: Optional[str] = None
    target_rooms: Optional[List[str]] = None
    requires_auth: bool = False


@dataclass
class ConnectionInfo:
    """WebSocket connection information"""
    connection_id: str
    user_id: Optional[str]
    rooms: Set[str]
    last_activity: float
    is_authenticated: bool
    device_info: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class WebSocketManager:
    """Manages WebSocket connections and real-time communication"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        self.room_subscriptions: Dict[str, Set[str]] = {}  # room_id -> set of connection_ids
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 300  # 5 minutes
        self.max_connections_per_user = 5
        self.max_rooms_per_connection = 10
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "rooms_created": 0,
            "authentication_failures": 0
        }
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._cleanup_loop())
    
    async def connect(self, websocket: WebSocket, connection_id: str = None) -> str:
        """Accept a new WebSocket connection"""
        try:
            await websocket.accept()
            
            if not connection_id:
                connection_id = str(uuid.uuid4())
            
            # Create connection info
            connection_info = ConnectionInfo(
                connection_id=connection_id,
                user_id=None,
                rooms=set(),
                last_activity=time.time(),
                is_authenticated=False,
                ip_address=websocket.client.host if websocket.client else None
            )
            
            # Store connection
            self.active_connections[connection_id] = websocket
            self.connection_info[connection_id] = connection_info
            
            # Update statistics
            self.stats["total_connections"] += 1
            self.stats["active_connections"] += 1
            
            logger.info(f"WebSocket connection established: {connection_id}")
            
            # Send welcome message
            await self._send_message(connection_id, MessageType.PONG, {
                "message": "Connected to Opinion Market WebSocket",
                "connection_id": connection_id,
                "server_time": time.time()
            })
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            raise e
    
    async def disconnect(self, connection_id: str, reason: str = "client_disconnect"):
        """Disconnect a WebSocket connection"""
        try:
            if connection_id in self.active_connections:
                # Remove from all rooms
                connection_info = self.connection_info.get(connection_id)
                if connection_info:
                    for room_id in list(connection_info.rooms):
                        await self.leave_room(connection_id, room_id)
                
                # Close WebSocket connection
                websocket = self.active_connections[connection_id]
                try:
                    await websocket.close()
                except:
                    pass
                
                # Remove from tracking
                del self.active_connections[connection_id]
                if connection_id in self.connection_info:
                    del self.connection_info[connection_id]
                
                # Update statistics
                self.stats["active_connections"] -= 1
                
                logger.info(f"WebSocket connection closed: {connection_id}, reason: {reason}")
                
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket {connection_id}: {e}")
    
    async def send_message(self, connection_id: str, message_type: MessageType, 
                         data: Dict[str, Any]) -> bool:
        """Send a message to a specific connection"""
        try:
            if connection_id not in self.active_connections:
                return False
            
            websocket = self.active_connections[connection_id]
            message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=message_type,
                data=data,
                timestamp=time.time(),
                sender_id="server"
            )
            
            await websocket.send_text(json.dumps({
                "message_id": message.message_id,
                "type": message.message_type.value,
                "data": message.data,
                "timestamp": message.timestamp
            }))
            
            # Update statistics
            self.stats["messages_sent"] += 1
            
            # Update last activity
            if connection_id in self.connection_info:
                self.connection_info[connection_id].last_activity = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            return False
    
    async def broadcast_to_room(self, room_id: str, message_type: MessageType, 
                              data: Dict[str, Any], exclude_connection: str = None) -> int:
        """Broadcast a message to all connections in a room"""
        try:
            if room_id not in self.room_subscriptions:
                return 0
            
            sent_count = 0
            connections = list(self.room_subscriptions[room_id])
            
            for connection_id in connections:
                if connection_id != exclude_connection:
                    if await self.send_message(connection_id, message_type, data):
                        sent_count += 1
            
            return sent_count
            
        except Exception as e:
            logger.error(f"Failed to broadcast to room {room_id}: {e}")
            return 0
    
    async def broadcast_to_user(self, user_id: str, message_type: MessageType, 
                              data: Dict[str, Any]) -> int:
        """Broadcast a message to all connections of a user"""
        try:
            sent_count = 0
            
            for connection_id, connection_info in self.connection_info.items():
                if connection_info.user_id == user_id:
                    if await self.send_message(connection_id, message_type, data):
                        sent_count += 1
            
            return sent_count
            
        except Exception as e:
            logger.error(f"Failed to broadcast to user {user_id}: {e}")
            return 0
    
    async def join_room(self, connection_id: str, room_id: str) -> bool:
        """Add a connection to a room"""
        try:
            if connection_id not in self.connection_info:
                return False
            
            connection_info = self.connection_info[connection_id]
            
            # Check room limit
            if len(connection_info.rooms) >= self.max_rooms_per_connection:
                await self.send_message(connection_id, MessageType.ERROR, {
                    "error": "Maximum room limit reached",
                    "max_rooms": self.max_rooms_per_connection
                })
                return False
            
            # Add to room
            connection_info.rooms.add(room_id)
            
            if room_id not in self.room_subscriptions:
                self.room_subscriptions[room_id] = set()
                self.stats["rooms_created"] += 1
            
            self.room_subscriptions[room_id].add(connection_id)
            
            # Notify room members
            await self.broadcast_to_room(room_id, MessageType.NOTIFICATION, {
                "message": f"User joined room {room_id}",
                "room_id": room_id,
                "connection_id": connection_id
            }, exclude_connection=connection_id)
            
            logger.info(f"Connection {connection_id} joined room {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to join room {room_id}: {e}")
            return False
    
    async def leave_room(self, connection_id: str, room_id: str) -> bool:
        """Remove a connection from a room"""
        try:
            if connection_id not in self.connection_info:
                return False
            
            connection_info = self.connection_info[connection_id]
            connection_info.rooms.discard(room_id)
            
            if room_id in self.room_subscriptions:
                self.room_subscriptions[room_id].discard(connection_id)
                
                # Clean up empty rooms
                if not self.room_subscriptions[room_id]:
                    del self.room_subscriptions[room_id]
            
            # Notify room members
            await self.broadcast_to_room(room_id, MessageType.NOTIFICATION, {
                "message": f"User left room {room_id}",
                "room_id": room_id,
                "connection_id": connection_id
            }, exclude_connection=connection_id)
            
            logger.info(f"Connection {connection_id} left room {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to leave room {room_id}: {e}")
            return False
    
    async def authenticate_connection(self, connection_id: str, user_id: str, 
                                    token: str) -> bool:
        """Authenticate a WebSocket connection"""
        try:
            if connection_id not in self.connection_info:
                return False
            
            # Check if user already has too many connections
            user_connections = sum(
                1 for conn_info in self.connection_info.values() 
                if conn_info.user_id == user_id
            )
            
            if user_connections >= self.max_connections_per_user:
                await self.send_message(connection_id, MessageType.ERROR, {
                    "error": "Maximum connections per user reached",
                    "max_connections": self.max_connections_per_user
                })
                return False
            
            # Validate token (simplified - in real implementation, verify JWT)
            if not self._validate_token(token):
                self.stats["authentication_failures"] += 1
                await self.send_message(connection_id, MessageType.ERROR, {
                    "error": "Invalid authentication token"
                })
                return False
            
            # Update connection info
            connection_info = self.connection_info[connection_id]
            connection_info.user_id = user_id
            connection_info.is_authenticated = True
            
            # Send authentication success message
            await self.send_message(connection_id, MessageType.AUTH, {
                "message": "Authentication successful",
                "user_id": user_id,
                "authenticated": True
            })
            
            logger.info(f"Connection {connection_id} authenticated as user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to authenticate connection {connection_id}: {e}")
            return False
    
    def _validate_token(self, token: str) -> bool:
        """Validate authentication token (simplified)"""
        # In real implementation, verify JWT signature and expiration
        return len(token) > 10
    
    async def handle_message(self, connection_id: str, message_data: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        try:
            if connection_id not in self.connection_info:
                return
            
            # Update last activity
            self.connection_info[connection_id].last_activity = time.time()
            
            # Parse message
            message_type_str = message_data.get("type")
            if not message_type_str:
                await self.send_message(connection_id, MessageType.ERROR, {
                    "error": "Missing message type"
                })
                return
            
            try:
                message_type = MessageType(message_type_str)
            except ValueError:
                await self.send_message(connection_id, MessageType.ERROR, {
                    "error": f"Invalid message type: {message_type_str}"
                })
                return
            
            data = message_data.get("data", {})
            
            # Update statistics
            self.stats["messages_received"] += 1
            
            # Handle message based on type
            await self._handle_message_by_type(connection_id, message_type, data)
            
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            await self.send_message(connection_id, MessageType.ERROR, {
                "error": "Internal server error"
            })
    
    async def _handle_message_by_type(self, connection_id: str, message_type: MessageType, 
                                    data: Dict[str, Any]):
        """Handle message based on its type"""
        if message_type == MessageType.PING:
            await self.send_message(connection_id, MessageType.PONG, {
                "message": "pong",
                "server_time": time.time()
            })
        
        elif message_type == MessageType.AUTH:
            user_id = data.get("user_id")
            token = data.get("token")
            if user_id and token:
                await self.authenticate_connection(connection_id, user_id, token)
            else:
                await self.send_message(connection_id, MessageType.ERROR, {
                    "error": "Missing user_id or token"
                })
        
        elif message_type == MessageType.SUBSCRIBE:
            room_id = data.get("room_id")
            if room_id:
                await self.join_room(connection_id, room_id)
            else:
                await self.send_message(connection_id, MessageType.ERROR, {
                    "error": "Missing room_id"
                })
        
        elif message_type == MessageType.UNSUBSCRIBE:
            room_id = data.get("room_id")
            if room_id:
                await self.leave_room(connection_id, room_id)
            else:
                await self.send_message(connection_id, MessageType.ERROR, {
                    "error": "Missing room_id"
                })
        
        elif message_type == MessageType.HEARTBEAT:
            await self.send_message(connection_id, MessageType.HEARTBEAT, {
                "message": "heartbeat_ack",
                "server_time": time.time()
            })
        
        else:
            # Call registered message handlers
            if message_type in self.message_handlers:
                for handler in self.message_handlers[message_type]:
                    try:
                        await handler(connection_id, data)
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for a specific message type"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
        logger.info(f"Registered message handler for {message_type.value}")
    
    async def _send_message(self, connection_id: str, message_type: MessageType, 
                          data: Dict[str, Any]):
        """Internal method to send message (used by other methods)"""
        await self.send_message(connection_id, message_type, data)
    
    async def _heartbeat_loop(self):
        """Background task to send heartbeat messages"""
        while True:
            try:
                current_time = time.time()
                inactive_connections = []
                
                for connection_id, connection_info in self.connection_info.items():
                    # Check if connection is inactive
                    if current_time - connection_info.last_activity > self.connection_timeout:
                        inactive_connections.append(connection_id)
                        continue
                    
                    # Send heartbeat
                    await self.send_message(connection_id, MessageType.HEARTBEAT, {
                        "message": "heartbeat",
                        "server_time": current_time
                    })
                
                # Disconnect inactive connections
                for connection_id in inactive_connections:
                    await self.disconnect(connection_id, "timeout")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background task to cleanup resources"""
        while True:
            try:
                # Clean up disconnected connections
                disconnected_connections = []
                for connection_id, websocket in self.active_connections.items():
                    try:
                        # Try to send a ping to check if connection is alive
                        await websocket.ping()
                    except:
                        disconnected_connections.append(connection_id)
                
                for connection_id in disconnected_connections:
                    await self.disconnect(connection_id, "connection_lost")
                
                # Clean up empty rooms
                empty_rooms = [
                    room_id for room_id, connections in self.room_subscriptions.items()
                    if not connections
                ]
                for room_id in empty_rooms:
                    del self.room_subscriptions[room_id]
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_connections": self.stats["total_connections"],
            "messages_sent": self.stats["messages_sent"],
            "messages_received": self.stats["messages_received"],
            "rooms_created": self.stats["rooms_created"],
            "authentication_failures": self.stats["authentication_failures"],
            "rooms_count": len(self.room_subscriptions),
            "authenticated_connections": sum(
                1 for conn_info in self.connection_info.values() 
                if conn_info.is_authenticated
            )
        }
    
    def get_room_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific room"""
        if room_id not in self.room_subscriptions:
            return None
        
        connections = self.room_subscriptions[room_id]
        return {
            "room_id": room_id,
            "connection_count": len(connections),
            "connections": list(connections),
            "created_at": time.time()  # Simplified - in real implementation, track creation time
        }
    
    def get_user_connections(self, user_id: str) -> List[str]:
        """Get all connection IDs for a user"""
        return [
            conn_id for conn_id, conn_info in self.connection_info.items()
            if conn_info.user_id == user_id
        ]


# Global WebSocket manager instance
websocket_manager = WebSocketManager()
