"""
Advanced WebSocket Real-time Communication System
Provides sophisticated WebSocket management with rooms, broadcasting, and presence tracking
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import threading

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    PING = "ping"
    PONG = "pong"
    HEARTBEAT = "heartbeat"
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    ROOM_JOIN = "room_join"
    ROOM_LEAVE = "room_leave"
    BROADCAST = "broadcast"
    PRIVATE_MESSAGE = "private_message"
    SYSTEM_MESSAGE = "system_message"

class ConnectionStatus(Enum):
    """Connection status types"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

@dataclass
class WebSocketMessage:
    """Represents a WebSocket message"""
    id: str
    type: MessageType
    content: Any
    sender_id: Optional[str] = None
    recipient_id: Optional[str] = None
    room_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "room_id": self.room_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection"""
    id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    status: ConnectionStatus = ConnectionStatus.CONNECTING
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    rooms: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

@dataclass
class Room:
    """Represents a WebSocket room"""
    id: str
    name: str
    description: str = ""
    max_connections: int = 1000
    created_at: datetime = field(default_factory=datetime.utcnow)
    connections: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_history: deque = field(default_factory=lambda: deque(maxlen=1000))

class AdvancedWebSocketManager:
    """Advanced WebSocket manager with room support and real-time features"""
    
    def __init__(self, max_connections: int = 10000, heartbeat_interval: int = 30):
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Room management
        self.rooms: Dict[str, Room] = {}
        self.connection_rooms: Dict[str, Set[str]] = defaultdict(set)
        
        # Message handling
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.message_history: deque = deque(maxlen=10000)
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_messages": 0,
            "total_rooms": 0,
            "bytes_sent": 0,
            "bytes_received": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._heartbeat_task = None
        self._cleanup_task = None
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # Event loop not running yet
            pass
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Accept WebSocket connection"""
        try:
            await websocket.accept()
            
            # Generate connection ID
            connection_id = str(uuid.uuid4())
            
            # Create connection object
            connection = WebSocketConnection(
                id=connection_id,
                websocket=websocket,
                user_id=user_id,
                status=ConnectionStatus.CONNECTED,
                metadata=metadata or {}
            )
            
            # Store connection
            with self._lock:
                self.connections[connection_id] = connection
                if user_id:
                    self.user_connections[user_id].add(connection_id)
                
                self.stats["total_connections"] += 1
                self.stats["active_connections"] += 1
            
            logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
            
            # Send welcome message
            welcome_message = WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.SYSTEM_MESSAGE,
                content="Connected to WebSocket server",
                metadata={"connection_id": connection_id}
            )
            
            await self._send_message(connection_id, welcome_message)
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {e}")
            raise
    
    async def disconnect(self, connection_id: str, reason: str = "Client disconnect"):
        """Disconnect WebSocket connection"""
        try:
            with self._lock:
                if connection_id not in self.connections:
                    return
                
                connection = self.connections[connection_id]
                connection.status = ConnectionStatus.DISCONNECTING
                
                # Remove from rooms
                for room_id in connection.rooms.copy():
                    await self.leave_room(connection_id, room_id)
                
                # Remove from user connections
                if connection.user_id:
                    self.user_connections[connection.user_id].discard(connection_id)
                    if not self.user_connections[connection.user_id]:
                        del self.user_connections[connection.user_id]
                
                # Close WebSocket
                if connection.websocket.client_state == WebSocketState.CONNECTED:
                    await connection.websocket.close()
                
                # Remove connection
                del self.connections[connection_id]
                self.stats["active_connections"] -= 1
                
                # Clean up connection rooms
                if connection_id in self.connection_rooms:
                    del self.connection_rooms[connection_id]
            
            logger.info(f"WebSocket disconnected: {connection_id} (reason: {reason})")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def send_message(self, connection_id: str, message: Union[str, dict, WebSocketMessage]) -> bool:
        """Send message to specific connection"""
        try:
            if isinstance(message, WebSocketMessage):
                ws_message = message
            else:
                ws_message = WebSocketMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.TEXT if isinstance(message, str) else MessageType.JSON,
                    content=message
                )
            
            return await self._send_message(connection_id, ws_message)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def _send_message(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Internal method to send message"""
        try:
            with self._lock:
                if connection_id not in self.connections:
                    return False
                
                connection = self.connections[connection_id]
                
                if connection.status != ConnectionStatus.CONNECTED:
                    return False
            
            # Send message
            if message.type == MessageType.TEXT:
                await connection.websocket.send_text(message.content)
            elif message.type == MessageType.JSON:
                await connection.websocket.send_json(message.to_dict())
            elif message.type == MessageType.BINARY:
                await connection.websocket.send_bytes(message.content)
            
            # Update statistics
            with self._lock:
                connection.message_count += 1
                connection.last_activity = datetime.utcnow()
                connection.bytes_sent += len(str(message.content))
                self.stats["total_messages"] += 1
                self.stats["bytes_sent"] += len(str(message.content))
            
            # Store in message history
            self.message_history.append(message)
            
            return True
            
        except WebSocketDisconnect:
            await self.disconnect(connection_id, "WebSocket disconnect")
            return False
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            return False
    
    async def broadcast_to_room(self, room_id: str, message: Union[str, dict, WebSocketMessage], 
                               exclude_connection: Optional[str] = None) -> int:
        """Broadcast message to all connections in a room"""
        try:
            if room_id not in self.rooms:
                return 0
            
            room = self.rooms[room_id]
            sent_count = 0
            
            for connection_id in room.connections.copy():
                if connection_id != exclude_connection:
                    if await self.send_message(connection_id, message):
                        sent_count += 1
            
            return sent_count
            
        except Exception as e:
            logger.error(f"Error broadcasting to room {room_id}: {e}")
            return 0
    
    async def broadcast_to_user(self, user_id: str, message: Union[str, dict, WebSocketMessage]) -> int:
        """Broadcast message to all connections of a user"""
        try:
            if user_id not in self.user_connections:
                return 0
            
            sent_count = 0
            
            for connection_id in self.user_connections[user_id].copy():
                if await self.send_message(connection_id, message):
                    sent_count += 1
            
            return sent_count
            
        except Exception as e:
            logger.error(f"Error broadcasting to user {user_id}: {e}")
            return 0
    
    async def create_room(self, room_id: str, name: str, description: str = "", 
                         max_connections: int = 1000, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new room"""
        try:
            with self._lock:
                if room_id in self.rooms:
                    return False
                
                room = Room(
                    id=room_id,
                    name=name,
                    description=description,
                    max_connections=max_connections,
                    metadata=metadata or {}
                )
                
                self.rooms[room_id] = room
                self.stats["total_rooms"] += 1
            
            logger.info(f"Room created: {room_id} ({name})")
            return True
            
        except Exception as e:
            logger.error(f"Error creating room {room_id}: {e}")
            return False
    
    async def join_room(self, connection_id: str, room_id: str) -> bool:
        """Join a connection to a room"""
        try:
            with self._lock:
                if connection_id not in self.connections:
                    return False
                
                if room_id not in self.rooms:
                    return False
                
                connection = self.connections[connection_id]
                room = self.rooms[room_id]
                
                # Check room capacity
                if len(room.connections) >= room.max_connections:
                    return False
                
                # Add to room
                room.connections.add(connection_id)
                connection.rooms.add(room_id)
                self.connection_rooms[connection_id].add(room_id)
            
            # Notify room members
            join_message = WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.USER_JOIN,
                content=f"User joined room {room_id}",
                sender_id=connection.user_id,
                room_id=room_id
            )
            
            await self.broadcast_to_room(room_id, join_message, exclude_connection=connection_id)
            
            logger.info(f"Connection {connection_id} joined room {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error joining room {room_id}: {e}")
            return False
    
    async def leave_room(self, connection_id: str, room_id: str) -> bool:
        """Remove connection from a room"""
        try:
            with self._lock:
                if connection_id not in self.connections:
                    return False
                
                if room_id not in self.rooms:
                    return False
                
                connection = self.connections[connection_id]
                room = self.rooms[room_id]
                
                # Remove from room
                room.connections.discard(connection_id)
                connection.rooms.discard(room_id)
                self.connection_rooms[connection_id].discard(room_id)
            
            # Notify room members
            leave_message = WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.USER_LEAVE,
                content=f"User left room {room_id}",
                sender_id=connection.user_id,
                room_id=room_id
            )
            
            await self.broadcast_to_room(room_id, leave_message, exclude_connection=connection_id)
            
            logger.info(f"Connection {connection_id} left room {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error leaving room {room_id}: {e}")
            return False
    
    async def delete_room(self, room_id: str) -> bool:
        """Delete a room and disconnect all members"""
        try:
            with self._lock:
                if room_id not in self.rooms:
                    return False
                
                room = self.rooms[room_id]
                
                # Notify all members
                delete_message = WebSocketMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.SYSTEM_MESSAGE,
                    content=f"Room {room_id} is being deleted",
                    room_id=room_id
                )
                
                await self.broadcast_to_room(room_id, delete_message)
                
                # Remove all connections from room
                for connection_id in room.connections.copy():
                    await self.leave_room(connection_id, room_id)
                
                # Delete room
                del self.rooms[room_id]
                self.stats["total_rooms"] -= 1
            
            logger.info(f"Room deleted: {room_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting room {room_id}: {e}")
            return False
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type].append(handler)
        logger.info(f"Registered message handler for {message_type.value}")
    
    async def handle_message(self, connection_id: str, message: WebSocketMessage):
        """Handle incoming message"""
        try:
            # Update connection activity
            with self._lock:
                if connection_id in self.connections:
                    connection = self.connections[connection_id]
                    connection.last_activity = datetime.utcnow()
                    connection.bytes_received += len(str(message.content))
                    self.stats["bytes_received"] += len(str(message.content))
            
            # Call registered handlers
            if message.type in self.message_handlers:
                for handler in self.message_handlers[message.type]:
                    try:
                        await handler(connection_id, message)
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}")
            
            # Handle system messages
            if message.type == MessageType.PING:
                pong_message = WebSocketMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.PONG,
                    content="pong"
                )
                await self.send_message(connection_id, pong_message)
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _heartbeat_loop(self):
        """Background heartbeat loop"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send heartbeat to all connections
                heartbeat_message = WebSocketMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.HEARTBEAT,
                    content="heartbeat"
                )
                
                with self._lock:
                    connection_ids = list(self.connections.keys())
                
                for connection_id in connection_ids:
                    await self.send_message(connection_id, heartbeat_message)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean up inactive connections
                current_time = datetime.utcnow()
                inactive_threshold = timedelta(minutes=5)
                
                with self._lock:
                    inactive_connections = [
                        conn_id for conn_id, conn in self.connections.items()
                        if current_time - conn.last_activity > inactive_threshold
                    ]
                
                for connection_id in inactive_connections:
                    await self.disconnect(connection_id, "Inactive connection")
                
                # Clean up empty rooms
                with self._lock:
                    empty_rooms = [
                        room_id for room_id, room in self.rooms.items()
                        if not room.connections
                    ]
                
                for room_id in empty_rooms:
                    await self.delete_room(room_id)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        with self._lock:
            return {
                "connections": {
                    "total": self.stats["total_connections"],
                    "active": self.stats["active_connections"],
                    "by_user": len(self.user_connections)
                },
                "rooms": {
                    "total": self.stats["total_rooms"],
                    "active": len([r for r in self.rooms.values() if r.connections])
                },
                "messages": {
                    "total": self.stats["total_messages"],
                    "history_size": len(self.message_history)
                },
                "bandwidth": {
                    "bytes_sent": self.stats["bytes_sent"],
                    "bytes_received": self.stats["bytes_received"]
                }
            }
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information"""
        with self._lock:
            if connection_id not in self.connections:
                return None
            
            connection = self.connections[connection_id]
            return {
                "id": connection.id,
                "user_id": connection.user_id,
                "status": connection.status.value,
                "connected_at": connection.connected_at.isoformat(),
                "last_activity": connection.last_activity.isoformat(),
                "rooms": list(connection.rooms),
                "message_count": connection.message_count,
                "bytes_sent": connection.bytes_sent,
                "bytes_received": connection.bytes_received,
                "metadata": connection.metadata
            }
    
    def get_room_info(self, room_id: str) -> Optional[Dict[str, Any]]:
        """Get room information"""
        with self._lock:
            if room_id not in self.rooms:
                return None
            
            room = self.rooms[room_id]
            return {
                "id": room.id,
                "name": room.name,
                "description": room.description,
                "max_connections": room.max_connections,
                "current_connections": len(room.connections),
                "created_at": room.created_at.isoformat(),
                "metadata": room.metadata
            }

# Global WebSocket manager instance
websocket_manager = AdvancedWebSocketManager()

# Convenience functions
async def connect_websocket(websocket: WebSocket, user_id: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
    """Connect WebSocket"""
    return await websocket_manager.connect(websocket, user_id, metadata)

async def disconnect_websocket(connection_id: str, reason: str = "Client disconnect"):
    """Disconnect WebSocket"""
    await websocket_manager.disconnect(connection_id, reason)

async def send_websocket_message(connection_id: str, message: Union[str, dict, WebSocketMessage]) -> bool:
    """Send WebSocket message"""
    return await websocket_manager.send_message(connection_id, message)

async def broadcast_to_room(room_id: str, message: Union[str, dict, WebSocketMessage], 
                           exclude_connection: Optional[str] = None) -> int:
    """Broadcast to room"""
    return await websocket_manager.broadcast_to_room(room_id, message, exclude_connection)

async def broadcast_to_user(user_id: str, message: Union[str, dict, WebSocketMessage]) -> int:
    """Broadcast to user"""
    return await websocket_manager.broadcast_to_user(user_id, message)

async def create_websocket_room(room_id: str, name: str, description: str = "", 
                               max_connections: int = 1000, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Create WebSocket room"""
    return await websocket_manager.create_room(room_id, name, description, max_connections, metadata)

async def join_websocket_room(connection_id: str, room_id: str) -> bool:
    """Join WebSocket room"""
    return await websocket_manager.join_room(connection_id, room_id)

async def leave_websocket_room(connection_id: str, room_id: str) -> bool:
    """Leave WebSocket room"""
    return await websocket_manager.leave_room(connection_id, room_id)
