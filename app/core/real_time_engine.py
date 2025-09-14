"""
Real-time data processing and streaming engine
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
import uuid
import secrets

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams"""
    MARKET_DATA = "market_data"
    PRICE_UPDATES = "price_updates"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    ANALYTICS = "analytics"
    NOTIFICATIONS = "notifications"
    SYSTEM_METRICS = "system_metrics"


@dataclass
class StreamSubscription:
    """Stream subscription information"""
    id: str
    user_id: str
    stream_type: StreamType
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class StreamMessage:
    """Real-time stream message"""
    id: str
    stream_type: StreamType
    data: Dict[str, Any]
    timestamp: datetime
    filters: Dict[str, Any] = field(default_factory=dict)


class RealTimeEngine:
    """Advanced real-time data processing and streaming engine"""
    
    def __init__(self):
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.websocket_connections: Dict[str, WebSocketServerProtocol] = {}
        self.message_handlers: Dict[StreamType, List[Callable]] = {}
        self.data_cache: Dict[str, Any] = {}
        self.stream_processors: Dict[StreamType, asyncio.Task] = {}
        self.is_running = False
        self._initialize_handlers()
    
    def _initialize_handlers(self):
        """Initialize message handlers for each stream type"""
        for stream_type in StreamType:
            self.message_handlers[stream_type] = []
    
    async def start(self):
        """Start the real-time engine"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting real-time engine")
        
        # Start stream processors
        for stream_type in StreamType:
            processor_task = asyncio.create_task(
                self._process_stream(stream_type)
            )
            self.stream_processors[stream_type] = processor_task
        
        logger.info("Real-time engine started successfully")
    
    async def stop(self):
        """Stop the real-time engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping real-time engine")
        
        # Cancel all stream processors
        for task in self.stream_processors.values():
            task.cancel()
        
        # Close all WebSocket connections
        for connection in self.websocket_connections.values():
            await connection.close()
        
        self.stream_processors.clear()
        self.websocket_connections.clear()
        
        logger.info("Real-time engine stopped")
    
    async def subscribe(
        self,
        user_id: str,
        stream_type: StreamType,
        filters: Optional[Dict[str, Any]] = None,
        websocket: Optional[WebSocketServerProtocol] = None
    ) -> str:
        """Subscribe to a real-time stream"""
        try:
            subscription_id = str(uuid.uuid4())
            
            subscription = StreamSubscription(
                id=subscription_id,
                user_id=user_id,
                stream_type=stream_type,
                filters=filters or {}
            )
            
            self.subscriptions[subscription_id] = subscription
            
            if websocket:
                self.websocket_connections[subscription_id] = websocket
            
            logger.info(f"User {user_id} subscribed to {stream_type.value} stream")
            
            return subscription_id
            
        except Exception as e:
            logger.error(f"Failed to create subscription: {e}")
            raise
    
    async def unsubscribe(self, subscription_id: str):
        """Unsubscribe from a stream"""
        try:
            if subscription_id in self.subscriptions:
                subscription = self.subscriptions[subscription_id]
                logger.info(f"User {subscription.user_id} unsubscribed from {subscription.stream_type.value}")
                
                del self.subscriptions[subscription_id]
            
            if subscription_id in self.websocket_connections:
                connection = self.websocket_connections[subscription_id]
                await connection.close()
                del self.websocket_connections[subscription_id]
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")
    
    async def publish_message(
        self,
        stream_type: StreamType,
        data: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None
    ):
        """Publish a message to a stream"""
        try:
            message = StreamMessage(
                id=str(uuid.uuid4()),
                stream_type=stream_type,
                data=data,
                timestamp=datetime.now(),
                filters=filters or {}
            )
            
            # Cache the message
            self._cache_message(message)
            
            # Process message through handlers
            await self._process_message(message)
            
            # Send to WebSocket subscribers
            await self._broadcast_to_subscribers(message)
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
    
    async def _process_stream(self, stream_type: StreamType):
        """Process a specific stream type"""
        while self.is_running:
            try:
                # Generate mock data for the stream
                data = await self._generate_stream_data(stream_type)
                
                if data:
                    await self.publish_message(stream_type, data)
                
                # Wait before next update
                await asyncio.sleep(self._get_stream_interval(stream_type))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing {stream_type.value} stream: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _generate_stream_data(self, stream_type: StreamType) -> Optional[Dict[str, Any]]:
        """Generate mock data for a stream type"""
        try:
            if stream_type == StreamType.MARKET_DATA:
                return {
                    "market_id": f"market_{secrets.randbelow(100)}",
                    "price": 100.0 + secrets.randbelow(50),
                    "volume": 1000 + secrets.randbelow(5000),
                    "change": secrets.randbelow(20) - 10,
                    "change_percent": (secrets.randbelow(20) - 10) / 100
                }
            
            elif stream_type == StreamType.PRICE_UPDATES:
                return {
                    "symbol": f"SYMBOL_{secrets.randbelow(10)}",
                    "price": 50.0 + secrets.randbelow(100),
                    "timestamp": datetime.now().isoformat(),
                    "source": "real_time_feed"
                }
            
            elif stream_type == StreamType.ORDER_BOOK:
                return {
                    "market_id": f"market_{secrets.randbelow(100)}",
                    "bids": [
                        {"price": 99.5, "quantity": 100},
                        {"price": 99.0, "quantity": 200}
                    ],
                    "asks": [
                        {"price": 100.5, "quantity": 150},
                        {"price": 101.0, "quantity": 250}
                    ],
                    "timestamp": datetime.now().isoformat()
                }
            
            elif stream_type == StreamType.TRADES:
                return {
                    "trade_id": str(uuid.uuid4()),
                    "market_id": f"market_{secrets.randbelow(100)}",
                    "price": 100.0 + secrets.randbelow(10),
                    "quantity": 10 + secrets.randbelow(100),
                    "side": "buy" if secrets.randbelow(2) else "sell",
                    "timestamp": datetime.now().isoformat()
                }
            
            elif stream_type == StreamType.ANALYTICS:
                return {
                    "metric": "market_volume",
                    "value": 10000 + secrets.randbelow(50000),
                    "trend": "increasing" if secrets.randbelow(2) else "decreasing",
                    "timestamp": datetime.now().isoformat()
                }
            
            elif stream_type == StreamType.NOTIFICATIONS:
                return {
                    "notification_id": str(uuid.uuid4()),
                    "type": "market_alert",
                    "title": "Price Alert",
                    "message": "Market price has reached your target",
                    "timestamp": datetime.now().isoformat()
                }
            
            elif stream_type == StreamType.SYSTEM_METRICS:
                return {
                    "cpu_usage": secrets.randbelow(100),
                    "memory_usage": secrets.randbelow(100),
                    "active_connections": len(self.websocket_connections),
                    "messages_per_second": secrets.randbelow(1000),
                    "timestamp": datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate stream data: {e}")
            return None
    
    def _get_stream_interval(self, stream_type: StreamType) -> float:
        """Get update interval for a stream type"""
        intervals = {
            StreamType.MARKET_DATA: 1.0,      # 1 second
            StreamType.PRICE_UPDATES: 0.5,    # 0.5 seconds
            StreamType.ORDER_BOOK: 2.0,       # 2 seconds
            StreamType.TRADES: 0.1,           # 0.1 seconds
            StreamType.ANALYTICS: 5.0,        # 5 seconds
            StreamType.NOTIFICATIONS: 10.0,   # 10 seconds
            StreamType.SYSTEM_METRICS: 3.0    # 3 seconds
        }
        return intervals.get(stream_type, 1.0)
    
    def _cache_message(self, message: StreamMessage):
        """Cache the latest message for each stream type"""
        cache_key = f"{message.stream_type.value}_latest"
        self.data_cache[cache_key] = {
            "message": message,
            "cached_at": datetime.now()
        }
    
    async def _process_message(self, message: StreamMessage):
        """Process message through registered handlers"""
        handlers = self.message_handlers.get(message.stream_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
    
    async def _broadcast_to_subscribers(self, message: StreamMessage):
        """Broadcast message to WebSocket subscribers"""
        try:
            # Find subscribers for this stream type
            subscribers = [
                sub for sub in self.subscriptions.values()
                if sub.stream_type == message.stream_type
            ]
            
            # Filter subscribers based on message filters
            filtered_subscribers = []
            for subscriber in subscribers:
                if self._matches_filters(message, subscriber.filters):
                    filtered_subscribers.append(subscriber)
            
            # Send to WebSocket connections
            for subscriber in filtered_subscribers:
                if subscriber.id in self.websocket_connections:
                    connection = self.websocket_connections[subscriber.id]
                    try:
                        await connection.send(json.dumps({
                            "type": message.stream_type.value,
                            "data": message.data,
                            "timestamp": message.timestamp.isoformat(),
                            "message_id": message.id
                        }))
                        
                        # Update last activity
                        subscriber.last_activity = datetime.now()
                        
                    except websockets.exceptions.ConnectionClosed:
                        # Remove closed connections
                        await self.unsubscribe(subscriber.id)
                    except Exception as e:
                        logger.error(f"Error sending to subscriber {subscriber.id}: {e}")
            
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
    
    def _matches_filters(self, message: StreamMessage, filters: Dict[str, Any]) -> bool:
        """Check if message matches subscription filters"""
        if not filters:
            return True
        
        try:
            for key, value in filters.items():
                if key in message.data:
                    if isinstance(value, list):
                        if message.data[key] not in value:
                            return False
                    else:
                        if message.data[key] != value:
                            return False
                else:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking filters: {e}")
            return True
    
    def register_handler(self, stream_type: StreamType, handler: Callable):
        """Register a message handler for a stream type"""
        if stream_type not in self.message_handlers:
            self.message_handlers[stream_type] = []
        
        self.message_handlers[stream_type].append(handler)
        logger.info(f"Registered handler for {stream_type.value}")
    
    def unregister_handler(self, stream_type: StreamType, handler: Callable):
        """Unregister a message handler"""
        if stream_type in self.message_handlers:
            if handler in self.message_handlers[stream_type]:
                self.message_handlers[stream_type].remove(handler)
                logger.info(f"Unregistered handler for {stream_type.value}")
    
    async def get_stream_status(self) -> Dict[str, Any]:
        """Get real-time engine status"""
        return {
            "is_running": self.is_running,
            "total_subscriptions": len(self.subscriptions),
            "active_connections": len(self.websocket_connections),
            "stream_types": [stream_type.value for stream_type in StreamType],
            "cached_data": {
                stream_type.value: "available" if f"{stream_type.value}_latest" in self.data_cache else "none"
                for stream_type in StreamType
            },
            "uptime": datetime.now().isoformat()
        }
    
    async def get_latest_data(self, stream_type: StreamType) -> Optional[Dict[str, Any]]:
        """Get latest cached data for a stream type"""
        cache_key = f"{stream_type.value}_latest"
        cached = self.data_cache.get(cache_key)
        
        if cached:
            return {
                "data": cached["message"].data,
                "timestamp": cached["message"].timestamp.isoformat(),
                "cached_at": cached["cached_at"].isoformat()
            }
        
        return None
    
    async def cleanup_inactive_subscriptions(self, timeout_minutes: int = 30):
        """Clean up inactive subscriptions"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=timeout_minutes)
            
            inactive_subscriptions = [
                sub_id for sub_id, sub in self.subscriptions.items()
                if sub.last_activity < cutoff_time
            ]
            
            for sub_id in inactive_subscriptions:
                await self.unsubscribe(sub_id)
                logger.info(f"Cleaned up inactive subscription: {sub_id}")
            
            return len(inactive_subscriptions)
            
        except Exception as e:
            logger.error(f"Error cleaning up subscriptions: {e}")
            return 0


# Global real-time engine
real_time_engine = RealTimeEngine()
