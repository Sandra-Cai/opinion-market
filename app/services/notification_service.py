"""
Notification Service for Opinion Market
Handles user notifications, alerts, and communication
"""

import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc
from fastapi import HTTPException, status

from app.core.database import get_db, get_redis_client
from app.core.config import settings
from app.core.cache import cache
from app.core.logging import log_system_metric
from app.core.websocket import websocket_service
from app.models.user import User
from app.models.market import Market
from app.models.trade import Trade


class NotificationType(str, Enum):
    """Types of notifications"""
    TRADE_EXECUTED = "trade_executed"
    MARKET_CREATED = "market_created"
    MARKET_CLOSED = "market_closed"
    PRICE_ALERT = "price_alert"
    SYSTEM_UPDATE = "system_update"
    SECURITY_ALERT = "security_alert"
    MARKET_TRENDING = "market_trending"
    USER_MENTION = "user_mention"
    FOLLOW_UPDATE = "follow_update"
    ACHIEVEMENT = "achievement"


class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notification data structure"""
    notification_id: str
    user_id: int
    notification_type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Dict[str, Any]
    created_at: datetime
    read_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivery_methods: List[str] = None
    
    def __post_init__(self):
        if self.delivery_methods is None:
            self.delivery_methods = ["websocket"]


@dataclass
class NotificationTemplate:
    """Notification template data structure"""
    template_id: str
    notification_type: NotificationType
    title_template: str
    message_template: str
    variables: List[str]
    delivery_methods: List[str]


class NotificationService:
    """Service for notification operations"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.cache_ttl = 300  # 5 minutes
        self.notification_templates = self._load_notification_templates()
        
    def _load_notification_templates(self) -> Dict[NotificationType, NotificationTemplate]:
        """Load notification templates"""
        templates = {}
        
        # Trade executed template
        templates[NotificationType.TRADE_EXECUTED] = NotificationTemplate(
            template_id="trade_executed",
            notification_type=NotificationType.TRADE_EXECUTED,
            title_template="Trade Executed",
            message_template="Your {trade_type} order for {amount} shares of {market_title} has been executed at {price}.",
            variables=["trade_type", "amount", "market_title", "price"],
            delivery_methods=["websocket", "email"]
        )
        
        # Market created template
        templates[NotificationType.MARKET_CREATED] = NotificationTemplate(
            template_id="market_created",
            notification_type=NotificationType.MARKET_CREATED,
            title_template="New Market Created",
            message_template="A new market '{market_title}' has been created in {category}.",
            variables=["market_title", "category"],
            delivery_methods=["websocket", "email"]
        )
        
        # Price alert template
        templates[NotificationType.PRICE_ALERT] = NotificationTemplate(
            template_id="price_alert",
            notification_type=NotificationType.PRICE_ALERT,
            title_template="Price Alert",
            message_template="Price alert triggered for {market_title}: {current_price} {condition} {target_price}.",
            variables=["market_title", "current_price", "condition", "target_price"],
            delivery_methods=["websocket", "email", "push"]
        )
        
        # System update template
        templates[NotificationType.SYSTEM_UPDATE] = NotificationTemplate(
            template_id="system_update",
            notification_type=NotificationType.SYSTEM_UPDATE,
            title_template="System Update",
            message_template="{message}",
            variables=["message"],
            delivery_methods=["websocket", "email"]
        )
        
        return templates
    
    async def send_notification(
        self,
        user_id: int,
        notification_type: NotificationType,
        title: str,
        message: str,
        data: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        delivery_methods: Optional[List[str]] = None
    ) -> Notification:
        """Send notification to a user"""
        try:
            # Create notification
            notification = Notification(
                notification_id=str(uuid.uuid4()),
                user_id=user_id,
                notification_type=notification_type,
                priority=priority,
                title=title,
                message=message,
                data=data,
                created_at=datetime.utcnow(),
                delivery_methods=delivery_methods or ["websocket"]
            )
            
            # Store notification
            await self._store_notification(notification)
            
            # Send via specified methods
            for method in notification.delivery_methods:
                await self._send_via_method(notification, method)
            
            # Log notification
            log_system_metric("notification_sent", 1, {
                "notification_id": notification.notification_id,
                "user_id": user_id,
                "type": notification_type.value,
                "priority": priority.value
            })
            
            return notification
            
        except Exception as e:
            log_system_metric("notification_send_error", 1, {"error": str(e)})
            raise
    
    async def send_notification_from_template(
        self,
        user_id: int,
        notification_type: NotificationType,
        variables: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.MEDIUM
    ) -> Notification:
        """Send notification using a template"""
        try:
            if notification_type not in self.notification_templates:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No template found for notification type: {notification_type}"
                )
            
            template = self.notification_templates[notification_type]
            
            # Render template
            title = template.title_template.format(**variables)
            message = template.message_template.format(**variables)
            
            # Send notification
            return await self.send_notification(
                user_id=user_id,
                notification_type=notification_type,
                title=title,
                message=message,
                data=variables,
                priority=priority,
                delivery_methods=template.delivery_methods
            )
            
        except Exception as e:
            log_system_metric("template_notification_error", 1, {"error": str(e)})
            raise
    
    async def get_user_notifications(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 20,
        unread_only: bool = False,
        db: Session = None
    ) -> Tuple[List[Notification], int]:
        """Get user notifications with pagination"""
        try:
            # Get notifications from cache first
            cache_key = f"user_notifications:{user_id}:{skip}:{limit}:{unread_only}"
            cached_notifications = cache.get(cache_key)
            if cached_notifications:
                return cached_notifications
            
            # Get notifications from database
            notifications = []
            total = 0
            
            # In a real implementation, you'd have a notifications table
            # For now, we'll return empty results
            notifications = []
            total = 0
            
            # Cache results
            cache.set(cache_key, (notifications, total), ttl=self.cache_ttl)
            
            return notifications, total
            
        except Exception as e:
            log_system_metric("get_notifications_error", 1, {"error": str(e)})
            raise
    
    async def mark_notification_read(self, notification_id: str, user_id: int) -> bool:
        """Mark notification as read"""
        try:
            # In a real implementation, you'd update the database
            # For now, we'll just log the action
            log_system_metric("notification_marked_read", 1, {
                "notification_id": notification_id,
                "user_id": user_id
            })
            
            # Clear cache
            cache.delete(f"user_notifications:{user_id}")
            
            return True
            
        except Exception as e:
            log_system_metric("mark_notification_read_error", 1, {"error": str(e)})
            raise
    
    async def mark_all_notifications_read(self, user_id: int) -> bool:
        """Mark all notifications as read for a user"""
        try:
            # In a real implementation, you'd update the database
            # For now, we'll just log the action
            log_system_metric("all_notifications_marked_read", 1, {
                "user_id": user_id
            })
            
            # Clear cache
            cache.delete(f"user_notifications:{user_id}")
            
            return True
            
        except Exception as e:
            log_system_metric("mark_all_notifications_read_error", 1, {"error": str(e)})
            raise
    
    async def get_notification_stats(self, user_id: int) -> Dict[str, Any]:
        """Get notification statistics for a user"""
        try:
            # In a real implementation, you'd query the database
            # For now, we'll return mock data
            return {
                "total_notifications": 0,
                "unread_notifications": 0,
                "notifications_by_type": {},
                "notifications_by_priority": {}
            }
            
        except Exception as e:
            log_system_metric("notification_stats_error", 1, {"error": str(e)})
            raise
    
    async def _store_notification(self, notification: Notification):
        """Store notification in database and cache"""
        try:
            # In a real implementation, you'd store in database
            # For now, we'll just store in cache
            cache_key = f"notification:{notification.notification_id}"
            cache.set(cache_key, notification, ttl=7 * 24 * 3600)  # 7 days
            
            # Add to user's notification list
            user_notifications_key = f"user_notifications_list:{notification.user_id}"
            notifications = cache.get(user_notifications_key) or []
            notifications.append(notification.notification_id)
            cache.set(user_notifications_key, notifications, ttl=7 * 24 * 3600)
            
        except Exception as e:
            log_system_metric("store_notification_error", 1, {"error": str(e)})
            raise
    
    async def _send_via_method(self, notification: Notification, method: str):
        """Send notification via specific method"""
        try:
            if method == "websocket":
                await self._send_via_websocket(notification)
            elif method == "email":
                await self._send_via_email(notification)
            elif method == "push":
                await self._send_via_push(notification)
            elif method == "sms":
                await self._send_via_sms(notification)
            
        except Exception as e:
            log_system_metric("send_via_method_error", 1, {
                "method": method,
                "error": str(e)
            })
    
    async def _send_via_websocket(self, notification: Notification):
        """Send notification via WebSocket"""
        try:
            await websocket_service.send_user_notification(
                notification.user_id,
                {
                    "notification_id": notification.notification_id,
                    "type": notification.notification_type.value,
                    "priority": notification.priority.value,
                    "title": notification.title,
                    "message": notification.message,
                    "data": notification.data,
                    "created_at": notification.created_at.isoformat()
                }
            )
            
        except Exception as e:
            log_system_metric("websocket_notification_error", 1, {"error": str(e)})
    
    async def _send_via_email(self, notification: Notification):
        """Send notification via email"""
        try:
            # In a real implementation, you'd send actual emails
            # For now, we'll just log the action
            log_system_metric("email_notification_sent", 1, {
                "notification_id": notification.notification_id,
                "user_id": notification.user_id
            })
            
        except Exception as e:
            log_system_metric("email_notification_error", 1, {"error": str(e)})
    
    async def _send_via_push(self, notification: Notification):
        """Send notification via push notification"""
        try:
            # In a real implementation, you'd send push notifications
            # For now, we'll just log the action
            log_system_metric("push_notification_sent", 1, {
                "notification_id": notification.notification_id,
                "user_id": notification.user_id
            })
            
        except Exception as e:
            log_system_metric("push_notification_error", 1, {"error": str(e)})
    
    async def _send_via_sms(self, notification: Notification):
        """Send notification via SMS"""
        try:
            # In a real implementation, you'd send SMS
            # For now, we'll just log the action
            log_system_metric("sms_notification_sent", 1, {
                "notification_id": notification.notification_id,
                "user_id": notification.user_id
            })
            
        except Exception as e:
            log_system_metric("sms_notification_error", 1, {"error": str(e)})
    
    async def send_trade_notification(self, user_id: int, trade_data: Dict[str, Any]):
        """Send trade execution notification"""
        try:
            await self.send_notification_from_template(
                user_id=user_id,
                notification_type=NotificationType.TRADE_EXECUTED,
                variables=trade_data,
                priority=NotificationPriority.MEDIUM
            )
            
        except Exception as e:
            log_system_metric("trade_notification_error", 1, {"error": str(e)})
    
    async def send_market_notification(self, user_id: int, market_data: Dict[str, Any]):
        """Send market-related notification"""
        try:
            await self.send_notification_from_template(
                user_id=user_id,
                notification_type=NotificationType.MARKET_CREATED,
                variables=market_data,
                priority=NotificationPriority.LOW
            )
            
        except Exception as e:
            log_system_metric("market_notification_error", 1, {"error": str(e)})
    
    async def send_price_alert(self, user_id: int, alert_data: Dict[str, Any]):
        """Send price alert notification"""
        try:
            await self.send_notification_from_template(
                user_id=user_id,
                notification_type=NotificationType.PRICE_ALERT,
                variables=alert_data,
                priority=NotificationPriority.HIGH
            )
            
        except Exception as e:
            log_system_metric("price_alert_error", 1, {"error": str(e)})
    
    async def send_system_notification(self, message: str, priority: NotificationPriority = NotificationPriority.MEDIUM):
        """Send system-wide notification"""
        try:
            # Get all active users
            # In a real implementation, you'd query the database
            active_users = []  # Mock data
            
            for user_id in active_users:
                await self.send_notification(
                    user_id=user_id,
                    notification_type=NotificationType.SYSTEM_UPDATE,
                    title="System Update",
                    message=message,
                    data={"message": message},
                    priority=priority,
                    delivery_methods=["websocket", "email"]
                )
            
        except Exception as e:
            log_system_metric("system_notification_error", 1, {"error": str(e)})


# Global notification service instance
notification_service = NotificationService()