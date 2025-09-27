"""
Advanced Notification Manager
Manages multi-channel notifications with scheduling and personalization
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

from app.core.enhanced_cache import enhanced_cache
from app.core.database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(Enum):
    """Notification status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NotificationChannel(Enum):
    """Notification channels"""
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"
    WEBSOCKET = "websocket"
    IN_APP = "in_app"


@dataclass
class NotificationRecipient:
    """Notification recipient information"""
    user_id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


@dataclass
class NotificationTemplate:
    """Notification template"""
    template_id: str
    name: str
    subject: str
    content: str
    channels: List[NotificationChannel]
    variables: List[str]
    created_at: float
    updated_at: float


@dataclass
class Notification:
    """Notification data structure"""
    notification_id: str
    template_id: str
    recipient: NotificationRecipient
    channels: List[NotificationChannel]
    priority: NotificationPriority
    data: Dict[str, Any]
    scheduled_at: Optional[float] = None
    expires_at: Optional[float] = None
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: float = None
    sent_at: Optional[float] = None
    delivery_attempts: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None


class NotificationManager:
    """Advanced notification management system"""
    
    def __init__(self):
        self.templates: Dict[str, NotificationTemplate] = {}
        self.pending_notifications: Dict[str, Notification] = {}
        self.notification_channels = {}
        self.scheduled_notifications = {}
        
        # Configuration
        self.max_retry_attempts = 3
        self.retry_delay = 300  # 5 minutes
        self.batch_size = 100
        self.rate_limit_per_user = 100  # notifications per hour
        
        # Statistics
        self.stats = {
            "notifications_sent": 0,
            "notifications_failed": 0,
            "notifications_delivered": 0,
            "templates_used": 0,
            "channels_used": {}
        }
        
        # Start background tasks
        asyncio.create_task(self._process_notifications_loop())
        asyncio.create_task(self._cleanup_loop())
    
    async def create_template(self, template_data: Dict[str, Any]) -> str:
        """Create a new notification template"""
        try:
            template_id = str(uuid.uuid4())
            current_time = time.time()
            
            template = NotificationTemplate(
                template_id=template_id,
                name=template_data["name"],
                subject=template_data["subject"],
                content=template_data["content"],
                channels=[NotificationChannel(ch) for ch in template_data.get("channels", [])],
                variables=template_data.get("variables", []),
                created_at=current_time,
                updated_at=current_time
            )
            
            # Store template
            self.templates[template_id] = template
            
            # Store in database
            await self._store_template(template)
            
            logger.info(f"Created notification template: {template_id}")
            return template_id
            
        except Exception as e:
            logger.error(f"Failed to create notification template: {e}")
            raise e
    
    async def send_notification(self, template_id: str, recipient_data: Dict[str, Any], 
                              data: Dict[str, Any], channels: Optional[List[str]] = None,
                              priority: str = "normal", scheduled_at: Optional[float] = None) -> str:
        """Send a notification"""
        try:
            if template_id not in self.templates:
                raise ValueError(f"Template {template_id} not found")
            
            template = self.templates[template_id]
            
            # Create recipient
            recipient = NotificationRecipient(
                user_id=recipient_data["user_id"],
                email=recipient_data.get("email"),
                phone=recipient_data.get("phone"),
                push_token=recipient_data.get("push_token"),
                preferences=recipient_data.get("preferences", {})
            )
            
            # Determine channels
            if channels:
                notification_channels = [NotificationChannel(ch) for ch in channels]
            else:
                notification_channels = template.channels
            
            # Create notification
            notification_id = str(uuid.uuid4())
            notification = Notification(
                notification_id=notification_id,
                template_id=template_id,
                recipient=recipient,
                channels=notification_channels,
                priority=NotificationPriority(priority),
                data=data,
                scheduled_at=scheduled_at,
                expires_at=time.time() + (7 * 24 * 3600) if not scheduled_at else scheduled_at + (7 * 24 * 3600),  # 7 days
                created_at=time.time()
            )
            
            # Check rate limits
            if not await self._check_rate_limit(recipient.user_id):
                raise Exception("Rate limit exceeded for user")
            
            # Store notification
            self.pending_notifications[notification_id] = notification
            await self._store_notification(notification)
            
            # Schedule or send immediately
            if scheduled_at and scheduled_at > time.time():
                self.scheduled_notifications[notification_id] = scheduled_at
                logger.info(f"Scheduled notification {notification_id} for {scheduled_at}")
            else:
                asyncio.create_task(self._process_notification(notification))
            
            return notification_id
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            raise e
    
    async def send_bulk_notifications(self, template_id: str, recipients: List[Dict[str, Any]], 
                                    data: Dict[str, Any], channels: Optional[List[str]] = None,
                                    priority: str = "normal") -> List[str]:
        """Send notifications to multiple recipients"""
        try:
            notification_ids = []
            
            # Process in batches
            for i in range(0, len(recipients), self.batch_size):
                batch = recipients[i:i + self.batch_size]
                
                # Create tasks for batch
                tasks = []
                for recipient_data in batch:
                    task = asyncio.create_task(
                        self.send_notification(template_id, recipient_data, data, channels, priority)
                    )
                    tasks.append(task)
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                for result in batch_results:
                    if isinstance(result, str):
                        notification_ids.append(result)
                    elif isinstance(result, Exception):
                        logger.error(f"Failed to send notification in batch: {result}")
            
            logger.info(f"Sent {len(notification_ids)} bulk notifications")
            return notification_ids
            
        except Exception as e:
            logger.error(f"Failed to send bulk notifications: {e}")
            raise e
    
    async def _process_notification(self, notification: Notification):
        """Process a single notification"""
        try:
            # Update status
            notification.status = NotificationStatus.PENDING
            notification.delivery_attempts += 1
            
            # Process each channel
            for channel in notification.channels:
                try:
                    success = await self._send_to_channel(notification, channel)
                    if success:
                        logger.info(f"Sent notification {notification.notification_id} via {channel.value}")
                    else:
                        logger.warning(f"Failed to send notification {notification.notification_id} via {channel.value}")
                except Exception as e:
                    logger.error(f"Error sending to channel {channel.value}: {e}")
            
            # Update notification status
            if notification.delivery_attempts >= notification.max_attempts:
                notification.status = NotificationStatus.FAILED
                notification.error_message = "Max delivery attempts reached"
            else:
                notification.status = NotificationStatus.SENT
                notification.sent_at = time.time()
            
            # Update statistics
            if notification.status == NotificationStatus.SENT:
                self.stats["notifications_sent"] += 1
            else:
                self.stats["notifications_failed"] += 1
            
            # Update in database
            await self._update_notification(notification)
            
        except Exception as e:
            logger.error(f"Error processing notification {notification.notification_id}: {e}")
            notification.status = NotificationStatus.FAILED
            notification.error_message = str(e)
            await self._update_notification(notification)
    
    async def _send_to_channel(self, notification: Notification, 
                             channel: NotificationChannel) -> bool:
        """Send notification to a specific channel"""
        try:
            # Get channel handler
            channel_handler = self.notification_channels.get(channel)
            if not channel_handler:
                logger.error(f"No handler found for channel {channel.value}")
                return False
            
            # Get template
            template = self.templates.get(notification.template_id)
            if not template:
                logger.error(f"Template {notification.template_id} not found")
                return False
            
            # Render template
            rendered_content = await self._render_template(template, notification.data)
            
            # Send via channel
            success = await channel_handler.send(
                recipient=notification.recipient,
                subject=rendered_content["subject"],
                content=rendered_content["content"],
                data=notification.data
            )
            
            # Update channel statistics
            if channel.value not in self.stats["channels_used"]:
                self.stats["channels_used"][channel.value] = 0
            self.stats["channels_used"][channel.value] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending to channel {channel.value}: {e}")
            return False
    
    async def _render_template(self, template: NotificationTemplate, 
                             data: Dict[str, Any]) -> Dict[str, str]:
        """Render notification template with data"""
        try:
            # Simple template rendering (in real implementation, use proper templating engine)
            subject = template.subject
            content = template.content
            
            # Replace variables
            for key, value in data.items():
                placeholder = f"{{{key}}}"
                subject = subject.replace(placeholder, str(value))
                content = content.replace(placeholder, str(value))
            
            return {
                "subject": subject,
                "content": content
            }
            
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return {
                "subject": template.subject,
                "content": template.content
            }
    
    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limits"""
        try:
            cache_key = f"notification_rate_limit_{user_id}"
            current_count = await enhanced_cache.get(cache_key) or 0
            
            if current_count >= self.rate_limit_per_user:
                return False
            
            # Increment counter
            await enhanced_cache.set(
                cache_key,
                current_count + 1,
                ttl=3600,  # 1 hour
                tags=["rate_limit", "notifications"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow on error
    
    async def _process_notifications_loop(self):
        """Background task to process pending notifications"""
        while True:
            try:
                current_time = time.time()
                
                # Process scheduled notifications
                ready_notifications = []
                for notification_id, scheduled_time in list(self.scheduled_notifications.items()):
                    if scheduled_time <= current_time:
                        ready_notifications.append(notification_id)
                        del self.scheduled_notifications[notification_id]
                
                # Process ready notifications
                for notification_id in ready_notifications:
                    if notification_id in self.pending_notifications:
                        notification = self.pending_notifications[notification_id]
                        asyncio.create_task(self._process_notification(notification))
                
                # Process pending notifications
                pending_count = len(self.pending_notifications)
                if pending_count > 0:
                    logger.info(f"Processing {pending_count} pending notifications")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in notification processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background task to cleanup old notifications"""
        while True:
            try:
                current_time = time.time()
                expired_notifications = []
                
                # Find expired notifications
                for notification_id, notification in self.pending_notifications.items():
                    if notification.expires_at and notification.expires_at < current_time:
                        expired_notifications.append(notification_id)
                
                # Remove expired notifications
                for notification_id in expired_notifications:
                    del self.pending_notifications[notification_id]
                    logger.info(f"Removed expired notification: {notification_id}")
                
                # Clean up old scheduled notifications
                old_scheduled = [
                    notif_id for notif_id, scheduled_time in self.scheduled_notifications.items()
                    if scheduled_time < current_time - 3600  # 1 hour old
                ]
                for notification_id in old_scheduled:
                    del self.scheduled_notifications[notification_id]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _store_template(self, template: NotificationTemplate):
        """Store template in database"""
        try:
            query = """
            INSERT INTO notification_templates (template_id, name, subject, content, 
                                              channels, variables, created_at, updated_at)
            VALUES (:template_id, :name, :subject, :content, :channels, :variables, 
                   :created_at, :updated_at)
            ON CONFLICT (template_id) 
            DO UPDATE SET name = :name, subject = :subject, content = :content,
                         channels = :channels, variables = :variables, updated_at = :updated_at
            """
            
            with engine.connect() as conn:
                conn.execute(text(query), {
                    "template_id": template.template_id,
                    "name": template.name,
                    "subject": template.subject,
                    "content": template.content,
                    "channels": json.dumps([ch.value for ch in template.channels]),
                    "variables": json.dumps(template.variables),
                    "created_at": template.created_at,
                    "updated_at": template.updated_at
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store template: {e}")
    
    async def _store_notification(self, notification: Notification):
        """Store notification in database"""
        try:
            query = """
            INSERT INTO notifications (notification_id, template_id, user_id, channels, 
                                     priority, data, scheduled_at, expires_at, status, 
                                     created_at, delivery_attempts, max_attempts)
            VALUES (:notification_id, :template_id, :user_id, :channels, :priority, 
                   :data, :scheduled_at, :expires_at, :status, :created_at, 
                   :delivery_attempts, :max_attempts)
            """
            
            with engine.connect() as conn:
                conn.execute(text(query), {
                    "notification_id": notification.notification_id,
                    "template_id": notification.template_id,
                    "user_id": notification.recipient.user_id,
                    "channels": json.dumps([ch.value for ch in notification.channels]),
                    "priority": notification.priority.value,
                    "data": json.dumps(notification.data),
                    "scheduled_at": notification.scheduled_at,
                    "expires_at": notification.expires_at,
                    "status": notification.status.value,
                    "created_at": notification.created_at,
                    "delivery_attempts": notification.delivery_attempts,
                    "max_attempts": notification.max_attempts
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store notification: {e}")
    
    async def _update_notification(self, notification: Notification):
        """Update notification in database"""
        try:
            query = """
            UPDATE notifications 
            SET status = :status, sent_at = :sent_at, delivery_attempts = :delivery_attempts,
                error_message = :error_message
            WHERE notification_id = :notification_id
            """
            
            with engine.connect() as conn:
                conn.execute(text(query), {
                    "notification_id": notification.notification_id,
                    "status": notification.status.value,
                    "sent_at": notification.sent_at,
                    "delivery_attempts": notification.delivery_attempts,
                    "error_message": notification.error_message
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update notification: {e}")
    
    def register_channel(self, channel: NotificationChannel, handler):
        """Register a notification channel handler"""
        self.notification_channels[channel] = handler
        logger.info(f"Registered notification channel: {channel.value}")
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        return {
            "pending_notifications": len(self.pending_notifications),
            "scheduled_notifications": len(self.scheduled_notifications),
            "templates_count": len(self.templates),
            "registered_channels": list(self.notification_channels.keys()),
            "statistics": self.stats
        }
    
    async def get_user_notifications(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get notifications for a user"""
        try:
            query = """
            SELECT notification_id, template_id, channels, priority, data, status, 
                   created_at, sent_at, error_message
            FROM notifications 
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit
            """
            
            with engine.connect() as conn:
                result = conn.execute(text(query), {
                    "user_id": user_id,
                    "limit": limit
                })
                
                notifications = []
                for row in result:
                    notifications.append({
                        "notification_id": row[0],
                        "template_id": row[1],
                        "channels": json.loads(row[2]),
                        "priority": row[3],
                        "data": json.loads(row[4]),
                        "status": row[5],
                        "created_at": row[6],
                        "sent_at": row[7],
                        "error_message": row[8]
                    })
                
                return notifications
                
        except Exception as e:
            logger.error(f"Failed to get user notifications: {e}")
            return []


# Global notification manager instance
notification_manager = NotificationManager()
