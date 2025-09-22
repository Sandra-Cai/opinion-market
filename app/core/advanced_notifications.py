"""
Advanced Notification System
Provides sophisticated notification management with multiple channels, templates, and scheduling
"""

import asyncio
import json
import smtplib
import ssl
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from collections import defaultdict, deque
import uuid
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    IN_APP = "in_app"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

class NotificationStatus(Enum):
    """Notification status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SCHEDULED = "scheduled"

class NotificationType(Enum):
    """Notification types"""
    SYSTEM = "system"
    MARKET = "market"
    TRADE = "trade"
    ORDER = "order"
    USER = "user"
    SECURITY = "security"
    MAINTENANCE = "maintenance"
    PROMOTIONAL = "promotional"

@dataclass
class NotificationTemplate:
    """Notification template"""
    id: str
    name: str
    type: NotificationType
    channel: NotificationChannel
    subject: str
    content: str
    html_content: Optional[str] = None
    variables: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Notification:
    """Notification instance"""
    id: str
    user_id: str
    type: NotificationType
    channel: NotificationChannel
    priority: NotificationPriority
    subject: str
    content: str
    html_content: Optional[str] = None
    status: NotificationStatus = NotificationStatus.PENDING
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class NotificationPreferences:
    """User notification preferences"""
    user_id: str
    channels: Dict[NotificationChannel, bool] = field(default_factory=dict)
    types: Dict[NotificationType, bool] = field(default_factory=dict)
    quiet_hours: Dict[str, Any] = field(default_factory=dict)
    frequency_limits: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

class AdvancedNotificationManager:
    """Advanced notification manager with multiple channels and scheduling"""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        
        # Notification storage
        self.notifications: Dict[str, Notification] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.user_preferences: Dict[str, NotificationPreferences] = {}
        
        # Channel handlers
        self.channel_handlers: Dict[NotificationChannel, Callable] = {}
        self.channel_configs: Dict[NotificationChannel, Dict[str, Any]] = {}
        
        # Scheduling and queuing
        self.scheduled_notifications: Dict[str, Notification] = {}
        self.notification_queue: deque = deque()
        self.rate_limits: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Statistics
        self.stats = {
            "total_sent": 0,
            "total_delivered": 0,
            "total_failed": 0,
            "by_channel": defaultdict(int),
            "by_type": defaultdict(int),
            "by_priority": defaultdict(int)
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._scheduler_task = None
        self._processor_task = None
        self._cleanup_task = None
        
        # Initialize default templates and handlers
        self._initialize_default_templates()
        self._initialize_channel_handlers()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            self._processor_task = asyncio.create_task(self._processor_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # Event loop not running yet
            pass
    
    def _initialize_default_templates(self):
        """Initialize default notification templates"""
        # Welcome email template
        welcome_template = NotificationTemplate(
            id="welcome_email",
            name="Welcome Email",
            type=NotificationType.USER,
            channel=NotificationChannel.EMAIL,
            subject="Welcome to Opinion Market!",
            content="Hello {{username}}, welcome to Opinion Market! We're excited to have you join our community.",
            html_content="<h1>Welcome to Opinion Market!</h1><p>Hello {{username}}, welcome to Opinion Market! We're excited to have you join our community.</p>",
            variables={"username", "email"}
        )
        self.templates["welcome_email"] = welcome_template
        
        # Market update template
        market_template = NotificationTemplate(
            id="market_update",
            name="Market Update",
            type=NotificationType.MARKET,
            channel=NotificationChannel.IN_APP,
            subject="Market Update: {{market_name}}",
            content="Market '{{market_name}}' has been updated. Current price: {{current_price}}",
            variables={"market_name", "current_price", "change_percent"}
        )
        self.templates["market_update"] = market_template
        
        # Trade confirmation template
        trade_template = NotificationTemplate(
            id="trade_confirmation",
            name="Trade Confirmation",
            type=NotificationType.TRADE,
            channel=NotificationChannel.EMAIL,
            subject="Trade Confirmation - {{market_name}}",
            content="Your trade has been executed. Market: {{market_name}}, Amount: {{amount}}, Price: {{price}}",
            html_content="<h2>Trade Confirmation</h2><p>Market: {{market_name}}</p><p>Amount: {{amount}}</p><p>Price: {{price}}</p>",
            variables={"market_name", "amount", "price", "trade_id"}
        )
        self.templates["trade_confirmation"] = trade_template
        
        # Security alert template
        security_template = NotificationTemplate(
            id="security_alert",
            name="Security Alert",
            type=NotificationType.SECURITY,
            channel=NotificationChannel.EMAIL,
            subject="Security Alert - {{alert_type}}",
            content="Security alert: {{alert_type}}. Time: {{timestamp}}. Please review your account.",
            variables={"alert_type", "timestamp", "ip_address"}
        )
        self.templates["security_alert"] = security_template
    
    def _initialize_channel_handlers(self):
        """Initialize channel handlers"""
        self.channel_handlers[NotificationChannel.EMAIL] = self._send_email
        self.channel_handlers[NotificationChannel.WEBHOOK] = self._send_webhook
        self.channel_handlers[NotificationChannel.WEBSOCKET] = self._send_websocket
        self.channel_handlers[NotificationChannel.IN_APP] = self._send_in_app
        self.channel_handlers[NotificationChannel.SLACK] = self._send_slack
        self.channel_handlers[NotificationChannel.DISCORD] = self._send_discord
        self.channel_handlers[NotificationChannel.TELEGRAM] = self._send_telegram
    
    async def send_notification(self, user_id: str, template_id: str, 
                              variables: Optional[Dict[str, Any]] = None,
                              channel: Optional[NotificationChannel] = None,
                              priority: NotificationPriority = NotificationPriority.NORMAL,
                              scheduled_at: Optional[datetime] = None) -> str:
        """Send notification using template"""
        try:
            # Get template
            if template_id not in self.templates:
                raise ValueError(f"Template '{template_id}' not found")
            
            template = self.templates[template_id]
            
            # Check user preferences
            if not self._check_user_preferences(user_id, template.type, channel or template.channel):
                logger.info(f"Notification blocked by user preferences: {user_id}")
                return None
            
            # Check rate limits
            if not self._check_rate_limits(user_id, template.type, channel or template.channel):
                logger.warning(f"Notification rate limited: {user_id}")
                return None
            
            # Render template
            rendered_content = self._render_template(template, variables or {})
            
            # Create notification
            notification_id = str(uuid.uuid4())
            notification = Notification(
                id=notification_id,
                user_id=user_id,
                type=template.type,
                channel=channel or template.channel,
                priority=priority,
                subject=rendered_content["subject"],
                content=rendered_content["content"],
                html_content=rendered_content.get("html_content"),
                scheduled_at=scheduled_at,
                metadata={"template_id": template_id, "variables": variables or {}}
            )
            
            # Store notification
            with self._lock:
                self.notifications[notification_id] = notification
            
            # Schedule or send immediately
            if scheduled_at and scheduled_at > datetime.utcnow():
                notification.status = NotificationStatus.SCHEDULED
                self.scheduled_notifications[notification_id] = notification
            else:
                await self._queue_notification(notification)
            
            logger.info(f"Notification created: {notification_id} for user {user_id}")
            return notification_id
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            raise
    
    async def send_custom_notification(self, user_id: str, subject: str, content: str,
                                     channel: NotificationChannel,
                                     notification_type: NotificationType = NotificationType.SYSTEM,
                                     priority: NotificationPriority = NotificationPriority.NORMAL,
                                     html_content: Optional[str] = None,
                                     scheduled_at: Optional[datetime] = None) -> str:
        """Send custom notification without template"""
        try:
            # Check user preferences
            if not self._check_user_preferences(user_id, notification_type, channel):
                logger.info(f"Custom notification blocked by user preferences: {user_id}")
                return None
            
            # Check rate limits
            if not self._check_rate_limits(user_id, notification_type, channel):
                logger.warning(f"Custom notification rate limited: {user_id}")
                return None
            
            # Create notification
            notification_id = str(uuid.uuid4())
            notification = Notification(
                id=notification_id,
                user_id=user_id,
                type=notification_type,
                channel=channel,
                priority=priority,
                subject=subject,
                content=content,
                html_content=html_content,
                scheduled_at=scheduled_at
            )
            
            # Store notification
            with self._lock:
                self.notifications[notification_id] = notification
            
            # Schedule or send immediately
            if scheduled_at and scheduled_at > datetime.utcnow():
                notification.status = NotificationStatus.SCHEDULED
                self.scheduled_notifications[notification_id] = notification
            else:
                await self._queue_notification(notification)
            
            logger.info(f"Custom notification created: {notification_id} for user {user_id}")
            return notification_id
            
        except Exception as e:
            logger.error(f"Error sending custom notification: {e}")
            raise
    
    def _render_template(self, template: NotificationTemplate, variables: Dict[str, Any]) -> Dict[str, str]:
        """Render template with variables"""
        try:
            rendered = {
                "subject": template.subject,
                "content": template.content
            }
            
            if template.html_content:
                rendered["html_content"] = template.html_content
            
            # Replace variables in content
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                rendered["subject"] = rendered["subject"].replace(placeholder, str(value))
                rendered["content"] = rendered["content"].replace(placeholder, str(value))
                if "html_content" in rendered:
                    rendered["html_content"] = rendered["html_content"].replace(placeholder, str(value))
            
            return rendered
            
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return {"subject": template.subject, "content": template.content}
    
    def _check_user_preferences(self, user_id: str, notification_type: NotificationType, 
                               channel: NotificationChannel) -> bool:
        """Check if user allows this type of notification on this channel"""
        if user_id not in self.user_preferences:
            return True  # Default to allow if no preferences set
        
        preferences = self.user_preferences[user_id]
        
        # Check channel preference
        if channel in preferences.channels and not preferences.channels[channel]:
            return False
        
        # Check type preference
        if notification_type in preferences.types and not preferences.types[notification_type]:
            return False
        
        # Check quiet hours
        if self._is_quiet_hours(preferences):
            return False
        
        return True
    
    def _check_rate_limits(self, user_id: str, notification_type: NotificationType, 
                          channel: NotificationChannel) -> bool:
        """Check rate limits for user"""
        now = datetime.utcnow()
        key = f"{user_id}:{notification_type.value}:{channel.value}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                "count": 0,
                "window_start": now,
                "limit": self._get_rate_limit(notification_type, channel)
            }
        
        rate_limit = self.rate_limits[key]
        
        # Reset window if needed
        if now - rate_limit["window_start"] > timedelta(hours=1):
            rate_limit["count"] = 0
            rate_limit["window_start"] = now
        
        # Check limit
        if rate_limit["count"] >= rate_limit["limit"]:
            return False
        
        rate_limit["count"] += 1
        return True
    
    def _get_rate_limit(self, notification_type: NotificationType, channel: NotificationChannel) -> int:
        """Get rate limit for notification type and channel"""
        # Default rate limits
        limits = {
            (NotificationType.SYSTEM, NotificationChannel.EMAIL): 10,
            (NotificationType.MARKET, NotificationChannel.IN_APP): 100,
            (NotificationType.TRADE, NotificationChannel.EMAIL): 50,
            (NotificationType.SECURITY, NotificationChannel.EMAIL): 20,
            (NotificationType.PROMOTIONAL, NotificationChannel.EMAIL): 5
        }
        
        return limits.get((notification_type, channel), 20)
    
    def _is_quiet_hours(self, preferences: NotificationPreferences) -> bool:
        """Check if current time is in quiet hours"""
        if not preferences.quiet_hours:
            return False
        
        now = datetime.utcnow()
        current_hour = now.hour
        
        start_hour = preferences.quiet_hours.get("start", 22)
        end_hour = preferences.quiet_hours.get("end", 8)
        
        if start_hour <= end_hour:
            return start_hour <= current_hour < end_hour
        else:
            return current_hour >= start_hour or current_hour < end_hour
    
    async def _queue_notification(self, notification: Notification):
        """Queue notification for processing"""
        with self._lock:
            self.notification_queue.append(notification)
    
    async def _scheduler_loop(self):
        """Background task to process scheduled notifications"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = datetime.utcnow()
                to_send = []
                
                with self._lock:
                    for notification_id, notification in self.scheduled_notifications.items():
                        if notification.scheduled_at and notification.scheduled_at <= now:
                            to_send.append(notification_id)
                
                for notification_id in to_send:
                    notification = self.scheduled_notifications.pop(notification_id)
                    await self._queue_notification(notification)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(60)
    
    async def _processor_loop(self):
        """Background task to process notification queue"""
        while True:
            try:
                if self.notification_queue:
                    notification = self.notification_queue.popleft()
                    await self._process_notification(notification)
                else:
                    await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in processor loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_notification(self, notification: Notification):
        """Process a single notification"""
        try:
            # Get channel handler
            handler = self.channel_handlers.get(notification.channel)
            if not handler:
                logger.error(f"No handler for channel: {notification.channel}")
                notification.status = NotificationStatus.FAILED
                notification.error_message = f"No handler for channel: {notification.channel}"
                return
            
            # Send notification
            success = await handler(notification)
            
            if success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.utcnow()
                self.stats["total_sent"] += 1
                self.stats["by_channel"][notification.channel.value] += 1
                self.stats["by_type"][notification.type.value] += 1
                self.stats["by_priority"][notification.priority.value] += 1
            else:
                notification.retry_count += 1
                if notification.retry_count < notification.max_retries:
                    # Retry later
                    await asyncio.sleep(60 * notification.retry_count)
                    await self._queue_notification(notification)
                else:
                    notification.status = NotificationStatus.FAILED
                    notification.failed_at = datetime.utcnow()
                    self.stats["total_failed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing notification {notification.id}: {e}")
            notification.status = NotificationStatus.FAILED
            notification.error_message = str(e)
            notification.failed_at = datetime.utcnow()
            self.stats["total_failed"] += 1
    
    async def _send_email(self, notification: Notification) -> bool:
        """Send email notification"""
        try:
            # Get email config
            config = self.channel_configs.get(NotificationChannel.EMAIL, {})
            if not config:
                logger.error("Email configuration not found")
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = notification.subject
            msg['From'] = config.get('from_email', 'noreply@opinionmarket.com')
            msg['To'] = notification.user_id  # Assuming user_id is email
            
            # Add text content
            text_part = MIMEText(notification.content, 'plain')
            msg.attach(text_part)
            
            # Add HTML content if available
            if notification.html_content:
                html_part = MIMEText(notification.html_content, 'html')
                msg.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(config.get('smtp_host', 'localhost'), config.get('smtp_port', 587)) as server:
                server.starttls(context=context)
                server.login(config.get('username', ''), config.get('password', ''))
                server.send_message(msg)
            
            logger.info(f"Email sent to {notification.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_webhook(self, notification: Notification) -> bool:
        """Send webhook notification"""
        try:
            if not REQUESTS_AVAILABLE:
                logger.error("Requests library not available for webhook")
                return False
            
            config = self.channel_configs.get(NotificationChannel.WEBHOOK, {})
            webhook_url = config.get('url')
            
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
            
            payload = {
                "notification_id": notification.id,
                "user_id": notification.user_id,
                "type": notification.type.value,
                "priority": notification.priority.value,
                "subject": notification.subject,
                "content": notification.content,
                "timestamp": notification.created_at.isoformat()
            }
            
            response = requests.post(webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Webhook sent to {webhook_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False
    
    async def _send_websocket(self, notification: Notification) -> bool:
        """Send WebSocket notification"""
        try:
            # This would integrate with the WebSocket manager
            from app.core.advanced_websocket_manager import websocket_manager
            
            message = {
                "type": "notification",
                "notification_id": notification.id,
                "subject": notification.subject,
                "content": notification.content,
                "priority": notification.priority.value,
                "timestamp": notification.created_at.isoformat()
            }
            
            sent_count = await websocket_manager.broadcast_to_user(notification.user_id, message)
            
            logger.info(f"WebSocket notification sent to {notification.user_id}")
            return sent_count > 0
            
        except Exception as e:
            logger.error(f"Error sending WebSocket notification: {e}")
            return False
    
    async def _send_in_app(self, notification: Notification) -> bool:
        """Send in-app notification"""
        try:
            # Store in-app notification for user to retrieve
            # This would typically be stored in a database
            logger.info(f"In-app notification stored for {notification.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending in-app notification: {e}")
            return False
    
    async def _send_slack(self, notification: Notification) -> bool:
        """Send Slack notification"""
        try:
            if not REQUESTS_AVAILABLE:
                logger.error("Requests library not available for Slack")
                return False
            
            config = self.channel_configs.get(NotificationChannel.SLACK, {})
            webhook_url = config.get('webhook_url')
            
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False
            
            payload = {
                "text": f"*{notification.subject}*",
                "attachments": [
                    {
                        "color": self._get_slack_color(notification.priority),
                        "text": notification.content,
                        "footer": "Opinion Market",
                        "ts": int(notification.created_at.timestamp())
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_discord(self, notification: Notification) -> bool:
        """Send Discord notification"""
        try:
            if not REQUESTS_AVAILABLE:
                logger.error("Requests library not available for Discord")
                return False
            
            config = self.channel_configs.get(NotificationChannel.DISCORD, {})
            webhook_url = config.get('webhook_url')
            
            if not webhook_url:
                logger.error("Discord webhook URL not configured")
                return False
            
            payload = {
                "embeds": [
                    {
                        "title": notification.subject,
                        "description": notification.content,
                        "color": self._get_discord_color(notification.priority),
                        "timestamp": notification.created_at.isoformat(),
                        "footer": {"text": "Opinion Market"}
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Discord notification sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False
    
    async def _send_telegram(self, notification: Notification) -> bool:
        """Send Telegram notification"""
        try:
            if not REQUESTS_AVAILABLE:
                logger.error("Requests library not available for Telegram")
                return False
            
            config = self.channel_configs.get(NotificationChannel.TELEGRAM, {})
            bot_token = config.get('bot_token')
            chat_id = config.get('chat_id')
            
            if not bot_token or not chat_id:
                logger.error("Telegram bot token or chat ID not configured")
                return False
            
            message = f"*{notification.subject}*\n\n{notification.content}"
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Telegram notification sent")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False
    
    def _get_slack_color(self, priority: NotificationPriority) -> str:
        """Get Slack color for priority"""
        colors = {
            NotificationPriority.LOW: "good",
            NotificationPriority.NORMAL: "#36a64f",
            NotificationPriority.HIGH: "warning",
            NotificationPriority.URGENT: "danger",
            NotificationPriority.CRITICAL: "#ff0000"
        }
        return colors.get(priority, "#36a64f")
    
    def _get_discord_color(self, priority: NotificationPriority) -> int:
        """Get Discord color for priority"""
        colors = {
            NotificationPriority.LOW: 0x00ff00,
            NotificationPriority.NORMAL: 0x36a64f,
            NotificationPriority.HIGH: 0xffaa00,
            NotificationPriority.URGENT: 0xff0000,
            NotificationPriority.CRITICAL: 0x8b0000
        }
        return colors.get(priority, 0x36a64f)
    
    async def _cleanup_loop(self):
        """Background task to cleanup old notifications"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                
                with self._lock:
                    # Clean up old notifications
                    old_notifications = [
                        nid for nid, notification in self.notifications.items()
                        if notification.created_at < cutoff_time
                    ]
                    
                    for nid in old_notifications:
                        del self.notifications[nid]
                    
                    # Clean up old rate limits
                    old_rate_limits = [
                        key for key, rate_limit in self.rate_limits.items()
                        if rate_limit["window_start"] < cutoff_time
                    ]
                    
                    for key in old_rate_limits:
                        del self.rate_limits[key]
                
                logger.info(f"Cleaned up {len(old_notifications)} old notifications")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    def set_user_preferences(self, user_id: str, preferences: NotificationPreferences):
        """Set user notification preferences"""
        with self._lock:
            self.user_preferences[user_id] = preferences
        logger.info(f"Notification preferences set for user {user_id}")
    
    def get_user_preferences(self, user_id: str) -> Optional[NotificationPreferences]:
        """Get user notification preferences"""
        return self.user_preferences.get(user_id)
    
    def configure_channel(self, channel: NotificationChannel, config: Dict[str, Any]):
        """Configure notification channel"""
        self.channel_configs[channel] = config
        logger.info(f"Channel configured: {channel.value}")
    
    def create_template(self, template: NotificationTemplate):
        """Create notification template"""
        with self._lock:
            self.templates[template.id] = template
        logger.info(f"Template created: {template.id}")
    
    def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Get notification by ID"""
        return self.notifications.get(notification_id)
    
    def get_user_notifications(self, user_id: str, limit: int = 100) -> List[Notification]:
        """Get user notifications"""
        with self._lock:
            user_notifications = [
                notification for notification in self.notifications.values()
                if notification.user_id == user_id
            ]
            
            # Sort by creation time (newest first)
            user_notifications.sort(key=lambda n: n.created_at, reverse=True)
            
            return user_notifications[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        with self._lock:
            return {
                "notifications": {
                    "total": len(self.notifications),
                    "pending": len([n for n in self.notifications.values() if n.status == NotificationStatus.PENDING]),
                    "sent": len([n for n in self.notifications.values() if n.status == NotificationStatus.SENT]),
                    "failed": len([n for n in self.notifications.values() if n.status == NotificationStatus.FAILED]),
                    "scheduled": len(self.scheduled_notifications)
                },
                "templates": {
                    "total": len(self.templates),
                    "active": len([t for t in self.templates.values() if t.is_active])
                },
                "users": {
                    "with_preferences": len(self.user_preferences)
                },
                "queue": {
                    "size": len(self.notification_queue)
                },
                "stats": dict(self.stats)
            }

# Global notification manager instance
notification_manager = AdvancedNotificationManager()

# Convenience functions
async def send_notification(user_id: str, template_id: str, 
                          variables: Optional[Dict[str, Any]] = None,
                          channel: Optional[NotificationChannel] = None,
                          priority: NotificationPriority = NotificationPriority.NORMAL,
                          scheduled_at: Optional[datetime] = None) -> str:
    """Send notification using template"""
    return await notification_manager.send_notification(
        user_id, template_id, variables, channel, priority, scheduled_at
    )

async def send_custom_notification(user_id: str, subject: str, content: str,
                                 channel: NotificationChannel,
                                 notification_type: NotificationType = NotificationType.SYSTEM,
                                 priority: NotificationPriority = NotificationPriority.NORMAL,
                                 html_content: Optional[str] = None,
                                 scheduled_at: Optional[datetime] = None) -> str:
    """Send custom notification"""
    return await notification_manager.send_custom_notification(
        user_id, subject, content, channel, notification_type, priority, html_content, scheduled_at
    )

def set_user_preferences(user_id: str, preferences: NotificationPreferences):
    """Set user notification preferences"""
    notification_manager.set_user_preferences(user_id, preferences)

def configure_channel(channel: NotificationChannel, config: Dict[str, Any]):
    """Configure notification channel"""
    notification_manager.configure_channel(channel, config)
