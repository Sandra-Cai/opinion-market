from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, Enum, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base

class NotificationType(str, enum.Enum):
    MARKET_RESOLVED = "market_resolved"
    TRADE_EXECUTED = "trade_executed"
    POSITION_UPDATE = "position_update"
    PRICE_ALERT = "price_alert"
    DISPUTE_CREATED = "dispute_created"
    DISPUTE_RESOLVED = "dispute_resolved"
    MARKET_VERIFIED = "market_verified"
    SYSTEM_ANNOUNCEMENT = "system_announcement"
    SECURITY_ALERT = "security_alert"

class NotificationPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Notification(Base):
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Notification details
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    notification_type = Column(Enum(NotificationType), nullable=False)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    priority = Column(Enum(NotificationPriority), default=NotificationPriority.MEDIUM)
    
    # Related entities
    market_id = Column(Integer, ForeignKey("markets.id"))
    trade_id = Column(Integer, ForeignKey("trades.id"))
    dispute_id = Column(Integer, ForeignKey("market_disputes.id"))
    
    # Additional data
    data = Column(JSON, default=dict)  # Additional notification data
    
    # Status
    is_read = Column(Boolean, default=False)
    is_sent = Column(Boolean, default=False)  # Email/push notification sent
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime)
    sent_at = Column(DateTime)
    
    # Relationships
    user = relationship("User")
    market = relationship("Market")
    trade = relationship("Trade")
    dispute = relationship("MarketDispute")
    
    @property
    def is_urgent(self) -> bool:
        return self.priority == NotificationPriority.URGENT
    
    @property
    def age_hours(self) -> float:
        """Age of notification in hours"""
        return (datetime.utcnow() - self.created_at).total_seconds() / 3600
    
    def mark_as_read(self):
        """Mark notification as read"""
        self.is_read = True
        self.read_at = datetime.utcnow()
    
    def mark_as_sent(self):
        """Mark notification as sent"""
        self.is_sent = True
        self.sent_at = datetime.utcnow()

class NotificationPreference(Base):
    __tablename__ = "notification_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # User preferences
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Email preferences
    email_enabled = Column(Boolean, default=True)
    email_market_resolved = Column(Boolean, default=True)
    email_trade_executed = Column(Boolean, default=True)
    email_price_alerts = Column(Boolean, default=True)
    email_disputes = Column(Boolean, default=True)
    email_system = Column(Boolean, default=True)
    
    # Push notification preferences
    push_enabled = Column(Boolean, default=True)
    push_market_resolved = Column(Boolean, default=True)
    push_trade_executed = Column(Boolean, default=True)
    push_price_alerts = Column(Boolean, default=True)
    push_disputes = Column(Boolean, default=True)
    push_system = Column(Boolean, default=True)
    
    # Price alert settings
    price_alert_threshold = Column(Float, default=0.05)  # 5% price change
    price_alert_frequency = Column(String, default="hourly")  # hourly, daily
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User")
    
    def should_send_email(self, notification_type: NotificationType) -> bool:
        """Check if email should be sent for this notification type"""
        if not self.email_enabled:
            return False
        
        if notification_type == NotificationType.MARKET_RESOLVED:
            return self.email_market_resolved
        elif notification_type == NotificationType.TRADE_EXECUTED:
            return self.email_trade_executed
        elif notification_type == NotificationType.PRICE_ALERT:
            return self.email_price_alerts
        elif notification_type in [NotificationType.DISPUTE_CREATED, NotificationType.DISPUTE_RESOLVED]:
            return self.email_disputes
        elif notification_type == NotificationType.SYSTEM_ANNOUNCEMENT:
            return self.email_system
        
        return True
    
    def should_send_push(self, notification_type: NotificationType) -> bool:
        """Check if push notification should be sent for this notification type"""
        if not self.push_enabled:
            return False
        
        if notification_type == NotificationType.MARKET_RESOLVED:
            return self.push_market_resolved
        elif notification_type == NotificationType.TRADE_EXECUTED:
            return self.push_trade_executed
        elif notification_type == NotificationType.PRICE_ALERT:
            return self.push_price_alerts
        elif notification_type in [NotificationType.DISPUTE_CREATED, NotificationType.DISPUTE_RESOLVED]:
            return self.push_disputes
        elif notification_type == NotificationType.SYSTEM_ANNOUNCEMENT:
            return self.push_system
        
        return True
