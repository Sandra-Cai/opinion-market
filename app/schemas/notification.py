from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.notification import NotificationType, NotificationPriority


class NotificationBase(BaseModel):
    notification_type: NotificationType
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=1000)
    priority: NotificationPriority = NotificationPriority.MEDIUM


class NotificationCreate(NotificationBase):
    user_id: int
    market_id: Optional[int] = None
    trade_id: Optional[int] = None
    dispute_id: Optional[int] = None
    data: Optional[Dict[str, Any]] = None


class NotificationResponse(NotificationBase):
    id: int
    user_id: int
    market_id: Optional[int] = None
    trade_id: Optional[int] = None
    dispute_id: Optional[int] = None
    data: Dict[str, Any]
    is_read: bool
    is_sent: bool
    is_urgent: bool
    age_hours: float
    created_at: datetime
    read_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class NotificationListResponse(BaseModel):
    notifications: List[NotificationResponse]
    total: int
    unread_count: int
    page: int
    per_page: int


class NotificationPreferenceUpdate(BaseModel):
    email_enabled: Optional[bool] = None
    email_market_resolved: Optional[bool] = None
    email_trade_executed: Optional[bool] = None
    email_price_alerts: Optional[bool] = None
    email_disputes: Optional[bool] = None
    email_system: Optional[bool] = None
    push_enabled: Optional[bool] = None
    push_market_resolved: Optional[bool] = None
    push_trade_executed: Optional[bool] = None
    push_price_alerts: Optional[bool] = None
    push_disputes: Optional[bool] = None
    push_system: Optional[bool] = None
    price_alert_threshold: Optional[float] = None
    price_alert_frequency: Optional[str] = None


class NotificationPreferenceResponse(BaseModel):
    id: int
    user_id: int
    email_enabled: bool
    email_market_resolved: bool
    email_trade_executed: bool
    email_price_alerts: bool
    email_disputes: bool
    email_system: bool
    push_enabled: bool
    push_market_resolved: bool
    push_trade_executed: bool
    push_price_alerts: bool
    push_disputes: bool
    push_system: bool
    price_alert_threshold: float
    price_alert_frequency: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
