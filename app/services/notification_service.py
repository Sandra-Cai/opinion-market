from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.notification import (
    Notification,
    NotificationPreference,
    NotificationType,
    NotificationPriority,
)
from app.models.user import User


class NotificationService:
    def __init__(self):
        self.db = SessionLocal()

    def create_notification(
        self,
        user_id: int,
        notification_type: NotificationType,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        market_id: Optional[int] = None,
        trade_id: Optional[int] = None,
        dispute_id: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Notification:
        """Create a new notification"""
        notification = Notification(
            user_id=user_id,
            notification_type=notification_type,
            title=title,
            message=message,
            priority=priority,
            market_id=market_id,
            trade_id=trade_id,
            dispute_id=dispute_id,
            data=data or {},
        )

        self.db.add(notification)
        self.db.commit()
        self.db.refresh(notification)

        # Check if we should send email/push notification
        self._check_and_send_notification(notification)

        return notification

    def _check_and_send_notification(self, notification: Notification):
        """Check user preferences and send notification if enabled"""
        preferences = (
            self.db.query(NotificationPreference)
            .filter(NotificationPreference.user_id == notification.user_id)
            .first()
        )

        if not preferences:
            return

        # Check email preferences
        if preferences.should_send_email(notification.notification_type):
            self._send_email_notification(notification)

        # Check push notification preferences
        if preferences.should_send_push(notification.notification_type):
            self._send_push_notification(notification)

    def _send_email_notification(self, notification: Notification):
        """Send email notification (placeholder for email service integration)"""
        # TODO: Integrate with email service (SendGrid, AWS SES, etc.)
        print(f"📧 Email notification sent: {notification.title}")
        notification.mark_as_sent()
        self.db.commit()

    def _send_push_notification(self, notification: Notification):
        """Send push notification (placeholder for push service integration)"""
        # TODO: Integrate with push notification service (Firebase, OneSignal, etc.)
        print(f"📱 Push notification sent: {notification.title}")
        notification.mark_as_sent()
        self.db.commit()

    def notify_market_resolved(self, market_id: int, outcome: str):
        """Notify all traders when a market is resolved"""
        from app.models.trade import Trade

        # Get all unique traders for this market
        traders = (
            self.db.query(Trade.user_id)
            .filter(Trade.market_id == market_id)
            .distinct()
            .all()
        )

        market = self.db.query(Market).filter(Market.id == market_id).first()
        if not market:
            return

        for trader_id in traders:
            self.create_notification(
                user_id=trader_id[0],
                notification_type=NotificationType.MARKET_RESOLVED,
                title=f"Market Resolved: {market.title}",
                message=f"The market '{market.title}' has been resolved with outcome: {outcome}",
                priority=NotificationPriority.HIGH,
                market_id=market_id,
                data={"outcome": outcome},
            )

    def notify_trade_executed(self, trade_id: int):
        """Notify user when their trade is executed"""
        from app.models.trade import Trade

        trade = self.db.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            return

        market = self.db.query(Market).filter(Market.id == trade.market_id).first()
        if not market:
            return

        self.create_notification(
            user_id=trade.user_id,
            notification_type=NotificationType.TRADE_EXECUTED,
            title="Trade Executed",
            message=f"Your {trade.trade_type} order for {trade.amount} shares in '{market.title}' has been executed at ${trade.price_per_share}",
            priority=NotificationPriority.MEDIUM,
            trade_id=trade_id,
            market_id=trade.market_id,
            data={
                "trade_type": trade.trade_type,
                "amount": trade.amount,
                "price": trade.price_per_share,
                "total_value": trade.total_value,
            },
        )

    def notify_price_alert(
        self, user_id: int, market_id: int, old_price: float, new_price: float
    ):
        """Notify user of significant price changes"""
        market = self.db.query(Market).filter(Market.id == market_id).first()
        if not market:
            return

        price_change = ((new_price - old_price) / old_price) * 100

        self.create_notification(
            user_id=user_id,
            notification_type=NotificationType.PRICE_ALERT,
            title=f"Price Alert: {market.title}",
            message=f"Price changed by {price_change:.2f}% from ${old_price:.3f} to ${new_price:.3f}",
            priority=NotificationPriority.MEDIUM,
            market_id=market_id,
            data={
                "old_price": old_price,
                "new_price": new_price,
                "price_change_percent": price_change,
            },
        )

    def notify_dispute_created(self, dispute_id: int):
        """Notify relevant users when a dispute is created"""
        from app.models.dispute import MarketDispute

        dispute = (
            self.db.query(MarketDispute).filter(MarketDispute.id == dispute_id).first()
        )
        if not dispute:
            return

        market = self.db.query(Market).filter(Market.id == dispute.market_id).first()
        if not market:
            return

        # Notify market creator
        self.create_notification(
            user_id=market.creator_id,
            notification_type=NotificationType.DISPUTE_CREATED,
            title=f"Dispute Filed: {market.title}",
            message=f"A dispute has been filed for your market '{market.title}'",
            priority=NotificationPriority.HIGH,
            market_id=dispute.market_id,
            dispute_id=dispute_id,
        )

    def notify_system_announcement(
        self, title: str, message: str, user_ids: Optional[list] = None
    ):
        """Send system-wide announcement"""
        if user_ids:
            # Send to specific users
            for user_id in user_ids:
                self.create_notification(
                    user_id=user_id,
                    notification_type=NotificationType.SYSTEM_ANNOUNCEMENT,
                    title=title,
                    message=message,
                    priority=NotificationPriority.MEDIUM,
                )
        else:
            # Send to all users (use with caution)
            users = self.db.query(User).filter(User.is_active == True).all()
            for user in users:
                self.create_notification(
                    user_id=user.id,
                    notification_type=NotificationType.SYSTEM_ANNOUNCEMENT,
                    title=title,
                    message=message,
                    priority=NotificationPriority.MEDIUM,
                )

    def cleanup_old_notifications(self, days: int = 30):
        """Clean up old notifications"""
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Delete old read notifications
        old_notifications = (
            self.db.query(Notification)
            .filter(Notification.created_at < cutoff_date, Notification.is_read == True)
            .all()
        )

        for notification in old_notifications:
            self.db.delete(notification)

        self.db.commit()
        return len(old_notifications)


# Global notification service instance
notification_service = NotificationService()
