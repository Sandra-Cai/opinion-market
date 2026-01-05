import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.core.database import SessionLocal
from app.models.user import User
from app.models.market import Market
from app.models.trade import Trade
from app.models.position import Position
from app.models.notification import Notification, NotificationType
from app.services.rewards_system import get_rewards_system

logger = logging.getLogger(__name__)


class MobileAPIService:
    """Mobile-optimized API service with push notifications and mobile features"""

    def __init__(self):
        self.push_tokens = {}  # user_id -> push_token mapping
        self.device_info = {}  # user_id -> device info mapping

    def register_device(self, user_id: int, push_token: str, device_info: Dict) -> Dict:
        """Register a mobile device for push notifications"""
        self.push_tokens[user_id] = push_token
        self.device_info[user_id] = device_info

        return {
            "success": True,
            "message": "Device registered successfully",
            "device_id": f"device_{user_id}_{int(datetime.utcnow().timestamp())}",
        }

    def unregister_device(self, user_id: int) -> Dict:
        """Unregister a mobile device"""
        if user_id in self.push_tokens:
            del self.push_tokens[user_id]
        if user_id in self.device_info:
            del self.device_info[user_id]

        return {"success": True, "message": "Device unregistered successfully"}

    def get_mobile_dashboard(self, user_id: int) -> Dict:
        """Get mobile-optimized dashboard data"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}

            # Get recent markets
            recent_markets = (
                db.query(Market)
                .filter(Market.status == "open")
                .order_by(desc(Market.created_at))
                .limit(10)
                .all()
            )

            # Get user's recent trades
            recent_trades = (
                db.query(Trade)
                .filter(Trade.user_id == user_id)
                .order_by(desc(Trade.created_at))
                .limit(5)
                .all()
            )

            # Get trending markets
            trending_markets = (
                db.query(Market)
                .filter(Market.status == "open")
                .order_by(desc(Market.trending_score))
                .limit(5)
                .all()
            )

            # Get user's portfolio summary
            positions = (
                db.query(Position)
                .filter(Position.user_id == user_id, Position.is_active == True)
                .all()
            )

            portfolio_value = sum(p.current_value for p in positions)
            total_pnl = sum(p.total_pnl for p in positions)

            # Get notifications count
            unread_notifications = (
                db.query(Notification)
                .filter(Notification.user_id == user_id, Notification.is_read == False)
                .count()
            )

            return {
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "available_balance": user.available_balance,
                    "portfolio_value": portfolio_value,
                    "total_pnl": total_pnl,
                    "reputation_score": user.reputation_score,
                },
                "recent_markets": [
                    {
                        "id": market.id,
                        "title": market.title,
                        "category": market.category,
                        "current_price_a": market.current_price_a,
                        "current_price_b": market.current_price_b,
                        "volume_24h": market.volume_24h,
                        "trending_score": market.trending_score,
                    }
                    for market in recent_markets
                ],
                "trending_markets": [
                    {
                        "id": market.id,
                        "title": market.title,
                        "category": market.category,
                        "current_price_a": market.current_price_a,
                        "current_price_b": market.current_price_b,
                        "volume_24h": market.volume_24h,
                        "trending_score": market.trending_score,
                    }
                    for market in trending_markets
                ],
                "recent_trades": [
                    {
                        "id": trade.id,
                        "market_id": trade.market_id,
                        "market_title": trade.market.title,
                        "trade_type": trade.trade_type,
                        "outcome": trade.outcome,
                        "amount": trade.total_value,
                        "profit_loss": trade.profit_loss,
                        "created_at": trade.created_at,
                    }
                    for trade in recent_trades
                ],
                "notifications": {"unread_count": unread_notifications},
                "quick_actions": [
                    "create_market",
                    "place_trade",
                    "view_portfolio",
                    "check_leaderboard",
                ],
            }

        finally:
            db.close()

    def get_mobile_market_details(
        self, market_id: int, user_id: Optional[int] = None
    ) -> Dict:
        """Get mobile-optimized market details"""
        db = SessionLocal()
        try:
            market = db.query(Market).filter(Market.id == market_id).first()
            if not market:
                return {"error": "Market not found"}

            # Get recent trades for this market
            recent_trades = (
                db.query(Trade)
                .filter(Trade.market_id == market_id)
                .order_by(desc(Trade.created_at))
                .limit(20)
                .all()
            )

            # Get user's position in this market (if logged in)
            user_position = None
            if user_id:
                position = (
                    db.query(Position)
                    .filter(
                        Position.user_id == user_id,
                        Position.market_id == market_id,
                        Position.is_active == True,
                    )
                    .first()
                )
                if position:
                    user_position = {
                        "outcome": position.outcome,
                        "shares_owned": position.shares_owned,
                        "average_price": position.average_price,
                        "current_value": position.current_value,
                        "total_pnl": position.total_pnl,
                    }

            # Calculate price history (simplified)
            price_history = self._calculate_price_history(market_id, db)

            return {
                "market": {
                    "id": market.id,
                    "title": market.title,
                    "description": market.description,
                    "category": market.category,
                    "outcome_a": market.outcome_a,
                    "outcome_b": market.outcome_b,
                    "current_price_a": market.current_price_a,
                    "current_price_b": market.current_price_b,
                    "volume_24h": market.volume_24h,
                    "volume_total": market.volume_total,
                    "unique_traders": market.unique_traders,
                    "closes_at": market.closes_at,
                    "status": market.status,
                    "is_verified": market.is_verified,
                    "trending_score": market.trending_score,
                },
                "user_position": user_position,
                "recent_trades": [
                    {
                        "id": trade.id,
                        "trade_type": trade.trade_type,
                        "outcome": trade.outcome,
                        "amount": trade.total_value,
                        "price_per_share": trade.price_per_share,
                        "created_at": trade.created_at,
                        "user_username": (
                            trade.user.username if trade.user else "Anonymous"
                        ),
                    }
                    for trade in recent_trades
                ],
                "price_history": price_history,
                "market_stats": {
                    "total_trades": len(recent_trades),
                    "avg_trade_size": (
                        sum(t.total_value for t in recent_trades) / len(recent_trades)
                        if recent_trades
                        else 0
                    ),
                    "price_change_24h": self._calculate_price_change(market_id, db),
                },
            }

        finally:
            db.close()

    def _calculate_price_history(self, market_id: int, db: Session) -> List[Dict]:
        """Calculate price history for mobile charts"""
        # Get trades from last 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)
        trades = (
            db.query(Trade)
            .filter(Trade.market_id == market_id, Trade.created_at >= week_ago)
            .order_by(Trade.created_at)
            .all()
        )

        # Group by hour and calculate average prices
        price_history = []
        if trades:
            current_hour = trades[0].created_at.replace(
                minute=0, second=0, microsecond=0
            )
            end_time = datetime.utcnow()

            while current_hour <= end_time:
                hour_trades = [
                    t
                    for t in trades
                    if t.created_at >= current_hour
                    and t.created_at < current_hour + timedelta(hours=1)
                ]

                if hour_trades:
                    outcome_a_trades = [
                        t for t in hour_trades if t.outcome == "outcome_a"
                    ]
                    outcome_b_trades = [
                        t for t in hour_trades if t.outcome == "outcome_b"
                    ]

                    avg_price_a = (
                        sum(t.price_per_share for t in outcome_a_trades)
                        / len(outcome_a_trades)
                        if outcome_a_trades
                        else 0
                    )
                    avg_price_b = (
                        sum(t.price_per_share for t in outcome_b_trades)
                        / len(outcome_b_trades)
                        if outcome_b_trades
                        else 0
                    )

                    price_history.append(
                        {
                            "timestamp": current_hour.isoformat(),
                            "price_a": avg_price_a,
                            "price_b": avg_price_b,
                            "volume": sum(t.total_value for t in hour_trades),
                        }
                    )

                current_hour += timedelta(hours=1)

        return price_history

    def _calculate_price_change(self, market_id: int, db: Session) -> Dict:
        """Calculate 24h price change"""
        day_ago = datetime.utcnow() - timedelta(days=1)

        # Get current prices
        market = db.query(Market).filter(Market.id == market_id).first()
        current_price_a = market.current_price_a
        current_price_b = market.current_price_b

        # Get prices from 24h ago (simplified - would need historical price table)
        # For now, return placeholder
        return {
            "outcome_a_change": 0.0,
            "outcome_b_change": 0.0,
            "outcome_a_change_percent": 0.0,
            "outcome_b_change_percent": 0.0,
        }

    def send_push_notification(
        self, user_id: int, title: str, body: str, data: Optional[Dict] = None
    ) -> Dict:
        """Send push notification to mobile device"""
        if user_id not in self.push_tokens:
            return {"error": "No device registered for user"}

        push_token = self.push_tokens[user_id]
        device_info = self.device_info.get(user_id, {})

        # In a real implementation, this would send to FCM/APNS
        notification_data = {
            "title": title,
            "body": body,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat(),
            "device_info": device_info,
        }

        # Simulate sending notification
        logger.info(f"Push notification sent to user {user_id}: {title} - {body}")

        return {
            "success": True,
            "message": "Push notification sent",
            "notification_id": f"push_{user_id}_{int(datetime.utcnow().timestamp())}",
        }

    def get_mobile_portfolio(self, user_id: int) -> Dict:
        """Get mobile-optimized portfolio view"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}

            # Get all active positions
            positions = (
                db.query(Position)
                .filter(Position.user_id == user_id, Position.is_active == True)
                .all()
            )

            # Calculate portfolio metrics
            total_value = sum(p.current_value for p in positions)
            total_invested = sum(p.total_invested for p in positions)
            total_pnl = sum(p.total_pnl for p in positions)
            total_pnl_percent = (
                (total_pnl / total_invested * 100) if total_invested > 0 else 0
            )

            # Group positions by market
            positions_by_market = {}
            for position in positions:
                market_id = position.market_id
                if market_id not in positions_by_market:
                    positions_by_market[market_id] = []
                positions_by_market[market_id].append(position)

            # Get market details for each position
            portfolio_positions = []
            for market_id, market_positions in positions_by_market.items():
                market = db.query(Market).filter(Market.id == market_id).first()
                if market:
                    for position in market_positions:
                        portfolio_positions.append(
                            {
                                "position_id": position.id,
                                "market_id": market.id,
                                "market_title": market.title,
                                "market_category": market.category,
                                "outcome": position.outcome,
                                "shares_owned": position.shares_owned,
                                "average_price": position.average_price,
                                "current_price": position.current_price,
                                "current_value": position.current_value,
                                "total_invested": position.total_invested,
                                "total_pnl": position.total_pnl,
                                "pnl_percent": position.profit_loss_percentage,
                                "is_profitable": position.is_profitable,
                            }
                        )

            return {
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "available_balance": user.available_balance,
                },
                "portfolio_summary": {
                    "total_value": total_value,
                    "total_invested": total_invested,
                    "total_pnl": total_pnl,
                    "total_pnl_percent": total_pnl_percent,
                    "position_count": len(positions),
                    "profitable_positions": len(
                        [p for p in positions if p.is_profitable]
                    ),
                },
                "positions": portfolio_positions,
                "performance_chart": self._generate_performance_chart(user_id, db),
            }

        finally:
            db.close()

    def _generate_performance_chart(self, user_id: int, db: Session) -> List[Dict]:
        """Generate performance chart data for mobile"""
        # Get trades from last 30 days
        month_ago = datetime.utcnow() - timedelta(days=30)
        trades = (
            db.query(Trade)
            .filter(Trade.user_id == user_id, Trade.created_at >= month_ago)
            .order_by(Trade.created_at)
            .all()
        )

        # Group by day and calculate cumulative P&L
        performance_data = []
        cumulative_pnl = 0

        if trades:
            current_date = trades[0].created_at.date()
            end_date = datetime.utcnow().date()

            while current_date <= end_date:
                day_trades = [t for t in trades if t.created_at.date() == current_date]
                day_pnl = sum(t.profit_loss for t in day_trades)
                cumulative_pnl += day_pnl

                performance_data.append(
                    {
                        "date": current_date.isoformat(),
                        "daily_pnl": day_pnl,
                        "cumulative_pnl": cumulative_pnl,
                        "trade_count": len(day_trades),
                    }
                )

                current_date += timedelta(days=1)

        return performance_data

    def get_mobile_leaderboard(
        self, category: str = "traders", period: str = "7d", limit: int = 20
    ) -> Dict:
        """Get mobile-optimized leaderboard"""
        db = SessionLocal()
        try:
            if category == "traders":
                # Get top traders by profit
                users = (
                    db.query(User)
                    .filter(User.is_active == True)
                    .order_by(desc(User.total_profit))
                    .limit(limit)
                    .all()
                )

                leaderboard_data = [
                    {
                        "rank": i + 1,
                        "user_id": user.id,
                        "username": user.username,
                        "total_profit": user.total_profit,
                        "total_volume": user.total_volume,
                        "win_rate": user.win_rate,
                        "reputation_score": user.reputation_score,
                    }
                    for i, user in enumerate(users)
                ]

            elif category == "volume":
                # Get top traders by volume
                users = (
                    db.query(User)
                    .filter(User.is_active == True)
                    .order_by(desc(User.total_volume))
                    .limit(limit)
                    .all()
                )

                leaderboard_data = [
                    {
                        "rank": i + 1,
                        "user_id": user.id,
                        "username": user.username,
                        "total_volume": user.total_volume,
                        "total_profit": user.total_profit,
                        "avg_trade_size": user.avg_trade_size,
                    }
                    for i, user in enumerate(users)
                ]

            elif category == "win_rate":
                # Get top traders by win rate (minimum trades required)
                users = (
                    db.query(User)
                    .filter(User.is_active == True, User.total_trades >= 10)
                    .order_by(desc(User.win_rate))
                    .limit(limit)
                    .all()
                )

                leaderboard_data = [
                    {
                        "rank": i + 1,
                        "user_id": user.id,
                        "username": user.username,
                        "win_rate": user.win_rate,
                        "total_trades": user.total_trades,
                        "total_profit": user.total_profit,
                    }
                    for i, user in enumerate(users)
                ]

            else:
                return {"error": "Invalid category"}

            return {
                "category": category,
                "period": period,
                "leaderboard": leaderboard_data,
                "total_users": len(leaderboard_data),
            }

        finally:
            db.close()


# Global mobile API service instance
mobile_api_service = MobileAPIService()


def get_mobile_api_service() -> MobileAPIService:
    """Get the global mobile API service instance"""
    return mobile_api_service
