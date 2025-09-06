from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.market import Market, MarketCategory
from app.models.trade import Trade
from app.models.user import User
from app.models.position import Position


class AnalyticsService:
    def __init__(self):
        self.db = SessionLocal()

    def get_market_analytics(self, market_id: int) -> Dict:
        """Get comprehensive analytics for a specific market"""
        market = self.db.query(Market).filter(Market.id == market_id).first()
        if not market:
            return {}

        # Get trade statistics
        trades = self.db.query(Trade).filter(Trade.market_id == market_id).all()

        # Calculate analytics
        total_volume = sum(trade.total_value for trade in trades)
        total_trades = len(trades)
        unique_traders = len(set(trade.user_id for trade in trades))

        # Price movement analysis
        if len(trades) > 1:
            first_trade = min(trades, key=lambda t: t.created_at)
            last_trade = max(trades, key=lambda t: t.created_at)
            price_change = (
                (last_trade.price_per_share - first_trade.price_per_share)
                / first_trade.price_per_share
            ) * 100
        else:
            price_change = 0

        # Volume over time
        volume_by_hour = self._get_volume_by_time(market_id, "hour")
        volume_by_day = self._get_volume_by_time(market_id, "day")

        # Trader sentiment
        buy_volume = sum(
            trade.total_value for trade in trades if trade.trade_type == "buy"
        )
        sell_volume = sum(
            trade.total_value for trade in trades if trade.trade_type == "sell"
        )
        sentiment_ratio = (
            buy_volume / (buy_volume + sell_volume)
            if (buy_volume + sell_volume) > 0
            else 0.5
        )

        return {
            "market_id": market_id,
            "total_volume": total_volume,
            "total_trades": total_trades,
            "unique_traders": unique_traders,
            "price_change_percent": price_change,
            "sentiment_ratio": sentiment_ratio,
            "volume_by_hour": volume_by_hour,
            "volume_by_day": volume_by_day,
            "current_price_a": market.current_price_a,
            "current_price_b": market.current_price_b,
            "liquidity_pool_a": market.liquidity_pool_a,
            "liquidity_pool_b": market.liquidity_pool_b,
        }

    def get_user_analytics(self, user_id: int) -> Dict:
        """Get comprehensive analytics for a specific user"""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return {}

        # Get user's trades
        trades = self.db.query(Trade).filter(Trade.user_id == user_id).all()

        # Calculate trading statistics
        total_volume = sum(trade.total_value for trade in trades)
        total_trades = len(trades)
        successful_trades = user.successful_trades
        win_rate = user.win_rate

        # Profit/Loss analysis
        total_profit = user.total_profit
        avg_trade_size = user.avg_trade_size

        # Trading frequency
        if trades:
            first_trade = min(trades, key=lambda t: t.created_at)
            last_trade = max(trades, key=lambda t: t.created_at)
            trading_days = (last_trade.created_at - first_trade.created_at).days + 1
            trades_per_day = total_trades / trading_days if trading_days > 0 else 0
        else:
            trades_per_day = 0

        # Portfolio analysis
        positions = (
            self.db.query(Position)
            .filter(Position.user_id == user_id, Position.is_active == True)
            .all()
        )

        portfolio_value = sum(pos.current_value for pos in positions)
        total_invested = sum(pos.total_invested for pos in positions)
        portfolio_return = (
            ((portfolio_value - total_invested) / total_invested * 100)
            if total_invested > 0
            else 0
        )

        # Market preferences
        market_categories = self._get_user_market_preferences(user_id)

        return {
            "user_id": user_id,
            "total_volume": total_volume,
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "avg_trade_size": avg_trade_size,
            "trades_per_day": trades_per_day,
            "portfolio_value": portfolio_value,
            "total_invested": total_invested,
            "portfolio_return_percent": portfolio_return,
            "reputation_score": user.reputation_score,
            "market_preferences": market_categories,
        }

    def get_platform_analytics(self) -> Dict:
        """Get platform-wide analytics"""
        # User statistics
        total_users = self.db.query(User).count()
        active_users_24h = (
            self.db.query(User)
            .filter(User.last_login >= datetime.utcnow() - timedelta(days=1))
            .count()
        )

        # Market statistics
        total_markets = self.db.query(Market).count()
        active_markets = self.db.query(Market).filter(Market.status == "open").count()
        resolved_markets = (
            self.db.query(Market).filter(Market.status == "resolved").count()
        )

        # Trading statistics
        total_volume = self.db.query(func.sum(Trade.total_value)).scalar() or 0
        total_trades = self.db.query(Trade).count()

        # Volume by category
        volume_by_category = self._get_volume_by_category()

        # Trending markets
        trending_markets = self._get_trending_markets(limit=10)

        return {
            "users": {"total": total_users, "active_24h": active_users_24h},
            "markets": {
                "total": total_markets,
                "active": active_markets,
                "resolved": resolved_markets,
            },
            "trading": {"total_volume": total_volume, "total_trades": total_trades},
            "volume_by_category": volume_by_category,
            "trending_markets": trending_markets,
        }

    def get_market_predictions(self, market_id: int) -> Dict:
        """Get market prediction analytics"""
        market = self.db.query(Market).filter(Market.id == market_id).first()
        if not market:
            return {}

        # Get recent trades for prediction
        recent_trades = (
            self.db.query(Trade)
            .filter(
                Trade.market_id == market_id,
                Trade.created_at >= datetime.utcnow() - timedelta(hours=24),
            )
            .order_by(Trade.created_at)
            .all()
        )

        if len(recent_trades) < 2:
            return {"prediction": "insufficient_data"}

        # Simple linear regression for price prediction
        prices = [trade.price_per_share for trade in recent_trades]
        times = [
            (trade.created_at - recent_trades[0].created_at).total_seconds()
            for trade in recent_trades
        ]

        # Calculate trend
        if len(prices) > 1:
            price_trend = (
                (prices[-1] - prices[0]) / (times[-1] - times[0])
                if times[-1] > times[0]
                else 0
            )
            predicted_price = prices[-1] + (price_trend * 3600)  # Predict 1 hour ahead
        else:
            predicted_price = prices[0]

        # Confidence based on volume and trader count
        confidence = min(95, len(recent_trades) * 2 + market.unique_traders * 0.5)

        return {
            "market_id": market_id,
            "current_price": market.current_price_a,
            "predicted_price": max(0, min(1, predicted_price)),
            "confidence": confidence,
            "trend": "up" if predicted_price > market.current_price_a else "down",
            "volume_24h": market.volume_24h,
        }

    def _get_volume_by_time(self, market_id: int, time_unit: str) -> List[Dict]:
        """Get volume data grouped by time unit"""
        if time_unit == "hour":
            trades = (
                self.db.query(
                    func.date_trunc("hour", Trade.created_at).label("time"),
                    func.sum(Trade.total_value).label("volume"),
                )
                .filter(Trade.market_id == market_id)
                .group_by(func.date_trunc("hour", Trade.created_at))
                .order_by("time")
                .all()
            )
        else:  # day
            trades = (
                self.db.query(
                    func.date_trunc("day", Trade.created_at).label("time"),
                    func.sum(Trade.total_value).label("volume"),
                )
                .filter(Trade.market_id == market_id)
                .group_by(func.date_trunc("day", Trade.created_at))
                .order_by("time")
                .all()
            )

        return [
            {"time": str(trade.time), "volume": float(trade.volume)} for trade in trades
        ]

    def _get_volume_by_category(self) -> List[Dict]:
        """Get trading volume by market category"""
        result = (
            self.db.query(Market.category, func.sum(Trade.total_value).label("volume"))
            .join(Trade, Market.id == Trade.market_id)
            .group_by(Market.category)
            .order_by(desc("volume"))
            .all()
        )

        return [
            {"category": str(r.category), "volume": float(r.volume)} for r in result
        ]

    def _get_trending_markets(self, limit: int = 10) -> List[Dict]:
        """Get trending markets based on recent activity"""
        trending = (
            self.db.query(Market)
            .filter(Market.status == "open")
            .order_by(desc(Market.trending_score))
            .limit(limit)
            .all()
        )

        return [
            {
                "id": market.id,
                "title": market.title,
                "trending_score": market.trending_score,
                "volume_24h": market.volume_24h,
                "current_price_a": market.current_price_a,
            }
            for market in trending
        ]

    def _get_user_market_preferences(self, user_id: int) -> List[Dict]:
        """Get user's preferred market categories"""
        result = (
            self.db.query(
                Market.category,
                func.count(Trade.id).label("trade_count"),
                func.sum(Trade.total_value).label("total_volume"),
            )
            .join(Trade, Market.id == Trade.market_id)
            .filter(Trade.user_id == user_id)
            .group_by(Market.category)
            .order_by(desc("total_volume"))
            .all()
        )

        return [
            {
                "category": str(r.category),
                "trade_count": r.trade_count,
                "total_volume": float(r.total_volume),
            }
            for r in result
        ]


# Global analytics service instance
analytics_service = AnalyticsService()
