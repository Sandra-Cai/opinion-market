import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from sqlalchemy.orm import Session
import websockets
import aiohttp
import redis.asyncio as redis

from app.core.database import SessionLocal
from app.models.market import Market, MarketStatus
from app.models.trade import Trade
from app.services.price_feed import price_feed_manager

logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Represents a single market data point"""

    market_id: int
    timestamp: datetime
    price_a: float
    price_b: float
    volume_24h: float
    volume_total: float
    unique_traders: int
    price_change_24h: float
    price_change_1h: float
    liquidity_a: float
    liquidity_b: float
    spread: float
    volatility: float


@dataclass
class MarketAlert:
    """Represents a market alert"""

    market_id: int
    alert_type: str  # price_spike, volume_surge, liquidity_drop, etc.
    message: str
    severity: str  # low, medium, high, critical
    timestamp: datetime
    data: Dict[str, Any]


class MarketDataFeed:
    """Real-time market data feed service"""

    def __init__(self):
        self.redis_client: Optional[redis_sync.Redis] = None
        self.price_feed_manager = price_feed_manager
        self.subscribers: Dict[str, List[Callable]] = {}
        self.market_alerts: List[MarketAlert] = []
        self.data_cache: Dict[int, MarketDataPoint] = {}
        self.alert_thresholds = {
            "price_spike": 0.1,  # 10% price change
            "volume_surge": 5.0,  # 5x normal volume
            "liquidity_drop": 0.3,  # 30% liquidity drop
            "spread_widening": 0.05,  # 5% spread increase
        }

    async def initialize(self, redis_url: str):
        """Initialize the market data feed"""
        self.redis_client = redis.from_url(redis_url)
        await self.redis_client.ping()
        logger.info("Market data feed initialized")

    async def start_data_feed(self):
        """Start the market data feed"""
        asyncio.create_task(self._market_data_loop())
        asyncio.create_task(self._alert_monitoring_loop())
        asyncio.create_task(self._data_cleanup_loop())
        logger.info("Market data feed started")

    async def _market_data_loop(self):
        """Main loop for collecting and broadcasting market data"""
        while True:
            try:
                await self._collect_market_data()
                await self._broadcast_market_updates()
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in market data loop: {e}")
                await asyncio.sleep(10)

    async def _collect_market_data(self):
        """Collect market data from database"""
        db = SessionLocal()
        try:
            # Get active markets
            markets = (
                db.query(Market).filter(Market.status == MarketStatus.ACTIVE).all()
            )

            for market in markets:
                # Calculate current prices
                current_price_a = market.current_price_a
                current_price_b = market.current_price_b

                # Get 24h volume
                yesterday = datetime.utcnow() - timedelta(days=1)
                volume_24h = (
                    db.query(Trade)
                    .filter(Trade.market_id == market.id, Trade.created_at >= yesterday)
                    .with_entities(db.func.sum(Trade.total_value))
                    .scalar()
                    or 0.0
                )

                # Get unique traders in last 24h
                unique_traders = (
                    db.query(Trade)
                    .filter(Trade.market_id == market.id, Trade.created_at >= yesterday)
                    .with_entities(db.func.count(db.func.distinct(Trade.user_id)))
                    .scalar()
                    or 0
                )

                # Calculate price changes
                price_change_24h = self._calculate_price_change(market.id, 24)
                price_change_1h = self._calculate_price_change(market.id, 1)

                # Calculate liquidity
                liquidity_a = market.liquidity_pool_a or 0.0
                liquidity_b = market.liquidity_pool_b or 0.0

                # Calculate spread
                spread = abs(current_price_a - current_price_b) / max(
                    current_price_a, current_price_b
                )

                # Calculate volatility
                volatility = self._calculate_volatility(market.id)

                # Create data point
                data_point = MarketDataPoint(
                    market_id=market.id,
                    timestamp=datetime.utcnow(),
                    price_a=current_price_a,
                    price_b=current_price_b,
                    volume_24h=volume_24h,
                    volume_total=market.volume_total or 0.0,
                    unique_traders=unique_traders,
                    price_change_24h=price_change_24h,
                    price_change_1h=price_change_1h,
                    liquidity_a=liquidity_a,
                    liquidity_b=liquidity_b,
                    spread=spread,
                    volatility=volatility,
                )

                # Cache data point
                self.data_cache[market.id] = data_point

                # Check for alerts
                await self._check_market_alerts(data_point)

        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
        finally:
            db.close()

    async def _broadcast_market_updates(self):
        """Broadcast market updates to subscribers"""
        if not self.redis_client:
            return

        try:
            # Prepare market data for broadcast
            market_updates = {}
            for market_id, data_point in self.data_cache.items():
                market_updates[market_id] = {
                    "timestamp": data_point.timestamp.isoformat(),
                    "price_a": data_point.price_a,
                    "price_b": data_point.price_b,
                    "volume_24h": data_point.volume_24h,
                    "unique_traders": data_point.unique_traders,
                    "price_change_24h": data_point.price_change_24h,
                    "price_change_1h": data_point.price_change_1h,
                    "liquidity_a": data_point.liquidity_a,
                    "liquidity_b": data_point.liquidity_b,
                    "spread": data_point.spread,
                    "volatility": data_point.volatility,
                }

            # Publish to Redis
            await self.redis_client.publish(
                "market_data_updates", json.dumps(market_updates)
            )

            # Notify WebSocket subscribers
            await self.price_feed_manager.broadcast_market_data(market_updates)

        except Exception as e:
            logger.error(f"Error broadcasting market updates: {e}")

    async def _check_market_alerts(self, data_point: MarketDataPoint):
        """Check for market alerts based on data point"""
        try:
            # Price spike alert
            if abs(data_point.price_change_1h) > self.alert_thresholds["price_spike"]:
                alert = MarketAlert(
                    market_id=data_point.market_id,
                    alert_type="price_spike",
                    message=f"Price spike detected: {data_point.price_change_1h:.2%} change in 1h",
                    severity=(
                        "high" if abs(data_point.price_change_1h) > 0.2 else "medium"
                    ),
                    timestamp=datetime.utcnow(),
                    data={"price_change_1h": data_point.price_change_1h},
                )
                await self._create_alert(alert)

            # Volume surge alert
            if (
                data_point.volume_24h
                > self._get_average_volume(data_point.market_id)
                * self.alert_thresholds["volume_surge"]
            ):
                alert = MarketAlert(
                    market_id=data_point.market_id,
                    alert_type="volume_surge",
                    message=f"Volume surge detected: {data_point.volume_24h:.2f} in 24h",
                    severity="medium",
                    timestamp=datetime.utcnow(),
                    data={"volume_24h": data_point.volume_24h},
                )
                await self._create_alert(alert)

            # Liquidity drop alert
            total_liquidity = data_point.liquidity_a + data_point.liquidity_b
            if total_liquidity < self._get_average_liquidity(data_point.market_id) * (
                1 - self.alert_thresholds["liquidity_drop"]
            ):
                alert = MarketAlert(
                    market_id=data_point.market_id,
                    alert_type="liquidity_drop",
                    message=f"Liquidity drop detected: {total_liquidity:.2f} total liquidity",
                    severity="high",
                    timestamp=datetime.utcnow(),
                    data={"total_liquidity": total_liquidity},
                )
                await self._create_alert(alert)

            # Spread widening alert
            if data_point.spread > self._get_average_spread(data_point.market_id) * (
                1 + self.alert_thresholds["spread_widening"]
            ):
                alert = MarketAlert(
                    market_id=data_point.market_id,
                    alert_type="spread_widening",
                    message=f"Spread widening detected: {data_point.spread:.2%} spread",
                    severity="medium",
                    timestamp=datetime.utcnow(),
                    data={"spread": data_point.spread},
                )
                await self._create_alert(alert)

        except Exception as e:
            logger.error(f"Error checking market alerts: {e}")

    async def _create_alert(self, alert: MarketAlert):
        """Create and broadcast a market alert"""
        try:
            # Add to alerts list
            self.market_alerts.append(alert)

            # Keep only last 100 alerts
            if len(self.market_alerts) > 100:
                self.market_alerts = self.market_alerts[-100:]

            # Publish alert to Redis
            if self.redis_client:
                await self.redis_client.publish(
                    "market_alerts",
                    json.dumps(
                        {
                            "market_id": alert.market_id,
                            "alert_type": alert.alert_type,
                            "message": alert.message,
                            "severity": alert.severity,
                            "timestamp": alert.timestamp.isoformat(),
                            "data": alert.data,
                        }
                    ),
                )

            # Log alert
            logger.warning(f"Market Alert: {alert.message}")

        except Exception as e:
            logger.error(f"Error creating alert: {e}")

    async def _alert_monitoring_loop(self):
        """Monitor and process alerts"""
        while True:
            try:
                # Process critical alerts
                critical_alerts = [
                    a for a in self.market_alerts if a.severity == "critical"
                ]
                for alert in critical_alerts:
                    await self._handle_critical_alert(alert)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _handle_critical_alert(self, alert: MarketAlert):
        """Handle critical market alerts"""
        try:
            # Send notifications to admins
            await self._send_admin_notification(alert)

            # Potentially pause trading if needed
            if alert.alert_type in ["liquidity_drop", "price_spike"]:
                await self._consider_trading_pause(alert)

        except Exception as e:
            logger.error(f"Error handling critical alert: {e}")

    async def _data_cleanup_loop(self):
        """Clean up old data periodically"""
        while True:
            try:
                # Clean up old data points (keep last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                old_market_ids = []

                for market_id, data_point in self.data_cache.items():
                    if data_point.timestamp < cutoff_time:
                        old_market_ids.append(market_id)

                for market_id in old_market_ids:
                    del self.data_cache[market_id]

                # Clean up old alerts (keep last 7 days)
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.market_alerts = [
                    alert
                    for alert in self.market_alerts
                    if alert.timestamp > cutoff_time
                ]

                await asyncio.sleep(3600)  # Clean up every hour

            except Exception as e:
                logger.error(f"Error in data cleanup loop: {e}")
                await asyncio.sleep(3600)

    def _calculate_price_change(self, market_id: int, hours: int) -> float:
        """Calculate price change over specified hours"""
        try:
            db = SessionLocal()
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            # Get average price from trades in the period
            trades = (
                db.query(Trade)
                .filter(Trade.market_id == market_id, Trade.created_at >= cutoff_time)
                .all()
            )

            if not trades:
                return 0.0

            avg_price = sum(trade.price_per_share for trade in trades) / len(trades)

            # Get current price
            market = db.query(Market).filter(Market.id == market_id).first()
            current_price = market.current_price_a if market else 0.5

            return (current_price - avg_price) / avg_price if avg_price > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating price change: {e}")
            return 0.0
        finally:
            db.close()

    def _calculate_volatility(self, market_id: int) -> float:
        """Calculate market volatility"""
        try:
            db = SessionLocal()
            # Get price data from last 24 hours
            yesterday = datetime.utcnow() - timedelta(days=1)

            trades = (
                db.query(Trade)
                .filter(Trade.market_id == market_id, Trade.created_at >= yesterday)
                .order_by(Trade.created_at)
                .all()
            )

            if len(trades) < 2:
                return 0.0

            # Calculate price changes
            price_changes = []
            for i in range(1, len(trades)):
                change = (
                    trades[i].price_per_share - trades[i - 1].price_per_share
                ) / trades[i - 1].price_per_share
                price_changes.append(change)

            # Calculate standard deviation
            if price_changes:
                mean = sum(price_changes) / len(price_changes)
                variance = sum((x - mean) ** 2 for x in price_changes) / len(
                    price_changes
                )
                return variance**0.5

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
        finally:
            db.close()

    def _get_average_volume(self, market_id: int) -> float:
        """Get average daily volume for market"""
        try:
            db = SessionLocal()
            # Get volume from last 7 days
            week_ago = datetime.utcnow() - timedelta(days=7)

            total_volume = (
                db.query(Trade)
                .filter(Trade.market_id == market_id, Trade.created_at >= week_ago)
                .with_entities(db.func.sum(Trade.total_value))
                .scalar()
                or 0.0
            )

            return total_volume / 7  # Average daily volume

        except Exception as e:
            logger.error(f"Error getting average volume: {e}")
            return 1000.0  # Default value
        finally:
            db.close()

    def _get_average_liquidity(self, market_id: int) -> float:
        """Get average liquidity for market"""
        try:
            db = SessionLocal()
            market = db.query(Market).filter(Market.id == market_id).first()
            if market:
                return (market.liquidity_pool_a or 0.0) + (
                    market.liquidity_pool_b or 0.0
                )
            return 1000.0  # Default value

        except Exception as e:
            logger.error(f"Error getting average liquidity: {e}")
            return 1000.0
        finally:
            db.close()

    def _get_average_spread(self, market_id: int) -> float:
        """Get average spread for market"""
        try:
            db = SessionLocal()
            market = db.query(Market).filter(Market.id == market_id).first()
            if market:
                return abs(market.current_price_a - market.current_price_b) / max(
                    market.current_price_a, market.current_price_b
                )
            return 0.05  # Default 5% spread

        except Exception as e:
            logger.error(f"Error getting average spread: {e}")
            return 0.05
        finally:
            db.close()

    async def _send_admin_notification(self, alert: MarketAlert):
        """Send notification to administrators"""
        # This would integrate with the notification service
        logger.info(f"Admin notification: {alert.message}")

    async def _consider_trading_pause(self, alert: MarketAlert):
        """Consider pausing trading for critical alerts"""
        # This would integrate with market management
        logger.warning(
            f"Considering trading pause for market {alert.market_id}: {alert.message}"
        )

    async def get_market_data(self, market_id: int) -> Optional[MarketDataPoint]:
        """Get current market data for a specific market"""
        return self.data_cache.get(market_id)

    async def get_all_market_data(self) -> Dict[int, MarketDataPoint]:
        """Get current market data for all markets"""
        return self.data_cache.copy()

    async def get_market_alerts(
        self, market_id: Optional[int] = None, severity: Optional[str] = None
    ) -> List[MarketAlert]:
        """Get market alerts with optional filtering"""
        alerts = self.market_alerts

        if market_id:
            alerts = [a for a in alerts if a.market_id == market_id]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    async def subscribe_to_market_data(self, callback: Callable):
        """Subscribe to market data updates"""
        if "market_data" not in self.subscribers:
            self.subscribers["market_data"] = []
        self.subscribers["market_data"].append(callback)

    async def subscribe_to_alerts(self, callback: Callable):
        """Subscribe to market alerts"""
        if "alerts" not in self.subscribers:
            self.subscribers["alerts"] = []
        self.subscribers["alerts"].append(callback)


# Global market data feed instance
market_data_feed = MarketDataFeed()


def get_market_data_feed() -> MarketDataFeed:
    """Get the global market data feed instance"""
    return market_data_feed
