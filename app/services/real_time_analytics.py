"""
Real-time Analytics Service for Opinion Market
Provides live market insights, user behavior analysis, and performance metrics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis as redis_sync
import redis.asyncio as redis
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import websockets
from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class MarketMetrics:
    """Real-time market metrics"""

    market_id: int
    current_price: float
    price_change_24h: float
    volume_24h: float
    volume_change_24h: float
    participant_count: int
    active_traders: int
    volatility_score: float
    momentum_score: float
    liquidity_score: float
    social_activity: int
    news_mentions: int
    sentiment_score: float
    prediction_accuracy: float
    last_updated: datetime


@dataclass
class UserMetrics:
    """User behavior metrics"""

    user_id: int
    total_trades: int
    successful_trades: int
    total_volume: float
    avg_trade_size: float
    win_rate: float
    profit_loss: float
    risk_score: float
    trading_frequency: float
    preferred_categories: List[str]
    last_active: datetime


@dataclass
class SystemMetrics:
    """System performance metrics"""

    total_users: int
    active_users: int
    total_markets: int
    active_markets: int
    total_volume_24h: float
    total_trades_24h: int
    avg_response_time: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    database_connections: int
    cache_hit_rate: float
    last_updated: datetime


class RealTimeAnalyticsService:
    """Real-time analytics service for market insights"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.market_data_cache: Dict[int, MarketMetrics] = {}
        self.user_data_cache: Dict[int, UserMetrics] = {}
        self.system_metrics = SystemMetrics(
            total_users=0,
            active_users=0,
            total_markets=0,
            active_markets=0,
            total_volume_24h=0.0,
            total_trades_24h=0,
            avg_response_time=0.0,
            error_rate=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            database_connections=0,
            cache_hit_rate=0.0,
            last_updated=datetime.utcnow(),
        )

        # Data structures for real-time analysis
        self.price_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.user_activity: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))

    async def initialize(self):
        """Initialize the real-time analytics service"""
        logger.info("Initializing Real-time Analytics Service")

        # Start background tasks
        asyncio.create_task(self._update_market_metrics())
        asyncio.create_task(self._update_user_metrics())
        asyncio.create_task(self._update_system_metrics())
        asyncio.create_task(self._broadcast_updates())

        # Load initial data
        await self._load_initial_data()

        logger.info("Real-time Analytics Service initialized successfully")

    async def _load_initial_data(self):
        """Load initial market and user data"""
        try:
            # Load market data
            markets = await self._get_all_markets()
            for market in markets:
                await self._calculate_market_metrics(market["id"])

            # Load user data
            users = await self._get_active_users()
            for user in users:
                await self._calculate_user_metrics(user["id"])

            # Calculate system metrics
            await self._calculate_system_metrics()

        except Exception as e:
            logger.error(f"Error loading initial data: {e}")

    async def _update_market_metrics(self):
        """Periodically update market metrics"""
        while True:
            try:
                await asyncio.sleep(30)  # Update every 30 seconds

                markets = await self._get_active_markets()
                for market in markets:
                    await self._calculate_market_metrics(market["id"])

                # Cache updated metrics
                await self._cache_market_metrics()

            except Exception as e:
                logger.error(f"Error updating market metrics: {e}")

    async def _update_user_metrics(self):
        """Periodically update user metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute

                users = await self._get_active_users()
                for user in users:
                    await self._calculate_user_metrics(user["id"])

                # Cache updated metrics
                await self._cache_user_metrics()

            except Exception as e:
                logger.error(f"Error updating user metrics: {e}")

    async def _update_system_metrics(self):
        """Periodically update system metrics"""
        while True:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds

                await self._calculate_system_metrics()

                # Cache system metrics
                await self._cache_system_metrics()

            except Exception as e:
                logger.error(f"Error updating system metrics: {e}")

    async def _calculate_market_metrics(self, market_id: int):
        """Calculate real-time metrics for a specific market"""
        try:
            # Get current market data
            market_data = await self._get_market_data(market_id)
            if not market_data:
                return

            # Get price history
            price_history = await self._get_price_history(market_id)
            if price_history:
                self.price_history[market_id].extend(price_history)

            # Calculate metrics
            current_price = market_data.get("current_price", 0.5)
            price_24h_ago = self._get_price_24h_ago(market_id)
            price_change_24h = (
                ((current_price - price_24h_ago) / price_24h_ago * 100)
                if price_24h_ago
                else 0
            )

            volume_24h = await self._get_volume_24h(market_id)
            volume_change_24h = await self._calculate_volume_change(market_id)

            participant_count = market_data.get("participant_count", 0)
            active_traders = await self._get_active_traders(market_id)

            volatility_score = self._calculate_volatility(market_id)
            momentum_score = self._calculate_momentum(market_id)
            liquidity_score = self._calculate_liquidity(market_id)

            social_activity = await self._get_social_activity(market_id)
            news_mentions = await self._get_news_mentions(market_id)
            sentiment_score = await self._get_sentiment_score(market_id)

            prediction_accuracy = await self._get_prediction_accuracy(market_id)

            metrics = MarketMetrics(
                market_id=market_id,
                current_price=current_price,
                price_change_24h=price_change_24h,
                volume_24h=volume_24h,
                volume_change_24h=volume_change_24h,
                participant_count=participant_count,
                active_traders=active_traders,
                volatility_score=volatility_score,
                momentum_score=momentum_score,
                liquidity_score=liquidity_score,
                social_activity=social_activity,
                news_mentions=news_mentions,
                sentiment_score=sentiment_score,
                prediction_accuracy=prediction_accuracy,
                last_updated=datetime.utcnow(),
            )

            self.market_data_cache[market_id] = metrics

        except Exception as e:
            logger.error(f"Error calculating market metrics for {market_id}: {e}")

    async def _calculate_user_metrics(self, user_id: int):
        """Calculate metrics for a specific user"""
        try:
            # Get user trading data
            trading_data = await self._get_user_trading_data(user_id)
            if not trading_data:
                return

            total_trades = trading_data.get("total_trades", 0)
            successful_trades = trading_data.get("successful_trades", 0)
            total_volume = trading_data.get("total_volume", 0.0)
            avg_trade_size = total_volume / total_trades if total_trades > 0 else 0.0
            win_rate = (
                (successful_trades / total_trades * 100) if total_trades > 0 else 0.0
            )
            profit_loss = trading_data.get("profit_loss", 0.0)

            risk_score = self._calculate_user_risk_score(user_id)
            trading_frequency = await self._calculate_trading_frequency(user_id)
            preferred_categories = await self._get_preferred_categories(user_id)

            metrics = UserMetrics(
                user_id=user_id,
                total_trades=total_trades,
                successful_trades=successful_trades,
                total_volume=total_volume,
                avg_trade_size=avg_trade_size,
                win_rate=win_rate,
                profit_loss=profit_loss,
                risk_score=risk_score,
                trading_frequency=trading_frequency,
                preferred_categories=preferred_categories,
                last_active=datetime.utcnow(),
            )

            self.user_data_cache[user_id] = metrics

        except Exception as e:
            logger.error(f"Error calculating user metrics for {user_id}: {e}")

    async def _calculate_system_metrics(self):
        """Calculate system-wide metrics"""
        try:
            total_users = await self._get_total_users()
            active_users = await self._get_active_users_count()
            total_markets = await self._get_total_markets()
            active_markets = await self._get_active_markets_count()
            total_volume_24h = await self._get_total_volume_24h()
            total_trades_24h = await self._get_total_trades_24h()

            avg_response_time = await self._get_avg_response_time()
            error_rate = await self._get_error_rate()
            cpu_usage = await self._get_cpu_usage()
            memory_usage = await self._get_memory_usage()
            database_connections = await self._get_database_connections()
            cache_hit_rate = await self._get_cache_hit_rate()

            self.system_metrics = SystemMetrics(
                total_users=total_users,
                active_users=active_users,
                total_markets=total_markets,
                active_markets=active_markets,
                total_volume_24h=total_volume_24h,
                total_trades_24h=total_trades_24h,
                avg_response_time=avg_response_time,
                error_rate=error_rate,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                database_connections=database_connections,
                cache_hit_rate=cache_hit_rate,
                last_updated=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error calculating system metrics: {e}")

    async def get_market_metrics(self, market_id: int) -> Optional[MarketMetrics]:
        """Get metrics for a specific market"""
        return self.market_data_cache.get(market_id)

    async def get_user_metrics(self, user_id: int) -> Optional[UserMetrics]:
        """Get metrics for a specific user"""
        return self.user_data_cache.get(user_id)

    async def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics"""
        return self.system_metrics

    async def get_top_markets(self, limit: int = 10) -> List[MarketMetrics]:
        """Get top markets by volume"""
        markets = list(self.market_data_cache.values())
        markets.sort(key=lambda x: x.volume_24h, reverse=True)
        return markets[:limit]

    async def get_top_traders(self, limit: int = 10) -> List[UserMetrics]:
        """Get top traders by volume"""
        users = list(self.user_data_cache.values())
        users.sort(key=lambda x: x.total_volume, reverse=True)
        return users[:limit]

    async def get_market_trends(
        self, market_id: int, hours: int = 24
    ) -> Dict[str, List[float]]:
        """Get market trends over time"""
        try:
            # Get price and volume data for the specified time period
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            price_data = await self._get_historical_prices(
                market_id, start_time, end_time
            )
            volume_data = await self._get_historical_volumes(
                market_id, start_time, end_time
            )

            return {
                "prices": [p["price"] for p in price_data],
                "volumes": [v["volume"] for v in volume_data],
                "timestamps": [p["timestamp"] for p in price_data],
            }

        except Exception as e:
            logger.error(f"Error getting market trends: {e}")
            return {"prices": [], "volumes": [], "timestamps": []}

    async def add_websocket_connection(self, connection_id: str, websocket: WebSocket):
        """Add a new WebSocket connection for real-time updates"""
        self.websocket_connections[connection_id] = websocket

    async def remove_websocket_connection(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.websocket_connections:
            del self.websocket_connections[connection_id]

    async def _broadcast_updates(self):
        """Broadcast real-time updates to connected clients"""
        while True:
            try:
                await asyncio.sleep(5)  # Broadcast every 5 seconds

                if not self.websocket_connections:
                    continue

                # Prepare update data
                update_data = {
                    "type": "analytics_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "system_metrics": {
                        "total_users": self.system_metrics.total_users,
                        "active_users": self.system_metrics.active_users,
                        "total_volume_24h": self.system_metrics.total_volume_24h,
                        "total_trades_24h": self.system_metrics.total_trades_24h,
                    },
                    "top_markets": [
                        {
                            "market_id": m.market_id,
                            "current_price": m.current_price,
                            "volume_24h": m.volume_24h,
                            "price_change_24h": m.price_change_24h,
                        }
                        for m in (await self.get_top_markets(5))
                    ],
                }

                # Broadcast to all connected clients
                disconnected = []
                for connection_id, websocket in self.websocket_connections.items():
                    try:
                        await websocket.send_text(json.dumps(update_data))
                    except Exception as e:
                        logger.error(f"Error broadcasting to {connection_id}: {e}")
                        disconnected.append(connection_id)

                # Remove disconnected clients
                for connection_id in disconnected:
                    await self.remove_websocket_connection(connection_id)

            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")

    async def _cache_market_metrics(self):
        """Cache market metrics in Redis"""
        try:
            for market_id, metrics in self.market_data_cache.items():
                cache_key = f"market_metrics:{market_id}"
                cache_data = {
                    "current_price": metrics.current_price,
                    "price_change_24h": metrics.price_change_24h,
                    "volume_24h": metrics.volume_24h,
                    "participant_count": metrics.participant_count,
                    "volatility_score": metrics.volatility_score,
                    "last_updated": metrics.last_updated.isoformat(),
                }
                await self.redis.setex(
                    cache_key, 300, json.dumps(cache_data)
                )  # 5 minutes TTL
        except Exception as e:
            logger.error(f"Error caching market metrics: {e}")

    async def _cache_user_metrics(self):
        """Cache user metrics in Redis"""
        try:
            for user_id, metrics in self.user_data_cache.items():
                cache_key = f"user_metrics:{user_id}"
                cache_data = {
                    "total_trades": metrics.total_trades,
                    "win_rate": metrics.win_rate,
                    "total_volume": metrics.total_volume,
                    "profit_loss": metrics.profit_loss,
                    "last_active": metrics.last_active.isoformat(),
                }
                await self.redis.setex(
                    cache_key, 600, json.dumps(cache_data)
                )  # 10 minutes TTL
        except Exception as e:
            logger.error(f"Error caching user metrics: {e}")

    async def _cache_system_metrics(self):
        """Cache system metrics in Redis"""
        try:
            cache_key = "system_metrics"
            cache_data = {
                "total_users": self.system_metrics.total_users,
                "active_users": self.system_metrics.active_users,
                "total_volume_24h": self.system_metrics.total_volume_24h,
                "total_trades_24h": self.system_metrics.total_trades_24h,
                "avg_response_time": self.system_metrics.avg_response_time,
                "error_rate": self.system_metrics.error_rate,
                "last_updated": self.system_metrics.last_updated.isoformat(),
            }
            await self.redis.setex(
                cache_key, 60, json.dumps(cache_data)
            )  # 1 minute TTL
        except Exception as e:
            logger.error(f"Error caching system metrics: {e}")

    # Helper methods for data retrieval and calculations
    def _get_price_24h_ago(self, market_id: int) -> float:
        """Get price from 24 hours ago"""
        prices = list(self.price_history[market_id])
        if len(prices) >= 24:
            return prices[-24]
        return prices[0] if prices else 0.5

    def _calculate_volatility(self, market_id: int) -> float:
        """Calculate price volatility"""
        prices = list(self.price_history[market_id])
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

    def _calculate_momentum(self, market_id: int) -> float:
        """Calculate price momentum"""
        prices = list(self.price_history[market_id])
        if len(prices) < 2:
            return 0.0
        return (prices[-1] - prices[0]) / prices[0]

    def _calculate_liquidity(self, market_id: int) -> float:
        """Calculate liquidity score"""
        volumes = list(self.volume_history[market_id])
        if not volumes:
            return 0.0
        return np.mean(volumes) / 1000  # Normalize

    def _calculate_user_risk_score(self, user_id: int) -> float:
        """Calculate user risk score"""
        # Implementation depends on your risk model
        return 0.5

    # Database query methods (implementations depend on your models)
    async def _get_all_markets(self) -> List[Dict]:
        """Get all markets from database"""
        # Implementation depends on your database models
        return [{"id": 1, "title": "Test Market"}]

    async def _get_active_markets(self) -> List[Dict]:
        """Get active markets from database"""
        # Implementation depends on your database models
        return [{"id": 1, "title": "Test Market"}]

    async def _get_market_data(self, market_id: int) -> Optional[Dict]:
        """Get market data from database"""
        # Implementation depends on your database models
        return {
            "current_price": 0.55,
            "participant_count": 150,
            "total_volume": 100000.0,
        }

    async def _get_price_history(self, market_id: int) -> List[float]:
        """Get price history from database"""
        # Implementation depends on your database models
        return [0.5, 0.52, 0.48, 0.55, 0.53]

    async def _get_volume_24h(self, market_id: int) -> float:
        """Get 24-hour volume from database"""
        # Implementation depends on your database models
        return 50000.0

    async def _calculate_volume_change(self, market_id: int) -> float:
        """Calculate volume change percentage"""
        # Implementation depends on your database models
        return 5.2

    async def _get_active_traders(self, market_id: int) -> int:
        """Get number of active traders"""
        # Implementation depends on your database models
        return 45

    async def _get_social_activity(self, market_id: int) -> int:
        """Get social media activity count"""
        # Implementation depends on your database models
        return 25

    async def _get_news_mentions(self, market_id: int) -> int:
        """Get news mentions count"""
        # Implementation depends on your database models
        return 10

    async def _get_sentiment_score(self, market_id: int) -> float:
        """Get sentiment score"""
        # Implementation depends on your database models
        return 0.6

    async def _get_prediction_accuracy(self, market_id: int) -> float:
        """Get prediction accuracy"""
        # Implementation depends on your database models
        return 0.75

    async def _get_active_users(self) -> List[Dict]:
        """Get active users from database"""
        # Implementation depends on your database models
        return [{"id": 1, "username": "testuser"}]

    async def _get_user_trading_data(self, user_id: int) -> Optional[Dict]:
        """Get user trading data from database"""
        # Implementation depends on your database models
        return {
            "total_trades": 50,
            "successful_trades": 35,
            "total_volume": 25000.0,
            "profit_loss": 1500.0,
        }

    async def _calculate_trading_frequency(self, user_id: int) -> float:
        """Calculate user trading frequency"""
        # Implementation depends on your database models
        return 2.5

    async def _get_preferred_categories(self, user_id: int) -> List[str]:
        """Get user's preferred trading categories"""
        # Implementation depends on your database models
        return ["technology", "finance"]

    async def _get_total_users(self) -> int:
        """Get total user count"""
        # Implementation depends on your database models
        return 1000

    async def _get_active_users_count(self) -> int:
        """Get active user count"""
        # Implementation depends on your database models
        return 150

    async def _get_total_markets(self) -> int:
        """Get total market count"""
        # Implementation depends on your database models
        return 50

    async def _get_active_markets_count(self) -> int:
        """Get active market count"""
        # Implementation depends on your database models
        return 25

    async def _get_total_volume_24h(self) -> float:
        """Get total 24-hour volume"""
        # Implementation depends on your database models
        return 500000.0

    async def _get_total_trades_24h(self) -> int:
        """Get total 24-hour trades"""
        # Implementation depends on your database models
        return 1000

    async def _get_avg_response_time(self) -> float:
        """Get average response time"""
        # Implementation depends on your monitoring
        return 0.15

    async def _get_error_rate(self) -> float:
        """Get error rate"""
        # Implementation depends on your monitoring
        return 0.02

    async def _get_cpu_usage(self) -> float:
        """Get CPU usage"""
        # Implementation depends on your monitoring
        return 0.45

    async def _get_memory_usage(self) -> float:
        """Get memory usage"""
        # Implementation depends on your monitoring
        return 0.60

    async def _get_database_connections(self) -> int:
        """Get database connection count"""
        # Implementation depends on your monitoring
        return 25

    async def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        # Implementation depends on your monitoring
        return 0.85

    async def _get_historical_prices(
        self, market_id: int, start_time: datetime, end_time: datetime
    ) -> List[Dict]:
        """Get historical price data"""
        # Implementation depends on your database models
        return [
            {"price": 0.5, "timestamp": start_time.isoformat()},
            {"price": 0.52, "timestamp": end_time.isoformat()},
        ]

    async def _get_historical_volumes(
        self, market_id: int, start_time: datetime, end_time: datetime
    ) -> List[Dict]:
        """Get historical volume data"""
        # Implementation depends on your database models
        return [
            {"volume": 1000.0, "timestamp": start_time.isoformat()},
            {"volume": 1200.0, "timestamp": end_time.isoformat()},
        ]


# Factory function
async def get_real_time_analytics_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> RealTimeAnalyticsService:
    """Get real-time analytics service instance"""
    service = RealTimeAnalyticsService(redis_client, db_session)
    await service.initialize()
    return service
