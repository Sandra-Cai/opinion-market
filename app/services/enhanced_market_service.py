"""
Enhanced Market Service for Opinion Market
Provides advanced market management, analytics, and real-time updates
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_

from app.services.base_service import BaseService, CacheableService, MetricsService, BackgroundTaskService
from app.core.database import get_db_session
from app.core.cache import cache
from app.core.logging import log_trading_event, log_system_metric
from app.core.config import settings
from app.models.market import Market, MarketStatus, MarketCategory
from app.models.trade import Trade
from app.models.user import User


class EnhancedMarketService(BaseService, CacheableService, MetricsService, BackgroundTaskService):
    """Enhanced market service with advanced features"""
    
    def __init__(self):
        BaseService.__init__(self)
        CacheableService.__init__(self, cache_ttl=300)
        MetricsService.__init__(self)
        BackgroundTaskService.__init__(self)
        
        self.trending_markets = []
        self.market_analytics = {}
        self.price_alerts = {}
    
    async def _initialize_internal(self) -> None:
        """Initialize market service"""
        # Start background tasks
        self.start_background_task("update_trending_markets", self._update_trending_markets, interval=300)  # 5 minutes
        self.start_background_task("update_market_analytics", self._update_market_analytics, interval=600)  # 10 minutes
        self.start_background_task("check_price_alerts", self._check_price_alerts, interval=60)  # 1 minute
        self.start_background_task("update_market_quality_scores", self._update_market_quality_scores, interval=1800)  # 30 minutes
        
        # Load initial data
        await self._load_initial_data()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup market service"""
        # Background tasks are cleaned up by parent class
        pass
    
    async def _load_initial_data(self):
        """Load initial market data"""
        with get_db_session() as db:
            # Load trending markets
            await self._update_trending_markets()
            
            # Load market analytics
            await self._update_market_analytics()
    
    async def _update_trending_markets(self):
        """Update trending markets list"""
        try:
            with get_db_session() as db:
                markets = (
                    db.query(Market)
                    .filter(Market.status == MarketStatus.OPEN)
                    .filter(Market.trending_score >= settings.MARKET_TRENDING_THRESHOLD)
                    .order_by(desc(Market.trending_score))
                    .limit(20)
                    .all()
                )
                
                self.trending_markets = [market.id for market in markets]
                
                # Cache trending markets
                cache_key = self.get_cache_key("trending_markets")
                self.set_cached(cache_key, self.trending_markets, ttl=300)
                
                self.record_metric("trending_markets_count", len(self.trending_markets))
                
        except Exception as e:
            self.logger.error(f"Failed to update trending markets: {e}")
    
    async def _update_market_analytics(self):
        """Update market analytics"""
        try:
            with get_db_session() as db:
                # Calculate platform-wide analytics
                total_markets = db.query(Market).count()
                active_markets = db.query(Market).filter(Market.status == MarketStatus.OPEN).count()
                total_volume = db.query(func.sum(Market.volume_total)).scalar() or 0
                
                # 24h volume
                yesterday = datetime.utcnow() - timedelta(days=1)
                volume_24h = db.query(func.sum(Market.volume_24h)).scalar() or 0
                
                # Category breakdown
                category_stats = (
                    db.query(Market.category, func.count(Market.id), func.sum(Market.volume_total))
                    .group_by(Market.category)
                    .all()
                )
                
                analytics = {
                    "total_markets": total_markets,
                    "active_markets": active_markets,
                    "total_volume": float(total_volume),
                    "volume_24h": float(volume_24h),
                    "categories": [
                        {
                            "category": cat.value,
                            "count": count,
                            "volume": float(volume or 0)
                        }
                        for cat, count, volume in category_stats
                    ],
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                self.market_analytics = analytics
                
                # Cache analytics
                cache_key = self.get_cache_key("market_analytics")
                self.set_cached(cache_key, analytics, ttl=600)
                
                self.record_metric("total_markets", total_markets)
                self.record_metric("active_markets", active_markets)
                self.record_metric("total_volume", total_volume)
                self.record_metric("volume_24h", volume_24h)
                
        except Exception as e:
            self.logger.error(f"Failed to update market analytics: {e}")
    
    async def _check_price_alerts(self):
        """Check and trigger price alerts"""
        try:
            with get_db_session() as db:
                # Get all active price alerts
                # This would typically be stored in a separate table
                # For now, we'll check for significant price movements
                
                markets = db.query(Market).filter(Market.status == MarketStatus.OPEN).all()
                
                for market in markets:
                    # Check for significant price movements (5% change)
                    price_change_a = abs(market.current_price_a - 0.5) / 0.5
                    price_change_b = abs(market.current_price_b - 0.5) / 0.5
                    
                    if price_change_a > 0.05 or price_change_b > 0.05:
                        # Log significant price movement
                        log_trading_event(
                            "significant_price_movement",
                            market.id,
                            0,  # No specific user
                            0,  # No specific amount
                            price_change_a=price_change_a,
                            price_change_b=price_change_b,
                            current_price_a=market.current_price_a,
                            current_price_b=market.current_price_b
                        )
                        
                        # Clear market cache to force refresh
                        cache_key = self.get_cache_key("market", market.id)
                        self.delete_cached(cache_key)
                
        except Exception as e:
            self.logger.error(f"Failed to check price alerts: {e}")
    
    async def _update_market_quality_scores(self):
        """Update market quality scores"""
        try:
            with get_db_session() as db:
                markets = db.query(Market).filter(Market.status == MarketStatus.OPEN).all()
                
                for market in markets:
                    old_score = market.market_quality_score
                    new_score = market.calculate_quality_score()
                    
                    if abs(new_score - old_score) > 5:  # Significant change
                        db.commit()
                        
                        # Log quality score change
                        log_system_metric("market_quality_score_change", new_score - old_score, {
                            "market_id": market.id,
                            "old_score": old_score,
                            "new_score": new_score
                        })
                
        except Exception as e:
            self.logger.error(f"Failed to update market quality scores: {e}")
    
    def get_trending_markets(self, limit: int = 10) -> List[int]:
        """Get trending market IDs"""
        return self.trending_markets[:limit]
    
    def get_market_analytics(self) -> Dict[str, Any]:
        """Get market analytics"""
        return self.market_analytics
    
    def get_market_summary(self, market_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive market summary"""
        cache_key = self.get_cache_key("market_summary", market_id)
        cached_summary = self.get_cached(cache_key)
        
        if cached_summary:
            return cached_summary
        
        try:
            with get_db_session() as db:
                market = db.query(Market).filter(Market.id == market_id).first()
                if not market:
                    return None
                
                summary = market.get_market_summary()
                
                # Add additional analytics
                summary["analytics"] = {
                    "price_impact": market.price_impact,
                    "implied_probability": market.calculate_implied_probability(),
                    "trading_limits": market.get_trading_limits(),
                    "is_trending": market.is_trending,
                    "time_until_close": market.time_until_close
                }
                
                # Cache the summary
                self.set_cached(cache_key, summary, ttl=60)
                
                return summary
                
        except Exception as e:
            self.logger.error(f"Failed to get market summary for {market_id}: {e}")
            return None
    
    def get_market_price_history(self, market_id: int, hours: int = 24) -> List[Dict[str, Any]]:
        """Get market price history"""
        cache_key = self.get_cache_key("price_history", market_id, hours)
        cached_history = self.get_cached(cache_key)
        
        if cached_history:
            return cached_history
        
        try:
            with get_db_session() as db:
                # Get recent trades for price history
                since = datetime.utcnow() - timedelta(hours=hours)
                trades = (
                    db.query(Trade)
                    .filter(Trade.market_id == market_id)
                    .filter(Trade.created_at >= since)
                    .order_by(Trade.created_at)
                    .all()
                )
                
                # Build price history from trades
                price_history = []
                for trade in trades:
                    price_history.append({
                        "timestamp": trade.created_at.isoformat(),
                        "price_a": trade.price_a,
                        "price_b": trade.price_b,
                        "volume": trade.amount,
                        "trade_type": trade.trade_type
                    })
                
                # Cache the history
                self.set_cached(cache_key, price_history, ttl=300)
                
                return price_history
                
        except Exception as e:
            self.logger.error(f"Failed to get price history for market {market_id}: {e}")
            return []
    
    def get_market_volume_analytics(self, market_id: int) -> Dict[str, Any]:
        """Get market volume analytics"""
        cache_key = self.get_cache_key("volume_analytics", market_id)
        cached_analytics = self.get_cached(cache_key)
        
        if cached_analytics:
            return cached_analytics
        
        try:
            with get_db_session() as db:
                market = db.query(Market).filter(Market.id == market_id).first()
                if not market:
                    return {}
                
                # Calculate volume metrics
                now = datetime.utcnow()
                volume_1h = 0
                volume_24h = market.volume_24h
                volume_7d = 0
                
                # Get 1h volume
                since_1h = now - timedelta(hours=1)
                trades_1h = (
                    db.query(Trade)
                    .filter(Trade.market_id == market_id)
                    .filter(Trade.created_at >= since_1h)
                    .all()
                )
                volume_1h = sum(trade.amount for trade in trades_1h)
                
                # Get 7d volume
                since_7d = now - timedelta(days=7)
                trades_7d = (
                    db.query(Trade)
                    .filter(Trade.market_id == market_id)
                    .filter(Trade.created_at >= since_7d)
                    .all()
                )
                volume_7d = sum(trade.amount for trade in trades_7d)
                
                analytics = {
                    "volume_1h": volume_1h,
                    "volume_24h": volume_24h,
                    "volume_7d": volume_7d,
                    "total_volume": market.volume_total,
                    "unique_traders": market.unique_traders,
                    "avg_trade_size": volume_24h / len(trades_1h) if trades_1h else 0,
                    "updated_at": now.isoformat()
                }
                
                # Cache the analytics
                self.set_cached(cache_key, analytics, ttl=300)
                
                return analytics
                
        except Exception as e:
            self.logger.error(f"Failed to get volume analytics for market {market_id}: {e}")
            return {}
    
    def get_market_sentiment(self, market_id: int) -> Dict[str, Any]:
        """Get market sentiment analysis"""
        cache_key = self.get_cache_key("sentiment", market_id)
        cached_sentiment = self.get_cached(cache_key)
        
        if cached_sentiment:
            return cached_sentiment
        
        try:
            with get_db_session() as db:
                market = db.query(Market).filter(Market.id == market_id).first()
                if not market:
                    return {}
                
                # Calculate sentiment based on recent trading activity
                now = datetime.utcnow()
                since_24h = now - timedelta(hours=24)
                
                # Get recent trades
                recent_trades = (
                    db.query(Trade)
                    .filter(Trade.market_id == market_id)
                    .filter(Trade.created_at >= since_24h)
                    .all()
                )
                
                if not recent_trades:
                    return {"sentiment": "neutral", "confidence": 0.0}
                
                # Analyze trade patterns
                buy_volume_a = sum(t.amount for t in recent_trades if t.outcome == "outcome_a" and t.trade_type == "buy")
                sell_volume_a = sum(t.amount for t in recent_trades if t.outcome == "outcome_a" and t.trade_type == "sell")
                buy_volume_b = sum(t.amount for t in recent_trades if t.outcome == "outcome_b" and t.trade_type == "buy")
                sell_volume_b = sum(t.amount for t in recent_trades if t.outcome == "outcome_b" and t.trade_type == "sell")
                
                # Calculate sentiment
                net_sentiment_a = buy_volume_a - sell_volume_a
                net_sentiment_b = buy_volume_b - sell_volume_b
                
                if net_sentiment_a > net_sentiment_b:
                    sentiment = "bullish_a"
                    confidence = min(abs(net_sentiment_a) / (buy_volume_a + sell_volume_a + 1), 1.0)
                elif net_sentiment_b > net_sentiment_a:
                    sentiment = "bullish_b"
                    confidence = min(abs(net_sentiment_b) / (buy_volume_b + sell_volume_b + 1), 1.0)
                else:
                    sentiment = "neutral"
                    confidence = 0.0
                
                sentiment_data = {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "net_sentiment_a": net_sentiment_a,
                    "net_sentiment_b": net_sentiment_b,
                    "total_volume_24h": sum(t.amount for t in recent_trades),
                    "trade_count_24h": len(recent_trades),
                    "updated_at": now.isoformat()
                }
                
                # Cache the sentiment
                self.set_cached(cache_key, sentiment_data, ttl=300)
                
                return sentiment_data
                
        except Exception as e:
            self.logger.error(f"Failed to get sentiment for market {market_id}: {e}")
            return {"sentiment": "neutral", "confidence": 0.0}
    
    def invalidate_market_cache(self, market_id: int):
        """Invalidate cache for a specific market"""
        patterns = [
            self.get_cache_key("market_summary", market_id),
            self.get_cache_key("price_history", market_id, "*"),
            self.get_cache_key("volume_analytics", market_id),
            self.get_cache_key("sentiment", market_id)
        ]
        
        for pattern in patterns:
            self.delete_cached(pattern)
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service-specific metrics"""
        return {
            "trending_markets_count": len(self.trending_markets),
            "cached_analytics": len(self.market_analytics),
            "background_tasks": len(self.tasks),
            "uptime": self.get_uptime(),
            "metrics": self.get_all_metrics()
        }


# Global market service instance
enhanced_market_service = EnhancedMarketService()
