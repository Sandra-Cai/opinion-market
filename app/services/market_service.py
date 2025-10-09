"""
Market Service for Opinion Market
Handles market-related business logic including creation, management, and analytics
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import math

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc, asc
from fastapi import HTTPException, status

from app.core.database import get_db, get_redis_client
from app.core.config import settings
from app.core.cache import cache
from app.core.logging import log_trading_event, log_system_metric
from app.core.security_audit import security_auditor, SecurityEventType, SecuritySeverity
from app.models.market import Market, MarketStatus, MarketCategory
from app.models.trade import Trade
from app.models.user import User
from app.schemas.market import MarketCreate, MarketUpdate, MarketResponse, MarketStats


@dataclass
class MarketAnalytics:
    """Market analytics data structure"""
    total_volume: float
    volume_24h: float
    volume_7d: float
    total_trades: int
    trades_24h: int
    trades_7d: int
    unique_traders: int
    price_movement: float
    price_volatility: float
    trending_score: float
    sentiment_score: float
    liquidity_score: float
    risk_score: float
    market_depth: Dict[str, float]
    price_history: List[Dict[str, Any]]
    top_traders: List[Dict[str, Any]]
    category_performance: Dict[str, float]


@dataclass
class MarketRecommendation:
    """Market recommendation data structure"""
    market_id: int
    recommendation_type: str  # "buy", "sell", "hold"
    confidence_score: float
    reasoning: str
    risk_level: str
    expected_return: float
    time_horizon: str


class MarketService:
    """Service for market-related operations"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.cache_ttl = 300  # 5 minutes
        
    async def create_market(self, market_data: MarketCreate, creator_id: int, db: Session) -> Market:
        """Create a new prediction market with comprehensive validation"""
        try:
            # Validate market data
            await self._validate_market_data(market_data)
            
            # Check if user can create markets
            await self._check_market_creation_permissions(creator_id, db)
            
            # Create market
            market = Market(
                title=market_data.title,
                description=market_data.description,
                question=market_data.question,
                category=market_data.category,
                outcome_a=market_data.outcome_a,
                outcome_b=market_data.outcome_b,
                creator_id=creator_id,
                closes_at=market_data.closes_at,
                resolution_criteria=market_data.resolution_criteria,
                initial_liquidity=market_data.initial_liquidity or settings.DEFAULT_LIQUIDITY,
                trading_fee=market_data.trading_fee or settings.DEFAULT_TRADING_FEE,
                status=MarketStatus.OPEN,
                price_a=0.5,  # Initial price
                price_b=0.5,  # Initial price
                volume_total=0.0,
                volume_24h=0.0,
                trending_score=0.0,
                sentiment_score=0.0,
                liquidity_score=0.0,
                risk_score=0.0
            )
            
            db.add(market)
            db.commit()
            db.refresh(market)
            
            # Log market creation
            log_trading_event("market_created", {
                "market_id": market.id,
                "creator_id": creator_id,
                "category": market.category.value,
                "closes_at": market.closes_at.isoformat()
            })
            
            # Clear cache
            await self._clear_market_cache(market.id)
            
            return market
            
        except Exception as e:
            db.rollback()
            log_system_metric("market_creation_error", 1, {"error": str(e)})
            raise
    
    async def get_market_by_id(self, market_id: int, db: Session) -> Optional[Market]:
        """Get market by ID with caching"""
        cache_key = f"market:{market_id}"
        
        # Try cache first
        cached_market = cache.get(cache_key)
        if cached_market:
            return cached_market
        
        # Get from database
        market = db.query(Market).filter(Market.id == market_id).first()
        
        if market:
            # Cache market data
            cache.set(cache_key, market, ttl=self.cache_ttl)
        
        return market
    
    async def list_markets(
        self,
        skip: int = 0,
        limit: int = 20,
        category: Optional[MarketCategory] = None,
        status: Optional[MarketStatus] = None,
        trending: bool = False,
        search: Optional[str] = None,
        db: Session = None
    ) -> Tuple[List[Market], int]:
        """List markets with filtering and pagination"""
        try:
            query = db.query(Market)
            
            # Apply filters
            if category:
                query = query.filter(Market.category == category)
            
            if status:
                query = query.filter(Market.status == status)
            
            if trending:
                query = query.filter(Market.trending_score >= settings.MARKET_TRENDING_THRESHOLD)
            
            if search:
                search_term = f"%{search}%"
                query = query.filter(
                    or_(
                        Market.title.ilike(search_term),
                        Market.description.ilike(search_term),
                        Market.question.ilike(search_term)
                    )
                )
            
            # Get total count
            total = query.count()
            
            # Apply pagination and ordering
            markets = query.order_by(desc(Market.created_at)).offset(skip).limit(limit).all()
            
            return markets, total
            
        except Exception as e:
            log_system_metric("market_listing_error", 1, {"error": str(e)})
            raise
    
    async def get_trending_markets(self, limit: int = 10, db: Session = None) -> List[Market]:
        """Get trending markets based on activity and volume"""
        try:
            markets = (
                db.query(Market)
                .filter(Market.status == MarketStatus.OPEN)
                .filter(Market.trending_score >= settings.MARKET_TRENDING_THRESHOLD)
                .order_by(desc(Market.trending_score))
                .limit(limit)
                .all()
            )
            
            return markets
            
        except Exception as e:
            log_system_metric("trending_markets_error", 1, {"error": str(e)})
            raise
    
    async def update_market(self, market_id: int, market_data: MarketUpdate, user_id: int, db: Session) -> Market:
        """Update market with permission checks"""
        try:
            market = await self.get_market_by_id(market_id, db)
            if not market:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Market not found"
                )
            
            # Check permissions
            if market.creator_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to update this market"
                )
            
            # Check if market can be updated
            if market.status != MarketStatus.OPEN:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot update closed market"
                )
            
            # Update fields
            if market_data.title is not None:
                market.title = market_data.title
            
            if market_data.description is not None:
                market.description = market_data.description
            
            if market_data.question is not None:
                market.question = market_data.question
            
            if market_data.category is not None:
                market.category = market_data.category
            
            if market_data.outcome_a is not None:
                market.outcome_a = market_data.outcome_a
            
            if market_data.outcome_b is not None:
                market.outcome_b = market_data.outcome_b
            
            if market_data.closes_at is not None:
                market.closes_at = market_data.closes_at
            
            if market_data.resolution_criteria is not None:
                market.resolution_criteria = market_data.resolution_criteria
            
            # Update timestamp
            market.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(market)
            
            # Clear cache
            await self._clear_market_cache(market.id)
            
            # Log update
            log_trading_event("market_updated", {
                "market_id": market.id,
                "user_id": user_id
            })
            
            return market
            
        except Exception as e:
            db.rollback()
            log_system_metric("market_update_error", 1, {"error": str(e)})
            raise
    
    async def close_market(self, market_id: int, user_id: int, db: Session) -> Market:
        """Close a market (only creator can close)"""
        try:
            market = await self.get_market_by_id(market_id, db)
            if not market:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Market not found"
                )
            
            # Check permissions
            if market.creator_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to close this market"
                )
            
            # Check if market can be closed
            if market.status != MarketStatus.OPEN:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Market is already closed"
                )
            
            # Close market
            market.status = MarketStatus.CLOSED
            market.closed_at = datetime.utcnow()
            market.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(market)
            
            # Clear cache
            await self._clear_market_cache(market.id)
            
            # Log closure
            log_trading_event("market_closed", {
                "market_id": market.id,
                "user_id": user_id,
                "closed_at": market.closed_at.isoformat()
            })
            
            return market
            
        except Exception as e:
            db.rollback()
            log_system_metric("market_closure_error", 1, {"error": str(e)})
            raise
    
    async def get_market_analytics(self, market_id: int, db: Session) -> MarketAnalytics:
        """Get comprehensive market analytics"""
        try:
            market = await self.get_market_by_id(market_id, db)
            if not market:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Market not found"
                )
            
            # Get trades for this market
            trades = db.query(Trade).filter(Trade.market_id == market_id).all()
            
            # Calculate basic metrics
            total_volume = sum(trade.total_value for trade in trades)
            total_trades = len(trades)
            unique_traders = len(set(trade.user_id for trade in trades))
            
            # Calculate 24h metrics
            yesterday = datetime.utcnow() - timedelta(days=1)
            trades_24h = [t for t in trades if t.created_at >= yesterday]
            volume_24h = sum(trade.total_value for trade in trades_24h)
            trades_24h_count = len(trades_24h)
            
            # Calculate 7d metrics
            week_ago = datetime.utcnow() - timedelta(days=7)
            trades_7d = [t for t in trades if t.created_at >= week_ago]
            volume_7d = sum(trade.total_value for trade in trades_7d)
            trades_7d_count = len(trades_7d)
            
            # Calculate price movement and volatility
            price_movement = self._calculate_price_movement(market, trades)
            price_volatility = self._calculate_price_volatility(trades)
            
            # Calculate trending score
            trending_score = self._calculate_trending_score(market, trades, volume_24h, trades_24h_count)
            
            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(market, trades)
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(market, trades)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(market, trades, price_volatility)
            
            # Get market depth
            market_depth = self._calculate_market_depth(market, trades)
            
            # Get price history
            price_history = self._get_price_history(market, trades)
            
            # Get top traders
            top_traders = self._get_top_traders(market_id, db)
            
            # Get category performance
            category_performance = self._get_category_performance(market.category, db)
            
            return MarketAnalytics(
                total_volume=total_volume,
                volume_24h=volume_24h,
                volume_7d=volume_7d,
                total_trades=total_trades,
                trades_24h=trades_24h_count,
                trades_7d=trades_7d_count,
                unique_traders=unique_traders,
                price_movement=price_movement,
                price_volatility=price_volatility,
                trending_score=trending_score,
                sentiment_score=sentiment_score,
                liquidity_score=liquidity_score,
                risk_score=risk_score,
                market_depth=market_depth,
                price_history=price_history,
                top_traders=top_traders,
                category_performance=category_performance
            )
            
        except Exception as e:
            log_system_metric("market_analytics_error", 1, {"error": str(e)})
            raise
    
    async def get_market_stats(self, db: Session) -> MarketStats:
        """Get overall market statistics"""
        try:
            # Total markets
            total_markets = db.query(Market).count()
            
            # Active markets
            active_markets = db.query(Market).filter(Market.status == MarketStatus.OPEN).count()
            
            # Total volume
            total_volume = db.query(func.sum(Market.volume_total)).scalar() or 0.0
            
            # 24h volume
            yesterday = datetime.utcnow() - timedelta(days=1)
            volume_24h = db.query(func.sum(Market.volume_24h)).scalar() or 0.0
            
            # Top categories
            category_stats = (
                db.query(Market.category, func.count(Market.id), func.sum(Market.volume_total))
                .group_by(Market.category)
                .order_by(desc(func.count(Market.id)))
                .limit(5)
                .all()
            )
            
            return MarketStats(
                total_markets=total_markets,
                active_markets=active_markets,
                total_volume=total_volume,
                volume_24h=volume_24h,
                top_categories=[
                    {"category": cat.value, "count": count, "volume": volume or 0}
                    for cat, count, volume in category_stats
                ]
            )
            
        except Exception as e:
            log_system_metric("market_stats_error", 1, {"error": str(e)})
            raise
    
    async def _validate_market_data(self, market_data: MarketCreate):
        """Validate market creation data"""
        # Check closing date
        if market_data.closes_at <= datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Market closing date must be in the future"
            )
        
        # Check minimum duration
        min_duration = timedelta(hours=1)
        if market_data.closes_at - datetime.utcnow() < min_duration:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Market must be open for at least 1 hour"
            )
        
        # Check maximum duration
        max_duration = timedelta(days=365)
        if market_data.closes_at - datetime.utcnow() > max_duration:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Market cannot be open for more than 1 year"
            )
        
        # Validate outcomes
        if not market_data.outcome_a or not market_data.outcome_b:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both outcomes must be specified"
            )
        
        if market_data.outcome_a == market_data.outcome_b:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Outcomes must be different"
            )
    
    async def _check_market_creation_permissions(self, user_id: int, db: Session):
        """Check if user can create markets"""
        # Check if user exists and is active
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User not found or inactive"
            )
        
        # Check market creation limits
        today = datetime.utcnow().date()
        markets_today = db.query(Market).filter(
            Market.creator_id == user_id,
            func.date(Market.created_at) == today
        ).count()
        
        if markets_today >= 10:  # Max 10 markets per day
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Daily market creation limit reached"
            )
    
    async def _clear_market_cache(self, market_id: int):
        """Clear market-related cache entries"""
        cache_keys = [
            f"market:{market_id}",
            f"market_analytics:{market_id}",
            f"market_stats:{market_id}",
            "trending_markets",
            "market_list"
        ]
        
        for key in cache_keys:
            cache.delete(key)
    
    def _calculate_price_movement(self, market: Market, trades: List[Trade]) -> float:
        """Calculate price movement percentage"""
        if not trades:
            return 0.0
        
        # Get first and last trade prices
        sorted_trades = sorted(trades, key=lambda t: t.created_at)
        first_price = sorted_trades[0].price_per_share
        last_price = sorted_trades[-1].price_per_share
        
        if first_price == 0:
            return 0.0
        
        return ((last_price - first_price) / first_price) * 100
    
    def _calculate_price_volatility(self, trades: List[Trade]) -> float:
        """Calculate price volatility"""
        if len(trades) < 2:
            return 0.0
        
        prices = [trade.price_per_share for trade in trades]
        mean_price = sum(prices) / len(prices)
        
        variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
        return math.sqrt(variance)
    
    def _calculate_trending_score(self, market: Market, trades: List[Trade], volume_24h: float, trades_24h: int) -> float:
        """Calculate trending score based on recent activity"""
        # Base score from volume and trade count
        volume_score = min(volume_24h / 1000, 100)  # Normalize to 0-100
        trade_score = min(trades_24h * 10, 100)  # Normalize to 0-100
        
        # Time decay factor
        hours_since_creation = (datetime.utcnow() - market.created_at).total_seconds() / 3600
        time_factor = max(0.1, 1.0 - (hours_since_creation / 168))  # Decay over 1 week
        
        trending_score = (volume_score * 0.6 + trade_score * 0.4) * time_factor
        return min(100.0, trending_score)
    
    def _calculate_sentiment_score(self, market: Market, trades: List[Trade]) -> float:
        """Calculate sentiment score based on trading patterns"""
        if not trades:
            return 0.5  # Neutral
        
        # Analyze buy vs sell patterns
        buy_trades = [t for t in trades if t.trade_type == "buy"]
        sell_trades = [t for t in trades if t.trade_type == "sell"]
        
        buy_volume = sum(t.total_value for t in buy_trades)
        sell_volume = sum(t.total_value for t in sell_trades)
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.5
        
        sentiment = buy_volume / total_volume
        return sentiment
    
    def _calculate_liquidity_score(self, market: Market, trades: List[Trade]) -> float:
        """Calculate liquidity score based on trading activity"""
        if not trades:
            return 0.0
        
        # Calculate average trade size and frequency
        avg_trade_size = sum(t.total_value for t in trades) / len(trades)
        trade_frequency = len(trades) / max(1, (datetime.utcnow() - market.created_at).days)
        
        # Normalize scores
        size_score = min(avg_trade_size / 100, 100)
        frequency_score = min(trade_frequency * 10, 100)
        
        liquidity_score = (size_score * 0.4 + frequency_score * 0.6)
        return min(100.0, liquidity_score)
    
    def _calculate_risk_score(self, market: Market, trades: List[Trade], volatility: float) -> float:
        """Calculate risk score based on volatility and other factors"""
        # Base risk from volatility
        volatility_risk = min(volatility * 100, 100)
        
        # Time to expiration risk
        time_to_expiry = (market.closes_at - datetime.utcnow()).total_seconds() / 3600
        time_risk = max(0, 100 - (time_to_expiry / 24))  # Higher risk as expiry approaches
        
        # Volume risk (low volume = higher risk)
        total_volume = sum(t.total_value for t in trades)
        volume_risk = max(0, 100 - (total_volume / 1000))
        
        risk_score = (volatility_risk * 0.5 + time_risk * 0.3 + volume_risk * 0.2)
        return min(100.0, risk_score)
    
    def _calculate_market_depth(self, market: Market, trades: List[Trade]) -> Dict[str, float]:
        """Calculate market depth for both outcomes"""
        outcome_a_trades = [t for t in trades if t.outcome == "outcome_a"]
        outcome_b_trades = [t for t in trades if t.outcome == "outcome_b"]
        
        return {
            "outcome_a": sum(t.total_value for t in outcome_a_trades),
            "outcome_b": sum(t.total_value for t in outcome_b_trades)
        }
    
    def _get_price_history(self, market: Market, trades: List[Trade]) -> List[Dict[str, Any]]:
        """Get price history for the market"""
        if not trades:
            return []
        
        # Group trades by hour and calculate average prices
        price_history = []
        sorted_trades = sorted(trades, key=lambda t: t.created_at)
        
        current_hour = None
        hour_trades = []
        
        for trade in sorted_trades:
            trade_hour = trade.created_at.replace(minute=0, second=0, microsecond=0)
            
            if current_hour != trade_hour:
                if hour_trades:
                    avg_price = sum(t.price_per_share for t in hour_trades) / len(hour_trades)
                    price_history.append({
                        "timestamp": current_hour.isoformat(),
                        "price": avg_price,
                        "volume": sum(t.total_value for t in hour_trades)
                    })
                
                current_hour = trade_hour
                hour_trades = [trade]
            else:
                hour_trades.append(trade)
        
        # Add last hour
        if hour_trades:
            avg_price = sum(t.price_per_share for t in hour_trades) / len(hour_trades)
            price_history.append({
                "timestamp": current_hour.isoformat(),
                "price": avg_price,
                "volume": sum(t.total_value for t in hour_trades)
            })
        
        return price_history
    
    def _get_top_traders(self, market_id: int, db: Session) -> List[Dict[str, Any]]:
        """Get top traders for a market"""
        try:
            top_traders = (
                db.query(User, func.sum(Trade.total_value).label('total_volume'))
                .join(Trade, User.id == Trade.user_id)
                .filter(Trade.market_id == market_id)
                .group_by(User.id)
                .order_by(desc('total_volume'))
                .limit(10)
                .all()
            )
            
            return [
                {
                    "user_id": trader[0].id,
                    "username": trader[0].username,
                    "total_volume": float(trader[1])
                }
                for trader in top_traders
            ]
        except Exception:
            return []
    
    def _get_category_performance(self, category: MarketCategory, db: Session) -> Dict[str, float]:
        """Get performance metrics for a category"""
        try:
            category_markets = db.query(Market).filter(Market.category == category).all()
            
            if not category_markets:
                return {"total_volume": 0.0, "avg_volume": 0.0, "success_rate": 0.0}
            
            total_volume = sum(market.volume_total for market in category_markets)
            avg_volume = total_volume / len(category_markets)
            
            # Calculate success rate (simplified)
            successful_markets = len([m for m in category_markets if m.status == MarketStatus.RESOLVED])
            success_rate = (successful_markets / len(category_markets)) * 100
            
            return {
                "total_volume": total_volume,
                "avg_volume": avg_volume,
                "success_rate": success_rate
            }
        except Exception:
            return {"total_volume": 0.0, "avg_volume": 0.0, "success_rate": 0.0}


# Global market service instance
market_service = MarketService()