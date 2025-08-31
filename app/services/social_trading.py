"""
Social Trading Service
Provides copy trading, social signals, leaderboards, and community features
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json

logger = logging.getLogger(__name__)


@dataclass
class TraderProfile:
    """Trader profile information"""
    user_id: int
    username: str
    display_name: str
    bio: str
    avatar_url: str
    join_date: datetime
    total_trades: int
    win_rate: float
    total_profit: float
    followers_count: int
    following_count: int
    is_verified: bool
    risk_level: str  # 'conservative', 'moderate', 'aggressive'
    preferred_markets: List[str]
    performance_rating: float
    last_active: datetime


@dataclass
class SocialSignal:
    """Social trading signal"""
    signal_id: str
    trader_id: int
    market_id: int
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    reasoning: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    created_at: datetime
    expires_at: datetime
    followers_count: int
    success_rate: float
    tags: List[str]


@dataclass
class CopyTrade:
    """Copy trade information"""
    copy_trade_id: str
    follower_id: int
    leader_id: int
    market_id: int
    original_trade_id: str
    copied_amount: float
    copy_percentage: float
    status: str  # 'active', 'completed', 'cancelled'
    created_at: datetime
    completed_at: Optional[datetime]
    profit_loss: Optional[float]
    performance_ratio: float


@dataclass
class LeaderboardEntry:
    """Leaderboard entry"""
    user_id: int
    username: str
    display_name: str
    avatar_url: str
    rank: int
    total_profit: float
    win_rate: float
    total_trades: int
    followers_count: int
    performance_score: float
    period: str  # 'daily', 'weekly', 'monthly', 'all_time'
    last_updated: datetime


@dataclass
class CommunityPost:
    """Community post"""
    post_id: str
    user_id: int
    username: str
    display_name: str
    avatar_url: str
    content: str
    post_type: str  # 'analysis', 'prediction', 'discussion', 'news'
    market_id: Optional[int]
    tags: List[str]
    likes_count: int
    comments_count: int
    shares_count: int
    created_at: datetime
    updated_at: datetime
    is_verified: bool


class SocialTradingService:
    """Comprehensive social trading service"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.trader_profiles: Dict[int, TraderProfile] = {}
        self.social_signals: Dict[str, SocialSignal] = {}
        self.copy_trades: Dict[str, CopyTrade] = {}
        self.leaderboards: Dict[str, List[LeaderboardEntry]] = {}
        self.community_posts: Dict[str, CommunityPost] = {}
        
        # Performance tracking
        self.performance_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.signal_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Social metrics
        self.follower_relationships: Dict[int, List[int]] = defaultdict(list)
        self.trader_activity: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        
    async def initialize(self):
        """Initialize the social trading service"""
        logger.info("Initializing Social Trading Service")
        
        # Load existing data
        await self._load_trader_profiles()
        await self._load_social_signals()
        await self._load_copy_trades()
        await self._load_community_posts()
        
        # Start background tasks
        asyncio.create_task(self._update_leaderboards())
        asyncio.create_task(self._track_performance())
        asyncio.create_task(self._update_social_metrics())
        asyncio.create_task(self._cleanup_expired_signals())
        
        logger.info("Social Trading Service initialized successfully")
    
    async def create_trader_profile(self, user_id: int, username: str, display_name: str,
                                  bio: str = "", avatar_url: str = "", 
                                  risk_level: str = "moderate") -> TraderProfile:
        """Create a new trader profile"""
        try:
            profile = TraderProfile(
                user_id=user_id,
                username=username,
                display_name=display_name,
                bio=bio,
                avatar_url=avatar_url,
                join_date=datetime.utcnow(),
                total_trades=0,
                win_rate=0.0,
                total_profit=0.0,
                followers_count=0,
                following_count=0,
                is_verified=False,
                risk_level=risk_level,
                preferred_markets=[],
                performance_rating=0.0,
                last_active=datetime.utcnow()
            )
            
            self.trader_profiles[user_id] = profile
            await self._cache_trader_profile(profile)
            
            logger.info(f"Created trader profile for user {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating trader profile: {e}")
            raise
    
    async def update_trader_performance(self, user_id: int, trade_result: Dict[str, Any]):
        """Update trader performance after a trade"""
        try:
            if user_id not in self.trader_profiles:
                return
            
            profile = self.trader_profiles[user_id]
            
            # Update trade statistics
            profile.total_trades += 1
            profile.total_profit += trade_result.get('profit_loss', 0.0)
            
            # Calculate win rate
            if trade_result.get('profit_loss', 0.0) > 0:
                wins = profile.win_rate * (profile.total_trades - 1) + 1
                profile.win_rate = wins / profile.total_trades
            else:
                profile.win_rate = profile.win_rate * (profile.total_trades - 1) / profile.total_trades
            
            # Update performance rating
            profile.performance_rating = self._calculate_performance_rating(profile)
            profile.last_active = datetime.utcnow()
            
            # Store performance history
            self.performance_history[user_id].append({
                'timestamp': datetime.utcnow(),
                'profit_loss': trade_result.get('profit_loss', 0.0),
                'trade_id': trade_result.get('trade_id'),
                'market_id': trade_result.get('market_id')
            })
            
            await self._cache_trader_profile(profile)
            
            logger.info(f"Updated performance for trader {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating trader performance: {e}")
    
    async def create_social_signal(self, trader_id: int, market_id: int, signal_type: str,
                                 confidence: float, reasoning: str, target_price: Optional[float] = None,
                                 stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                                 tags: List[str] = None) -> SocialSignal:
        """Create a social trading signal"""
        try:
            signal_id = f"signal_{trader_id}_{market_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            signal = SocialSignal(
                signal_id=signal_id,
                trader_id=trader_id,
                market_id=market_id,
                signal_type=signal_type,
                confidence=confidence,
                reasoning=reasoning,
                target_price=target_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=24),
                followers_count=0,
                success_rate=0.0,
                tags=tags or []
            )
            
            self.social_signals[signal_id] = signal
            await self._cache_social_signal(signal)
            
            # Update trader activity
            self.trader_activity[trader_id].append({
                'type': 'signal_created',
                'signal_id': signal_id,
                'timestamp': datetime.utcnow()
            })
            
            logger.info(f"Created social signal {signal_id} by trader {trader_id}")
            return signal
            
        except Exception as e:
            logger.error(f"Error creating social signal: {e}")
            raise
    
    async def follow_trader(self, follower_id: int, leader_id: int) -> bool:
        """Follow a trader"""
        try:
            if follower_id == leader_id:
                return False
            
            # Add to follower relationships
            if leader_id not in self.follower_relationships:
                self.follower_relationships[leader_id] = []
            
            if follower_id not in self.follower_relationships[leader_id]:
                self.follower_relationships[leader_id].append(follower_id)
                
                # Update follower counts
                if leader_id in self.trader_profiles:
                    self.trader_profiles[leader_id].followers_count += 1
                    await self._cache_trader_profile(self.trader_profiles[leader_id])
                
                if follower_id in self.trader_profiles:
                    self.trader_profiles[follower_id].following_count += 1
                    await self._cache_trader_profile(self.trader_profiles[follower_id])
                
                logger.info(f"User {follower_id} started following trader {leader_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error following trader: {e}")
            return False
    
    async def unfollow_trader(self, follower_id: int, leader_id: int) -> bool:
        """Unfollow a trader"""
        try:
            if leader_id in self.follower_relationships and follower_id in self.follower_relationships[leader_id]:
                self.follower_relationships[leader_id].remove(follower_id)
                
                # Update follower counts
                if leader_id in self.trader_profiles:
                    self.trader_profiles[leader_id].followers_count = max(0, self.trader_profiles[leader_id].followers_count - 1)
                    await self._cache_trader_profile(self.trader_profiles[leader_id])
                
                if follower_id in self.trader_profiles:
                    self.trader_profiles[follower_id].following_count = max(0, self.trader_profiles[follower_id].following_count - 1)
                    await self._cache_trader_profile(self.trader_profiles[follower_id])
                
                logger.info(f"User {follower_id} unfollowed trader {leader_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error unfollowing trader: {e}")
            return False
    
    async def create_copy_trade(self, follower_id: int, leader_id: int, market_id: int,
                              original_trade_id: str, copy_percentage: float) -> CopyTrade:
        """Create a copy trade"""
        try:
            copy_trade_id = f"copy_{follower_id}_{leader_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Get original trade details (this would come from your trading system)
            original_trade = await self._get_trade_details(original_trade_id)
            copied_amount = original_trade.get('amount', 0) * copy_percentage / 100
            
            copy_trade = CopyTrade(
                copy_trade_id=copy_trade_id,
                follower_id=follower_id,
                leader_id=leader_id,
                market_id=market_id,
                original_trade_id=original_trade_id,
                copied_amount=copied_amount,
                copy_percentage=copy_percentage,
                status='active',
                created_at=datetime.utcnow(),
                completed_at=None,
                profit_loss=None,
                performance_ratio=1.0
            )
            
            self.copy_trades[copy_trade_id] = copy_trade
            await self._cache_copy_trade(copy_trade)
            
            # Update trader activity
            self.trader_activity[follower_id].append({
                'type': 'copy_trade_created',
                'copy_trade_id': copy_trade_id,
                'leader_id': leader_id,
                'timestamp': datetime.utcnow()
            })
            
            logger.info(f"Created copy trade {copy_trade_id}")
            return copy_trade
            
        except Exception as e:
            logger.error(f"Error creating copy trade: {e}")
            raise
    
    async def get_leaderboard(self, period: str = "weekly", limit: int = 50) -> List[LeaderboardEntry]:
        """Get leaderboard for a specific period"""
        try:
            cache_key = f"leaderboard:{period}"
            cached_leaderboard = await self.redis.get(cache_key)
            
            if cached_leaderboard:
                leaderboard_data = json.loads(cached_leaderboard)
                return [LeaderboardEntry(**entry) for entry in leaderboard_data]
            
            # Calculate leaderboard
            leaderboard = await self._calculate_leaderboard(period, limit)
            
            # Cache leaderboard
            leaderboard_data = [
                {
                    'user_id': entry.user_id,
                    'username': entry.username,
                    'display_name': entry.display_name,
                    'avatar_url': entry.avatar_url,
                    'rank': entry.rank,
                    'total_profit': entry.total_profit,
                    'win_rate': entry.win_rate,
                    'total_trades': entry.total_trades,
                    'followers_count': entry.followers_count,
                    'performance_score': entry.performance_score,
                    'period': entry.period,
                    'last_updated': entry.last_updated.isoformat()
                }
                for entry in leaderboard
            ]
            
            await self.redis.setex(cache_key, 3600, json.dumps(leaderboard_data))  # Cache for 1 hour
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Error getting leaderboard: {e}")
            return []
    
    async def create_community_post(self, user_id: int, content: str, post_type: str,
                                  market_id: Optional[int] = None, tags: List[str] = None) -> CommunityPost:
        """Create a community post"""
        try:
            post_id = f"post_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Get user profile
            profile = self.trader_profiles.get(user_id)
            username = profile.username if profile else f"user_{user_id}"
            display_name = profile.display_name if profile else f"User {user_id}"
            avatar_url = profile.avatar_url if profile else ""
            is_verified = profile.is_verified if profile else False
            
            post = CommunityPost(
                post_id=post_id,
                user_id=user_id,
                username=username,
                display_name=display_name,
                avatar_url=avatar_url,
                content=content,
                post_type=post_type,
                market_id=market_id,
                tags=tags or [],
                likes_count=0,
                comments_count=0,
                shares_count=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                is_verified=is_verified
            )
            
            self.community_posts[post_id] = post
            await self._cache_community_post(post)
            
            # Update trader activity
            self.trader_activity[user_id].append({
                'type': 'post_created',
                'post_id': post_id,
                'timestamp': datetime.utcnow()
            })
            
            logger.info(f"Created community post {post_id}")
            return post
            
        except Exception as e:
            logger.error(f"Error creating community post: {e}")
            raise
    
    async def get_trader_feed(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get personalized feed for a trader"""
        try:
            # Get followed traders
            followed_traders = []
            for leader_id, followers in self.follower_relationships.items():
                if user_id in followers:
                    followed_traders.append(leader_id)
            
            # Get recent activity from followed traders
            feed_items = []
            
            # Get recent signals
            for signal in self.social_signals.values():
                if signal.trader_id in followed_traders:
                    feed_items.append({
                        'type': 'signal',
                        'data': signal,
                        'timestamp': signal.created_at
                    })
            
            # Get recent posts
            for post in self.community_posts.values():
                if post.user_id in followed_traders:
                    feed_items.append({
                        'type': 'post',
                        'data': post,
                        'timestamp': post.created_at
                    })
            
            # Sort by timestamp and limit
            feed_items.sort(key=lambda x: x['timestamp'], reverse=True)
            feed_items = feed_items[:limit]
            
            return feed_items
            
        except Exception as e:
            logger.error(f"Error getting trader feed: {e}")
            return []
    
    async def get_trader_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive analytics for a trader"""
        try:
            profile = self.trader_profiles.get(user_id)
            if not profile:
                return {}
            
            # Calculate analytics
            analytics = {
                'profile': {
                    'user_id': profile.user_id,
                    'username': profile.username,
                    'display_name': profile.display_name,
                    'join_date': profile.join_date.isoformat(),
                    'total_trades': profile.total_trades,
                    'win_rate': profile.win_rate,
                    'total_profit': profile.total_profit,
                    'followers_count': profile.followers_count,
                    'following_count': profile.following_count,
                    'performance_rating': profile.performance_rating,
                    'risk_level': profile.risk_level
                },
                'performance': {
                    'daily_profit': await self._calculate_daily_profit(user_id),
                    'weekly_profit': await self._calculate_weekly_profit(user_id),
                    'monthly_profit': await self._calculate_monthly_profit(user_id),
                    'best_trade': await self._get_best_trade(user_id),
                    'worst_trade': await self._get_worst_trade(user_id),
                    'avg_trade_size': await self._calculate_avg_trade_size(user_id)
                },
                'social': {
                    'signals_created': len([s for s in self.social_signals.values() if s.trader_id == user_id]),
                    'signals_followed': len([s for s in self.social_signals.values() if s.followers_count > 0]),
                    'posts_created': len([p for p in self.community_posts.values() if p.user_id == user_id]),
                    'copy_trades': len([c for c in self.copy_trades.values() if c.follower_id == user_id])
                },
                'markets': {
                    'preferred_markets': profile.preferred_markets,
                    'most_traded_market': await self._get_most_traded_market(user_id),
                    'best_performing_market': await self._get_best_performing_market(user_id)
                }
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting trader analytics: {e}")
            return {}
    
    def _calculate_performance_rating(self, profile: TraderProfile) -> float:
        """Calculate performance rating for a trader"""
        try:
            # Base score from win rate
            win_rate_score = profile.win_rate * 40
            
            # Profit score (normalized)
            profit_score = min(30, max(0, profile.total_profit / 1000 * 30))
            
            # Activity score
            activity_score = min(20, profile.total_trades / 10 * 20)
            
            # Social score
            social_score = min(10, profile.followers_count / 100 * 10)
            
            return win_rate_score + profit_score + activity_score + social_score
            
        except Exception as e:
            logger.error(f"Error calculating performance rating: {e}")
            return 0.0
    
    async def _calculate_leaderboard(self, period: str, limit: int) -> List[LeaderboardEntry]:
        """Calculate leaderboard for a specific period"""
        try:
            # Get all traders with performance data
            traders_with_performance = []
            
            for user_id, profile in self.trader_profiles.items():
                if profile.total_trades > 0:
                    # Calculate period-specific performance
                    period_profit = await self._calculate_period_profit(user_id, period)
                    
                    traders_with_performance.append({
                        'user_id': user_id,
                        'username': profile.username,
                        'display_name': profile.display_name,
                        'avatar_url': profile.avatar_url,
                        'total_profit': period_profit,
                        'win_rate': profile.win_rate,
                        'total_trades': profile.total_trades,
                        'followers_count': profile.followers_count,
                        'performance_score': profile.performance_rating
                    })
            
            # Sort by performance score
            traders_with_performance.sort(key=lambda x: x['performance_score'], reverse=True)
            
            # Create leaderboard entries
            leaderboard = []
            for i, trader in enumerate(traders_with_performance[:limit]):
                leaderboard.append(LeaderboardEntry(
                    user_id=trader['user_id'],
                    username=trader['username'],
                    display_name=trader['display_name'],
                    avatar_url=trader['avatar_url'],
                    rank=i + 1,
                    total_profit=trader['total_profit'],
                    win_rate=trader['win_rate'],
                    total_trades=trader['total_trades'],
                    followers_count=trader['followers_count'],
                    performance_score=trader['performance_score'],
                    period=period,
                    last_updated=datetime.utcnow()
                ))
            
            return leaderboard
            
        except Exception as e:
            logger.error(f"Error calculating leaderboard: {e}")
            return []
    
    async def _calculate_period_profit(self, user_id: int, period: str) -> float:
        """Calculate profit for a specific period"""
        try:
            if user_id not in self.performance_history:
                return 0.0
            
            now = datetime.utcnow()
            
            if period == "daily":
                start_time = now - timedelta(days=1)
            elif period == "weekly":
                start_time = now - timedelta(weeks=1)
            elif period == "monthly":
                start_time = now - timedelta(days=30)
            else:  # all_time
                start_time = datetime.min
            
            period_profit = 0.0
            for performance in self.performance_history[user_id]:
                if performance['timestamp'] >= start_time:
                    period_profit += performance['profit_loss']
            
            return period_profit
            
        except Exception as e:
            logger.error(f"Error calculating period profit: {e}")
            return 0.0
    
    async def _update_leaderboards(self):
        """Update leaderboards periodically"""
        while True:
            try:
                # Update all period leaderboards
                for period in ["daily", "weekly", "monthly", "all_time"]:
                    await self.get_leaderboard(period, 50)
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error updating leaderboards: {e}")
                await asyncio.sleep(7200)  # Retry in 2 hours
    
    async def _track_performance(self):
        """Track performance metrics"""
        while True:
            try:
                # Update performance ratings for all traders
                for user_id, profile in self.trader_profiles.items():
                    profile.performance_rating = self._calculate_performance_rating(profile)
                    await self._cache_trader_profile(profile)
                
                await asyncio.sleep(1800)  # Update every 30 minutes
                
            except Exception as e:
                logger.error(f"Error tracking performance: {e}")
                await asyncio.sleep(3600)
    
    async def _update_social_metrics(self):
        """Update social metrics"""
        while True:
            try:
                # Update signal performance
                for signal_id, signal in self.social_signals.items():
                    if signal.expires_at <= datetime.utcnow():
                        # Calculate signal performance
                        performance = await self._calculate_signal_performance(signal)
                        signal.success_rate = performance
                        await self._cache_social_signal(signal)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error updating social metrics: {e}")
                await asyncio.sleep(600)
    
    async def _cleanup_expired_signals(self):
        """Clean up expired signals"""
        while True:
            try:
                expired_signals = [
                    signal_id for signal_id, signal in self.social_signals.items()
                    if signal.expires_at <= datetime.utcnow()
                ]
                
                for signal_id in expired_signals:
                    del self.social_signals[signal_id]
                    await self.redis.delete(f"signal:{signal_id}")
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up expired signals: {e}")
                await asyncio.sleep(7200)
    
    # Helper methods (implementations would depend on your data models)
    async def _load_trader_profiles(self):
        """Load trader profiles from database"""
        # Implementation would query database
        pass
    
    async def _load_social_signals(self):
        """Load social signals from database"""
        # Implementation would query database
        pass
    
    async def _load_copy_trades(self):
        """Load copy trades from database"""
        # Implementation would query database
        pass
    
    async def _load_community_posts(self):
        """Load community posts from database"""
        # Implementation would query database
        pass
    
    async def _get_trade_details(self, trade_id: str) -> Dict[str, Any]:
        """Get trade details"""
        # Implementation would query trading system
        return {'amount': 100.0, 'market_id': 1}
    
    async def _calculate_signal_performance(self, signal: SocialSignal) -> float:
        """Calculate signal performance"""
        # Implementation would calculate based on market outcomes
        return 0.75  # Placeholder
    
    async def _calculate_daily_profit(self, user_id: int) -> float:
        """Calculate daily profit"""
        return await self._calculate_period_profit(user_id, "daily")
    
    async def _calculate_weekly_profit(self, user_id: int) -> float:
        """Calculate weekly profit"""
        return await self._calculate_period_profit(user_id, "weekly")
    
    async def _calculate_monthly_profit(self, user_id: int) -> float:
        """Calculate monthly profit"""
        return await self._calculate_period_profit(user_id, "monthly")
    
    async def _get_best_trade(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get best trade for user"""
        # Implementation would find best trade
        return None
    
    async def _get_worst_trade(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get worst trade for user"""
        # Implementation would find worst trade
        return None
    
    async def _calculate_avg_trade_size(self, user_id: int) -> float:
        """Calculate average trade size"""
        # Implementation would calculate average
        return 100.0
    
    async def _get_most_traded_market(self, user_id: int) -> Optional[int]:
        """Get most traded market for user"""
        # Implementation would find most traded market
        return None
    
    async def _get_best_performing_market(self, user_id: int) -> Optional[int]:
        """Get best performing market for user"""
        # Implementation would find best performing market
        return None
    
    # Caching methods
    async def _cache_trader_profile(self, profile: TraderProfile):
        """Cache trader profile"""
        try:
            cache_key = f"trader_profile:{profile.user_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps({
                    'username': profile.username,
                    'display_name': profile.display_name,
                    'bio': profile.bio,
                    'avatar_url': profile.avatar_url,
                    'join_date': profile.join_date.isoformat(),
                    'total_trades': profile.total_trades,
                    'win_rate': profile.win_rate,
                    'total_profit': profile.total_profit,
                    'followers_count': profile.followers_count,
                    'following_count': profile.following_count,
                    'is_verified': profile.is_verified,
                    'risk_level': profile.risk_level,
                    'preferred_markets': profile.preferred_markets,
                    'performance_rating': profile.performance_rating,
                    'last_active': profile.last_active.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching trader profile: {e}")
    
    async def _cache_social_signal(self, signal: SocialSignal):
        """Cache social signal"""
        try:
            cache_key = f"social_signal:{signal.signal_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'trader_id': signal.trader_id,
                    'market_id': signal.market_id,
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning,
                    'target_price': signal.target_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'created_at': signal.created_at.isoformat(),
                    'expires_at': signal.expires_at.isoformat(),
                    'followers_count': signal.followers_count,
                    'success_rate': signal.success_rate,
                    'tags': signal.tags
                })
            )
        except Exception as e:
            logger.error(f"Error caching social signal: {e}")
    
    async def _cache_copy_trade(self, copy_trade: CopyTrade):
        """Cache copy trade"""
        try:
            cache_key = f"copy_trade:{copy_trade.copy_trade_id}"
            await self.redis.setex(
                cache_key,
                86400,  # 24 hours TTL
                json.dumps({
                    'follower_id': copy_trade.follower_id,
                    'leader_id': copy_trade.leader_id,
                    'market_id': copy_trade.market_id,
                    'original_trade_id': copy_trade.original_trade_id,
                    'copied_amount': copy_trade.copied_amount,
                    'copy_percentage': copy_trade.copy_percentage,
                    'status': copy_trade.status,
                    'created_at': copy_trade.created_at.isoformat(),
                    'completed_at': copy_trade.completed_at.isoformat() if copy_trade.completed_at else None,
                    'profit_loss': copy_trade.profit_loss,
                    'performance_ratio': copy_trade.performance_ratio
                })
            )
        except Exception as e:
            logger.error(f"Error caching copy trade: {e}")
    
    async def _cache_community_post(self, post: CommunityPost):
        """Cache community post"""
        try:
            cache_key = f"community_post:{post.post_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'user_id': post.user_id,
                    'username': post.username,
                    'display_name': post.display_name,
                    'avatar_url': post.avatar_url,
                    'content': post.content,
                    'post_type': post.post_type,
                    'market_id': post.market_id,
                    'tags': post.tags,
                    'likes_count': post.likes_count,
                    'comments_count': post.comments_count,
                    'shares_count': post.shares_count,
                    'created_at': post.created_at.isoformat(),
                    'updated_at': post.updated_at.isoformat(),
                    'is_verified': post.is_verified
                })
            )
        except Exception as e:
            logger.error(f"Error caching community post: {e}")


# Factory function
async def get_social_trading_service(redis_client: redis.Redis, db_session: Session) -> SocialTradingService:
    """Get social trading service instance"""
    service = SocialTradingService(redis_client, db_session)
    await service.initialize()
    return service
