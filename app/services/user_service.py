"""
User Service for Opinion Market
Handles user-related business logic including authentication, profile management, and analytics
"""

import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc
from fastapi import HTTPException, status

from app.core.database import get_db, get_redis_client
from app.core.config import settings
from app.core.security import security_manager
from app.core.cache import cache
from app.core.logging import log_trading_event, log_system_metric
from app.core.security_audit import security_auditor, SecurityEventType, SecuritySeverity
from app.models.user import User
from app.models.trade import Trade
from app.models.market import Market
from app.schemas.user import UserCreate, UserUpdate, UserResponse, UserStats


@dataclass
class UserAnalytics:
    """User analytics data structure"""
    total_trades: int
    successful_trades: int
    total_profit: float
    total_volume: float
    reputation_score: float
    success_rate: float
    win_rate: float
    avg_trade_size: float
    largest_win: float
    largest_loss: float
    portfolio_value: float
    available_balance: float
    total_invested: float
    total_balance: float
    trading_streak: int
    favorite_categories: List[Dict[str, Any]]
    risk_tolerance: str
    trading_style: str


class UserService:
    """Service for user-related operations"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.cache_ttl = 300  # 5 minutes
        
    async def create_user(self, user_data: UserCreate, db: Session) -> User:
        """Create a new user with comprehensive validation and setup"""
        try:
            # Validate password strength
            password_check = security_manager.is_password_strong(user_data.password)
            if not password_check["is_strong"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Password is too weak: {', '.join(password_check['issues'])}"
                )
            
            # Check for existing user
            existing_user = db.query(User).filter(
                or_(User.username == user_data.username, User.email == user_data.email)
            ).first()
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already exists"
                )
            
            # Create user
            hashed_password = security_manager.hash_password(user_data.password)
            user = User(
                username=user_data.username,
                email=user_data.email,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                bio=user_data.bio,
                is_active=True,
                is_verified=False,
                preferences={
                    "theme": "light",
                    "language": "en",
                    "timezone": "UTC",
                    "notifications": True,
                    "privacy": "public"
                },
                notification_settings={
                    "email_notifications": True,
                    "push_notifications": True,
                    "market_updates": True,
                    "price_alerts": False,
                    "trade_notifications": True,
                    "social_notifications": True
                }
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Log user creation
            log_system_metric("user_created", 1, {
                "user_id": user.id,
                "username": user.username
            })
            
            # Clear cache
            await self._clear_user_cache(user.id, user.username, user.email)
            
            return user
            
        except Exception as e:
            db.rollback()
            log_system_metric("user_creation_error", 1, {"error": str(e)})
            raise
    
    async def authenticate_user(self, username: str, password: str, db: Session) -> Optional[User]:
        """Authenticate user with enhanced security"""
        try:
            # Find user by username or email
            user = db.query(User).filter(
                or_(User.username == username, User.email == username)
            ).first()
            
            if not user:
                return None
            
            # Verify password
            if not security_manager.verify_password(password, user.hashed_password):
                return None
            
            # Check if user is active
            if not user.is_active:
                return None
            
            # Update last login
            user.update_last_login()
            db.commit()
            
            # Log successful authentication
            log_system_metric("user_authenticated", 1, {
                "user_id": user.id,
                "username": user.username
            })
            
            return user
            
        except Exception as e:
            log_system_metric("user_authentication_error", 1, {"error": str(e)})
            return None
    
    async def get_user_by_id(self, user_id: int, db: Session) -> Optional[User]:
        """Get user by ID with caching"""
        cache_key = f"user:{user_id}"
        
        # Try cache first
        cached_user = cache.get(cache_key)
        if cached_user:
            return cached_user
        
        # Get from database
        user = db.query(User).filter(User.id == user_id).first()
        
        if user:
            # Cache user data
            cache.set(cache_key, user, ttl=self.cache_ttl)
        
        return user
    
    async def get_user_by_username(self, username: str, db: Session) -> Optional[User]:
        """Get user by username with caching"""
        cache_key = f"user_by_username:{username}"
        
        # Try cache first
        cached_user = cache.get(cache_key)
        if cached_user:
            return cached_user
        
        # Get from database
        user = db.query(User).filter(User.username == username).first()
        
        if user:
            # Cache user data
            cache.set(cache_key, user, ttl=self.cache_ttl)
        
        return user
    
    async def update_user(self, user_id: int, user_data: UserUpdate, db: Session) -> User:
        """Update user profile with validation"""
        try:
            user = await self.get_user_by_id(user_id, db)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Update fields
            if user_data.full_name is not None:
                user.full_name = user_data.full_name
            
            if user_data.bio is not None:
                user.bio = user_data.bio
            
            if user_data.avatar_url is not None:
                user.avatar_url = user_data.avatar_url
            
            if user_data.preferences is not None:
                user.preferences = {**user.preferences, **user_data.preferences}
            
            if user_data.notification_settings is not None:
                user.notification_settings = {**user.notification_settings, **user_data.notification_settings}
            
            # Update timestamp
            user.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(user)
            
            # Clear cache
            await self._clear_user_cache(user.id, user.username, user.email)
            
            # Log update
            log_system_metric("user_updated", 1, {
                "user_id": user.id,
                "username": user.username
            })
            
            return user
            
        except Exception as e:
            db.rollback()
            log_system_metric("user_update_error", 1, {"error": str(e)})
            raise
    
    async def change_password(self, user_id: int, current_password: str, new_password: str, db: Session) -> bool:
        """Change user password with security validation"""
        try:
            user = await self.get_user_by_id(user_id, db)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Verify current password
            if not security_manager.verify_password(current_password, user.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect"
                )
            
            # Validate new password
            password_check = security_manager.is_password_strong(new_password)
            if not password_check["is_strong"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"New password is too weak: {', '.join(password_check['issues'])}"
                )
            
            # Update password
            user.hashed_password = security_manager.hash_password(new_password)
            user.updated_at = datetime.utcnow()
            
            db.commit()
            
            # Clear cache
            await self._clear_user_cache(user.id, user.username, user.email)
            
            # Log password change
            log_system_metric("user_password_changed", 1, {
                "user_id": user.id,
                "username": user.username
            })
            
            return True
            
        except Exception as e:
            db.rollback()
            log_system_metric("user_password_change_error", 1, {"error": str(e)})
            raise
    
    async def deactivate_user(self, user_id: int, db: Session) -> bool:
        """Deactivate user account"""
        try:
            user = await self.get_user_by_id(user_id, db)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Deactivate user
            user.is_active = False
            user.updated_at = datetime.utcnow()
            
            db.commit()
            
            # Clear cache
            await self._clear_user_cache(user.id, user.username, user.email)
            
            # Log deactivation
            log_system_metric("user_deactivated", 1, {
                "user_id": user.id,
                "username": user.username
            })
            
            return True
            
        except Exception as e:
            db.rollback()
            log_system_metric("user_deactivation_error", 1, {"error": str(e)})
            raise
    
    async def get_user_analytics(self, user_id: int, db: Session) -> UserAnalytics:
        """Get comprehensive user analytics"""
        try:
            # Get user trades
            trades = db.query(Trade).filter(Trade.user_id == user_id).all()
            
            if not trades:
                return UserAnalytics(
                    total_trades=0,
                    successful_trades=0,
                    total_profit=0.0,
                    total_volume=0.0,
                    reputation_score=0.0,
                    success_rate=0.0,
                    win_rate=0.0,
                    avg_trade_size=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    portfolio_value=0.0,
                    available_balance=0.0,
                    total_invested=0.0,
                    total_balance=0.0,
                    trading_streak=0,
                    favorite_categories=[],
                    risk_tolerance="conservative",
                    trading_style="passive"
                )
            
            # Calculate basic metrics
            total_trades = len(trades)
            total_volume = sum(trade.total_value for trade in trades)
            avg_trade_size = total_volume / total_trades if total_trades > 0 else 0.0
            
            # Calculate profit/loss
            total_profit = 0.0
            successful_trades = 0
            largest_win = 0.0
            largest_loss = 0.0
            
            for trade in trades:
                # This is a simplified calculation - in reality, you'd need to track actual P&L
                if trade.status == "executed":
                    successful_trades += 1
                    # Simplified profit calculation
                    profit = trade.total_value * 0.1  # Assume 10% profit for successful trades
                    total_profit += profit
                    largest_win = max(largest_win, profit)
                else:
                    loss = trade.total_value * 0.05  # Assume 5% loss for failed trades
                    total_profit -= loss
                    largest_loss = max(largest_loss, loss)
            
            # Calculate rates
            success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0.0
            win_rate = success_rate  # Simplified - same as success rate
            
            # Calculate reputation score (0-100)
            reputation_score = min(100.0, (success_rate * 0.4) + (total_volume / 1000 * 0.3) + (total_trades * 0.3))
            
            # Calculate trading streak
            trading_streak = self._calculate_trading_streak(trades)
            
            # Get favorite categories
            favorite_categories = self._get_favorite_categories(user_id, db)
            
            # Determine risk tolerance and trading style
            risk_tolerance = self._determine_risk_tolerance(trades)
            trading_style = self._determine_trading_style(trades)
            
            # Calculate portfolio value (simplified)
            portfolio_value = total_profit + 1000  # Base portfolio value
            available_balance = portfolio_value * 0.3  # 30% available for trading
            total_invested = total_volume
            total_balance = portfolio_value
            
            return UserAnalytics(
                total_trades=total_trades,
                successful_trades=successful_trades,
                total_profit=total_profit,
                total_volume=total_volume,
                reputation_score=reputation_score,
                success_rate=success_rate,
                win_rate=win_rate,
                avg_trade_size=avg_trade_size,
                largest_win=largest_win,
                largest_loss=largest_loss,
                portfolio_value=portfolio_value,
                available_balance=available_balance,
                total_invested=total_invested,
                total_balance=total_balance,
                trading_streak=trading_streak,
                favorite_categories=favorite_categories,
                risk_tolerance=risk_tolerance,
                trading_style=trading_style
            )
            
        except Exception as e:
            log_system_metric("user_analytics_error", 1, {"error": str(e)})
            raise
    
    async def get_user_stats(self, db: Session) -> UserStats:
        """Get overall user statistics"""
        try:
            # Total users
            total_users = db.query(User).count()
            
            # Active users (last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            active_users_24h = db.query(User).filter(User.last_login >= yesterday).count()
            
            # Total volume (simplified)
            total_volume = db.query(func.sum(Trade.total_value)).scalar() or 0.0
            
            # Top traders (by volume)
            top_traders = (
                db.query(User, func.sum(Trade.total_value).label('total_volume'))
                .join(Trade, User.id == Trade.user_id)
                .group_by(User.id)
                .order_by(desc('total_volume'))
                .limit(10)
                .all()
            )
            
            return UserStats(
                total_users=total_users,
                active_users_24h=active_users_24h,
                total_volume_all_time=total_volume,
                top_traders=[trader[0] for trader in top_traders]
            )
            
        except Exception as e:
            log_system_metric("user_stats_error", 1, {"error": str(e)})
            raise
    
    async def _clear_user_cache(self, user_id: int, username: str, email: str):
        """Clear user-related cache entries"""
        cache_keys = [
            f"user:{user_id}",
            f"user_by_username:{username}",
            f"user_by_email:{email}",
            f"user_analytics:{user_id}",
            f"user_stats:{user_id}"
        ]
        
        for key in cache_keys:
            cache.delete(key)
    
    def _calculate_trading_streak(self, trades: List[Trade]) -> int:
        """Calculate current trading streak"""
        if not trades:
            return 0
        
        # Sort trades by date
        sorted_trades = sorted(trades, key=lambda t: t.created_at, reverse=True)
        
        streak = 0
        current_date = None
        
        for trade in sorted_trades:
            trade_date = trade.created_at.date()
            
            if current_date is None:
                current_date = trade_date
                streak = 1
            elif trade_date == current_date - timedelta(days=1):
                current_date = trade_date
                streak += 1
            else:
                break
        
        return streak
    
    def _get_favorite_categories(self, user_id: int, db: Session) -> List[Dict[str, Any]]:
        """Get user's favorite market categories"""
        try:
            category_stats = (
                db.query(Market.category, func.count(Trade.id).label('trade_count'))
                .join(Trade, Market.id == Trade.market_id)
                .filter(Trade.user_id == user_id)
                .group_by(Market.category)
                .order_by(desc('trade_count'))
                .limit(5)
                .all()
            )
            
            return [
                {"category": category.value, "trade_count": count}
                for category, count in category_stats
            ]
        except Exception:
            return []
    
    def _determine_risk_tolerance(self, trades: List[Trade]) -> str:
        """Determine user's risk tolerance based on trading patterns"""
        if not trades:
            return "conservative"
        
        # Analyze trade sizes and frequency
        avg_trade_size = sum(trade.total_value for trade in trades) / len(trades)
        trade_frequency = len(trades) / 30  # Trades per day (assuming 30-day period)
        
        if avg_trade_size > 1000 and trade_frequency > 5:
            return "aggressive"
        elif avg_trade_size > 500 and trade_frequency > 2:
            return "moderate"
        else:
            return "conservative"
    
    def _determine_trading_style(self, trades: List[Trade]) -> str:
        """Determine user's trading style based on patterns"""
        if not trades:
            return "passive"
        
        # Analyze trade timing and patterns
        recent_trades = [t for t in trades if (datetime.utcnow() - t.created_at).days < 7]
        
        if len(recent_trades) > 10:
            return "active"
        elif len(recent_trades) > 3:
            return "moderate"
        else:
            return "passive"


# Global user service instance
user_service = UserService()
