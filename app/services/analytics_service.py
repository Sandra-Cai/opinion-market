"""
Analytics Service for Opinion Market
Handles data analytics, reporting, and business intelligence
"""

import asyncio
import json
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
from app.core.logging import log_system_metric
from app.models.user import User
from app.models.market import Market, MarketCategory
from app.models.trade import Trade, TradeType


@dataclass
class AnalyticsReport:
    """Analytics report data structure"""
    report_id: str
    report_type: str
    generated_at: datetime
    data: Dict[str, Any]
    summary: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]


class AnalyticsService:
    """Service for analytics and reporting operations"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.cache_ttl = 600  # 10 minutes
        
    async def generate_market_analytics(self, db: Session) -> Dict[str, Any]:
        """Generate comprehensive market analytics"""
        try:
            # Get basic market statistics
            total_markets = db.query(Market).count()
            active_markets = db.query(Market).filter(Market.status == "open").count()
            closed_markets = db.query(Market).filter(Market.status == "closed").count()
            
            # Get volume statistics
            total_volume = db.query(func.sum(Market.volume_total)).scalar() or 0.0
            avg_volume_per_market = total_volume / total_markets if total_markets > 0 else 0.0
            
            # Get category breakdown
            category_stats = (
                db.query(Market.category, func.count(Market.id), func.sum(Market.volume_total))
                .group_by(Market.category)
                .all()
            )
            
            # Get trending markets
            trending_markets = (
                db.query(Market)
                .filter(Market.trending_score >= settings.MARKET_TRENDING_THRESHOLD)
                .order_by(desc(Market.trending_score))
                .limit(10)
                .all()
            )
            
            # Get market performance over time
            market_performance = self._calculate_market_performance(db)
            
            return {
                "total_markets": total_markets,
                "active_markets": active_markets,
                "closed_markets": closed_markets,
                "total_volume": total_volume,
                "avg_volume_per_market": avg_volume_per_market,
                "category_breakdown": [
                    {
                        "category": cat.value,
                        "count": count,
                        "volume": volume or 0
                    }
                    for cat, count, volume in category_stats
                ],
                "trending_markets": [
                    {
                        "id": market.id,
                        "title": market.title,
                        "trending_score": market.trending_score,
                        "volume": market.volume_total
                    }
                    for market in trending_markets
                ],
                "performance_over_time": market_performance
            }
            
        except Exception as e:
            log_system_metric("market_analytics_error", 1, {"error": str(e)})
            raise
    
    async def generate_trading_analytics(self, db: Session) -> Dict[str, Any]:
        """Generate comprehensive trading analytics"""
        try:
            # Get basic trade statistics
            total_trades = db.query(Trade).count()
            total_volume = db.query(func.sum(Trade.total_value)).scalar() or 0.0
            avg_trade_size = total_volume / total_trades if total_trades > 0 else 0.0
            
            # Get trade type breakdown
            trade_type_stats = (
                db.query(Trade.trade_type, func.count(Trade.id), func.sum(Trade.total_value))
                .group_by(Trade.trade_type)
                .all()
            )
            
            # Get active traders
            active_traders = db.query(func.count(func.distinct(Trade.user_id))).scalar() or 0
            
            # Get top traders
            top_traders = (
                db.query(User, func.sum(Trade.total_value).label('total_volume'))
                .join(Trade, User.id == Trade.user_id)
                .group_by(User.id)
                .order_by(desc('total_volume'))
                .limit(10)
                .all()
            )
            
            # Get trading volume over time
            volume_over_time = self._calculate_volume_over_time(db)
            
            return {
                "total_trades": total_trades,
                "total_volume": total_volume,
                "avg_trade_size": avg_trade_size,
                "active_traders": active_traders,
                "trade_type_breakdown": [
                    {
                        "type": trade_type.value,
                        "count": count,
                        "volume": volume or 0
                    }
                    for trade_type, count, volume in trade_type_stats
                ],
                "top_traders": [
                    {
                        "user_id": trader[0].id,
                        "username": trader[0].username,
                        "total_volume": float(trader[1])
                    }
                    for trader in top_traders
                ],
                "volume_over_time": volume_over_time
            }
            
        except Exception as e:
            log_system_metric("trading_analytics_error", 1, {"error": str(e)})
            raise
    
    async def generate_user_analytics(self, db: Session) -> Dict[str, Any]:
        """Generate comprehensive user analytics"""
        try:
            # Get basic user statistics
            total_users = db.query(User).count()
            active_users = db.query(User).filter(User.is_active == True).count()
            verified_users = db.query(User).filter(User.is_verified == True).count()
            
            # Get user registration over time
            registration_over_time = self._calculate_registration_over_time(db)
            
            # Get user activity levels
            activity_levels = self._calculate_user_activity_levels(db)
            
            # Get user retention
            retention_metrics = self._calculate_user_retention(db)
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "verified_users": verified_users,
                "registration_over_time": registration_over_time,
                "activity_levels": activity_levels,
                "retention_metrics": retention_metrics
            }
            
        except Exception as e:
            log_system_metric("user_analytics_error", 1, {"error": str(e)})
            raise
    
    async def generate_financial_analytics(self, db: Session) -> Dict[str, Any]:
        """Generate financial analytics and metrics"""
        try:
            # Get revenue metrics
            total_fees = db.query(func.sum(Trade.fee)).scalar() or 0.0
            
            # Get volume metrics
            total_volume = db.query(func.sum(Trade.total_value)).scalar() or 0.0
            
            # Calculate average fee rate
            avg_fee_rate = (total_fees / total_volume * 100) if total_volume > 0 else 0.0
            
            # Get daily revenue
            daily_revenue = self._calculate_daily_revenue(db)
            
            # Get revenue by category
            revenue_by_category = self._calculate_revenue_by_category(db)
            
            return {
                "total_fees": total_fees,
                "total_volume": total_volume,
                "avg_fee_rate": avg_fee_rate,
                "daily_revenue": daily_revenue,
                "revenue_by_category": revenue_by_category
            }
            
        except Exception as e:
            log_system_metric("financial_analytics_error", 1, {"error": str(e)})
            raise
    
    def _calculate_market_performance(self, db: Session) -> List[Dict[str, Any]]:
        """Calculate market performance over time"""
        try:
            # Get markets created over time
            markets_over_time = (
                db.query(
                    func.date(Market.created_at).label('date'),
                    func.count(Market.id).label('count'),
                    func.sum(Market.volume_total).label('volume')
                )
                .group_by(func.date(Market.created_at))
                .order_by('date')
                .limit(30)
                .all()
            )
            
            return [
                {
                    "date": date.isoformat(),
                    "markets_created": count,
                    "volume": volume or 0
                }
                for date, count, volume in markets_over_time
            ]
        except Exception:
            return []
    
    def _calculate_volume_over_time(self, db: Session) -> List[Dict[str, Any]]:
        """Calculate trading volume over time"""
        try:
            # Get volume over time
            volume_over_time = (
                db.query(
                    func.date(Trade.created_at).label('date'),
                    func.count(Trade.id).label('trade_count'),
                    func.sum(Trade.total_value).label('volume')
                )
                .group_by(func.date(Trade.created_at))
                .order_by('date')
                .limit(30)
                .all()
            )
            
            return [
                {
                    "date": date.isoformat(),
                    "trade_count": count,
                    "volume": volume or 0
                }
                for date, count, volume in volume_over_time
            ]
        except Exception:
            return []
    
    def _calculate_registration_over_time(self, db: Session) -> List[Dict[str, Any]]:
        """Calculate user registration over time"""
        try:
            # Get registrations over time
            registrations_over_time = (
                db.query(
                    func.date(User.created_at).label('date'),
                    func.count(User.id).label('count')
                )
                .group_by(func.date(User.created_at))
                .order_by('date')
                .limit(30)
                .all()
            )
            
            return [
                {
                    "date": date.isoformat(),
                    "registrations": count
                }
                for date, count in registrations_over_time
            ]
        except Exception:
            return []
    
    def _calculate_user_activity_levels(self, db: Session) -> Dict[str, int]:
        """Calculate user activity levels"""
        try:
            # Get users by activity level
            high_activity = db.query(User).join(Trade).group_by(User.id).having(func.count(Trade.id) > 50).count()
            medium_activity = db.query(User).join(Trade).group_by(User.id).having(func.count(Trade.id).between(10, 50)).count()
            low_activity = db.query(User).join(Trade).group_by(User.id).having(func.count(Trade.id).between(1, 9)).count()
            inactive = db.query(User).outerjoin(Trade).filter(Trade.id.is_(None)).count()
            
            return {
                "high_activity": high_activity,
                "medium_activity": medium_activity,
                "low_activity": low_activity,
                "inactive": inactive
            }
        except Exception:
            return {"high_activity": 0, "medium_activity": 0, "low_activity": 0, "inactive": 0}
    
    def _calculate_user_retention(self, db: Session) -> Dict[str, float]:
        """Calculate user retention metrics"""
        try:
            # Calculate retention rates (simplified)
            total_users = db.query(User).count()
            
            # Users who made at least one trade
            active_users = db.query(User).join(Trade).distinct().count()
            
            # Users who made trades in the last 30 days
            recent_active = db.query(User).join(Trade).filter(
                Trade.created_at >= datetime.utcnow() - timedelta(days=30)
            ).distinct().count()
            
            return {
                "overall_retention": (active_users / total_users * 100) if total_users > 0 else 0,
                "recent_retention": (recent_active / total_users * 100) if total_users > 0 else 0
            }
        except Exception:
            return {"overall_retention": 0, "recent_retention": 0}
    
    def _calculate_daily_revenue(self, db: Session) -> List[Dict[str, Any]]:
        """Calculate daily revenue"""
        try:
            # Get daily revenue
            daily_revenue = (
                db.query(
                    func.date(Trade.created_at).label('date'),
                    func.sum(Trade.fee).label('revenue')
                )
                .group_by(func.date(Trade.created_at))
                .order_by('date')
                .limit(30)
                .all()
            )
            
            return [
                {
                    "date": date.isoformat(),
                    "revenue": revenue or 0
                }
                for date, revenue in daily_revenue
            ]
        except Exception:
            return []
    
    def _calculate_revenue_by_category(self, db: Session) -> List[Dict[str, Any]]:
        """Calculate revenue by market category"""
        try:
            # Get revenue by category
            revenue_by_category = (
                db.query(
                    Market.category,
                    func.sum(Trade.fee).label('revenue'),
                    func.count(Trade.id).label('trade_count')
                )
                .join(Trade, Market.id == Trade.market_id)
                .group_by(Market.category)
                .all()
            )
            
            return [
                {
                    "category": category.value,
                    "revenue": revenue or 0,
                    "trade_count": count
                }
                for category, revenue, count in revenue_by_category
            ]
        except Exception:
            return []


# Global analytics service instance
analytics_service = AnalyticsService()