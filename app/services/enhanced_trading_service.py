"""
Enhanced Trading Service for Opinion Market
Provides advanced trading functionality, order management, and risk assessment
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
from app.models.market import Market, MarketStatus
from app.models.trade import Trade, TradeType, TradeStatus
from app.models.user import User
from app.models.position import Position


class EnhancedTradingService(BaseService, CacheableService, MetricsService, BackgroundTaskService):
    """Enhanced trading service with advanced features"""
    
    def __init__(self):
        BaseService.__init__(self)
        CacheableService.__init__(self, cache_ttl=60)  # Shorter TTL for trading data
        MetricsService.__init__(self)
        BackgroundTaskService.__init__(self)
        
        self.active_orders = {}
        self.price_feeds = {}
        self.risk_limits = {}
    
    async def _initialize_internal(self) -> None:
        """Initialize trading service"""
        # Start background tasks
        self.start_background_task("process_pending_orders", self._process_pending_orders, interval=10)  # 10 seconds
        self.start_background_task("update_price_feeds", self._update_price_feeds, interval=5)  # 5 seconds
        self.start_background_task("risk_monitoring", self._risk_monitoring, interval=30)  # 30 seconds
        self.start_background_task("cleanup_old_data", self._cleanup_old_data, interval=3600)  # 1 hour
        
        # Load initial data
        await self._load_initial_data()
    
    async def _cleanup_internal(self) -> None:
        """Cleanup trading service"""
        # Background tasks are cleaned up by parent class
        pass
    
    async def _load_initial_data(self):
        """Load initial trading data"""
        with get_db_session() as db:
            # Load active orders
            await self._load_active_orders()
            
            # Initialize risk limits
            await self._initialize_risk_limits()
    
    async def _load_active_orders(self):
        """Load active orders from database"""
        try:
            with get_db_session() as db:
                # This would typically load from an orders table
                # For now, we'll initialize empty
                self.active_orders = {}
                
        except Exception as e:
            self.logger.error(f"Failed to load active orders: {e}")
    
    async def _initialize_risk_limits(self):
        """Initialize risk management limits"""
        self.risk_limits = {
            "max_position_size": 10000.0,  # $10,000 max position
            "max_daily_loss": 1000.0,      # $1,000 max daily loss
            "max_trades_per_hour": 100,    # 100 trades per hour
            "max_trade_amount": 5000.0,    # $5,000 max single trade
            "min_trade_amount": 1.0,       # $1 min trade
            "max_leverage": 1.0,           # No leverage for now
        }
    
    async def _process_pending_orders(self):
        """Process pending orders"""
        try:
            # This would process pending orders from the database
            # For now, we'll just log the activity
            self.record_metric("orders_processed", 0)
            
        except Exception as e:
            self.logger.error(f"Failed to process pending orders: {e}")
    
    async def _update_price_feeds(self):
        """Update price feeds for all active markets"""
        try:
            with get_db_session() as db:
                active_markets = (
                    db.query(Market)
                    .filter(Market.status == MarketStatus.OPEN)
                    .all()
                )
                
                for market in active_markets:
                    # Update price feed data
                    self.price_feeds[market.id] = {
                        "price_a": market.current_price_a,
                        "price_b": market.current_price_b,
                        "volume_24h": market.volume_24h,
                        "last_updated": datetime.utcnow()
                    }
                
                self.record_metric("price_feeds_updated", len(active_markets))
                
        except Exception as e:
            self.logger.error(f"Failed to update price feeds: {e}")
    
    async def _risk_monitoring(self):
        """Monitor trading risks"""
        try:
            with get_db_session() as db:
                # Check for unusual trading patterns
                now = datetime.utcnow()
                since_1h = now - timedelta(hours=1)
                
                # Get recent trades
                recent_trades = (
                    db.query(Trade)
                    .filter(Trade.created_at >= since_1h)
                    .all()
                )
                
                # Analyze for risk patterns
                total_volume_1h = sum(trade.amount for trade in recent_trades)
                unique_traders = len(set(trade.user_id for trade in recent_trades))
                
                # Check for high volume alerts
                if total_volume_1h > 100000:  # $100k in 1 hour
                    log_system_metric("high_volume_alert", total_volume_1h, {
                        "timeframe": "1h",
                        "unique_traders": unique_traders
                    })
                
                # Check for unusual trader activity
                trader_volumes = {}
                for trade in recent_trades:
                    if trade.user_id not in trader_volumes:
                        trader_volumes[trade.user_id] = 0
                    trader_volumes[trade.user_id] += trade.amount
                
                # Alert on high individual trader volume
                for user_id, volume in trader_volumes.items():
                    if volume > 10000:  # $10k in 1 hour
                        log_system_metric("high_trader_volume", volume, {
                            "user_id": user_id,
                            "timeframe": "1h"
                        })
                
                self.record_metric("risk_checks_performed", 1)
                
        except Exception as e:
            self.logger.error(f"Failed to perform risk monitoring: {e}")
    
    async def _cleanup_old_data(self):
        """Cleanup old trading data"""
        try:
            # Clean up old price feeds (older than 1 hour)
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            old_feeds = [
                market_id for market_id, data in self.price_feeds.items()
                if data["last_updated"] < cutoff_time
            ]
            
            for market_id in old_feeds:
                del self.price_feeds[market_id]
            
            if old_feeds:
                self.logger.info(f"Cleaned up {len(old_feeds)} old price feeds")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def execute_trade(
        self, 
        user_id: int, 
        market_id: int, 
        outcome: str, 
        trade_type: str, 
        amount: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute a trade with comprehensive validation and risk checks"""
        try:
            with get_db_session() as db:
                # Get user and market
                user = db.query(User).filter(User.id == user_id).first()
                market = db.query(Market).filter(Market.id == market_id).first()
                
                if not user or not market:
                    return {"success": False, "error": "User or market not found"}
                
                # Validate trade
                validation_result = self._validate_trade(user, market, outcome, trade_type, amount, price)
                if not validation_result["valid"]:
                    return {"success": False, "error": validation_result["error"]}
                
                # Check risk limits
                risk_check = self._check_risk_limits(user, amount, market_id)
                if not risk_check["allowed"]:
                    return {"success": False, "error": risk_check["reason"]}
                
                # Calculate trade details
                trade_details = self._calculate_trade_details(market, outcome, trade_type, amount, price)
                
                # Execute the trade
                trade_result = self._execute_trade_internal(
                    db, user, market, outcome, trade_type, amount, trade_details
                )
                
                if trade_result["success"]:
                    # Log successful trade
                    log_trading_event(
                        "trade_executed",
                        market_id,
                        user_id,
                        amount,
                        outcome=outcome,
                        trade_type=trade_type,
                        price_a=trade_details["price_a"],
                        price_b=trade_details["price_b"],
                        fee=trade_details["fee"]
                    )
                    
                    # Update metrics
                    self.record_metric("trades_executed", 1)
                    self.record_metric("trade_volume", amount)
                    
                    # Invalidate caches
                    self.invalidate_trading_cache(user_id, market_id)
                
                return trade_result
                
        except Exception as e:
            self.logger.error(f"Failed to execute trade: {e}")
            return {"success": False, "error": "Internal error"}
    
    def _validate_trade(
        self, 
        user: User, 
        market: Market, 
        outcome: str, 
        trade_type: str, 
        amount: float, 
        price: Optional[float]
    ) -> Dict[str, Any]:
        """Validate trade parameters"""
        # Check if user can trade
        if not user.can_trade(amount):
            return {"valid": False, "error": "Insufficient balance or inactive account"}
        
        # Check if market is active
        if not market.is_active:
            return {"valid": False, "error": "Market is not active for trading"}
        
        # Validate amount
        if amount < market.min_trade_amount:
            return {"valid": False, "error": f"Amount below minimum: ${market.min_trade_amount}"}
        
        if amount > market.max_trade_amount:
            return {"valid": False, "error": f"Amount above maximum: ${market.max_trade_amount}"}
        
        # Validate outcome
        if outcome not in ["outcome_a", "outcome_b"]:
            return {"valid": False, "error": "Invalid outcome"}
        
        # Validate trade type
        if trade_type not in ["buy", "sell"]:
            return {"valid": False, "error": "Invalid trade type"}
        
        return {"valid": True}
    
    def _check_risk_limits(self, user: User, amount: float, market_id: int) -> Dict[str, Any]:
        """Check risk limits for the trade"""
        # Check maximum trade amount
        if amount > self.risk_limits["max_trade_amount"]:
            return {
                "allowed": False,
                "reason": f"Trade amount exceeds maximum: ${self.risk_limits['max_trade_amount']}"
            }
        
        # Check minimum trade amount
        if amount < self.risk_limits["min_trade_amount"]:
            return {
                "allowed": False,
                "reason": f"Trade amount below minimum: ${self.risk_limits['min_trade_amount']}"
            }
        
        # Check daily loss limit (simplified)
        if user.total_profit < -self.risk_limits["max_daily_loss"]:
            return {
                "allowed": False,
                "reason": "Daily loss limit exceeded"
            }
        
        return {"allowed": True}
    
    def _calculate_trade_details(
        self, 
        market: Market, 
        outcome: str, 
        trade_type: str, 
        amount: float, 
        price: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate trade details including prices and fees"""
        # Use current market prices if no specific price provided
        if price is None:
            if outcome == "outcome_a":
                price = market.current_price_a
            else:
                price = market.current_price_b
        
        # Calculate fee
        fee = amount * market.fee_rate
        
        # Calculate net amount (amount after fee)
        net_amount = amount - fee
        
        return {
            "price_a": market.current_price_a,
            "price_b": market.current_price_b,
            "trade_price": price,
            "amount": amount,
            "fee": fee,
            "net_amount": net_amount,
            "fee_rate": market.fee_rate
        }
    
    def _execute_trade_internal(
        self, 
        db: Session, 
        user: User, 
        market: Market, 
        outcome: str, 
        trade_type: str, 
        amount: float, 
        trade_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the trade in the database"""
        try:
            # Create trade record
            trade = Trade(
                user_id=user.id,
                market_id=market.id,
                outcome=outcome,
                trade_type=trade_type,
                amount=amount,
                price_a=trade_details["price_a"],
                price_b=trade_details["price_b"],
                fee=trade_details["fee"],
                total_value=amount,
                status=TradeStatus.COMPLETED,
                created_at=datetime.utcnow()
            )
            
            db.add(trade)
            
            # Update market prices and liquidity
            market.update_prices(amount, outcome, trade_type)
            market.volume_24h += amount
            market.volume_total += amount
            
            # Update user stats
            profit = 0.0  # Profit will be calculated when position is closed
            user.update_stats(amount, profit)
            
            # Update user balance
            if trade_type == "buy":
                user.available_balance -= amount
                user.total_invested += amount
            else:  # sell
                user.available_balance += amount
                user.total_invested -= amount
            
            # Update or create position
            position = db.query(Position).filter(
                and_(Position.user_id == user.id, Position.market_id == market.id)
            ).first()
            
            if not position:
                position = Position(
                    user_id=user.id,
                    market_id=market.id,
                    outcome_a_shares=0.0,
                    outcome_b_shares=0.0,
                    total_invested=0.0,
                    current_value=0.0
                )
                db.add(position)
            
            # Update position
            if outcome == "outcome_a":
                if trade_type == "buy":
                    position.outcome_a_shares += amount / trade_details["price_a"]
                else:
                    position.outcome_a_shares -= amount / trade_details["price_a"]
            else:
                if trade_type == "buy":
                    position.outcome_b_shares += amount / trade_details["price_b"]
                else:
                    position.outcome_b_shares -= amount / trade_details["price_b"]
            
            position.total_invested += amount if trade_type == "buy" else -amount
            position.current_value = (
                position.outcome_a_shares * market.current_price_a +
                position.outcome_b_shares * market.current_price_b
            )
            
            # Update user portfolio value
            user.portfolio_value = sum(
                pos.current_value for pos in db.query(Position).filter(Position.user_id == user.id).all()
            )
            
            db.commit()
            
            return {
                "success": True,
                "trade_id": trade.id,
                "trade_details": trade_details,
                "new_balance": user.available_balance,
                "new_portfolio_value": user.portfolio_value
            }
            
        except Exception as e:
            db.rollback()
            self.logger.error(f"Failed to execute trade in database: {e}")
            return {"success": False, "error": "Database error"}
    
    def get_user_trading_stats(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive trading statistics for a user"""
        cache_key = self.get_cache_key("user_trading_stats", user_id)
        cached_stats = self.get_cached(cache_key)
        
        if cached_stats:
            return cached_stats
        
        try:
            with get_db_session() as db:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return {}
                
                # Get recent trades
                since_24h = datetime.utcnow() - timedelta(hours=24)
                recent_trades = (
                    db.query(Trade)
                    .filter(Trade.user_id == user_id)
                    .filter(Trade.created_at >= since_24h)
                    .all()
                )
                
                # Calculate stats
                stats = {
                    "user_id": user_id,
                    "total_trades": user.total_trades,
                    "successful_trades": user.successful_trades,
                    "success_rate": user.success_rate,
                    "total_volume": user.total_volume,
                    "total_profit": user.total_profit,
                    "win_rate": user.win_rate,
                    "avg_trade_size": user.avg_trade_size,
                    "largest_win": user.largest_win,
                    "largest_loss": user.largest_loss,
                    "reputation_score": user.reputation_score,
                    "portfolio_value": user.portfolio_value,
                    "available_balance": user.available_balance,
                    "total_balance": user.total_balance,
                    "profit_loss_percentage": user.profit_loss_percentage,
                    "trades_24h": len(recent_trades),
                    "volume_24h": sum(trade.amount for trade in recent_trades),
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                # Cache the stats
                self.set_cached(cache_key, stats, ttl=300)
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get trading stats for user {user_id}: {e}")
            return {}
    
    def get_market_trading_stats(self, market_id: int) -> Dict[str, Any]:
        """Get comprehensive trading statistics for a market"""
        cache_key = self.get_cache_key("market_trading_stats", market_id)
        cached_stats = self.get_cached(cache_key)
        
        if cached_stats:
            return cached_stats
        
        try:
            with get_db_session() as db:
                market = db.query(Market).filter(Market.id == market_id).first()
                if not market:
                    return {}
                
                # Get recent trades
                since_24h = datetime.utcnow() - timedelta(hours=24)
                recent_trades = (
                    db.query(Trade)
                    .filter(Trade.market_id == market_id)
                    .filter(Trade.created_at >= since_24h)
                    .all()
                )
                
                # Calculate stats
                stats = {
                    "market_id": market_id,
                    "total_trades": len(market.trades),
                    "total_volume": market.volume_total,
                    "volume_24h": market.volume_24h,
                    "unique_traders": market.unique_traders,
                    "current_price_a": market.current_price_a,
                    "current_price_b": market.current_price_b,
                    "total_liquidity": market.total_liquidity,
                    "liquidity_pool_a": market.liquidity_pool_a,
                    "liquidity_pool_b": market.liquidity_pool_b,
                    "trades_24h": len(recent_trades),
                    "volume_24h_trades": sum(trade.amount for trade in recent_trades),
                    "unique_traders_24h": len(set(trade.user_id for trade in recent_trades)),
                    "avg_trade_size_24h": (
                        sum(trade.amount for trade in recent_trades) / len(recent_trades)
                        if recent_trades else 0
                    ),
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                # Cache the stats
                self.set_cached(cache_key, stats, ttl=300)
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get trading stats for market {market_id}: {e}")
            return {}
    
    def invalidate_trading_cache(self, user_id: int, market_id: int):
        """Invalidate trading-related cache"""
        patterns = [
            self.get_cache_key("user_trading_stats", user_id),
            self.get_cache_key("market_trading_stats", market_id),
        ]
        
        for pattern in patterns:
            self.delete_cached(pattern)
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service-specific metrics"""
        return {
            "active_orders": len(self.active_orders),
            "price_feeds": len(self.price_feeds),
            "background_tasks": len(self.tasks),
            "uptime": self.get_uptime(),
            "metrics": self.get_all_metrics()
        }


# Global trading service instance
enhanced_trading_service = EnhancedTradingService()
