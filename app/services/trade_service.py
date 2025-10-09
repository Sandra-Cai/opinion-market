"""
Trade Service for Opinion Market
Handles trading-related business logic including order execution, portfolio management, and analytics
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
from app.models.trade import Trade, TradeType, TradeOutcome, TradeStatus
from app.models.market import Market, MarketStatus
from app.models.user import User
from app.schemas.trade import TradeCreate, TradeResponse, TradeStats


@dataclass
class TradeAnalytics:
    """Trade analytics data structure"""
    total_trades: int
    successful_trades: int
    total_volume: float
    total_profit: float
    win_rate: float
    avg_trade_size: float
    largest_win: float
    largest_loss: float
    trading_frequency: float
    risk_reward_ratio: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    trade_distribution: Dict[str, int]
    performance_by_category: Dict[str, float]
    monthly_performance: List[Dict[str, Any]]


@dataclass
class PortfolioPosition:
    """Portfolio position data structure"""
    market_id: int
    market_title: str
    outcome: str
    shares: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    total_value: float
    percentage_of_portfolio: float


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    market_id: int
    signal_type: str  # "buy", "sell", "hold"
    confidence: float
    reasoning: str
    expected_return: float
    risk_level: str
    time_horizon: str


class TradeService:
    """Service for trade-related operations"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.cache_ttl = 300  # 5 minutes
        
    async def execute_trade(
        self,
        trade_data: TradeCreate,
        user_id: int,
        db: Session
    ) -> Trade:
        """Execute a trade with comprehensive validation and risk management"""
        try:
            # Validate trade data
            await self._validate_trade_data(trade_data, user_id, db)
            
            # Check user permissions and balance
            await self._check_trading_permissions(user_id, trade_data, db)
            
            # Get market information
            market = await self._get_market_for_trade(trade_data.market_id, db)
            
            # Calculate trade details
            trade_details = await self._calculate_trade_details(trade_data, market, db)
            
            # Execute the trade
            trade = await self._execute_trade_transaction(trade_data, user_id, trade_details, db)
            
            # Update market prices and volume
            await self._update_market_metrics(market, trade, db)
            
            # Update user portfolio
            await self._update_user_portfolio(user_id, trade, db)
            
            # Log trade execution
            log_trading_event("trade_executed", {
                "trade_id": trade.id,
                "user_id": user_id,
                "market_id": trade_data.market_id,
                "trade_type": trade_data.trade_type.value,
                "outcome": trade_data.outcome.value,
                "amount": trade_data.amount,
                "total_value": trade.total_value
            })
            
            # Clear cache
            await self._clear_trade_cache(user_id, trade_data.market_id)
            
            return trade
            
        except Exception as e:
            db.rollback()
            log_system_metric("trade_execution_error", 1, {"error": str(e)})
            raise
    
    async def get_user_trades(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 20,
        market_id: Optional[int] = None,
        trade_type: Optional[TradeType] = None,
        status: Optional[TradeStatus] = None,
        db: Session = None
    ) -> Tuple[List[Trade], int]:
        """Get user trades with filtering and pagination"""
        try:
            query = db.query(Trade).filter(Trade.user_id == user_id)
            
            # Apply filters
            if market_id:
                query = query.filter(Trade.market_id == market_id)
            
            if trade_type:
                query = query.filter(Trade.trade_type == trade_type)
            
            if status:
                query = query.filter(Trade.status == status)
            
            # Get total count
            total = query.count()
            
            # Apply pagination and ordering
            trades = query.order_by(desc(Trade.created_at)).offset(skip).limit(limit).all()
            
            return trades, total
            
        except Exception as e:
            log_system_metric("user_trades_error", 1, {"error": str(e)})
            raise
    
    async def get_trade_by_id(self, trade_id: int, user_id: int, db: Session) -> Optional[Trade]:
        """Get trade by ID with permission check"""
        try:
            trade = db.query(Trade).filter(
                Trade.id == trade_id,
                Trade.user_id == user_id
            ).first()
            
            return trade
            
        except Exception as e:
            log_system_metric("trade_by_id_error", 1, {"error": str(e)})
            raise
    
    async def cancel_trade(self, trade_id: int, user_id: int, db: Session) -> bool:
        """Cancel a pending trade"""
        try:
            trade = await self.get_trade_by_id(trade_id, user_id, db)
            if not trade:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Trade not found"
                )
            
            # Check if trade can be cancelled
            if trade.status != TradeStatus.PENDING:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Trade cannot be cancelled"
                )
            
            # Check time limit (5 minutes)
            time_limit = timedelta(minutes=5)
            if datetime.utcnow() - trade.created_at > time_limit:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Trade cancellation time limit exceeded"
                )
            
            # Cancel trade
            trade.status = TradeStatus.CANCELLED
            trade.updated_at = datetime.utcnow()
            
            db.commit()
            
            # Log cancellation
            log_trading_event("trade_cancelled", {
                "trade_id": trade.id,
                "user_id": user_id
            })
            
            # Clear cache
            await self._clear_trade_cache(user_id, trade.market_id)
            
            return True
            
        except Exception as e:
            db.rollback()
            log_system_metric("trade_cancellation_error", 1, {"error": str(e)})
            raise
    
    async def get_user_portfolio(self, user_id: int, db: Session) -> List[PortfolioPosition]:
        """Get user's current portfolio positions"""
        try:
            # Get all user trades
            trades = db.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.status == TradeStatus.COMPLETED
            ).all()
            
            # Group trades by market and outcome
            positions = {}
            
            for trade in trades:
                key = (trade.market_id, trade.outcome.value)
                
                if key not in positions:
                    positions[key] = {
                        "market_id": trade.market_id,
                        "outcome": trade.outcome.value,
                        "shares": 0.0,
                        "total_cost": 0.0,
                        "total_value": 0.0,
                        "trades": []
                    }
                
                position = positions[key]
                position["trades"].append(trade)
                
                # Calculate shares and cost
                if trade.trade_type == TradeType.BUY:
                    position["shares"] += trade.amount
                    position["total_cost"] += trade.total_value
                else:  # SELL
                    position["shares"] -= trade.amount
                    position["total_cost"] -= trade.total_value
                
                position["total_value"] = position["shares"] * trade.price_per_share
            
            # Convert to PortfolioPosition objects
            portfolio = []
            total_portfolio_value = sum(pos["total_value"] for pos in positions.values())
            
            for key, pos in positions.items():
                if pos["shares"] > 0:  # Only include positions with shares
                    # Get market information
                    market = db.query(Market).filter(Market.id == pos["market_id"]).first()
                    
                    if market:
                        avg_price = pos["total_cost"] / pos["shares"] if pos["shares"] > 0 else 0
                        current_price = market.price_a if pos["outcome"] == "outcome_a" else market.price_b
                        unrealized_pnl = (current_price - avg_price) * pos["shares"]
                        percentage = (pos["total_value"] / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                        
                        portfolio.append(PortfolioPosition(
                            market_id=pos["market_id"],
                            market_title=market.title,
                            outcome=pos["outcome"],
                            shares=pos["shares"],
                            avg_price=avg_price,
                            current_price=current_price,
                            unrealized_pnl=unrealized_pnl,
                            realized_pnl=0.0,  # Would need to track this separately
                            total_value=pos["total_value"],
                            percentage_of_portfolio=percentage
                        ))
            
            return portfolio
            
        except Exception as e:
            log_system_metric("portfolio_error", 1, {"error": str(e)})
            raise
    
    async def get_trade_analytics(self, user_id: int, db: Session) -> TradeAnalytics:
        """Get comprehensive trade analytics for a user"""
        try:
            # Get all user trades
            trades = db.query(Trade).filter(
                Trade.user_id == user_id,
                Trade.status == TradeStatus.COMPLETED
            ).all()
            
            if not trades:
                return TradeAnalytics(
                    total_trades=0,
                    successful_trades=0,
                    total_volume=0.0,
                    total_profit=0.0,
                    win_rate=0.0,
                    avg_trade_size=0.0,
                    largest_win=0.0,
                    largest_loss=0.0,
                    trading_frequency=0.0,
                    risk_reward_ratio=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    profit_factor=0.0,
                    trade_distribution={},
                    performance_by_category={},
                    monthly_performance=[]
                )
            
            # Calculate basic metrics
            total_trades = len(trades)
            total_volume = sum(trade.total_value for trade in trades)
            avg_trade_size = total_volume / total_trades
            
            # Calculate profit/loss (simplified)
            total_profit = 0.0
            successful_trades = 0
            largest_win = 0.0
            largest_loss = 0.0
            
            for trade in trades:
                # Simplified P&L calculation
                if trade.trade_type == TradeType.BUY:
                    # Assume 10% profit for successful trades
                    profit = trade.total_value * 0.1
                    total_profit += profit
                    successful_trades += 1
                    largest_win = max(largest_win, profit)
                else:
                    # Assume 5% loss for sell trades
                    loss = trade.total_value * 0.05
                    total_profit -= loss
                    largest_loss = max(largest_loss, loss)
            
            # Calculate rates
            win_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0.0
            
            # Calculate trading frequency
            trading_days = (datetime.utcnow() - trades[0].created_at).days
            trading_frequency = total_trades / max(1, trading_days)
            
            # Calculate risk metrics
            risk_reward_ratio = largest_win / max(largest_loss, 1)
            sharpe_ratio = self._calculate_sharpe_ratio(trades)
            max_drawdown = self._calculate_max_drawdown(trades)
            profit_factor = self._calculate_profit_factor(trades)
            
            # Trade distribution
            trade_distribution = {
                "buy": len([t for t in trades if t.trade_type == TradeType.BUY]),
                "sell": len([t for t in trades if t.trade_type == TradeType.SELL]),
                "outcome_a": len([t for t in trades if t.outcome == TradeOutcome.OUTCOME_A]),
                "outcome_b": len([t for t in trades if t.outcome == TradeOutcome.OUTCOME_B])
            }
            
            # Performance by category
            performance_by_category = self._calculate_performance_by_category(user_id, db)
            
            # Monthly performance
            monthly_performance = self._calculate_monthly_performance(trades)
            
            return TradeAnalytics(
                total_trades=total_trades,
                successful_trades=successful_trades,
                total_volume=total_volume,
                total_profit=total_profit,
                win_rate=win_rate,
                avg_trade_size=avg_trade_size,
                largest_win=largest_win,
                largest_loss=largest_loss,
                trading_frequency=trading_frequency,
                risk_reward_ratio=risk_reward_ratio,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                profit_factor=profit_factor,
                trade_distribution=trade_distribution,
                performance_by_category=performance_by_category,
                monthly_performance=monthly_performance
            )
            
        except Exception as e:
            log_system_metric("trade_analytics_error", 1, {"error": str(e)})
            raise
    
    async def get_trade_stats(self, db: Session) -> TradeStats:
        """Get overall trade statistics"""
        try:
            # Total trades
            total_trades = db.query(Trade).count()
            
            # Total volume
            total_volume = db.query(func.sum(Trade.total_value)).scalar() or 0.0
            
            # 24h volume
            yesterday = datetime.utcnow() - timedelta(days=1)
            volume_24h = db.query(func.sum(Trade.total_value)).filter(
                Trade.created_at >= yesterday
            ).scalar() or 0.0
            
            # Active traders
            active_traders = db.query(func.count(func.distinct(Trade.user_id))).filter(
                Trade.created_at >= yesterday
            ).scalar() or 0
            
            # Top traders
            top_traders = (
                db.query(User, func.sum(Trade.total_value).label('total_volume'))
                .join(Trade, User.id == Trade.user_id)
                .group_by(User.id)
                .order_by(desc('total_volume'))
                .limit(10)
                .all()
            )
            
            return TradeStats(
                total_trades=total_trades,
                total_volume=total_volume,
                volume_24h=volume_24h,
                active_traders_24h=active_traders,
                top_traders=[trader[0] for trader in top_traders]
            )
            
        except Exception as e:
            log_system_metric("trade_stats_error", 1, {"error": str(e)})
            raise
    
    async def _validate_trade_data(self, trade_data: TradeCreate, user_id: int, db: Session):
        """Validate trade data"""
        # Check minimum trade amount
        if trade_data.amount < settings.MIN_TRADE_AMOUNT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Trade amount must be at least {settings.MIN_TRADE_AMOUNT}"
            )
        
        # Check maximum trade amount
        if trade_data.amount > settings.MAX_TRADE_AMOUNT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Trade amount cannot exceed {settings.MAX_TRADE_AMOUNT}"
            )
        
        # Check if market exists and is open
        market = db.query(Market).filter(Market.id == trade_data.market_id).first()
        if not market:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Market not found"
            )
        
        if market.status != MarketStatus.OPEN:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Market is not open for trading"
            )
        
        # Check if market is not expired
        if market.closes_at <= datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Market has expired"
            )
    
    async def _check_trading_permissions(self, user_id: int, trade_data: TradeCreate, db: Session):
        """Check user trading permissions and balance"""
        # Check if user exists and is active
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User not found or inactive"
            )
        
        # Check daily trading limits
        today = datetime.utcnow().date()
        trades_today = db.query(Trade).filter(
            Trade.user_id == user_id,
            func.date(Trade.created_at) == today
        ).count()
        
        if trades_today >= 100:  # Max 100 trades per day
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Daily trading limit reached"
            )
        
        # Check user balance (simplified)
        # In a real system, you'd check actual user balance
        estimated_cost = trade_data.amount * 0.5  # Assume 50% of amount as cost
        if estimated_cost > 10000:  # Max trade value
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insufficient balance for this trade"
            )
    
    async def _get_market_for_trade(self, market_id: int, db: Session) -> Market:
        """Get market information for trade"""
        market = db.query(Market).filter(Market.id == market_id).first()
        if not market:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Market not found"
            )
        return market
    
    async def _calculate_trade_details(self, trade_data: TradeCreate, market: Market, db: Session) -> Dict[str, Any]:
        """Calculate trade details including price and fees"""
        # Get current market prices
        current_price_a = market.price_a
        current_price_b = market.price_b
        
        # Determine trade price based on outcome
        if trade_data.outcome == TradeOutcome.OUTCOME_A:
            trade_price = current_price_a
        else:
            trade_price = current_price_b
        
        # Calculate total value
        total_value = trade_data.amount * trade_price
        
        # Calculate trading fee
        trading_fee = total_value * market.trading_fee
        
        # Calculate final total
        final_total = total_value + trading_fee
        
        return {
            "price_per_share": trade_price,
            "price_a": current_price_a,
            "price_b": current_price_b,
            "total_value": total_value,
            "trading_fee": trading_fee,
            "final_total": final_total
        }
    
    async def _execute_trade_transaction(
        self,
        trade_data: TradeCreate,
        user_id: int,
        trade_details: Dict[str, Any],
        db: Session
    ) -> Trade:
        """Execute the trade transaction"""
        # Create trade record
        trade = Trade(
            trade_type=trade_data.trade_type,
            outcome=trade_data.outcome,
            amount=trade_data.amount,
            price_a=trade_details["price_a"],
            price_b=trade_details["price_b"],
            price_per_share=trade_details["price_per_share"],
            total_value=trade_details["total_value"],
            fee=trade_details["trading_fee"],
            market_id=trade_data.market_id,
            user_id=user_id,
            status=TradeStatus.COMPLETED,
            trade_hash=self._generate_trade_hash()
        )
        
        db.add(trade)
        db.commit()
        db.refresh(trade)
        
        return trade
    
    async def _update_market_metrics(self, market: Market, trade: Trade, db: Session):
        """Update market metrics after trade"""
        # Update volume
        market.volume_total += trade.total_value
        market.volume_24h += trade.total_value
        
        # Update prices based on trade (simplified)
        if trade.outcome == TradeOutcome.OUTCOME_A:
            if trade.trade_type == TradeType.BUY:
                market.price_a = min(1.0, market.price_a + 0.01)
                market.price_b = max(0.0, market.price_b - 0.01)
            else:
                market.price_a = max(0.0, market.price_a - 0.01)
                market.price_b = min(1.0, market.price_b + 0.01)
        else:
            if trade.trade_type == TradeType.BUY:
                market.price_b = min(1.0, market.price_b + 0.01)
                market.price_a = max(0.0, market.price_a - 0.01)
            else:
                market.price_b = max(0.0, market.price_b - 0.01)
                market.price_a = min(1.0, market.price_a + 0.01)
        
        # Update trending score
        market.trending_score = min(100.0, market.trending_score + 1.0)
        
        # Update liquidity score
        market.liquidity_score = min(100.0, market.liquidity_score + 0.5)
        
        db.commit()
    
    async def _update_user_portfolio(self, user_id: int, trade: Trade, db: Session):
        """Update user portfolio after trade"""
        # In a real system, you'd update user balance and portfolio
        # For now, we'll just log the update
        log_system_metric("user_portfolio_updated", 1, {
            "user_id": user_id,
            "trade_id": trade.id,
            "trade_value": trade.total_value
        })
    
    async def _clear_trade_cache(self, user_id: int, market_id: int):
        """Clear trade-related cache entries"""
        cache_keys = [
            f"user_trades:{user_id}",
            f"user_portfolio:{user_id}",
            f"user_trade_analytics:{user_id}",
            f"market_trades:{market_id}",
            "trade_stats"
        ]
        
        for key in cache_keys:
            cache.delete(key)
    
    def _generate_trade_hash(self) -> str:
        """Generate unique trade hash"""
        timestamp = int(datetime.utcnow().timestamp() * 1000)
        random_part = uuid.uuid4().hex[:8]
        return f"trade_{timestamp}_{random_part}"
    
    def _calculate_sharpe_ratio(self, trades: List[Trade]) -> float:
        """Calculate Sharpe ratio for trades"""
        if len(trades) < 2:
            return 0.0
        
        # Simplified calculation
        returns = [trade.total_value * 0.1 for trade in trades]  # Assume 10% return
        mean_return = sum(returns) / len(returns)
        
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return 0.0
        
        return mean_return / std_dev
    
    def _calculate_max_drawdown(self, trades: List[Trade]) -> float:
        """Calculate maximum drawdown"""
        if not trades:
            return 0.0
        
        # Simplified calculation
        peak = 0.0
        max_dd = 0.0
        
        for trade in trades:
            value = trade.total_value
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd * 100  # Return as percentage
    
    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """Calculate profit factor"""
        if not trades:
            return 0.0
        
        total_profit = 0.0
        total_loss = 0.0
        
        for trade in trades:
            if trade.trade_type == TradeType.BUY:
                total_profit += trade.total_value * 0.1  # Assume 10% profit
            else:
                total_loss += trade.total_value * 0.05  # Assume 5% loss
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0
        
        return total_profit / total_loss
    
    def _calculate_performance_by_category(self, user_id: int, db: Session) -> Dict[str, float]:
        """Calculate performance by market category"""
        try:
            # Get trades grouped by category
            category_performance = (
                db.query(Market.category, func.sum(Trade.total_value).label('total_volume'))
                .join(Trade, Market.id == Trade.market_id)
                .filter(Trade.user_id == user_id)
                .group_by(Market.category)
                .all()
            )
            
            return {
                category.value: float(volume) for category, volume in category_performance
            }
        except Exception:
            return {}
    
    def _calculate_monthly_performance(self, trades: List[Trade]) -> List[Dict[str, Any]]:
        """Calculate monthly performance"""
        if not trades:
            return []
        
        # Group trades by month
        monthly_trades = {}
        for trade in trades:
            month_key = trade.created_at.strftime("%Y-%m")
            if month_key not in monthly_trades:
                monthly_trades[month_key] = []
            monthly_trades[month_key].append(trade)
        
        # Calculate performance for each month
        monthly_performance = []
        for month, month_trades in monthly_trades.items():
            total_volume = sum(trade.total_value for trade in month_trades)
            total_trades = len(month_trades)
            avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
            
            monthly_performance.append({
                "month": month,
                "total_trades": total_trades,
                "total_volume": total_volume,
                "avg_trade_size": avg_trade_size
            })
        
        return sorted(monthly_performance, key=lambda x: x["month"])


# Global trade service instance
trade_service = TradeService()
