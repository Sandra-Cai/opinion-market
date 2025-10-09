"""
Enhanced Position model for Opinion Market
Provides comprehensive position tracking and portfolio management
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, ForeignKey, Index, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from app.core.database import Base


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    
    # Position identification
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    
    # Position details
    outcome_a_shares = Column(Float, default=0.0)  # Shares in outcome A
    outcome_b_shares = Column(Float, default=0.0)  # Shares in outcome B
    
    # Investment tracking
    total_invested = Column(Float, default=0.0)  # Total amount invested
    total_fees_paid = Column(Float, default=0.0)  # Total fees paid
    total_withdrawn = Column(Float, default=0.0)  # Total amount withdrawn
    
    # Current values
    current_value = Column(Float, default=0.0)  # Current position value
    unrealized_pnl = Column(Float, default=0.0)  # Unrealized profit/loss
    realized_pnl = Column(Float, default=0.0)  # Realized profit/loss
    
    # Position statistics
    entry_price_a = Column(Float, default=0.0)  # Average entry price for outcome A
    entry_price_b = Column(Float, default=0.0)  # Average entry price for outcome B
    exit_price_a = Column(Float, default=0.0)  # Exit price for outcome A (if closed)
    exit_price_b = Column(Float, default=0.0)  # Exit price for outcome B (if closed)
    
    # Position status
    is_closed = Column(Boolean, default=False)  # Whether position is closed
    is_hedged = Column(Boolean, default=False)  # Whether position is hedged
    risk_level = Column(String, default="medium")  # Risk level: low, medium, high
    
    # Position metadata
    position_type = Column(String, default="long")  # long, short, neutral
    strategy = Column(String, nullable=True)  # Trading strategy used
    notes = Column(String, nullable=True)  # User notes
    tags = Column(JSON, default=list)  # Position tags
    
    # Performance metrics
    max_profit = Column(Float, default=0.0)  # Maximum profit achieved
    max_loss = Column(Float, default=0.0)  # Maximum loss incurred
    sharpe_ratio = Column(Float, default=0.0)  # Risk-adjusted return
    win_rate = Column(Float, default=0.0)  # Win rate for this position
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    closed_at = Column(DateTime, nullable=True)
    last_trade_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="positions")
    market = relationship("Market", back_populates="positions")
    
    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint('outcome_a_shares >= 0', name='check_non_negative_shares_a'),
        CheckConstraint('outcome_b_shares >= 0', name='check_non_negative_shares_b'),
        CheckConstraint('total_invested >= 0', name='check_non_negative_invested'),
        CheckConstraint('total_fees_paid >= 0', name='check_non_negative_fees'),
        CheckConstraint('total_withdrawn >= 0', name='check_non_negative_withdrawn'),
        CheckConstraint('current_value >= 0', name='check_non_negative_value'),
        CheckConstraint('entry_price_a >= 0 AND entry_price_a <= 1', name='check_entry_price_a_range'),
        CheckConstraint('entry_price_b >= 0 AND entry_price_b <= 1', name='check_entry_price_b_range'),
        CheckConstraint('exit_price_a >= 0 AND exit_price_a <= 1', name='check_exit_price_a_range'),
        CheckConstraint('exit_price_b >= 0 AND exit_price_b <= 1', name='check_exit_price_b_range'),
        CheckConstraint('max_profit >= 0', name='check_non_negative_max_profit'),
        CheckConstraint('win_rate >= 0 AND win_rate <= 100', name='check_win_rate_range'),
        Index('idx_position_user', 'user_id'),
        Index('idx_position_market', 'market_id'),
        Index('idx_position_closed', 'is_closed'),
        Index('idx_position_created_at', 'created_at'),
        Index('idx_position_updated_at', 'updated_at'),
        Index('idx_position_user_market', 'user_id', 'market_id', unique=True),
    )
    
    @property
    def total_shares(self) -> float:
        """Total shares across both outcomes"""
        return self.outcome_a_shares + self.outcome_b_shares
    
    @property
    def net_investment(self) -> float:
        """Net investment (invested - withdrawn)"""
        return self.total_invested - self.total_withdrawn
    
    @property
    def total_pnl(self) -> float:
        """Total profit/loss (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_percentage(self) -> float:
        """Profit/loss as percentage of investment"""
        if self.net_investment == 0:
            return 0.0
        return (self.total_pnl / self.net_investment) * 100
    
    @property
    def is_profitable(self) -> bool:
        """Whether position is currently profitable"""
        return self.total_pnl > 0
    
    @property
    def is_active(self) -> bool:
        """Whether position is active (not closed)"""
        return not self.is_closed and self.total_shares > 0
    
    @property
    def is_hedged_position(self) -> bool:
        """Whether position is hedged (has shares in both outcomes)"""
        return self.outcome_a_shares > 0 and self.outcome_b_shares > 0
    
    @property
    def exposure_percentage(self) -> Dict[str, float]:
        """Percentage exposure to each outcome"""
        total = self.total_shares
        if total == 0:
            return {"outcome_a": 0.0, "outcome_b": 0.0}
        
        return {
            "outcome_a": (self.outcome_a_shares / total) * 100,
            "outcome_b": (self.outcome_b_shares / total) * 100
        }
    
    def update_position(self, outcome: str, shares: float, price: float, trade_type: str, fee: float = 0.0):
        """Update position with new trade"""
        if outcome not in ["outcome_a", "outcome_b"]:
            raise ValueError("Invalid outcome")
        
        # Update shares
        if outcome == "outcome_a":
            if trade_type == "buy":
                # Update average entry price for outcome A
                if self.outcome_a_shares > 0:
                    total_value = self.outcome_a_shares * self.entry_price_a
                    new_value = shares * price
                    self.entry_price_a = (total_value + new_value) / (self.outcome_a_shares + shares)
                else:
                    self.entry_price_a = price
                self.outcome_a_shares += shares
            else:  # sell
                self.outcome_a_shares -= shares
                if self.outcome_a_shares < 0:
                    self.outcome_a_shares = 0
        else:  # outcome_b
            if trade_type == "buy":
                # Update average entry price for outcome B
                if self.outcome_b_shares > 0:
                    total_value = self.outcome_b_shares * self.entry_price_b
                    new_value = shares * price
                    self.entry_price_b = (total_value + new_value) / (self.outcome_b_shares + shares)
                else:
                    self.entry_price_b = price
                self.outcome_b_shares += shares
            else:  # sell
                self.outcome_b_shares -= shares
                if self.outcome_b_shares < 0:
                    self.outcome_b_shares = 0
        
        # Update investment tracking
        if trade_type == "buy":
            self.total_invested += shares * price
        else:  # sell
            self.total_withdrawn += shares * price
        
        # Update fees
        self.total_fees_paid += fee
        
        # Update timestamps
        self.updated_at = datetime.utcnow()
        self.last_trade_at = datetime.utcnow()
        
        # Recalculate current value and PnL
        self._recalculate_values()
    
    def _recalculate_values(self):
        """Recalculate current values and PnL"""
        # This would typically use current market prices
        # For now, we'll use a simplified calculation
        
        # Calculate current value based on current market prices
        if hasattr(self, 'market') and self.market:
            current_value_a = self.outcome_a_shares * self.market.current_price_a
            current_value_b = self.outcome_b_shares * self.market.current_price_b
            self.current_value = current_value_a + current_value_b
        else:
            # Fallback calculation
            self.current_value = self.outcome_a_shares * self.entry_price_a + self.outcome_b_shares * self.entry_price_b
        
        # Calculate unrealized PnL
        self.unrealized_pnl = self.current_value - self.net_investment
        
        # Update max profit/loss
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        if self.unrealized_pnl < -self.max_loss:
            self.max_loss = abs(self.unrealized_pnl)
    
    def close_position(self, exit_prices: Dict[str, float]):
        """Close the position"""
        if self.is_closed:
            raise ValueError("Position is already closed")
        
        # Set exit prices
        self.exit_price_a = exit_prices.get("outcome_a", 0.0)
        self.exit_price_b = exit_prices.get("outcome_b", 0.0)
        
        # Calculate final value
        final_value = (
            self.outcome_a_shares * self.exit_price_a +
            self.outcome_b_shares * self.exit_price_b
        )
        
        # Calculate realized PnL
        self.realized_pnl = final_value - self.net_investment
        
        # Close the position
        self.is_closed = True
        self.closed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Clear shares
        self.outcome_a_shares = 0.0
        self.outcome_b_shares = 0.0
        self.current_value = 0.0
        self.unrealized_pnl = 0.0
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "market_id": self.market_id,
            "outcome_a_shares": self.outcome_a_shares,
            "outcome_b_shares": self.outcome_b_shares,
            "total_shares": self.total_shares,
            "total_invested": self.total_invested,
            "total_fees_paid": self.total_fees_paid,
            "total_withdrawn": self.total_withdrawn,
            "net_investment": self.net_investment,
            "current_value": self.current_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "pnl_percentage": self.pnl_percentage,
            "entry_price_a": self.entry_price_a,
            "entry_price_b": self.entry_price_b,
            "exit_price_a": self.exit_price_a,
            "exit_price_b": self.exit_price_b,
            "is_closed": self.is_closed,
            "is_hedged": self.is_hedged,
            "is_active": self.is_active,
            "is_profitable": self.is_profitable,
            "is_hedged_position": self.is_hedged_position,
            "exposure_percentage": self.exposure_percentage,
            "risk_level": self.risk_level,
            "position_type": self.position_type,
            "strategy": self.strategy,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
            "tags": self.tags,
            "notes": self.notes
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get position performance metrics"""
        duration = (self.updated_at - self.created_at).total_seconds() / 3600  # hours
        
        return {
            "total_return": self.pnl_percentage,
            "absolute_return": self.total_pnl,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "duration_hours": duration,
            "annualized_return": (self.pnl_percentage / duration) * 8760 if duration > 0 else 0,  # 8760 hours in a year
            "volatility": self._calculate_volatility(),
            "drawdown": self._calculate_drawdown()
        }
    
    def _calculate_volatility(self) -> float:
        """Calculate position volatility (simplified)"""
        # This would typically use historical price data
        # For now, return a placeholder
        return 0.0
    
    def _calculate_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        # This would typically use historical data
        # For now, return max loss as drawdown
        return self.max_loss
    
    def validate_position_data(self) -> Dict[str, Any]:
        """Validate position data integrity"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check shares
        if self.outcome_a_shares < 0 or self.outcome_b_shares < 0:
            validation_result["valid"] = False
            validation_result["errors"].append("Shares cannot be negative")
        
        # Check investment amounts
        if self.total_invested < 0 or self.total_fees_paid < 0 or self.total_withdrawn < 0:
            validation_result["valid"] = False
            validation_result["errors"].append("Investment amounts cannot be negative")
        
        # Check price ranges
        prices = [self.entry_price_a, self.entry_price_b, self.exit_price_a, self.exit_price_b]
        for price in prices:
            if price < 0 or price > 1:
                validation_result["warnings"].append(f"Price {price} is outside valid range [0, 1]")
        
        # Check win rate
        if self.win_rate < 0 or self.win_rate > 100:
            validation_result["warnings"].append("Win rate should be between 0 and 100")
        
        # Check position consistency
        if self.is_closed and self.total_shares > 0:
            validation_result["warnings"].append("Closed position should have zero shares")
        
        return validation_result
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get position risk metrics"""
        return {
            "risk_level": self.risk_level,
            "exposure_percentage": self.exposure_percentage,
            "is_hedged": self.is_hedged_position,
            "max_loss": self.max_loss,
            "sharpe_ratio": self.sharpe_ratio,
            "volatility": self._calculate_volatility(),
            "drawdown": self._calculate_drawdown(),
            "concentration_risk": self._calculate_concentration_risk()
        }
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk"""
        # Higher risk if position is concentrated in one outcome
        if self.total_shares == 0:
            return 0.0
        
        exposure = self.exposure_percentage
        concentration = max(exposure["outcome_a"], exposure["outcome_b"])
        
        # Risk increases as concentration increases
        if concentration > 80:
            return 1.0  # High risk
        elif concentration > 60:
            return 0.7  # Medium-high risk
        elif concentration > 40:
            return 0.4  # Medium risk
        else:
            return 0.1  # Low risk