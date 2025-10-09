from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Enum, JSON, ForeignKey, Index, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum
import hashlib
import secrets
from typing import Optional, Dict, Any
from app.core.database import Base


class TradeType(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class TradeOutcome(str, enum.Enum):
    OUTCOME_A = "outcome_a"
    OUTCOME_B = "outcome_b"


class TradeStatus(str, enum.Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PARTIALLY_FILLED = "partially_filled"


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)

    # Trade details
    trade_type = Column(Enum(TradeType), nullable=False)
    outcome = Column(
        Enum(TradeOutcome), nullable=False
    )  # Which outcome they're trading
    amount = Column(Float, nullable=False)  # Number of shares
    price_a = Column(Float, nullable=False)  # Price for outcome A at time of trade
    price_b = Column(Float, nullable=False)  # Price for outcome B at time of trade
    price_per_share = Column(Float, nullable=False)  # Price at time of trade (for compatibility)
    total_value = Column(Float, nullable=False)  # amount * price_per_share

    # Advanced trade details
    status = Column(Enum(TradeStatus), default=TradeStatus.COMPLETED)
    fee = Column(Float, default=0.0)  # Trading fee (renamed from fee_amount)
    fee_amount = Column(Float, default=0.0)  # Trading fee (for compatibility)
    price_impact = Column(Float, default=0.0)  # Price impact of this trade
    slippage = Column(Float, default=0.0)  # Slippage from expected price
    
    # Order details
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=True)  # Link to order if applicable
    fill_amount = Column(Float, nullable=True)  # Actual amount filled (for partial fills)
    fill_price = Column(Float, nullable=True)  # Actual fill price

    # Market and user
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Trade metadata
    trade_hash = Column(String, unique=True)  # Unique trade identifier
    gas_fee = Column(Float, default=0.0)  # For future blockchain integration
    additional_data = Column(JSON, default=dict)  # Additional trade metadata

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    executed_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    market = relationship("Market", back_populates="trades")
    user = relationship("User", back_populates="trades")
    order = relationship("Order", back_populates="trades", foreign_keys=[order_id])

    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint('amount > 0', name='check_positive_amount'),
        CheckConstraint('price_per_share >= 0 AND price_per_share <= 1', name='check_price_range'),
        CheckConstraint('total_value > 0', name='check_positive_total_value'),
        CheckConstraint('fee_amount >= 0', name='check_non_negative_fee'),
        CheckConstraint('price_impact >= 0', name='check_non_negative_price_impact'),
        Index('idx_trade_user', 'user_id'),
        Index('idx_trade_market', 'market_id'),
        Index('idx_trade_created_at', 'created_at'),
        Index('idx_trade_status', 'status'),
        Index('idx_trade_type', 'trade_type'),
        Index('idx_trade_outcome', 'outcome'),
    )

    @property
    def is_buy(self) -> bool:
        return self.trade_type == TradeType.BUY

    @property
    def is_sell(self) -> bool:
        return self.trade_type == TradeType.SELL

    @property
    def net_value(self) -> float:
        """Total value including fees"""
        return self.total_value + self.fee_amount

    @property
    def profit_loss(self) -> float:
        """Calculate profit/loss for this trade (if resolved)"""
        if not self.market.resolved_outcome:
            return 0.0

        if self.outcome.value == f"outcome_{self.market.resolved_outcome.lower()}":
            # Won the trade
            return self.amount - self.total_value
        else:
            # Lost the trade
            return -self.total_value

    def calculate_fee(self, fee_rate: float) -> float:
        """Calculate trading fee"""
        return self.total_value * fee_rate

    def calculate_price_impact(self, market_liquidity: float) -> float:
        """Calculate price impact of this trade"""
        if market_liquidity == 0:
            return 0.0
        return (self.total_value / market_liquidity) * 100

    def generate_trade_hash(self) -> str:
        """Generate unique trade hash"""
        if not self.trade_hash:
            data = f"{self.user_id}_{self.market_id}_{self.amount}_{self.price_per_share}_{self.created_at}_{secrets.token_hex(8)}"
            self.trade_hash = hashlib.sha256(data.encode()).hexdigest()
        return self.trade_hash

    def get_trade_summary(self) -> Dict[str, Any]:
        """Get comprehensive trade summary"""
        return {
            "id": self.id,
            "trade_hash": self.trade_hash,
            "user_id": self.user_id,
            "market_id": self.market_id,
            "trade_type": self.trade_type.value,
            "outcome": self.outcome.value,
            "amount": self.amount,
            "price_per_share": self.price_per_share,
            "total_value": self.total_value,
            "fee_amount": self.fee_amount,
            "net_value": self.net_value,
            "price_impact": self.price_impact,
            "slippage": self.slippage,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat(),
            "profit_loss": self.profit_loss,
            "gas_fee": self.gas_fee,
        }

    def is_executed(self) -> bool:
        """Check if trade is executed"""
        return self.status == TradeStatus.EXECUTED

    def is_pending(self) -> bool:
        """Check if trade is pending"""
        return self.status == TradeStatus.PENDING

    def is_cancelled(self) -> bool:
        """Check if trade is cancelled"""
        return self.status == TradeStatus.CANCELLED

    def is_failed(self) -> bool:
        """Check if trade failed"""
        return self.status == TradeStatus.FAILED

    def calculate_total_cost(self) -> float:
        """Calculate total cost including fees and gas"""
        return self.net_value + self.gas_fee

    def get_execution_time(self) -> Optional[float]:
        """Get trade execution time in seconds"""
        if self.executed_at and self.created_at:
            return (self.executed_at - self.created_at).total_seconds()
        return None

    def update_status(self, new_status: TradeStatus):
        """Update trade status with timestamp"""
        self.status = new_status
        if new_status == TradeStatus.EXECUTED:
            self.executed_at = datetime.utcnow()

    def validate_trade_data(self) -> Dict[str, Any]:
        """Validate trade data integrity"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check amount
        if self.amount <= 0:
            validation_result["valid"] = False
            validation_result["errors"].append("Trade amount must be positive")

        # Check price
        if self.price_per_share < 0 or self.price_per_share > 1:
            validation_result["valid"] = False
            validation_result["errors"].append("Price per share must be between 0 and 1")

        # Check total value consistency
        expected_total = self.amount * self.price_per_share
        if abs(self.total_value - expected_total) > 0.01:  # Allow small floating point differences
            validation_result["warnings"].append("Total value doesn't match amount * price_per_share")

        # Check fee
        if self.fee_amount < 0:
            validation_result["valid"] = False
            validation_result["errors"].append("Fee amount cannot be negative")

        return validation_result
