from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Enum, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
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
    CANCELLED = "cancelled"
    FAILED = "failed"


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)

    # Trade details
    trade_type = Column(Enum(TradeType), nullable=False)
    outcome = Column(
        Enum(TradeOutcome), nullable=False
    )  # Which outcome they're trading
    amount = Column(Float, nullable=False)  # Number of shares
    price_per_share = Column(Float, nullable=False)  # Price at time of trade
    total_value = Column(Float, nullable=False)  # amount * price_per_share

    # Advanced trade details
    status = Column(Enum(TradeStatus), default=TradeStatus.EXECUTED)
    fee_amount = Column(Float, default=0.0)  # Trading fee
    price_impact = Column(Float, default=0.0)  # Price impact of this trade
    slippage = Column(Float, default=0.0)  # Slippage from expected price

    # Market and user
    market_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)

    # Trade metadata
    trade_hash = Column(String, unique=True)  # Unique trade identifier
    gas_fee = Column(Float, default=0.0)  # For future blockchain integration
    metadata = Column(JSON, default=dict)  # Additional trade metadata

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    market = relationship("Market", back_populates="trades")
    user = relationship("User", back_populates="trades")

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
