from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Boolean,
    Text,
    Enum,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class MarketInstrument(str, enum.Enum):
    SPOT = "spot"  # Regular prediction market
    FUTURES = "futures"  # Futures contract
    OPTIONS = "options"  # Options contract
    CONDITIONAL = "conditional"  # Conditional market
    SPREAD = "spread"  # Spread betting


class OptionType(str, enum.Enum):
    CALL = "call"  # Call option
    PUT = "put"  # Put option


class FuturesContract(Base):
    __tablename__ = "futures_contracts"

    id = Column(Integer, primary_key=True, index=True)

    # Contract details
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    contract_size = Column(Float, nullable=False)  # Size of one contract
    tick_size = Column(Float, default=0.01)  # Minimum price movement
    margin_requirement = Column(Float, default=0.1)  # 10% margin requirement

    # Settlement
    settlement_date = Column(DateTime, nullable=False)
    settlement_price = Column(Float)  # Final settlement price
    cash_settlement = Column(Boolean, default=True)  # Cash vs physical settlement

    # Risk management
    max_position_size = Column(Float)  # Maximum position size per user
    daily_price_limit = Column(Float)  # Daily price movement limit

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    market = relationship("Market")
    positions = relationship("FuturesPosition", back_populates="contract")

    @property
    def is_settled(self) -> bool:
        """Check if contract is settled"""
        return self.settlement_date <= datetime.utcnow()

    @property
    def days_to_settlement(self) -> int:
        """Days remaining until settlement"""
        return max(0, (self.settlement_date - datetime.utcnow()).days)

    def calculate_margin_requirement(
        self, position_size: float, current_price: float
    ) -> float:
        """Calculate margin requirement for position"""
        return position_size * current_price * self.margin_requirement


class FuturesPosition(Base):
    __tablename__ = "futures_positions"

    id = Column(Integer, primary_key=True, index=True)

    # Position details
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    contract_id = Column(Integer, ForeignKey("futures_contracts.id"), nullable=False)

    # Position amounts
    long_contracts = Column(Float, default=0.0)  # Long position size
    short_contracts = Column(Float, default=0.0)  # Short position size
    average_entry_price = Column(Float, default=0.0)

    # Margin and P&L
    margin_used = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)

    # Risk management
    liquidation_price = Column(Float)  # Price at which position gets liquidated

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User")
    contract = relationship("FuturesContract", back_populates="positions")

    @property
    def net_position(self) -> float:
        """Net position (long - short)"""
        return self.long_contracts - self.short_contracts

    @property
    def position_value(self) -> float:
        """Current position value"""
        return abs(self.net_position) * self.contract.contract_size

    @property
    def is_liquidated(self) -> bool:
        """Check if position is liquidated"""
        if not self.liquidation_price:
            return False
        # This would need current market price to determine
        return False  # Placeholder

    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        if self.net_position > 0:  # Long position
            self.unrealized_pnl = (
                current_price - self.average_entry_price
            ) * self.net_position
        elif self.net_position < 0:  # Short position
            self.unrealized_pnl = (self.average_entry_price - current_price) * abs(
                self.net_position
            )
        else:
            self.unrealized_pnl = 0.0


class OptionsContract(Base):
    __tablename__ = "options_contracts"

    id = Column(Integer, primary_key=True, index=True)

    # Contract details
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    option_type = Column(Enum(OptionType), nullable=False)
    strike_price = Column(Float, nullable=False)
    expiration_date = Column(DateTime, nullable=False)

    # Contract specifications
    contract_size = Column(Float, default=1.0)
    premium = Column(Float, default=0.0)  # Current option premium

    # Greeks (for options pricing)
    delta = Column(Float, default=0.0)
    gamma = Column(Float, default=0.0)
    theta = Column(Float, default=0.0)
    vega = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    market = relationship("Market")
    positions = relationship("OptionsPosition", back_populates="contract")

    @property
    def is_expired(self) -> bool:
        """Check if option is expired"""
        return self.expiration_date <= datetime.utcnow()

    @property
    def is_in_the_money(self) -> bool:
        """Check if option is in the money"""
        current_price = self.market.current_price_a
        if self.option_type == OptionType.CALL:
            return current_price > self.strike_price
        else:  # PUT
            return current_price < self.strike_price

    @property
    def intrinsic_value(self) -> float:
        """Calculate intrinsic value"""
        current_price = self.market.current_price_a
        if self.option_type == OptionType.CALL:
            return max(0, current_price - self.strike_price)
        else:  # PUT
            return max(0, self.strike_price - current_price)

    @property
    def time_value(self) -> float:
        """Calculate time value"""
        return self.premium - self.intrinsic_value


class OptionsPosition(Base):
    __tablename__ = "options_positions"

    id = Column(Integer, primary_key=True, index=True)

    # Position details
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    contract_id = Column(Integer, ForeignKey("options_contracts.id"), nullable=False)

    # Position amounts
    long_contracts = Column(Float, default=0.0)  # Long position size
    short_contracts = Column(Float, default=0.0)  # Short position size
    average_entry_price = Column(Float, default=0.0)

    # P&L
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User")
    contract = relationship("OptionsContract", back_populates="positions")

    @property
    def net_position(self) -> float:
        """Net position (long - short)"""
        return self.long_contracts - self.short_contracts

    def update_pnl(self, current_premium: float):
        """Update unrealized P&L"""
        if self.net_position > 0:  # Long position
            self.unrealized_pnl = (
                current_premium - self.average_entry_price
            ) * self.net_position
        elif self.net_position < 0:  # Short position
            self.unrealized_pnl = (self.average_entry_price - current_premium) * abs(
                self.net_position
            )
        else:
            self.unrealized_pnl = 0.0


class ConditionalMarket(Base):
    __tablename__ = "conditional_markets"

    id = Column(Integer, primary_key=True, index=True)

    # Market details
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    condition_description = Column(Text, nullable=False)

    # Conditional logic
    trigger_condition = Column(JSON, nullable=False)  # Condition that must be met
    trigger_market_id = Column(
        Integer, ForeignKey("markets.id")
    )  # Market that triggers this one

    # Activation
    is_active = Column(Boolean, default=False)
    activated_at = Column(DateTime)
    activated_by = Column(Integer, ForeignKey("users.id"))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    market = relationship("Market", foreign_keys=[market_id])
    trigger_market = relationship("Market", foreign_keys=[trigger_market_id])
    activator = relationship("User")

    def check_trigger_condition(self) -> bool:
        """Check if trigger condition is met"""
        if not self.trigger_market:
            return False

        # Example: trigger when price reaches certain level
        if "price_threshold" in self.trigger_condition:
            threshold = self.trigger_condition["price_threshold"]
            current_price = self.trigger_market.current_price_a
            return current_price >= threshold

        return False

    def activate(self, user_id: int):
        """Activate the conditional market"""
        if not self.is_active and self.check_trigger_condition():
            self.is_active = True
            self.activated_at = datetime.utcnow()
            self.activated_by = user_id


class SpreadMarket(Base):
    __tablename__ = "spread_markets"

    id = Column(Integer, primary_key=True, index=True)

    # Market details
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)

    # Spread configuration
    spread_type = Column(String, nullable=False)  # "binary", "range", "index"
    min_value = Column(Float, nullable=False)
    max_value = Column(Float, nullable=False)
    tick_size = Column(Float, default=0.1)

    # Spread outcomes
    outcomes = Column(JSON, default=list)  # List of possible outcomes

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    market = relationship("Market")

    def generate_outcomes(self):
        """Generate spread outcomes based on configuration"""
        outcomes = []
        current = self.min_value

        while current <= self.max_value:
            outcomes.append(
                {"value": current, "label": f"{current:.1f}", "probability": 0.0}
            )
            current += self.tick_size

        self.outcomes = outcomes

    def get_outcome_probability(self, value: float) -> float:
        """Get probability for a specific outcome value"""
        for outcome in self.outcomes:
            if outcome["value"] == value:
                return outcome["probability"]
        return 0.0
