from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Enum, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class PositionType(str, enum.Enum):
    LONG = "long"  # Betting outcome will happen
    SHORT = "short"  # Betting outcome won't happen


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)

    # Position details
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    outcome = Column(String, nullable=False)  # Which outcome they hold

    # Position amounts
    shares_owned = Column(Float, default=0.0)  # Number of shares owned
    average_price = Column(Float, default=0.0)  # Average purchase price
    total_invested = Column(Float, default=0.0)  # Total amount invested

    # Current value
    current_price = Column(Float, default=0.0)  # Current market price
    current_value = Column(Float, default=0.0)  # Current position value

    # Profit/Loss tracking
    unrealized_pnl = Column(Float, default=0.0)  # Unrealized profit/loss
    realized_pnl = Column(Float, default=0.0)  # Realized profit/loss
    total_pnl = Column(Float, default=0.0)  # Total profit/loss

    # Position metadata
    is_active = Column(Boolean, default=True)  # Active position
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User")
    market = relationship("Market")

    @property
    def profit_loss_percentage(self) -> float:
        """Calculate profit/loss percentage"""
        if self.total_invested == 0:
            return 0.0
        return (self.total_pnl / self.total_invested) * 100

    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable"""
        return self.total_pnl > 0

    def update_position(self, shares_change: float, price: float, trade_value: float):
        """Update position when new trade occurs"""
        if shares_change > 0:  # Buying
            # Calculate new average price
            total_shares = self.shares_owned + shares_change
            total_cost = self.total_invested + trade_value
            self.average_price = total_cost / total_shares if total_shares > 0 else 0
            self.shares_owned = total_shares
            self.total_invested = total_cost
        else:  # Selling
            shares_sold = abs(shares_change)
            if shares_sold >= self.shares_owned:
                # Selling entire position
                realized_pnl = (price - self.average_price) * self.shares_owned
                self.realized_pnl += realized_pnl
                self.shares_owned = 0
                self.total_invested = 0
                self.average_price = 0
            else:
                # Partial sale
                realized_pnl = (price - self.average_price) * shares_sold
                self.realized_pnl += realized_pnl
                self.shares_owned -= shares_sold
                self.total_invested -= self.average_price * shares_sold

        # Update current values
        self.current_price = price
        self.current_value = self.shares_owned * price
        self.unrealized_pnl = self.current_value - self.total_invested
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        self.last_updated = datetime.utcnow()

    def close_position(self, close_price: float):
        """Close the entire position"""
        if self.shares_owned > 0:
            close_value = self.shares_owned * close_price
            realized_pnl = close_value - self.total_invested
            self.realized_pnl += realized_pnl
            self.unrealized_pnl = 0
            self.total_pnl = self.realized_pnl
            self.shares_owned = 0
            self.total_invested = 0
            self.current_value = 0
            self.is_active = False
            self.last_updated = datetime.utcnow()
