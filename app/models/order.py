from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Boolean,
    Enum,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base
from typing import Optional, Dict


class OrderType(str, enum.Enum):
    MARKET = "market"  # Execute immediately at current price
    LIMIT = "limit"  # Execute only at specified price or better
    STOP = "stop"  # Execute when price reaches trigger level


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)

    # Order details
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    order_type = Column(Enum(OrderType), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    outcome = Column(String, nullable=False)  # Which outcome to trade

    # Order amounts
    original_amount = Column(Float, nullable=False)  # Original order size
    remaining_amount = Column(Float, nullable=False)  # Remaining unfilled amount
    filled_amount = Column(Float, default=0.0)  # Amount already filled

    # Price information
    limit_price = Column(Float)  # For limit orders
    stop_price = Column(Float)  # For stop orders
    average_fill_price = Column(Float, default=0.0)  # Average price of fills

    # Order status
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime)  # Order expiration

    # Additional metadata
    order_hash = Column(String, unique=True)  # Unique order identifier
    additional_data = Column(JSON, default=dict)  # Additional order data

    # Relationships
    user = relationship("User")
    market = relationship("Market")
    fills = relationship("OrderFill", back_populates="order")

    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        if self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        ]:
            return False

        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False

        return True

    @property
    def total_value(self) -> float:
        """Calculate total order value"""
        if self.order_type == OrderType.MARKET:
            # For market orders, use current market price
            return self.original_amount * self._get_current_price()
        else:
            # For limit orders, use limit price
            return self.original_amount * (self.limit_price or 0)

    def _get_current_price(self) -> float:
        """Get current market price for the outcome"""
        if self.outcome == "outcome_a":
            return self.market.current_price_a
        else:
            return self.market.current_price_b

    def can_fill(self, price: float, amount: float) -> bool:
        """Check if order can be filled at given price and amount"""
        if not self.is_active or self.remaining_amount < amount:
            return False

        if self.order_type == OrderType.LIMIT:
            if self.side == OrderSide.BUY:
                return price <= self.limit_price  # Buy at limit price or lower
            else:
                return price >= self.limit_price  # Sell at limit price or higher

        return True

    def fill(self, price: float, amount: float):
        """Fill part of the order"""
        if amount > self.remaining_amount:
            raise ValueError("Fill amount exceeds remaining amount")

        # Update fill amounts
        self.filled_amount += amount
        self.remaining_amount -= amount

        # Update average fill price
        if self.filled_amount > 0:
            total_value = self.average_fill_price * (self.filled_amount - amount) + (
                price * amount
            )
            self.average_fill_price = total_value / self.filled_amount

        # Update status
        if self.remaining_amount == 0:
            self.status = OrderStatus.FILLED
        elif self.filled_amount > 0:
            self.status = OrderStatus.PARTIAL

        self.updated_at = datetime.utcnow()

    def cancel(self):
        """Cancel the order"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            raise ValueError("Cannot cancel filled or already cancelled order")

        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.utcnow()


class OrderFill(Base):
    __tablename__ = "order_fills"

    id = Column(Integer, primary_key=True, index=True)

    # Fill details
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False)

    # Fill amounts
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    order = relationship("Order", back_populates="fills")
    trade = relationship("Trade")


# OrderBook class moved to services/order_management_system.py as it's business logic, not a database model
