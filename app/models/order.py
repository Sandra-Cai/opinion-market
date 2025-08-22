from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Enum, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base
from typing import Optional, Dict

class OrderType(str, enum.Enum):
    MARKET = "market"  # Execute immediately at current price
    LIMIT = "limit"    # Execute only at specified price or better
    STOP = "stop"      # Execute when price reaches trigger level

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
    stop_price = Column(Float)   # For stop orders
    average_fill_price = Column(Float, default=0.0)  # Average price of fills
    
    # Order status
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime)  # Order expiration
    
    # Additional metadata
    order_hash = Column(String, unique=True)  # Unique order identifier
    metadata = Column(JSON, default=dict)  # Additional order data
    
    # Relationships
    user = relationship("User")
    market = relationship("Market")
    fills = relationship("OrderFill", back_populates="order")
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
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
            total_value = self.average_fill_price * (self.filled_amount - amount) + (price * amount)
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

class OrderBook(Base):
    """Order book for a market"""
    
    def __init__(self, market_id: int):
        self.market_id = market_id
        self.buy_orders = []  # Sorted by price (highest first)
        self.sell_orders = []  # Sorted by price (lowest first)
    
    def add_order(self, order: Order):
        """Add order to the book"""
        if order.side == OrderSide.BUY:
            self._add_buy_order(order)
        else:
            self._add_sell_order(order)
    
    def _add_buy_order(self, order: Order):
        """Add buy order to the book (sorted by price descending)"""
        # Insert in correct position to maintain price order
        for i, existing_order in enumerate(self.buy_orders):
            if order.limit_price > existing_order.limit_price:
                self.buy_orders.insert(i, order)
                return
        
        self.buy_orders.append(order)
    
    def _add_sell_order(self, order: Order):
        """Add sell order to the book (sorted by price ascending)"""
        # Insert in correct position to maintain price order
        for i, existing_order in enumerate(self.sell_orders):
            if order.limit_price < existing_order.limit_price:
                self.sell_orders.insert(i, order)
                return
        
        self.sell_orders.append(order)
    
    def get_best_bid(self) -> Optional[Order]:
        """Get best bid (highest buy price)"""
        return self.buy_orders[0] if self.buy_orders else None
    
    def get_best_ask(self) -> Optional[Order]:
        """Get best ask (lowest sell price)"""
        return self.sell_orders[0] if self.sell_orders else None
    
    def get_spread(self) -> float:
        """Get bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.limit_price - best_bid.limit_price
        
        return 0.0
    
    def get_depth(self, levels: int = 10) -> Dict:
        """Get order book depth"""
        bids = []
        asks = []
        
        for i, order in enumerate(self.buy_orders[:levels]):
            bids.append({
                "price": order.limit_price,
                "amount": order.remaining_amount,
                "total": order.limit_price * order.remaining_amount
            })
        
        for i, order in enumerate(self.sell_orders[:levels]):
            asks.append({
                "price": order.limit_price,
                "amount": order.remaining_amount,
                "total": order.limit_price * order.remaining_amount
            })
        
        return {
            "bids": bids,
            "asks": asks,
            "spread": self.get_spread()
        }
