"""
Enhanced Order model for Opinion Market
Provides comprehensive order management with advanced order types
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Enum, JSON, ForeignKey, Index, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import enum
import hashlib
import secrets
from typing import Optional, Dict, Any
from app.core.database import Base


class OrderType(str, enum.Enum):
    MARKET = "market"           # Execute immediately at current price
    LIMIT = "limit"            # Execute only at specified price or better
    STOP = "stop"              # Execute when price reaches stop level
    STOP_LIMIT = "stop_limit"  # Stop order with limit price
    TAKE_PROFIT = "take_profit" # Execute when profit target is reached
    TRAILING_STOP = "trailing_stop" # Dynamic stop that follows price
    BRACKET = "bracket"        # Entry with automatic stop-loss and take-profit


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, enum.Enum):
    PENDING = "pending"           # Order is waiting to be processed
    PARTIALLY_FILLED = "partially_filled"  # Order partially executed
    FILLED = "filled"            # Order completely executed
    CANCELLED = "cancelled"      # Order cancelled by user
    REJECTED = "rejected"        # Order rejected by system
    EXPIRED = "expired"          # Order expired
    FAILED = "failed"            # Order failed to execute


class OrderTimeInForce(str, enum.Enum):
    GTC = "gtc"                  # Good Till Cancelled
    IOC = "ioc"                  # Immediate or Cancel
    FOK = "fok"                  # Fill or Kill
    DAY = "day"                  # Good for Day
    GTD = "gtd"                  # Good Till Date


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    
    # Order identification
    order_id = Column(String, unique=True, nullable=False, index=True)  # External order ID
    client_order_id = Column(String, nullable=True)  # Client-provided order ID
    
    # Order details
    order_type = Column(Enum(OrderType), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    outcome = Column(String, nullable=False)  # outcome_a or outcome_b
    
    # Quantity and price
    quantity = Column(Float, nullable=False)  # Total quantity to trade
    filled_quantity = Column(Float, default=0.0)  # Quantity already filled
    remaining_quantity = Column(Float, nullable=False)  # Quantity remaining
    
    # Price information
    limit_price = Column(Float, nullable=True)  # Limit price for limit orders
    stop_price = Column(Float, nullable=True)   # Stop price for stop orders
    average_fill_price = Column(Float, default=0.0)  # Average price of fills
    
    # Order management
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING)
    time_in_force = Column(Enum(OrderTimeInForce), default=OrderTimeInForce.GTC)
    expires_at = Column(DateTime, nullable=True)  # Expiration time
    
    # Advanced order features
    is_post_only = Column(Boolean, default=False)  # Post-only order (maker only)
    reduce_only = Column(Boolean, default=False)   # Reduce position only
    iceberg = Column(Boolean, default=False)       # Iceberg order
    iceberg_visible_size = Column(Float, nullable=True)  # Visible size for iceberg
    
    # Trailing stop specific
    trailing_distance = Column(Float, nullable=True)  # Distance for trailing stop
    trailing_percentage = Column(Float, nullable=True)  # Percentage for trailing stop
    
    # Bracket order specific
    parent_order_id = Column(Integer, ForeignKey("orders.id"), nullable=True)
    take_profit_price = Column(Float, nullable=True)
    stop_loss_price = Column(Float, nullable=True)
    
    # Market and user
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Fees and costs
    total_fees = Column(Float, default=0.0)
    estimated_fees = Column(Float, default=0.0)
    
    # Metadata
    order_hash = Column(String, unique=True)  # Unique order hash
    additional_data = Column(JSON, default=dict)  # Additional order metadata
    notes = Column(String, nullable=True)  # User notes
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    filled_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    # Relationships
    market = relationship("Market", back_populates="orders")
    user = relationship("User", back_populates="orders")
    trades = relationship("Trade", back_populates="order", foreign_keys="Trade.order_id")
    parent_order = relationship("Order", remote_side=[id], back_populates="child_orders")
    child_orders = relationship("Order", back_populates="parent_order")
    
    # Table constraints and indexes
    __table_args__ = (
        CheckConstraint('quantity > 0', name='check_positive_quantity'),
        CheckConstraint('filled_quantity >= 0', name='check_non_negative_filled'),
        CheckConstraint('remaining_quantity >= 0', name='check_non_negative_remaining'),
        CheckConstraint('filled_quantity + remaining_quantity = quantity', name='check_quantity_consistency'),
        CheckConstraint('limit_price IS NULL OR (limit_price >= 0 AND limit_price <= 1)', name='check_limit_price_range'),
        CheckConstraint('stop_price IS NULL OR (stop_price >= 0 AND stop_price <= 1)', name='check_stop_price_range'),
        CheckConstraint('average_fill_price >= 0', name='check_non_negative_avg_price'),
        CheckConstraint('total_fees >= 0', name='check_non_negative_fees'),
        CheckConstraint('estimated_fees >= 0', name='check_non_negative_estimated_fees'),
        CheckConstraint('trailing_distance IS NULL OR trailing_distance > 0', name='check_positive_trailing_distance'),
        CheckConstraint('trailing_percentage IS NULL OR (trailing_percentage > 0 AND trailing_percentage <= 100)', name='check_trailing_percentage_range'),
        Index('idx_order_user', 'user_id'),
        Index('idx_order_market', 'market_id'),
        Index('idx_order_status', 'status'),
        Index('idx_order_type', 'order_type'),
        Index('idx_order_side', 'side'),
        Index('idx_order_created_at', 'created_at'),
        Index('idx_order_expires_at', 'expires_at'),
        Index('idx_order_parent', 'parent_order_id'),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.order_id:
            self.order_id = self.generate_order_id()
        if not self.order_hash:
            self.order_hash = self.generate_order_hash()
        if not self.remaining_quantity:
            self.remaining_quantity = self.quantity
    
    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_cancelled(self) -> bool:
        return self.status == OrderStatus.CANCELLED
    
    @property
    def is_expired(self) -> bool:
        return self.status == OrderStatus.EXPIRED or (
            self.expires_at and datetime.utcnow() > self.expires_at
        )
    
    @property
    def fill_percentage(self) -> float:
        """Percentage of order that has been filled"""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100
    
    @property
    def total_value(self) -> float:
        """Total value of the order"""
        return self.quantity * (self.limit_price or 0.5)  # Use limit price or default
    
    @property
    def filled_value(self) -> float:
        """Value of filled portion"""
        return self.filled_quantity * self.average_fill_price
    
    @property
    def remaining_value(self) -> float:
        """Value of remaining portion"""
        return self.remaining_quantity * (self.limit_price or 0.5)
    
    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        random_part = secrets.token_hex(4)
        return f"ORD_{timestamp}_{random_part}"
    
    def generate_order_hash(self) -> str:
        """Generate unique order hash"""
        data = f"{self.user_id}_{self.market_id}_{self.quantity}_{self.limit_price}_{self.created_at}_{secrets.token_hex(8)}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def update_fill(self, fill_quantity: float, fill_price: float, fee: float = 0.0):
        """Update order with new fill"""
        if fill_quantity > self.remaining_quantity:
            raise ValueError("Fill quantity exceeds remaining quantity")
        
        # Update quantities
        self.filled_quantity += fill_quantity
        self.remaining_quantity -= fill_quantity
        
        # Update average fill price
        if self.filled_quantity > 0:
            total_filled_value = (self.filled_quantity - fill_quantity) * self.average_fill_price
            new_fill_value = fill_quantity * fill_price
            self.average_fill_price = (total_filled_value + new_fill_value) / self.filled_quantity
        
        # Update fees
        self.total_fees += fee
        
        # Update status
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.utcnow()
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = datetime.utcnow()
    
    def cancel(self, reason: str = "User cancelled"):
        """Cancel the order"""
        if not self.is_active:
            raise ValueError("Cannot cancel inactive order")
        
        self.status = OrderStatus.CANCELLED
        self.cancelled_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Add cancellation reason to metadata
        if not self.additional_data:
            self.additional_data = {}
        self.additional_data["cancellation_reason"] = reason
    
    def reject(self, reason: str):
        """Reject the order"""
        self.status = OrderStatus.REJECTED
        self.updated_at = datetime.utcnow()
        
        # Add rejection reason to metadata
        if not self.additional_data:
            self.additional_data = {}
        self.additional_data["rejection_reason"] = reason
    
    def expire(self):
        """Mark order as expired"""
        if not self.is_active:
            raise ValueError("Cannot expire inactive order")
        
        self.status = OrderStatus.EXPIRED
        self.updated_at = datetime.utcnow()
    
    def calculate_estimated_fees(self, fee_rate: float) -> float:
        """Calculate estimated fees for the order"""
        return self.quantity * (self.limit_price or 0.5) * fee_rate
    
    def get_execution_time(self) -> Optional[float]:
        """Get order execution time in seconds"""
        if self.filled_at and self.created_at:
            return (self.filled_at - self.created_at).total_seconds()
        return None
    
    def get_order_summary(self) -> Dict[str, Any]:
        """Get comprehensive order summary"""
        return {
            "id": self.id,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "order_type": self.order_type.value,
            "side": self.side.value,
            "outcome": self.outcome,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "fill_percentage": self.fill_percentage,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "average_fill_price": self.average_fill_price,
            "status": self.status.value,
            "time_in_force": self.time_in_force.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "total_fees": self.total_fees,
            "estimated_fees": self.estimated_fees,
            "total_value": self.total_value,
            "filled_value": self.filled_value,
            "remaining_value": self.remaining_value,
            "is_active": self.is_active,
            "is_filled": self.is_filled,
            "is_cancelled": self.is_cancelled,
            "is_expired": self.is_expired,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "execution_time": self.get_execution_time(),
            "market_id": self.market_id,
            "user_id": self.user_id,
            "parent_order_id": self.parent_order_id,
            "additional_data": self.additional_data
        }
    
    def validate_order_data(self) -> Dict[str, Any]:
        """Validate order data integrity"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check quantity
        if self.quantity <= 0:
            validation_result["valid"] = False
            validation_result["errors"].append("Order quantity must be positive")
        
        # Check price constraints based on order type
        if self.order_type == OrderType.LIMIT and (self.limit_price is None or self.limit_price <= 0):
            validation_result["valid"] = False
            validation_result["errors"].append("Limit orders must have a valid limit price")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and (self.stop_price is None or self.stop_price <= 0):
            validation_result["valid"] = False
            validation_result["errors"].append("Stop orders must have a valid stop price")
        
        # Check trailing stop constraints
        if self.order_type == OrderType.TRAILING_STOP:
            if not self.trailing_distance and not self.trailing_percentage:
                validation_result["valid"] = False
                validation_result["errors"].append("Trailing stop orders must have trailing distance or percentage")
        
        # Check expiration
        if self.expires_at and self.expires_at <= datetime.utcnow():
            validation_result["warnings"].append("Order expiration time is in the past")
        
        # Check bracket order constraints
        if self.order_type == OrderType.BRACKET:
            if not self.take_profit_price and not self.stop_loss_price:
                validation_result["warnings"].append("Bracket orders should have take profit or stop loss prices")
        
        return validation_result
    
    def can_be_modified(self) -> bool:
        """Check if order can be modified"""
        return self.is_active and not self.is_expired
    
    def can_be_cancelled(self) -> bool:
        """Check if order can be cancelled"""
        return self.is_active and not self.is_expired
    
    def get_priority_score(self) -> float:
        """Calculate order priority score for matching"""
        # Higher priority for:
        # - Market orders
        # - Better prices (for limit orders)
        # - Earlier creation time
        # - Larger quantities
        
        base_score = 1000.0
        
        # Order type priority
        if self.order_type == OrderType.MARKET:
            base_score += 1000
        elif self.order_type == OrderType.LIMIT:
            base_score += 500
        
        # Price priority (for limit orders)
        if self.limit_price:
            if self.is_buy:
                base_score += self.limit_price * 100  # Higher price = higher priority
            else:
                base_score += (1 - self.limit_price) * 100  # Lower price = higher priority
        
        # Time priority (earlier orders get higher priority)
        time_diff = (datetime.utcnow() - self.created_at).total_seconds()
        base_score += max(0, 1000 - time_diff / 60)  # Decrease by 1 point per minute
        
        # Size priority (larger orders get slightly higher priority)
        base_score += min(self.quantity / 100, 100)  # Cap at 100 points
        
        return base_score