from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, Enum, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base

class MarketStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    RESOLVED = "resolved"
    DISPUTED = "disputed"
    SUSPENDED = "suspended"

class MarketCategory(str, enum.Enum):
    POLITICS = "politics"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SCIENCE = "science"
    CRYPTO = "crypto"
    WEATHER = "weather"
    OTHER = "other"

class Market(Base):
    __tablename__ = "markets"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    category = Column(Enum(MarketCategory), default=MarketCategory.OTHER)
    
    # Market details
    question = Column(String, nullable=False)
    outcome_a = Column(String, nullable=False)  # "Yes" or specific outcome
    outcome_b = Column(String, nullable=False)  # "No" or alternative outcome
    
    # Advanced pricing and liquidity (like Polymarket)
    current_price_a = Column(Float, default=0.5)  # Price for outcome A (0-1)
    current_price_b = Column(Float, default=0.5)  # Price for outcome B (0-1)
    total_liquidity = Column(Float, default=1000.0)
    liquidity_pool_a = Column(Float, default=500.0)  # Liquidity for outcome A
    liquidity_pool_b = Column(Float, default=500.0)  # Liquidity for outcome B
    
    # Market mechanics
    fee_rate = Column(Float, default=0.02)  # 2% trading fee
    min_trade_amount = Column(Float, default=1.0)
    max_trade_amount = Column(Float, default=10000.0)
    
    # Market status
    status = Column(Enum(MarketStatus), default=MarketStatus.OPEN)
    resolved_outcome = Column(String)  # Which outcome won
    resolution_source = Column(String)  # Source of resolution
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    closes_at = Column(DateTime, nullable=False)
    resolved_at = Column(DateTime)
    
    # Creator
    creator_id = Column(Integer, nullable=False)
    
    # Additional metadata
    tags = Column(JSON, default=list)  # Market tags
    image_url = Column(String)  # Market image
    volume_24h = Column(Float, default=0.0)  # 24h trading volume
    volume_total = Column(Float, default=0.0)  # Total trading volume
    
    # Relationships
    creator = relationship("User", back_populates="markets_created")
    trades = relationship("Trade", back_populates="market")
    votes = relationship("Vote", back_populates="market")
    
    @property
    def is_active(self) -> bool:
        return self.status == MarketStatus.OPEN and datetime.utcnow() < self.closes_at
    
    @property
    def total_volume(self) -> float:
        return sum(trade.total_value for trade in self.trades)
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def price_impact(self) -> float:
        """Calculate price impact for large trades"""
        total_liquidity = self.liquidity_pool_a + self.liquidity_pool_b
        return 0.1 / total_liquidity if total_liquidity > 0 else 0.1
    
    def update_prices(self, trade_amount: float, outcome: str, trade_type: str):
        """Update market prices based on trade (like Polymarket's AMM)"""
        if outcome == "outcome_a":
            if trade_type == "buy":
                # Buy outcome A - price goes up
                self.liquidity_pool_a += trade_amount
                self.liquidity_pool_b -= trade_amount
                # Calculate new price using constant product formula
                total_liquidity = self.liquidity_pool_a + self.liquidity_pool_b
                if total_liquidity > 0:
                    self.current_price_a = self.liquidity_pool_a / total_liquidity
                    self.current_price_b = self.liquidity_pool_b / total_liquidity
            else:
                # Sell outcome A - price goes down
                self.liquidity_pool_a -= trade_amount
                self.liquidity_pool_b += trade_amount
                total_liquidity = self.liquidity_pool_a + self.liquidity_pool_b
                if total_liquidity > 0:
                    self.current_price_a = self.liquidity_pool_a / total_liquidity
                    self.current_price_b = self.liquidity_pool_b / total_liquidity
        else:  # outcome_b
            if trade_type == "buy":
                # Buy outcome B - price goes up
                self.liquidity_pool_b += trade_amount
                self.liquidity_pool_a -= trade_amount
                total_liquidity = self.liquidity_pool_a + self.liquidity_pool_b
                if total_liquidity > 0:
                    self.current_price_a = self.liquidity_pool_a / total_liquidity
                    self.current_price_b = self.liquidity_pool_b / total_liquidity
            else:
                # Sell outcome B - price goes down
                self.liquidity_pool_b -= trade_amount
                self.liquidity_pool_a += trade_amount
                total_liquidity = self.liquidity_pool_a + self.liquidity_pool_b
                if total_liquidity > 0:
                    self.current_price_a = self.liquidity_pool_a / total_liquidity
                    self.current_price_b = self.liquidity_pool_b / total_liquidity
