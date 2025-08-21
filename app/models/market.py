from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base

class MarketStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    RESOLVED = "resolved"
    DISPUTED = "disputed"

class MarketCategory(str, enum.Enum):
    POLITICS = "politics"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SCIENCE = "science"
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
    
    # Pricing and liquidity
    current_price_a = Column(Float, default=0.5)  # Price for outcome A (0-1)
    current_price_b = Column(Float, default=0.5)  # Price for outcome B (0-1)
    total_liquidity = Column(Float, default=1000.0)
    
    # Market status
    status = Column(Enum(MarketStatus), default=MarketStatus.OPEN)
    resolved_outcome = Column(String)  # Which outcome won
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    closes_at = Column(DateTime, nullable=False)
    resolved_at = Column(DateTime)
    
    # Creator
    creator_id = Column(Integer, nullable=False)
    
    # Relationships
    creator = relationship("User", back_populates="markets_created")
    trades = relationship("Trade", back_populates="market")
    votes = relationship("Vote", back_populates="market")
    
    @property
    def is_active(self) -> bool:
        return self.status == MarketStatus.OPEN and datetime.utcnow() < self.closes_at
    
    @property
    def total_volume(self) -> float:
        return sum(trade.amount for trade in self.trades)
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
