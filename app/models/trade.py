from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Enum
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

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Trade details
    trade_type = Column(Enum(TradeType), nullable=False)
    outcome = Column(Enum(TradeOutcome), nullable=False)  # Which outcome they're trading
    amount = Column(Float, nullable=False)  # Number of shares
    price_per_share = Column(Float, nullable=False)  # Price at time of trade
    total_value = Column(Float, nullable=False)  # amount * price_per_share
    
    # Market and user
    market_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    market = relationship("Market", back_populates="trades")
    user = relationship("User", back_populates="trades")
    
    @property
    def is_buy(self) -> bool:
        return self.trade_type == TradeType.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.trade_type == TradeType.SELL
