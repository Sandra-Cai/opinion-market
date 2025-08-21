from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    bio = Column(Text)
    avatar_url = Column(String)
    
    # Trading stats
    total_trades = Column(Integer, default=0)
    successful_trades = Column(Integer, default=0)
    total_profit = Column(Float, default=0.0)
    reputation_score = Column(Float, default=100.0)
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    markets_created = relationship("Market", back_populates="creator")
    trades = relationship("Trade", back_populates="user")
    votes = relationship("Vote", back_populates="user")
    
    @property
    def success_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.successful_trades / self.total_trades) * 100
