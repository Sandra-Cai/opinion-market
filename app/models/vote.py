from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Enum, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class VoteOutcome(str, enum.Enum):
    OUTCOME_A = "outcome_a"
    OUTCOME_B = "outcome_b"


class Vote(Base):
    __tablename__ = "votes"

    id = Column(Integer, primary_key=True, index=True)

    # Vote details
    outcome = Column(Enum(VoteOutcome), nullable=False)  # Which outcome they voted for
    confidence = Column(Float, default=1.0)  # Confidence level (0-1)

    # Market and user
    market_id = Column(Integer, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    market = relationship("Market", back_populates="votes")
    user = relationship("User", back_populates="votes")
