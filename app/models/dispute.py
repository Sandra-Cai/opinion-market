from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, Enum, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base

class DisputeStatus(str, enum.Enum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"

class DisputeType(str, enum.Enum):
    INCORRECT_RESOLUTION = "incorrect_resolution"
    AMBIGUOUS_QUESTION = "ambiguous_question"
    INVALID_OUTCOME = "invalid_outcome"
    MANIPULATION = "manipulation"
    OTHER = "other"

class MarketDispute(Base):
    __tablename__ = "market_disputes"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Dispute details
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    dispute_type = Column(Enum(DisputeType), nullable=False)
    reason = Column(Text, nullable=False)
    evidence = Column(Text)  # Supporting evidence or links
    
    # Resolution details
    status = Column(Enum(DisputeStatus), default=DisputeStatus.OPEN)
    reviewed_by = Column(Integer, ForeignKey("users.id"))
    reviewed_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    # Voting system
    votes_for_dispute = Column(Integer, default=0)
    votes_against_dispute = Column(Integer, default=0)
    required_votes = Column(Integer, default=5)  # Votes needed to resolve
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    market = relationship("Market", back_populates="disputes")
    creator = relationship("User", foreign_keys=[created_by])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    votes = relationship("DisputeVote", back_populates="dispute")
    
    @property
    def is_resolved(self) -> bool:
        return self.status in [DisputeStatus.RESOLVED, DisputeStatus.DISMISSED]
    
    @property
    def total_votes(self) -> int:
        return self.votes_for_dispute + self.votes_against_dispute
    
    @property
    def consensus_reached(self) -> bool:
        return self.total_votes >= self.required_votes
    
    @property
    def dispute_wins(self) -> bool:
        return self.votes_for_dispute > self.votes_against_dispute and self.consensus_reached

class DisputeVote(Base):
    __tablename__ = "dispute_votes"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Vote details
    dispute_id = Column(Integer, ForeignKey("market_disputes.id"), nullable=False)
    voter_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    vote_for_dispute = Column(Boolean, nullable=False)  # True = support dispute, False = against
    reason = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dispute = relationship("MarketDispute", back_populates="votes")
    voter = relationship("User")
    
    class Meta:
        unique_together = ('dispute_id', 'voter_id')  # One vote per user per dispute
