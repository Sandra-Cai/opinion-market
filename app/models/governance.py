from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Boolean,
    Text,
    Enum,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class ProposalType(str, enum.Enum):
    PLATFORM_UPGRADE = "platform_upgrade"
    FEE_CHANGE = "fee_change"
    FEATURE_REQUEST = "feature_request"
    MARKET_RULE_CHANGE = "market_rule_change"
    GOVERNANCE_CHANGE = "governance_change"
    EMERGENCY_ACTION = "emergency_action"


class ProposalStatus(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    EXPIRED = "expired"


class VoteType(str, enum.Enum):
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


class GovernanceProposal(Base):
    __tablename__ = "governance_proposals"

    id = Column(Integer, primary_key=True, index=True)

    # Proposal details
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    proposal_type = Column(Enum(ProposalType), nullable=False)

    # Voting details
    status = Column(Enum(ProposalStatus), default=ProposalStatus.DRAFT)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Voting period
    voting_start = Column(DateTime, nullable=False)
    voting_end = Column(DateTime, nullable=False)
    quorum_required = Column(Float, default=0.1)  # 10% of total tokens
    majority_required = Column(Float, default=0.6)  # 60% majority

    # Voting results
    total_votes = Column(Integer, default=0)
    yes_votes = Column(Integer, default=0)
    no_votes = Column(Integer, default=0)
    abstain_votes = Column(Integer, default=0)
    total_voting_power = Column(Float, default=0.0)

    # Execution
    executed_by = Column(Integer, ForeignKey("users.id"))
    executed_at = Column(DateTime)
    execution_notes = Column(Text)

    # Additional data
    additional_data = Column(JSON, default=dict)  # Additional proposal data

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    executor = relationship("User", foreign_keys=[executed_by])
    votes = relationship("GovernanceVote", back_populates="proposal")

    @property
    def is_active(self) -> bool:
        """Check if proposal is currently active for voting"""
        now = datetime.utcnow()
        return (
            self.status == ProposalStatus.ACTIVE
            and self.voting_start <= now <= self.voting_end
        )

    @property
    def has_quorum(self) -> bool:
        """Check if proposal has reached quorum"""
        return self.total_voting_power >= self.quorum_required

    @property
    def has_majority(self) -> bool:
        """Check if proposal has majority support"""
        if self.total_votes == 0:
            return False
        return (self.yes_votes / self.total_votes) >= self.majority_required

    @property
    def voting_power_required(self) -> float:
        """Calculate voting power required for quorum"""
        # This would typically be based on total platform tokens
        return self.quorum_required * 1000000  # Example: 1M total tokens

    @property
    def time_remaining(self) -> float:
        """Time remaining for voting in seconds"""
        if not self.is_active:
            return 0
        return (self.voting_end - datetime.utcnow()).total_seconds()

    def calculate_results(self):
        """Calculate and update voting results"""
        from app.models.user import User

        # Get all votes for this proposal
        votes = self.votes

        self.total_votes = len(votes)
        self.yes_votes = len([v for v in votes if v.vote_type == VoteType.YES])
        self.no_votes = len([v for v in votes if v.vote_type == VoteType.NO])
        self.abstain_votes = len([v for v in votes if v.vote_type == VoteType.ABSTAIN])

        # Calculate total voting power
        self.total_voting_power = sum(v.voting_power for v in votes)

        # Update status based on results
        if self.voting_end <= datetime.utcnow():
            if self.has_quorum and self.has_majority:
                self.status = ProposalStatus.PASSED
            else:
                self.status = ProposalStatus.REJECTED


class GovernanceVote(Base):
    __tablename__ = "governance_votes"

    id = Column(Integer, primary_key=True, index=True)

    # Vote details
    proposal_id = Column(Integer, ForeignKey("governance_proposals.id"), nullable=False)
    voter_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    vote_type = Column(Enum(VoteType), nullable=False)

    # Voting power
    voting_power = Column(Float, nullable=False)  # Based on user's tokens/reputation
    voting_weight = Column(Float, default=1.0)  # Additional weight multiplier

    # Reasoning
    reason = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    proposal = relationship("GovernanceProposal", back_populates="votes")
    voter = relationship("User")

    class Meta:
        unique_together = ("proposal_id", "voter_id")  # One vote per user per proposal


class GovernanceToken(Base):
    """Governance token for voting power"""

    __tablename__ = "governance_tokens"

    id = Column(Integer, primary_key=True, index=True)

    # Token details
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_amount = Column(Float, default=0.0)
    locked_amount = Column(Float, default=0.0)  # Tokens locked in proposals

    # Staking
    staked_amount = Column(Float, default=0.0)
    staking_rewards = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User")

    @property
    def available_amount(self) -> float:
        """Available tokens for voting"""
        return self.token_amount - self.locked_amount

    @property
    def total_voting_power(self) -> float:
        """Total voting power including staked tokens"""
        return self.available_amount + (
            self.staked_amount * 1.5
        )  # Staked tokens have 1.5x voting power

    def lock_tokens(self, amount: float):
        """Lock tokens for voting"""
        if amount > self.available_amount:
            raise ValueError("Insufficient available tokens")
        self.locked_amount += amount

    def unlock_tokens(self, amount: float):
        """Unlock tokens after voting"""
        if amount > self.locked_amount:
            raise ValueError("Cannot unlock more than locked amount")
        self.locked_amount -= amount

    def stake_tokens(self, amount: float):
        """Stake tokens for additional voting power"""
        if amount > self.available_amount:
            raise ValueError("Insufficient available tokens")
        self.staked_amount += amount
        self.token_amount -= amount

    def unstake_tokens(self, amount: float):
        """Unstake tokens"""
        if amount > self.staked_amount:
            raise ValueError("Cannot unstake more than staked amount")
        self.staked_amount -= amount
        self.token_amount += amount
