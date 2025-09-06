from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.governance import ProposalType, ProposalStatus, VoteType


class GovernanceProposalBase(BaseModel):
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=20, max_length=2000)
    proposal_type: ProposalType
    voting_start: datetime
    voting_end: datetime
    quorum_required: float = Field(0.1, ge=0.01, le=1.0)
    majority_required: float = Field(0.6, ge=0.5, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class GovernanceProposalCreate(GovernanceProposalBase):
    pass


class GovernanceProposalUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=5, max_length=200)
    description: Optional[str] = Field(None, min_length=20, max_length=2000)
    voting_end: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class GovernanceVoteCreate(BaseModel):
    vote_type: VoteType
    voting_power: float = Field(..., gt=0)
    reason: Optional[str] = Field(None, max_length=500)


class GovernanceVoteResponse(BaseModel):
    id: int
    proposal_id: int
    voter_id: int
    vote_type: VoteType
    voting_power: float
    voting_weight: float
    reason: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class GovernanceProposalResponse(GovernanceProposalBase):
    id: int
    created_by: int
    status: ProposalStatus
    total_votes: int
    yes_votes: int
    no_votes: int
    abstain_votes: int
    total_voting_power: float
    voting_power_required: float
    time_remaining: float
    is_active: bool
    has_quorum: bool
    has_majority: bool
    executed_by: Optional[int] = None
    executed_at: Optional[datetime] = None
    execution_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class GovernanceProposalListResponse(BaseModel):
    proposals: List[GovernanceProposalResponse]
    total: int
    page: int
    per_page: int


class GovernanceTokenResponse(BaseModel):
    id: int
    user_id: int
    token_amount: float
    locked_amount: float
    available_amount: float
    staked_amount: float
    staking_rewards: float
    total_voting_power: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class GovernanceTokenStakeRequest(BaseModel):
    amount: float = Field(..., gt=0)


class GovernanceTokenUnstakeRequest(BaseModel):
    amount: float = Field(..., gt=0)


class GovernanceStatsResponse(BaseModel):
    total_proposals: int
    active_proposals: int
    passed_proposals: int
    rejected_proposals: int
    total_votes_cast: int
    total_voting_power: float
    average_participation_rate: float
    top_voters: List[Dict[str, Any]]
