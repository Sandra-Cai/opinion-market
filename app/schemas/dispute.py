from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from app.models.dispute import DisputeStatus, DisputeType


class DisputeBase(BaseModel):
    dispute_type: DisputeType
    reason: str = Field(..., min_length=10, max_length=1000)
    evidence: Optional[str] = None


class DisputeCreate(DisputeBase):
    market_id: int


class DisputeUpdate(BaseModel):
    status: Optional[DisputeStatus] = None
    resolution_notes: Optional[str] = None


class DisputeVoteCreate(BaseModel):
    vote_for_dispute: bool
    reason: Optional[str] = None


class DisputeVoteResponse(BaseModel):
    id: int
    dispute_id: int
    voter_id: int
    vote_for_dispute: bool
    reason: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DisputeResponse(DisputeBase):
    id: int
    market_id: int
    created_by: int
    status: DisputeStatus
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    votes_for_dispute: int
    votes_against_dispute: int
    required_votes: int
    total_votes: int
    consensus_reached: bool
    dispute_wins: bool
    is_resolved: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DisputeListResponse(BaseModel):
    disputes: List[DisputeResponse]
    total: int
    page: int
    per_page: int
