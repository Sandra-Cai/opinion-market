from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from app.models.vote import VoteOutcome

class VoteBase(BaseModel):
    outcome: VoteOutcome
    confidence: float = 1.0

class VoteCreate(VoteBase):
    market_id: int

class VoteResponse(VoteBase):
    id: int
    market_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class VoteListResponse(BaseModel):
    votes: List[VoteResponse]
    total: int
    page: int
    per_page: int
