from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from app.models.market import MarketStatus, MarketCategory

class MarketBase(BaseModel):
    title: str
    description: Optional[str] = None
    category: MarketCategory
    question: str
    outcome_a: str
    outcome_b: str
    closes_at: datetime

class MarketCreate(MarketBase):
    pass

class MarketUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[MarketCategory] = None
    status: Optional[MarketStatus] = None
    resolved_outcome: Optional[str] = None

class MarketResponse(MarketBase):
    id: int
    current_price_a: float
    current_price_b: float
    total_liquidity: float
    status: MarketStatus
    resolved_outcome: Optional[str] = None
    creator_id: int
    created_at: datetime
    resolved_at: Optional[datetime] = None
    total_volume: float
    total_trades: int
    is_active: bool
    
    class Config:
        from_attributes = True

class MarketListResponse(BaseModel):
    markets: List[MarketResponse]
    total: int
    page: int
    per_page: int
