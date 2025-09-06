from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from app.models.market import MarketStatus, MarketCategory


class MarketBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    category: MarketCategory
    question: str = Field(..., min_length=1, max_length=500)
    outcome_a: str = Field(..., min_length=1, max_length=100)
    outcome_b: str = Field(..., min_length=1, max_length=100)
    closes_at: datetime
    tags: Optional[List[str]] = []
    image_url: Optional[str] = None


class MarketCreate(MarketBase):
    total_liquidity: Optional[float] = 1000.0
    fee_rate: Optional[float] = 0.02
    min_trade_amount: Optional[float] = 1.0
    max_trade_amount: Optional[float] = 10000.0


class MarketUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    category: Optional[MarketCategory] = None
    status: Optional[MarketStatus] = None
    resolved_outcome: Optional[str] = None
    resolution_source: Optional[str] = None
    tags: Optional[List[str]] = None
    image_url: Optional[str] = None


class MarketResponse(MarketBase):
    id: int
    current_price_a: float
    current_price_b: float
    total_liquidity: float
    liquidity_pool_a: float
    liquidity_pool_b: float
    fee_rate: float
    min_trade_amount: float
    max_trade_amount: float
    status: MarketStatus
    resolved_outcome: Optional[str] = None
    resolution_source: Optional[str] = None
    creator_id: int
    created_at: datetime
    resolved_at: Optional[datetime] = None
    volume_24h: float
    volume_total: float
    total_volume: float
    total_trades: int
    is_active: bool
    price_impact: float

    class Config:
        from_attributes = True


class MarketListResponse(BaseModel):
    markets: List[MarketResponse]
    total: int
    page: int
    per_page: int


class MarketStats(BaseModel):
    total_markets: int
    active_markets: int
    total_volume_24h: float
    total_volume_all_time: float
    most_active_category: str
    trending_markets: List[MarketResponse]


class PriceHistory(BaseModel):
    timestamp: datetime
    price_a: float
    price_b: float
    volume: float
