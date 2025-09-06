from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from app.models.trade import TradeType, TradeOutcome


class TradeBase(BaseModel):
    trade_type: TradeType
    outcome: TradeOutcome
    amount: float
    price_per_share: float


class TradeCreate(TradeBase):
    market_id: int


class TradeResponse(TradeBase):
    id: int
    total_value: float
    market_id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class TradeListResponse(BaseModel):
    trades: List[TradeResponse]
    total: int
    page: int
    per_page: int
