from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class PositionBase(BaseModel):
    market_id: int
    outcome: str
    shares_owned: float
    average_price: float
    total_invested: float

class PositionCreate(BaseModel):
    market_id: int
    outcome: str

class PositionResponse(PositionBase):
    id: int
    user_id: int
    current_price: float
    current_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    profit_loss_percentage: float
    is_profitable: bool
    is_active: bool
    last_updated: datetime
    
    class Config:
        from_attributes = True

class PositionListResponse(BaseModel):
    positions: List[PositionResponse]
    total: int
    page: int
    per_page: int

class PortfolioSummary(BaseModel):
    total_positions: int
    active_positions: int
    total_portfolio_value: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    portfolio_return_percentage: float
    positions: List[PositionResponse]
