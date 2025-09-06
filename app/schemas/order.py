from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from app.models.order import OrderType, OrderStatus, OrderSide


class OrderBase(BaseModel):
    order_type: OrderType
    side: OrderSide
    outcome: str
    original_amount: float = Field(..., gt=0)
    limit_price: Optional[float] = Field(None, ge=0, le=1)
    stop_price: Optional[float] = Field(None, ge=0, le=1)
    expires_at: Optional[datetime] = None


class OrderCreate(OrderBase):
    market_id: int


class OrderUpdate(BaseModel):
    limit_price: Optional[float] = Field(None, ge=0, le=1)
    stop_price: Optional[float] = Field(None, ge=0, le=1)
    expires_at: Optional[datetime] = None


class OrderResponse(OrderBase):
    id: int
    user_id: int
    market_id: int
    remaining_amount: float
    filled_amount: float
    average_fill_price: float
    status: OrderStatus
    total_value: float
    is_active: bool
    order_hash: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class OrderListResponse(BaseModel):
    orders: List[OrderResponse]
    total: int
    page: int
    per_page: int


class OrderBookLevel(BaseModel):
    price: float
    amount: float
    total: float


class OrderBookResponse(BaseModel):
    market_id: int
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None


class OrderFillResponse(BaseModel):
    id: int
    order_id: int
    trade_id: int
    amount: float
    price: float
    total_value: float
    created_at: datetime

    class Config:
        from_attributes = True
