from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.advanced_markets import MarketInstrument, OptionType


class FuturesContractBase(BaseModel):
    market_id: int
    contract_size: float = Field(..., gt=0)
    tick_size: float = Field(0.01, gt=0)
    margin_requirement: float = Field(0.1, ge=0.01, le=1.0)
    settlement_date: datetime
    cash_settlement: bool = True
    max_position_size: Optional[float] = None
    daily_price_limit: Optional[float] = None


class FuturesContractCreate(FuturesContractBase):
    pass


class FuturesContractResponse(FuturesContractBase):
    id: int
    settlement_price: Optional[float] = None
    is_settled: bool
    days_to_settlement: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class FuturesPositionBase(BaseModel):
    contract_id: int
    long_contracts: float = Field(0, ge=0)
    short_contracts: float = Field(0, ge=0)
    average_entry_price: float = Field(0, ge=0)


class FuturesPositionCreate(FuturesPositionBase):
    pass


class FuturesPositionResponse(FuturesPositionBase):
    id: int
    user_id: int
    net_position: float
    position_value: float
    margin_used: float
    unrealized_pnl: float
    realized_pnl: float
    liquidation_price: Optional[float] = None
    is_liquidated: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class OptionsContractBase(BaseModel):
    market_id: int
    option_type: OptionType
    strike_price: float = Field(..., gt=0, le=1)
    expiration_date: datetime
    contract_size: float = Field(1.0, gt=0)
    premium: float = Field(0, ge=0)


class OptionsContractCreate(OptionsContractBase):
    pass


class OptionsContractResponse(OptionsContractBase):
    id: int
    delta: float
    gamma: float
    theta: float
    vega: float
    is_expired: bool
    is_in_the_money: bool
    intrinsic_value: float
    time_value: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class OptionsPositionBase(BaseModel):
    contract_id: int
    long_contracts: float = Field(0, ge=0)
    short_contracts: float = Field(0, ge=0)
    average_entry_price: float = Field(0, ge=0)


class OptionsPositionCreate(OptionsPositionBase):
    pass


class OptionsPositionResponse(OptionsPositionBase):
    id: int
    user_id: int
    net_position: float
    unrealized_pnl: float
    realized_pnl: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConditionalMarketBase(BaseModel):
    market_id: int
    condition_description: str = Field(..., min_length=10, max_length=500)
    trigger_condition: Dict[str, Any]
    trigger_market_id: Optional[int] = None


class ConditionalMarketCreate(ConditionalMarketBase):
    pass


class ConditionalMarketResponse(ConditionalMarketBase):
    id: int
    is_active: bool
    activated_at: Optional[datetime] = None
    activated_by: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SpreadMarketBase(BaseModel):
    market_id: int
    spread_type: str = Field(..., regex="^(binary|range|index)$")
    min_value: float
    max_value: float = Field(..., gt=0)
    tick_size: float = Field(0.1, gt=0)


class SpreadMarketCreate(SpreadMarketBase):
    pass


class SpreadMarketResponse(SpreadMarketBase):
    id: int
    outcomes: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MarketInstrumentResponse(BaseModel):
    instrument_type: MarketInstrument
    contracts: List[Dict[str, Any]]
    total_volume: float
    open_interest: float
    active_positions: int
