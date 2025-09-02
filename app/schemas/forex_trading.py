"""
Forex Trading Pydantic Schemas
Provides request and response models for FX trading operations
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from decimal import Decimal


# Currency Pair Schemas
class CurrencyPairCreate(BaseModel):
    """Schema for creating a currency pair"""
    base_currency: str = Field(..., description="Base currency (e.g., 'USD')")
    quote_currency: str = Field(..., description="Quote currency (e.g., 'EUR')")
    pip_value: float = Field(..., description="Value of one pip")
    lot_size: float = Field(..., description="Standard lot size")
    min_trade_size: float = Field(..., description="Minimum trade size")
    max_trade_size: float = Field(..., description="Maximum trade size")
    margin_requirement: float = Field(..., description="Margin requirement percentage")
    swap_long: float = Field(0.0, description="Overnight interest for long positions")
    swap_short: float = Field(0.0, description="Overnight interest for short positions")
    
    @validator('base_currency', 'quote_currency')
    def validate_currency_codes(cls, v):
        if len(v) != 3 or not v.isalpha():
            raise ValueError('Currency codes must be 3-letter alphabetic codes')
        return v.upper()


class CurrencyPairResponse(BaseModel):
    """Schema for currency pair response"""
    pair_id: str
    base_currency: str
    quote_currency: str
    pair_name: str
    pip_value: float
    lot_size: float
    min_trade_size: float
    max_trade_size: float
    margin_requirement: float
    swap_long: float
    swap_short: float
    is_active: bool
    trading_hours: Dict[str, List[str]]
    created_at: datetime
    last_updated: datetime
    
    class Config:
        from_attributes = True


# FX Price Schemas
class FXPriceCreate(BaseModel):
    """Schema for creating an FX price"""
    pair_id: str = Field(..., description="Currency pair ID")
    bid_price: float = Field(..., description="Bid price")
    ask_price: float = Field(..., description="Ask price")
    volume_24h: float = Field(..., description="24-hour trading volume")
    high_24h: float = Field(..., description="24-hour high price")
    low_24h: float = Field(..., description="24-hour low price")
    change_24h: float = Field(..., description="24-hour price change")
    source: str = Field(..., description="Price source")
    
    @validator('ask_price')
    def validate_ask_greater_than_bid(cls, v, values):
        if 'bid_price' in values and v <= values['bid_price']:
            raise ValueError('Ask price must be greater than bid price')
        return v


class FXPriceResponse(BaseModel):
    """Schema for FX price response"""
    price_id: str
    pair_id: str
    bid_price: float
    ask_price: float
    mid_price: float
    spread: float
    pip_value: float
    timestamp: datetime
    source: str
    volume_24h: float
    high_24h: float
    low_24h: float
    change_24h: float
    change_pct_24h: float
    
    class Config:
        from_attributes = True


# FX Position Schemas
class FXPositionCreate(BaseModel):
    """Schema for creating an FX position"""
    user_id: int = Field(..., description="User ID")
    pair_id: str = Field(..., description="Currency pair ID")
    position_type: str = Field(..., description="Position type: 'long' or 'short'")
    quantity: float = Field(..., description="Position quantity")
    entry_price: float = Field(..., description="Entry price")
    leverage: float = Field(1.0, description="Leverage used")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    
    @validator('position_type')
    def validate_position_type(cls, v):
        if v not in ['long', 'short']:
            raise ValueError('Position type must be "long" or "short"')
        return v
    
    @validator('leverage')
    def validate_leverage(cls, v):
        if v <= 0:
            raise ValueError('Leverage must be positive')
        return v


class FXPositionResponse(BaseModel):
    """Schema for FX position response"""
    position_id: str
    user_id: int
    pair_id: str
    position_type: str
    quantity: float
    entry_price: float
    current_price: float
    pip_value: float
    unrealized_pnl: float
    realized_pnl: float
    swap_charges: float
    margin_used: float
    leverage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    created_at: datetime
    last_updated: datetime
    
    class Config:
        from_attributes = True


# Forward Contract Schemas
class ForwardContractCreate(BaseModel):
    """Schema for creating a forward contract"""
    user_id: int = Field(..., description="User ID")
    pair_id: str = Field(..., description="Currency pair ID")
    quantity: float = Field(..., description="Contract quantity")
    forward_rate: float = Field(..., description="Forward rate")
    spot_rate: float = Field(..., description="Spot rate")
    value_date: datetime = Field(..., description="Value date")
    maturity_date: datetime = Field(..., description="Maturity date")
    contract_type: str = Field(..., description="Contract type: 'buy' or 'sell'")
    is_deliverable: bool = Field(True, description="Whether contract is deliverable")
    
    @validator('contract_type')
    def validate_contract_type(cls, v):
        if v not in ['buy', 'sell']:
            raise ValueError('Contract type must be "buy" or "sell"')
        return v
    
    @validator('maturity_date')
    def validate_maturity_after_value_date(cls, v, values):
        if 'value_date' in values and v <= values['value_date']:
            raise ValueError('Maturity date must be after value date')
        return v


class ForwardContractResponse(BaseModel):
    """Schema for forward contract response"""
    contract_id: str
    pair_id: str
    user_id: int
    quantity: float
    forward_rate: float
    spot_rate: float
    forward_points: float
    value_date: datetime
    maturity_date: datetime
    contract_type: str
    is_deliverable: bool
    created_at: datetime
    last_updated: datetime
    
    class Config:
        from_attributes = True


# Swap Contract Schemas
class SwapContractCreate(BaseModel):
    """Schema for creating a swap contract"""
    user_id: int = Field(..., description="User ID")
    pair_id: str = Field(..., description="Currency pair ID")
    near_leg: Dict[str, Any] = Field(..., description="Near leg details")
    far_leg: Dict[str, Any] = Field(..., description="Far leg details")
    swap_rate: float = Field(..., description="Swap rate")
    value_date: datetime = Field(..., description="Value date")
    maturity_date: datetime = Field(..., description="Maturity date")
    
    @validator('maturity_date')
    def validate_maturity_after_value_date(cls, v, values):
        if 'value_date' in values and v <= values['value_date']:
            raise ValueError('Maturity date must be after value date')
        return v


class SwapContractResponse(BaseModel):
    """Schema for swap contract response"""
    swap_id: str
    pair_id: str
    user_id: int
    near_leg: Dict[str, Any]
    far_leg: Dict[str, Any]
    swap_rate: float
    swap_points: float
    value_date: datetime
    maturity_date: datetime
    created_at: datetime
    last_updated: datetime
    
    class Config:
        from_attributes = True


# FX Order Schemas
class FXOrderCreate(BaseModel):
    """Schema for creating an FX order"""
    user_id: int = Field(..., description="User ID")
    pair_id: str = Field(..., description="Currency pair ID")
    order_type: str = Field(..., description="Order type: 'market', 'limit', 'stop', 'stop_limit'")
    side: str = Field(..., description="Order side: 'buy' or 'sell'")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = Field(None, description="Limit price")
    stop_price: Optional[float] = Field(None, description="Stop price")
    limit_price: Optional[float] = Field(None, description="Limit price for stop-limit orders")
    time_in_force: str = Field('GTC', description="Time in force: 'GTC', 'IOC', 'FOK'")
    
    @validator('order_type')
    def validate_order_type(cls, v):
        valid_types = ['market', 'limit', 'stop', 'stop_limit']
        if v not in valid_types:
            raise ValueError(f'Order type must be one of: {valid_types}')
        return v
    
    @validator('side')
    def validate_side(cls, v):
        if v not in ['buy', 'sell']:
            raise ValueError('Side must be "buy" or "sell"')
        return v
    
    @validator('time_in_force')
    def validate_time_in_force(cls, v):
        valid_tif = ['GTC', 'IOC', 'FOK']
        if v not in valid_tif:
            raise ValueError(f'Time in force must be one of: {valid_tif}')
        return v
    
    @validator('price', 'stop_price', 'limit_price')
    def validate_prices(cls, v, values):
        if v is not None and v <= 0:
            raise ValueError('Prices must be positive')
        return v


class FXOrderResponse(BaseModel):
    """Schema for FX order response"""
    order_id: str
    user_id: int
    pair_id: str
    order_type: str
    side: str
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    limit_price: Optional[float]
    time_in_force: str
    status: str
    filled_quantity: float
    filled_price: float
    created_at: datetime
    last_updated: datetime
    
    class Config:
        from_attributes = True


# FX Metrics Schemas
class CurrentPriceInfo(BaseModel):
    """Schema for current price information"""
    bid: float
    ask: float
    mid: float
    spread: float


class MarketDataInfo(BaseModel):
    """Schema for market data information"""
    volume_24h: float
    high_24h: float
    low_24h: float
    change_24h: float
    change_pct_24h: float


class TradingMetricsInfo(BaseModel):
    """Schema for trading metrics information"""
    pip_value: float
    lot_size: float
    min_trade_size: float
    max_trade_size: float
    margin_requirement: float


class RiskMetricsInfo(BaseModel):
    """Schema for risk metrics information"""
    volatility: float
    avg_spread: float
    spread_pct: float


class SwapRatesInfo(BaseModel):
    """Schema for swap rates information"""
    long: float
    short: float


class FXMetricsResponse(BaseModel):
    """Schema for FX metrics response"""
    pair_id: str
    pair_name: str
    base_currency: str
    quote_currency: str
    current_price: CurrentPriceInfo
    market_data: MarketDataInfo
    trading_metrics: TradingMetricsInfo
    risk_metrics: RiskMetricsInfo
    correlations: Dict[str, float]
    swap_rates: SwapRatesInfo
    trading_hours: Dict[str, List[str]]
    last_updated: str


# Cross Currency Rates Schemas
class CrossRateInfo(BaseModel):
    """Schema for cross rate information"""
    rate: float
    bid: float
    ask: float
    spread: float
    timestamp: str


class CrossCurrencyRatesResponse(BaseModel):
    """Schema for cross currency rates response"""
    base_currency: str
    cross_rates: Dict[str, CrossRateInfo]
    last_updated: str


# Forward Points Schemas
class ForwardPointsResponse(BaseModel):
    """Schema for forward points response"""
    pair_id: str
    spot_rate: float
    forward_rate: float
    forward_points: float
    annualized_points: float
    interest_rate_base: float
    interest_rate_quote: float
    days_to_maturity: int
    years_to_maturity: float
    calculation_method: str
    last_updated: str


# Trading Session Schemas
class TradingSessionInfo(BaseModel):
    """Schema for trading session information"""
    start: str
    end: str


class TradingSessionsResponse(BaseModel):
    """Schema for trading sessions response"""
    current_time_utc: str
    trading_sessions: Dict[str, TradingSessionInfo]
    active_sessions: List[str]
    next_session: str


# WebSocket Message Schemas
class FXUpdateMessage(BaseModel):
    """Schema for FX update WebSocket message"""
    type: str = "fx_update"
    pair_id: str
    timestamp: str
    message: str


# Error Response Schemas
class FXErrorResponse(BaseModel):
    """Schema for FX error response"""
    error: str
    detail: str
    timestamp: str


# Bulk Operations Schemas
class BulkFXPriceCreate(BaseModel):
    """Schema for bulk FX price creation"""
    prices: List[FXPriceCreate]


class BulkFXPositionCreate(BaseModel):
    """Schema for bulk FX position creation"""
    positions: List[FXPositionCreate]


class BulkFXOrderCreate(BaseModel):
    """Schema for bulk FX order creation"""
    orders: List[FXOrderCreate]


# Analytics Schemas
class FXAnalyticsRequest(BaseModel):
    """Schema for FX analytics request"""
    pair_ids: List[str] = Field(..., description="Currency pair IDs to analyze")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: datetime = Field(..., description="End date for analysis")
    metrics: List[str] = Field(..., description="Metrics to calculate")
    
    @validator('metrics')
    def validate_metrics(cls, v):
        valid_metrics = ['volatility', 'correlation', 'spread', 'volume', 'returns']
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f'Invalid metric: {metric}. Valid metrics: {valid_metrics}')
        return v


class FXAnalyticsResponse(BaseModel):
    """Schema for FX analytics response"""
    pair_ids: List[str]
    start_date: str
    end_date: str
    analytics: Dict[str, Dict[str, Any]]
    generated_at: str


# Portfolio Schemas
class FXPortfolioSummary(BaseModel):
    """Schema for FX portfolio summary"""
    user_id: int
    total_positions: int
    total_notional_value: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_swap_charges: float
    total_margin_used: float
    currency_exposure: Dict[str, float]
    risk_metrics: Dict[str, float]
    last_updated: str


class FXPortfolioRequest(BaseModel):
    """Schema for FX portfolio request"""
    user_id: int = Field(..., description="User ID")
    include_closed: bool = Field(False, description="Include closed positions")
    include_orders: bool = Field(True, description="Include pending orders")
    include_contracts: bool = Field(True, description="Include forward/swap contracts")


class FXPortfolioResponse(BaseModel):
    """Schema for FX portfolio response"""
    summary: FXPortfolioSummary
    positions: List[FXPositionResponse]
    orders: List[FXOrderResponse]
    forward_contracts: List[ForwardContractResponse]
    swap_contracts: List[SwapContractResponse]
    generated_at: str
