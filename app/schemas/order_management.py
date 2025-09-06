"""
Pydantic schemas for Order Management System
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class OrderTypeEnum(str, Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ADAPTIVE = "adaptive"
    PEG = "peg"
    HIDDEN = "hidden"
    DISPLAY = "display"


class OrderSideEnum(str, Enum):
    """Order sides"""

    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class OrderStatusEnum(str, Enum):
    """Order statuses"""

    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"


class TimeInForceEnum(str, Enum):
    """Time in force"""

    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    GTD = "gtd"
    ATC = "atc"
    ATO = "ato"


class ExecutionAlgorithmEnum(str, Enum):
    """Execution algorithms"""

    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ADAPTIVE = "adaptive"
    ICEBERG = "iceberg"
    PEG = "peg"
    HIDDEN = "hidden"
    DISPLAY = "display"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"


class ExecutionStrategyEnum(str, Enum):
    """Execution strategies"""

    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    NEUTRAL = "neutral"
    ADAPTIVE = "adaptive"


# Order Management Schemas
class OrderCreate(BaseModel):
    """Create order request"""

    user_id: int = Field(..., description="User ID")
    account_id: str = Field(..., description="Account ID")
    symbol: str = Field(..., description="Trading symbol")
    order_type: OrderTypeEnum = Field(..., description="Order type")
    side: OrderSideEnum = Field(..., description="Order side")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, gt=0, description="Order price")
    stop_price: Optional[float] = Field(None, gt=0, description="Stop price")
    time_in_force: TimeInForceEnum = Field(
        TimeInForceEnum.DAY, description="Time in force"
    )
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    algo_type: Optional[str] = Field(None, description="Algorithm type")
    algo_parameters: Optional[Dict[str, Any]] = Field(
        None, description="Algorithm parameters"
    )

    @validator("price")
    def validate_price(cls, v, values):
        if values.get("order_type") in ["limit", "stop_limit"] and v is None:
            raise ValueError("Price is required for limit and stop_limit orders")
        return v

    @validator("stop_price")
    def validate_stop_price(cls, v, values):
        if values.get("order_type") in ["stop", "stop_limit"] and v is None:
            raise ValueError("Stop price is required for stop and stop_limit orders")
        return v


class OrderResponse(BaseModel):
    """Order response"""

    order_id: str = Field(..., description="Order ID")
    user_id: int = Field(..., description="User ID")
    account_id: str = Field(..., description="Account ID")
    symbol: str = Field(..., description="Trading symbol")
    order_type: str = Field(..., description="Order type")
    side: str = Field(..., description="Order side")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = Field(None, description="Order price")
    stop_price: Optional[float] = Field(None, description="Stop price")
    time_in_force: str = Field(..., description="Time in force")
    status: str = Field(..., description="Order status")
    filled_quantity: float = Field(..., description="Filled quantity")
    remaining_quantity: float = Field(..., description="Remaining quantity")
    average_price: float = Field(..., description="Average fill price")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")


class OrderModify(BaseModel):
    """Modify order request"""

    user_id: int = Field(..., description="User ID")
    new_quantity: Optional[float] = Field(None, gt=0, description="New quantity")
    new_price: Optional[float] = Field(None, gt=0, description="New price")

    @validator("new_quantity", "new_price")
    def validate_modification(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Value must be positive")
        return v


class OrderCancel(BaseModel):
    """Cancel order request"""

    user_id: int = Field(..., description="User ID")


# Execution Management Schemas
class ExecutionCreate(BaseModel):
    """Create execution request"""

    parent_order_id: str = Field(..., description="Parent order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side")
    quantity: float = Field(..., gt=0, description="Execution quantity")
    algorithm: ExecutionAlgorithmEnum = Field(..., description="Execution algorithm")
    strategy: ExecutionStrategyEnum = Field(
        ExecutionStrategyEnum.NEUTRAL, description="Execution strategy"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Algorithm parameters"
    )


class ExecutionResponse(BaseModel):
    """Execution response"""

    execution_id: str = Field(..., description="Execution ID")
    parent_order_id: str = Field(..., description="Parent order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side")
    quantity: float = Field(..., description="Execution quantity")
    algorithm: str = Field(..., description="Execution algorithm")
    strategy: str = Field(..., description="Execution strategy")
    parameters: Dict[str, Any] = Field(..., description="Algorithm parameters")
    status: str = Field(..., description="Execution status")
    filled_quantity: float = Field(..., description="Filled quantity")
    remaining_quantity: float = Field(..., description="Remaining quantity")
    average_price: float = Field(..., description="Average fill price")
    venues: List[str] = Field(..., description="Selected venues")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")


class ExecutionMetricsResponse(BaseModel):
    """Execution metrics response"""

    execution_id: str = Field(..., description="Execution ID")
    symbol: str = Field(..., description="Trading symbol")
    total_quantity: float = Field(..., description="Total quantity")
    filled_quantity: float = Field(..., description="Filled quantity")
    average_price: float = Field(..., description="Average fill price")
    benchmark_price: float = Field(..., description="Benchmark price")
    implementation_shortfall: float = Field(..., description="Implementation shortfall")
    market_impact: float = Field(..., description="Market impact")
    timing_cost: float = Field(..., description="Timing cost")
    opportunity_cost: float = Field(..., description="Opportunity cost")
    total_cost: float = Field(..., description="Total cost")
    vwap_deviation: float = Field(..., description="VWAP deviation")
    participation_rate: float = Field(..., description="Participation rate")
    fill_rate: float = Field(..., description="Fill rate")
    execution_time: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Creation timestamp")


# Market Data Schemas
class MarketDataResponse(BaseModel):
    """Market data response"""

    symbol: str = Field(..., description="Trading symbol")
    bid_price: float = Field(..., description="Bid price")
    ask_price: float = Field(..., description="Ask price")
    bid_size: float = Field(..., description="Bid size")
    ask_size: float = Field(..., description="Ask size")
    last_price: float = Field(..., description="Last trade price")
    volume: float = Field(..., description="Volume")
    high_price: float = Field(..., description="High price")
    low_price: float = Field(..., description="Low price")
    open_price: float = Field(..., description="Open price")
    timestamp: datetime = Field(..., description="Data timestamp")


class OrderBookResponse(BaseModel):
    """Order book response"""

    symbol: str = Field(..., description="Trading symbol")
    bids: List[List[float]] = Field(..., description="Bid levels (price, quantity)")
    asks: List[List[float]] = Field(..., description="Ask levels (price, quantity)")
    last_trade_price: Optional[float] = Field(None, description="Last trade price")
    last_trade_quantity: Optional[float] = Field(
        None, description="Last trade quantity"
    )
    last_trade_time: Optional[datetime] = Field(None, description="Last trade time")
    volume: float = Field(..., description="Volume")
    timestamp: datetime = Field(..., description="Data timestamp")


class ExecutionReportResponse(BaseModel):
    """Execution report response"""

    report_id: str = Field(..., description="Report ID")
    order_id: str = Field(..., description="Order ID")
    execution_type: str = Field(..., description="Execution type")
    order_status: str = Field(..., description="Order status")
    filled_quantity: float = Field(..., description="Filled quantity")
    remaining_quantity: float = Field(..., description="Remaining quantity")
    average_price: float = Field(..., description="Average price")
    last_fill_price: Optional[float] = Field(None, description="Last fill price")
    last_fill_quantity: Optional[float] = Field(None, description="Last fill quantity")
    commission: float = Field(..., description="Commission")
    venue: str = Field(..., description="Execution venue")
    execution_time: datetime = Field(..., description="Execution time")
    text: Optional[str] = Field(None, description="Report text")
    created_at: datetime = Field(..., description="Creation timestamp")


# Bulk Operations Schemas
class BulkOrderCreate(BaseModel):
    """Bulk order creation request"""

    orders: List[OrderCreate] = Field(..., description="List of orders to create")

    @validator("orders")
    def validate_orders(cls, v):
        if len(v) == 0:
            raise ValueError("At least one order is required")
        if len(v) > 100:
            raise ValueError("Maximum 100 orders allowed per request")
        return v


class BulkOrderResponse(BaseModel):
    """Bulk order response"""

    orders: List[OrderResponse] = Field(..., description="Created orders")
    failed_orders: List[Dict[str, Any]] = Field(
        ..., description="Failed orders with errors"
    )
    total_created: int = Field(..., description="Total orders created")
    total_failed: int = Field(..., description="Total orders failed")


# Analytics Schemas
class OrderAnalyticsRequest(BaseModel):
    """Order analytics request"""

    user_id: Optional[int] = Field(None, description="User ID filter")
    account_id: Optional[str] = Field(None, description="Account ID filter")
    symbol: Optional[str] = Field(None, description="Symbol filter")
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    group_by: Optional[str] = Field("symbol", description="Group by field")


class OrderAnalyticsResponse(BaseModel):
    """Order analytics response"""

    total_orders: int = Field(..., description="Total orders")
    total_volume: float = Field(..., description="Total volume")
    total_value: float = Field(..., description="Total value")
    fill_rate: float = Field(..., description="Fill rate")
    average_execution_time: float = Field(..., description="Average execution time")
    breakdown: List[Dict[str, Any]] = Field(..., description="Breakdown by group")


class ExecutionAnalyticsRequest(BaseModel):
    """Execution analytics request"""

    execution_id: Optional[str] = Field(None, description="Execution ID filter")
    symbol: Optional[str] = Field(None, description="Symbol filter")
    algorithm: Optional[str] = Field(None, description="Algorithm filter")
    start_date: Optional[datetime] = Field(None, description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")


class ExecutionAnalyticsResponse(BaseModel):
    """Execution analytics response"""

    total_executions: int = Field(..., description="Total executions")
    average_implementation_shortfall: float = Field(
        ..., description="Average implementation shortfall"
    )
    average_market_impact: float = Field(..., description="Average market impact")
    average_execution_time: float = Field(..., description="Average execution time")
    algorithm_performance: List[Dict[str, Any]] = Field(
        ..., description="Algorithm performance"
    )
    venue_performance: List[Dict[str, Any]] = Field(
        ..., description="Venue performance"
    )


# Risk Management Schemas
class RiskLimitRequest(BaseModel):
    """Risk limit request"""

    account_id: str = Field(..., description="Account ID")
    symbol: Optional[str] = Field(None, description="Symbol (None for global limits)")
    max_position_size: Optional[float] = Field(
        None, gt=0, description="Maximum position size"
    )
    max_order_value: Optional[float] = Field(
        None, gt=0, description="Maximum order value"
    )
    max_daily_volume: Optional[float] = Field(
        None, gt=0, description="Maximum daily volume"
    )
    max_daily_trades: Optional[int] = Field(
        None, gt=0, description="Maximum daily trades"
    )


class RiskLimitResponse(BaseModel):
    """Risk limit response"""

    account_id: str = Field(..., description="Account ID")
    symbol: Optional[str] = Field(None, description="Symbol")
    max_position_size: Optional[float] = Field(
        None, description="Maximum position size"
    )
    max_order_value: Optional[float] = Field(None, description="Maximum order value")
    max_daily_volume: Optional[float] = Field(None, description="Maximum daily volume")
    max_daily_trades: Optional[int] = Field(None, description="Maximum daily trades")
    current_position: float = Field(..., description="Current position")
    daily_volume: float = Field(..., description="Daily volume")
    daily_trades: int = Field(..., description="Daily trades")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")


# Configuration Schemas
class VenueConfigurationRequest(BaseModel):
    """Venue configuration request"""

    venue_id: str = Field(..., description="Venue ID")
    name: str = Field(..., description="Venue name")
    venue_type: str = Field(..., description="Venue type")
    is_active: bool = Field(True, description="Is venue active")
    latency_ms: float = Field(..., ge=0, description="Latency in milliseconds")
    commission_rate: float = Field(..., ge=0, le=1, description="Commission rate")
    min_order_size: float = Field(..., gt=0, description="Minimum order size")
    max_order_size: float = Field(..., gt=0, description="Maximum order size")
    supported_algorithms: List[str] = Field(..., description="Supported algorithms")


class VenueConfigurationResponse(BaseModel):
    """Venue configuration response"""

    venue_id: str = Field(..., description="Venue ID")
    name: str = Field(..., description="Venue name")
    venue_type: str = Field(..., description="Venue type")
    is_active: bool = Field(..., description="Is venue active")
    latency_ms: float = Field(..., description="Latency in milliseconds")
    commission_rate: float = Field(..., description="Commission rate")
    min_order_size: float = Field(..., description="Minimum order size")
    max_order_size: float = Field(..., description="Maximum order size")
    supported_algorithms: List[str] = Field(..., description="Supported algorithms")
    performance_metrics: Dict[str, float] = Field(
        ..., description="Performance metrics"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")
