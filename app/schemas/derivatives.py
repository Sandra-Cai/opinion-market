"""
Derivatives Trading Pydantic Schemas
Advanced derivatives trading, pricing, and risk management schemas
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class DerivativeType(str, Enum):
    """Derivative types"""

    OPTION = "option"
    FUTURE = "future"
    FORWARD = "forward"
    SWAP = "swap"
    WARRANT = "warrant"
    CONVERTIBLE = "convertible"
    STRUCTURED_PRODUCT = "structured_product"


class OptionType(str, Enum):
    """Option types"""

    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    """Exercise styles"""

    AMERICAN = "american"
    EUROPEAN = "european"
    BERMUDAN = "bermudan"


class SwapType(str, Enum):
    """Swap types"""

    INTEREST_RATE = "interest_rate"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    EQUITY = "equity"
    CREDIT_DEFAULT = "credit_default"
    TOTAL_RETURN = "total_return"


class RiskType(str, Enum):
    """Risk types"""

    MARKET = "market"
    CREDIT = "credit"
    LIQUIDITY = "liquidity"
    OPERATIONAL = "operational"
    MODEL = "model"
    CONCENTRATION = "concentration"
    BASIS = "basis"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    GAMMA = "gamma"
    THETA = "theta"
    VEGA = "vega"


class StressTestType(str, Enum):
    """Stress test types"""

    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    SCENARIO = "scenario"
    SENSITIVITY = "sensitivity"


# Derivative Schemas
class DerivativeCreate(BaseModel):
    """Create derivative request"""

    symbol: str = Field(..., description="Derivative symbol")
    derivative_type: DerivativeType = Field(..., description="Type of derivative")
    underlying_asset: str = Field(..., description="Underlying asset symbol")
    strike_price: Optional[float] = Field(
        None, description="Strike price (for options)"
    )
    expiration_date: Optional[datetime] = Field(None, description="Expiration date")
    option_type: Optional[OptionType] = Field(
        None, description="Option type (call/put)"
    )
    exercise_style: Optional[ExerciseStyle] = Field(None, description="Exercise style")
    contract_size: float = Field(1.0, description="Contract size")
    multiplier: float = Field(1.0, description="Price multiplier")
    currency: str = Field("USD", description="Currency")
    exchange: str = Field("CBOE", description="Exchange")

    @validator("strike_price")
    def validate_strike_price(cls, v, values):
        if values.get("derivative_type") == DerivativeType.OPTION and v is None:
            raise ValueError("Strike price is required for options")
        return v

    @validator("expiration_date")
    def validate_expiration_date(cls, v, values):
        if (
            values.get("derivative_type")
            in [DerivativeType.OPTION, DerivativeType.FUTURE]
            and v is None
        ):
            raise ValueError("Expiration date is required for options and futures")
        return v

    @validator("option_type")
    def validate_option_type(cls, v, values):
        if values.get("derivative_type") == DerivativeType.OPTION and v is None:
            raise ValueError("Option type is required for options")
        return v


class DerivativeResponse(BaseModel):
    """Derivative response"""

    derivative_id: str = Field(..., description="Derivative ID")
    symbol: str = Field(..., description="Derivative symbol")
    derivative_type: str = Field(..., description="Type of derivative")
    underlying_asset: str = Field(..., description="Underlying asset symbol")
    strike_price: Optional[float] = Field(None, description="Strike price")
    expiration_date: Optional[datetime] = Field(None, description="Expiration date")
    option_type: Optional[str] = Field(None, description="Option type")
    exercise_style: Optional[str] = Field(None, description="Exercise style")
    contract_size: float = Field(..., description="Contract size")
    multiplier: float = Field(..., description="Price multiplier")
    currency: str = Field(..., description="Currency")
    exchange: str = Field(..., description="Exchange")
    is_active: bool = Field(..., description="Is active")
    created_at: datetime = Field(..., description="Created timestamp")
    last_updated: datetime = Field(..., description="Last updated timestamp")


# Pricing Schemas
class OptionPriceRequest(BaseModel):
    """Option price calculation request"""

    underlying_price: float = Field(..., description="Current underlying price", gt=0)
    risk_free_rate: float = Field(0.05, description="Risk-free rate", ge=0, le=1)
    dividend_yield: float = Field(0.0, description="Dividend yield", ge=0, le=1)
    volatility: Optional[float] = Field(
        None, description="Implied volatility", gt=0, le=5
    )


class GreeksResponse(BaseModel):
    """Option Greeks response"""

    delta: float = Field(..., description="Delta")
    gamma: float = Field(..., description="Gamma")
    theta: float = Field(..., description="Theta")
    vega: float = Field(..., description="Vega")
    rho: float = Field(..., description="Rho")
    vanna: float = Field(..., description="Vanna")
    volga: float = Field(..., description="Volga")
    charm: float = Field(..., description="Charm")


class OptionPriceResponse(BaseModel):
    """Option price calculation response"""

    derivative_id: str = Field(..., description="Derivative ID")
    theoretical_price: float = Field(..., description="Theoretical price")
    greeks: Optional[GreeksResponse] = Field(None, description="Option Greeks")
    underlying_price: float = Field(..., description="Underlying price")
    strike_price: float = Field(..., description="Strike price")
    time_to_expiry: float = Field(..., description="Time to expiry (years)")
    risk_free_rate: float = Field(..., description="Risk-free rate")
    dividend_yield: float = Field(..., description="Dividend yield")
    volatility: float = Field(..., description="Volatility")
    option_type: str = Field(..., description="Option type")
    exercise_style: str = Field(..., description="Exercise style")


class DerivativePriceResponse(BaseModel):
    """Derivative price response"""

    derivative_id: str = Field(..., description="Derivative ID")
    timestamp: datetime = Field(..., description="Price timestamp")
    bid_price: float = Field(..., description="Bid price")
    ask_price: float = Field(..., description="Ask price")
    mid_price: float = Field(..., description="Mid price")
    last_price: float = Field(..., description="Last traded price")
    volume: float = Field(..., description="Volume")
    open_interest: float = Field(..., description="Open interest")
    implied_volatility: Optional[float] = Field(None, description="Implied volatility")
    greeks: Optional[GreeksResponse] = Field(None, description="Option Greeks")
    theoretical_price: float = Field(..., description="Theoretical price")
    price_source: str = Field(..., description="Price source")


class VolatilitySurfaceResponse(BaseModel):
    """Volatility surface response"""

    underlying_asset: str = Field(..., description="Underlying asset")
    timestamp: datetime = Field(..., description="Surface timestamp")
    strikes: List[float] = Field(..., description="Strike prices")
    expirations: List[str] = Field(..., description="Expiration dates")
    implied_volatilities: List[List[float]] = Field(
        ..., description="Implied volatilities matrix"
    )
    risk_free_rate: float = Field(..., description="Risk-free rate")
    dividend_yield: float = Field(..., description="Dividend yield")


# Position Schemas
class DerivativePositionCreate(BaseModel):
    """Create derivative position request"""

    user_id: int = Field(..., description="User ID", gt=0)
    derivative_id: str = Field(..., description="Derivative ID")
    quantity: float = Field(..., description="Position quantity", ne=0)
    average_price: float = Field(..., description="Average price", gt=0)


class DerivativePositionResponse(BaseModel):
    """Derivative position response"""

    position_id: str = Field(..., description="Position ID")
    user_id: int = Field(..., description="User ID")
    derivative_id: str = Field(..., description="Derivative ID")
    quantity: float = Field(..., description="Position quantity")
    average_price: float = Field(..., description="Average price")
    current_price: float = Field(..., description="Current price")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    realized_pnl: float = Field(..., description="Realized P&L")
    margin_required: float = Field(..., description="Margin required")
    delta_exposure: float = Field(..., description="Delta exposure")
    gamma_exposure: float = Field(..., description="Gamma exposure")
    theta_exposure: float = Field(..., description="Theta exposure")
    vega_exposure: float = Field(..., description="Vega exposure")
    created_at: datetime = Field(..., description="Created timestamp")
    last_updated: datetime = Field(..., description="Last updated timestamp")


# Order Schemas
class DerivativeOrderCreate(BaseModel):
    """Create derivative order request"""

    user_id: int = Field(..., description="User ID", gt=0)
    derivative_id: str = Field(..., description="Derivative ID")
    order_type: str = Field(
        ..., description="Order type", regex="^(market|limit|stop|stop_limit)$"
    )
    side: str = Field(..., description="Order side", regex="^(buy|sell)$")
    quantity: float = Field(..., description="Order quantity", gt=0)
    price: Optional[float] = Field(None, description="Limit price", gt=0)
    stop_price: Optional[float] = Field(None, description="Stop price", gt=0)
    time_in_force: str = Field(
        "GTC", description="Time in force", regex="^(GTC|IOC|FOK|DAY)$"
    )

    @validator("price")
    def validate_price(cls, v, values):
        if values.get("order_type") in ["limit", "stop_limit"] and v is None:
            raise ValueError("Price is required for limit and stop-limit orders")
        return v

    @validator("stop_price")
    def validate_stop_price(cls, v, values):
        if values.get("order_type") == "stop_limit" and v is None:
            raise ValueError("Stop price is required for stop-limit orders")
        return v


class DerivativeOrderResponse(BaseModel):
    """Derivative order response"""

    order_id: str = Field(..., description="Order ID")
    user_id: int = Field(..., description="User ID")
    derivative_id: str = Field(..., description="Derivative ID")
    order_type: str = Field(..., description="Order type")
    side: str = Field(..., description="Order side")
    quantity: float = Field(..., description="Order quantity")
    price: Optional[float] = Field(None, description="Limit price")
    stop_price: Optional[float] = Field(None, description="Stop price")
    time_in_force: str = Field(..., description="Time in force")
    status: str = Field(..., description="Order status")
    filled_quantity: float = Field(..., description="Filled quantity")
    average_fill_price: float = Field(..., description="Average fill price")
    commission: float = Field(..., description="Commission")
    created_at: datetime = Field(..., description="Created timestamp")
    updated_at: datetime = Field(..., description="Updated timestamp")


# Risk Management Schemas
class RiskLimitCreate(BaseModel):
    """Create risk limit request"""

    user_id: int = Field(..., description="User ID", gt=0)
    risk_type: str = Field(..., description="Risk type")
    limit_name: str = Field(..., description="Limit name")
    limit_value: float = Field(..., description="Limit value", gt=0)
    limit_type: str = Field(
        "absolute", description="Limit type", regex="^(absolute|percentage|var)$"
    )
    time_horizon: str = Field(
        "daily", description="Time horizon", regex="^(intraday|daily|weekly|monthly)$"
    )


class RiskLimitResponse(BaseModel):
    """Risk limit response"""

    limit_id: str = Field(..., description="Limit ID")
    user_id: int = Field(..., description="User ID")
    risk_type: str = Field(..., description="Risk type")
    limit_name: str = Field(..., description="Limit name")
    limit_value: float = Field(..., description="Limit value")
    current_value: float = Field(..., description="Current value")
    limit_type: str = Field(..., description="Limit type")
    time_horizon: str = Field(..., description="Time horizon")
    is_active: bool = Field(..., description="Is active")
    breach_count: int = Field(..., description="Breach count")
    last_breach: Optional[datetime] = Field(None, description="Last breach timestamp")
    created_at: datetime = Field(..., description="Created timestamp")
    last_updated: datetime = Field(..., description="Last updated timestamp")


class RiskMetricResponse(BaseModel):
    """Risk metric response"""

    metric_id: str = Field(..., description="Metric ID")
    user_id: int = Field(..., description="User ID")
    risk_type: str = Field(..., description="Risk type")
    metric_name: str = Field(..., description="Metric name")
    metric_value: float = Field(..., description="Metric value")
    threshold: float = Field(..., description="Threshold value")
    is_breached: bool = Field(..., description="Is breached")
    confidence_level: float = Field(..., description="Confidence level")
    time_horizon: str = Field(..., description="Time horizon")
    calculation_method: str = Field(..., description="Calculation method")
    timestamp: datetime = Field(..., description="Timestamp")


class StressTestRequest(BaseModel):
    """Stress test request"""

    test_type: str = Field(..., description="Test type")
    test_name: str = Field(..., description="Test name")
    scenarios: List[Dict[str, Any]] = Field(..., description="Stress scenarios")


class StressTestResponse(BaseModel):
    """Stress test response"""

    test_id: str = Field(..., description="Test ID")
    user_id: int = Field(..., description="User ID")
    test_type: str = Field(..., description="Test type")
    test_name: str = Field(..., description="Test name")
    portfolio_value: float = Field(..., description="Portfolio value")
    stress_scenarios: List[Dict[str, Any]] = Field(..., description="Stress scenarios")
    results: Dict[str, Any] = Field(..., description="Test results")
    max_loss: float = Field(..., description="Maximum loss")
    var_95: float = Field(..., description="95% VaR")
    var_99: float = Field(..., description="99% VaR")
    expected_shortfall: float = Field(..., description="Expected shortfall")
    created_at: datetime = Field(..., description="Created timestamp")


class RiskReportResponse(BaseModel):
    """Risk report response"""

    report_id: str = Field(..., description="Report ID")
    user_id: int = Field(..., description="User ID")
    report_type: str = Field(..., description="Report type")
    report_date: datetime = Field(..., description="Report date")
    summary: Dict[str, Any] = Field(..., description="Report summary")
    risk_metrics: List[RiskMetricResponse] = Field(..., description="Risk metrics")
    risk_limits: List[RiskLimitResponse] = Field(..., description="Risk limits")
    stress_tests: List[StressTestResponse] = Field(..., description="Stress tests")
    recommendations: List[str] = Field(..., description="Recommendations")
    created_at: datetime = Field(..., description="Created timestamp")


class PortfolioRiskResponse(BaseModel):
    """Portfolio risk response"""

    user_id: int = Field(..., description="User ID")
    timestamp: datetime = Field(..., description="Timestamp")
    total_exposure: float = Field(..., description="Total exposure")
    net_exposure: float = Field(..., description="Net exposure")
    gross_exposure: float = Field(..., description="Gross exposure")
    leverage: float = Field(..., description="Leverage")
    var_95: float = Field(..., description="95% VaR")
    var_99: float = Field(..., description="99% VaR")
    expected_shortfall: float = Field(..., description="Expected shortfall")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    calmar_ratio: float = Field(..., description="Calmar ratio")
    portfolio_greeks: Dict[str, float] = Field(..., description="Portfolio Greeks")
    concentration_risk: Dict[str, Any] = Field(..., description="Concentration risk")
    correlation_risk: Dict[str, Any] = Field(..., description="Correlation risk")


# Analytics Schemas
class DerivativesSummaryResponse(BaseModel):
    """Derivatives summary response"""

    total_derivatives: int = Field(..., description="Total derivatives")
    active_derivatives: int = Field(..., description="Active derivatives")
    type_distribution: Dict[str, int] = Field(..., description="Type distribution")
    underlying_distribution: Dict[str, int] = Field(
        ..., description="Underlying distribution"
    )
    timestamp: datetime = Field(..., description="Timestamp")


class RiskSummaryResponse(BaseModel):
    """Risk summary response"""

    total_users: int = Field(..., description="Total users")
    total_risk_limits: int = Field(..., description="Total risk limits")
    active_risk_limits: int = Field(..., description="Active risk limits")
    total_breaches: int = Field(..., description="Total breaches")
    timestamp: datetime = Field(..., description="Timestamp")


# WebSocket Schemas
class WebSocketMessage(BaseModel):
    """WebSocket message"""

    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: str = Field(..., description="Timestamp")


class PortfolioGreeksUpdate(BaseModel):
    """Portfolio Greeks update"""

    user_id: int = Field(..., description="User ID")
    portfolio_greeks: Dict[str, float] = Field(..., description="Portfolio Greeks")
    timestamp: datetime = Field(..., description="Timestamp")


class PriceUpdate(BaseModel):
    """Price update"""

    derivative_id: str = Field(..., description="Derivative ID")
    price_data: DerivativePriceResponse = Field(..., description="Price data")
    timestamp: datetime = Field(..., description="Timestamp")


class RiskAlert(BaseModel):
    """Risk alert"""

    user_id: int = Field(..., description="User ID")
    alert_type: str = Field(..., description="Alert type")
    message: str = Field(..., description="Alert message")
    severity: str = Field(..., description="Alert severity")
    timestamp: datetime = Field(..., description="Timestamp")


# Error Schemas
class ErrorResponse(BaseModel):
    """Error response"""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error detail")
    timestamp: datetime = Field(..., description="Timestamp")


class ValidationErrorResponse(BaseModel):
    """Validation error response"""

    errors: List[Dict[str, Any]] = Field(..., description="Validation errors")
    timestamp: datetime = Field(..., description="Timestamp")
