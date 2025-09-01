"""
Cryptocurrency Trading Schemas
Pydantic models for cryptocurrency trading API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


# Cryptocurrency Schemas
class CryptocurrencyRequest(BaseModel):
    """Request model for creating a cryptocurrency"""
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC, ETH)")
    name: str = Field(..., description="Full name of the cryptocurrency")
    blockchain: str = Field(..., description="Blockchain network (e.g., Bitcoin, Ethereum)")
    contract_address: Optional[str] = Field(None, description="Smart contract address (for tokens)")
    decimals: int = Field(..., description="Number of decimal places")
    total_supply: float = Field(..., description="Total supply of the cryptocurrency")


class CryptocurrencyResponse(BaseModel):
    """Response model for cryptocurrency data"""
    crypto_id: str
    symbol: str
    name: str
    blockchain: str
    contract_address: Optional[str]
    decimals: int
    total_supply: float
    circulating_supply: float
    market_cap: float
    is_active: bool
    created_at: datetime
    last_updated: datetime


# Crypto Price Schemas
class CryptoPriceRequest(BaseModel):
    """Request model for adding cryptocurrency price data"""
    price_usd: float = Field(..., description="Price in USD")
    price_btc: float = Field(..., description="Price in BTC")
    price_eth: float = Field(..., description="Price in ETH")
    volume_24h: float = Field(..., description="24-hour trading volume")
    market_cap: float = Field(..., description="Market capitalization")
    price_change_24h: float = Field(..., description="24-hour price change")
    price_change_percent_24h: float = Field(..., description="24-hour price change percentage")
    high_24h: float = Field(..., description="24-hour high price")
    low_24h: float = Field(..., description="24-hour low price")
    source: str = Field(..., description="Data source")


class CryptoPriceResponse(BaseModel):
    """Response model for cryptocurrency price data"""
    price_id: str
    crypto_id: str
    price_usd: float
    price_btc: float
    price_eth: float
    volume_24h: float
    market_cap: float
    price_change_24h: float
    price_change_percent_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime
    source: str


# Crypto Position Schemas
class CryptoPositionRequest(BaseModel):
    """Request model for creating a cryptocurrency position"""
    crypto_id: str = Field(..., description="Cryptocurrency ID")
    position_type: str = Field(..., description="Position type: 'long' or 'short'")
    size: float = Field(..., description="Position size")
    entry_price: float = Field(..., description="Entry price")
    leverage: float = Field(1.0, description="Leverage multiplier")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")


class CryptoPositionResponse(BaseModel):
    """Response model for cryptocurrency position data"""
    position_id: str
    user_id: int
    crypto_id: str
    position_type: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    margin_used: float
    leverage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    created_at: datetime
    last_updated: datetime


# DeFi Protocol Schemas
class DeFiProtocolRequest(BaseModel):
    """Request model for creating a DeFi protocol"""
    name: str = Field(..., description="Protocol name")
    protocol_type: str = Field(..., description="Protocol type: 'DEX', 'Lending', 'Yield', 'Derivatives'")
    blockchain: str = Field(..., description="Blockchain network")
    tvl: float = Field(..., description="Total Value Locked")
    apy: float = Field(..., description="Annual Percentage Yield")
    risk_score: float = Field(..., description="Risk score (0-1)")


class DeFiProtocolResponse(BaseModel):
    """Response model for DeFi protocol data"""
    protocol_id: str
    name: str
    protocol_type: str
    blockchain: str
    tvl: float
    apy: float
    risk_score: float
    is_active: bool
    created_at: datetime
    last_updated: datetime


# Crypto Order Schemas
class CryptoOrderRequest(BaseModel):
    """Request model for creating a cryptocurrency order"""
    crypto_id: str = Field(..., description="Cryptocurrency ID")
    order_type: str = Field(..., description="Order type: 'market', 'limit', 'stop', 'stop_limit'")
    side: str = Field(..., description="Order side: 'buy' or 'sell'")
    size: float = Field(..., description="Order size")
    price: Optional[float] = Field(None, description="Order price (for limit orders)")
    stop_price: Optional[float] = Field(None, description="Stop price (for stop orders)")
    limit_price: Optional[float] = Field(None, description="Limit price (for stop-limit orders)")
    time_in_force: str = Field("GTC", description="Time in force: 'GTC', 'IOC', 'FOK'")


class CryptoOrderResponse(BaseModel):
    """Response model for cryptocurrency order data"""
    order_id: str
    user_id: int
    crypto_id: str
    order_type: str
    side: str
    size: float
    price: Optional[float]
    stop_price: Optional[float]
    limit_price: Optional[float]
    time_in_force: str
    status: str
    filled_size: float
    filled_price: float
    commission: float
    created_at: datetime
    last_updated: datetime


# Analytics Schemas
class PriceStatistics(BaseModel):
    """Price statistics data"""
    price_mean: Optional[float]
    price_std: Optional[float]
    price_min: Optional[float]
    price_max: Optional[float]
    volume_mean: Optional[float]


class TechnicalIndicators(BaseModel):
    """Technical indicators data"""
    sma_20: Optional[float]
    sma_50: Optional[float]
    sma_200: Optional[float]
    rsi: Optional[float]
    volatility: float


class PriceTrends(BaseModel):
    """Price trends data"""
    price_change: Optional[float]
    price_change_percent: Optional[float]
    trend_direction: Optional[str]
    sma_5: Optional[float]
    sma_10: Optional[float]
    sma_20: Optional[float]
    volatility: float


class DeFiIntegration(BaseModel):
    """DeFi integration data"""
    related_protocols: List[Dict[str, Any]]
    protocol_count: int
    total_tvl_integration: float


class CryptoMetricsResponse(BaseModel):
    """Response model for comprehensive cryptocurrency metrics"""
    crypto_id: str
    symbol: str
    name: str
    blockchain: str
    current_price_usd: Optional[float]
    current_price_btc: Optional[float]
    market_cap: float
    total_supply: float
    circulating_supply: float
    price_statistics: PriceStatistics
    technical_indicators: TechnicalIndicators
    price_trends: PriceTrends
    blockchain_metrics: Dict[str, Any]
    defi_integration: DeFiIntegration
    last_updated: str


# DeFi Analytics Schemas
class ProtocolTypeMetrics(BaseModel):
    """Protocol type metrics data"""
    count: int
    total_tvl: float
    avg_apy: float
    avg_risk_score: float


class TopProtocol(BaseModel):
    """Top protocol data"""
    protocol_id: str
    name: str
    protocol_type: str
    blockchain: str
    tvl: float
    apy: float
    risk_score: float


class BlockchainDistribution(BaseModel):
    """Blockchain distribution data"""
    count: int
    tvl: float
    avg_apy: float


class RiskAnalysis(BaseModel):
    """Risk analysis data"""
    avg_risk_score: float
    min_risk_score: float
    max_risk_score: float
    risk_distribution: Dict[str, int]


class DeFiAnalyticsResponse(BaseModel):
    """Response model for comprehensive DeFi analytics"""
    total_tvl: float
    total_protocols: int
    protocol_types: Dict[str, ProtocolTypeMetrics]
    top_protocols: List[TopProtocol]
    blockchain_distribution: Dict[str, BlockchainDistribution]
    risk_analysis: RiskAnalysis
    last_updated: str


# WebSocket Message Schemas
class CryptoUpdateMessage(BaseModel):
    """WebSocket message for cryptocurrency updates"""
    type: str = "crypto_update"
    crypto_id: str
    timestamp: str
    message: str
    data: Optional[Dict[str, Any]] = None


# Error Response Schemas
class CryptoErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    timestamp: str
    crypto_id: Optional[str] = None


# Health Check Schemas
class CryptoServiceHealth(BaseModel):
    """Cryptocurrency service health status"""
    service: str = "crypto_trading"
    status: str
    active_cryptocurrencies: int
    active_protocols: int
    last_price_update: Optional[str]
    last_updated: str


# Statistics Schemas
class CryptoStats(BaseModel):
    """Cryptocurrency statistics"""
    total_cryptocurrencies: int
    total_market_cap: float
    total_volume_24h: float
    top_gainers: List[Dict[str, Any]]
    top_losers: List[Dict[str, Any]]
    most_traded: List[Dict[str, Any]]
    last_updated: str
