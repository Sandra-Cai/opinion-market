"""
Pydantic schemas for Quantitative Trading API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class TradingStrategyRequest(BaseModel):
    """Request model for creating a trading strategy"""

    strategy_name: str = Field(..., description="Strategy name")
    strategy_type: str = Field(
        ..., description="Strategy type (momentum, mean_reversion, arbitrage, ml_based)"
    )
    parameters: Dict[str, Any] = Field(default={}, description="Strategy parameters")
    indicators: List[str] = Field(default=[], description="Technical indicators to use")
    timeframes: List[str] = Field(default=[], description="Trading timeframes")
    markets: List[int] = Field(default=[], description="Target markets")
    risk_management: Dict[str, Any] = Field(
        default={}, description="Risk management rules"
    )

    class Config:
        schema_extra = {
            "example": {
                "strategy_name": "Momentum Breakout Strategy",
                "strategy_type": "momentum",
                "parameters": {
                    "rsi_period": 14,
                    "sma_period": 20,
                    "volume_threshold": 1.5,
                },
                "indicators": ["rsi", "sma", "volume_sma"],
                "timeframes": ["1h", "4h", "1d"],
                "markets": [1, 2, 3],
                "risk_management": {
                    "max_position_size": 0.1,
                    "stop_loss": 0.05,
                    "take_profit": 0.15,
                },
            }
        }


class TradingStrategyResponse(BaseModel):
    """Response model for trading strategy"""

    strategy_id: str = Field(..., description="Strategy ID")
    strategy_name: str = Field(..., description="Strategy name")
    strategy_type: str = Field(..., description="Strategy type")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    indicators: List[str] = Field(..., description="Technical indicators")
    timeframes: List[str] = Field(..., description="Trading timeframes")
    markets: List[int] = Field(..., description="Target markets")
    risk_management: Dict[str, Any] = Field(..., description="Risk management rules")
    performance_metrics: Dict[str, float] = Field(
        ..., description="Performance metrics"
    )
    is_active: bool = Field(..., description="Active status")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")

    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "strategy_momentum_20240115_103000",
                "strategy_name": "Momentum Breakout Strategy",
                "strategy_type": "momentum",
                "parameters": {
                    "rsi_period": 14,
                    "sma_period": 20,
                    "volume_threshold": 1.5,
                },
                "indicators": ["rsi", "sma", "volume_sma"],
                "timeframes": ["1h", "4h", "1d"],
                "markets": [1, 2, 3],
                "risk_management": {
                    "max_position_size": 0.1,
                    "stop_loss": 0.05,
                    "take_profit": 0.15,
                },
                "performance_metrics": {
                    "total_return": 0.25,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": 0.08,
                    "win_rate": 0.65,
                    "profit_factor": 1.8,
                },
                "is_active": True,
                "created_at": "2024-01-15T10:30:00Z",
                "last_updated": "2024-01-15T10:30:00Z",
            }
        }


class SignalResponse(BaseModel):
    """Response model for trading signal"""

    signal_id: str = Field(..., description="Signal ID")
    strategy_id: str = Field(..., description="Strategy ID")
    market_id: int = Field(..., description="Market ID")
    signal_type: str = Field(..., description="Signal type (buy, sell, hold)")
    strength: float = Field(..., ge=0.0, le=1.0, description="Signal strength")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level")
    price_target: Optional[float] = Field(None, description="Price target")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    reasoning: str = Field(..., description="Signal reasoning")
    indicators: Dict[str, float] = Field(..., description="Indicator values")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: datetime = Field(..., description="Expiration timestamp")

    class Config:
        schema_extra = {
            "example": {
                "signal_id": "signal_strategy_1_20240115_103000",
                "strategy_id": "strategy_momentum_20240115_103000",
                "market_id": 1,
                "signal_type": "buy",
                "strength": 0.8,
                "confidence": 0.75,
                "price_target": 50000.0,
                "stop_loss": 45000.0,
                "take_profit": 55000.0,
                "reasoning": "Strong momentum with price above moving averages",
                "indicators": {"rsi": 65.5, "sma_20": 48000.0, "volume_sma": 1250000.0},
                "created_at": "2024-01-15T10:30:00Z",
                "expires_at": "2024-01-16T10:30:00Z",
            }
        }


class BacktestRequest(BaseModel):
    """Request model for running backtest"""

    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(default=100000.0, description="Initial capital")

    class Config:
        schema_extra = {
            "example": {
                "start_date": "2023-01-01T00:00:00Z",
                "end_date": "2023-12-31T23:59:59Z",
                "initial_capital": 100000.0,
            }
        }


class BacktestResponse(BaseModel):
    """Response model for backtest result"""

    backtest_id: str = Field(..., description="Backtest ID")
    strategy_id: str = Field(..., description="Strategy ID")
    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")
    initial_capital: float = Field(..., description="Initial capital")
    final_capital: float = Field(..., description="Final capital")
    total_return: float = Field(..., description="Total return")
    annualized_return: float = Field(..., description="Annualized return")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    win_rate: float = Field(..., description="Win rate")
    profit_factor: float = Field(..., description="Profit factor")
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    avg_win: float = Field(..., description="Average win")
    avg_loss: float = Field(..., description="Average loss")
    trade_history: List[Dict[str, Any]] = Field(..., description="Trade history")
    equity_curve: List[Dict[str, Any]] = Field(..., description="Equity curve")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        schema_extra = {
            "example": {
                "backtest_id": "backtest_strategy_1_20240115_103000",
                "strategy_id": "strategy_momentum_20240115_103000",
                "start_date": "2023-01-01T00:00:00Z",
                "end_date": "2023-12-31T23:59:59Z",
                "initial_capital": 100000.0,
                "final_capital": 125000.0,
                "total_return": 0.25,
                "annualized_return": 0.25,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.08,
                "win_rate": 0.65,
                "profit_factor": 1.8,
                "total_trades": 150,
                "winning_trades": 98,
                "losing_trades": 52,
                "avg_win": 2500.0,
                "avg_loss": 1500.0,
                "trade_history": [
                    {
                        "date": "2023-01-15T10:30:00Z",
                        "market_id": 1,
                        "signal_type": "buy",
                        "price": 48000.0,
                        "quantity": 2,
                        "pnl": 2500.0,
                    }
                ],
                "equity_curve": [
                    {
                        "date": "2023-01-01T00:00:00Z",
                        "equity": 100000.0,
                        "capital": 100000.0,
                        "positions": 0,
                    }
                ],
                "created_at": "2024-01-15T10:30:00Z",
            }
        }


class PortfolioOptimizationRequest(BaseModel):
    """Request model for portfolio optimization"""

    optimization_type: str = Field(
        ...,
        description="Optimization type (sharpe, min_variance, max_return, black_litterman)",
    )
    target_return: Optional[float] = Field(None, description="Target return")
    risk_tolerance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Risk tolerance"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        None, description="Optimization constraints"
    )

    class Config:
        schema_extra = {
            "example": {
                "optimization_type": "sharpe",
                "target_return": 0.15,
                "risk_tolerance": 0.5,
                "constraints": {"max_weight": 0.3, "min_weight": 0.05},
            }
        }


class PortfolioOptimizationResponse(BaseModel):
    """Response model for portfolio optimization"""

    optimization_id: str = Field(..., description="Optimization ID")
    user_id: int = Field(..., description="User ID")
    optimization_type: str = Field(..., description="Optimization type")
    target_return: Optional[float] = Field(None, description="Target return")
    risk_tolerance: float = Field(..., description="Risk tolerance")
    constraints: Dict[str, Any] = Field(..., description="Optimization constraints")
    optimal_weights: Dict[int, float] = Field(..., description="Optimal asset weights")
    expected_return: float = Field(..., description="Expected return")
    expected_volatility: float = Field(..., description="Expected volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    efficient_frontier: List[Dict[str, float]] = Field(
        ..., description="Efficient frontier points"
    )
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        schema_extra = {
            "example": {
                "optimization_id": "optimization_user_1_20240115_103000",
                "user_id": 1,
                "optimization_type": "sharpe",
                "target_return": 0.15,
                "risk_tolerance": 0.5,
                "constraints": {"max_weight": 0.3, "min_weight": 0.05},
                "optimal_weights": {1: 0.25, 2: 0.30, 3: 0.20, 4: 0.15, 5: 0.10},
                "expected_return": 0.15,
                "expected_volatility": 0.12,
                "sharpe_ratio": 1.25,
                "efficient_frontier": [
                    {"return": 0.10, "volatility": 0.08, "sharpe": 1.25},
                    {"return": 0.15, "volatility": 0.12, "sharpe": 1.25},
                    {"return": 0.20, "volatility": 0.18, "sharpe": 1.11},
                ],
                "created_at": "2024-01-15T10:30:00Z",
            }
        }


class TechnicalIndicatorsResponse(BaseModel):
    """Response model for technical indicators"""

    market_id: int = Field(..., description="Market ID")
    indicators: Dict[str, Optional[float]] = Field(..., description="Indicator values")
    calculated_at: datetime = Field(..., description="Calculation timestamp")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "indicators": {
                    "rsi": 65.5,
                    "sma_20": 48000.0,
                    "sma_50": 47000.0,
                    "macd": 500.0,
                    "macd_signal": 450.0,
                    "bb_upper": 52000.0,
                    "bb_middle": 48000.0,
                    "bb_lower": 44000.0,
                    "atr": 2000.0,
                },
                "calculated_at": "2024-01-15T10:30:00Z",
            }
        }


class StrategyUpdateRequest(BaseModel):
    """Request model for updating strategy"""

    strategy_name: Optional[str] = Field(None, description="Strategy name")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Strategy parameters"
    )
    indicators: Optional[List[str]] = Field(None, description="Technical indicators")
    timeframes: Optional[List[str]] = Field(None, description="Trading timeframes")
    markets: Optional[List[int]] = Field(None, description="Target markets")
    risk_management: Optional[Dict[str, Any]] = Field(
        None, description="Risk management rules"
    )
    is_active: Optional[bool] = Field(None, description="Active status")

    class Config:
        schema_extra = {
            "example": {
                "strategy_name": "Updated Momentum Strategy",
                "parameters": {
                    "rsi_period": 21,
                    "sma_period": 50,
                    "volume_threshold": 2.0,
                },
                "is_active": True,
            }
        }


class SignalFilterRequest(BaseModel):
    """Request model for filtering signals"""

    strategy_id: Optional[str] = Field(None, description="Filter by strategy ID")
    market_id: Optional[int] = Field(None, description="Filter by market ID")
    signal_type: Optional[str] = Field(None, description="Filter by signal type")
    min_strength: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum signal strength"
    )
    min_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum confidence"
    )
    limit: int = Field(default=20, description="Number of signals to return")

    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "strategy_momentum_20240115_103000",
                "signal_type": "buy",
                "min_strength": 0.7,
                "min_confidence": 0.6,
                "limit": 20,
            }
        }


class BacktestFilterRequest(BaseModel):
    """Request model for filtering backtest results"""

    strategy_id: Optional[str] = Field(None, description="Filter by strategy ID")
    min_return: Optional[float] = Field(None, description="Minimum total return")
    min_sharpe: Optional[float] = Field(None, description="Minimum Sharpe ratio")
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    limit: int = Field(default=20, description="Number of backtest results to return")

    class Config:
        schema_extra = {
            "example": {
                "strategy_id": "strategy_momentum_20240115_103000",
                "min_return": 0.1,
                "min_sharpe": 1.0,
                "max_drawdown": 0.15,
                "limit": 20,
            }
        }


class WebSocketQuantitativeTradingMessage(BaseModel):
    """Base model for WebSocket quantitative trading messages"""

    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(..., description="Message timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")


class WebSocketStrategySubscription(BaseModel):
    """Request model for WebSocket strategy subscription"""

    type: str = Field("subscribe_strategy_updates", description="Message type")
    strategy_id: str = Field(..., description="Strategy ID to subscribe to")


class WebSocketSignalSubscription(BaseModel):
    """Request model for WebSocket signal subscription"""

    type: str = Field("subscribe_signals", description="Message type")


class WebSocketBacktestRequest(BaseModel):
    """Request model for WebSocket backtest updates"""

    type: str = Field("get_backtest_updates", description="Message type")
    strategy_id: str = Field(..., description="Strategy ID")


class QuantitativeTradingHealthResponse(BaseModel):
    """Response model for quantitative trading health check"""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(..., description="Health check timestamp")
    features: List[str] = Field(..., description="Available features")
    metrics: Dict[str, Any] = Field(..., description="Service metrics")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "quantitative_trading",
                "timestamp": "2024-01-15T10:30:00Z",
                "features": [
                    "trading_strategies",
                    "signal_generation",
                    "backtesting",
                    "portfolio_optimization",
                    "technical_indicators",
                    "real_time_updates",
                ],
                "metrics": {
                    "active_strategies": 15,
                    "total_signals": 250,
                    "completed_backtests": 50,
                    "portfolio_optimizations": 25,
                },
            }
        }


class QuantitativeTradingStatsResponse(BaseModel):
    """Response model for quantitative trading statistics"""

    total_strategies: int = Field(..., description="Total number of strategies")
    active_strategies: int = Field(..., description="Number of active strategies")
    total_signals: int = Field(..., description="Total number of signals")
    total_backtests: int = Field(..., description="Total number of backtests")
    total_optimizations: int = Field(..., description="Total number of optimizations")
    active_signals: int = Field(..., description="Number of active signals")
    top_performing_strategies: List[Dict[str, Any]] = Field(
        ..., description="Top performing strategies"
    )
    recent_signals: List[Dict[str, Any]] = Field(..., description="Recent signals")

    class Config:
        schema_extra = {
            "example": {
                "total_strategies": 25,
                "active_strategies": 15,
                "total_signals": 250,
                "total_backtests": 50,
                "total_optimizations": 25,
                "active_signals": 45,
                "top_performing_strategies": [
                    {
                        "strategy_id": "strategy_momentum_20240115_103000",
                        "strategy_name": "Momentum Breakout Strategy",
                        "total_return": 0.25,
                        "sharpe_ratio": 1.5,
                    }
                ],
                "recent_signals": [
                    {
                        "signal_id": "signal_strategy_1_20240115_103000",
                        "strategy_id": "strategy_momentum_20240115_103000",
                        "market_id": 1,
                        "signal_type": "buy",
                        "strength": 0.8,
                        "confidence": 0.75,
                    }
                ],
            }
        }
