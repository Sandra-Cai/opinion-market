"""
Pydantic schemas for Risk Management API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class RiskProfileRequest(BaseModel):
    """Request model for creating a risk profile"""

    risk_tolerance: str = Field(
        default="moderate",
        description="Risk tolerance (conservative, moderate, aggressive)",
    )
    max_portfolio_risk: float = Field(
        default=0.02, ge=0.001, le=0.1, description="Maximum portfolio risk percentage"
    )
    max_position_size: float = Field(
        default=0.05, ge=0.01, le=0.5, description="Maximum position size percentage"
    )
    max_drawdown: float = Field(
        default=0.15, ge=0.05, le=0.5, description="Maximum acceptable drawdown"
    )
    stop_loss_percentage: float = Field(
        default=0.10, ge=0.01, le=0.5, description="Default stop loss percentage"
    )
    take_profit_percentage: float = Field(
        default=0.20, ge=0.01, le=1.0, description="Default take profit percentage"
    )
    correlation_threshold: float = Field(
        default=0.7, ge=0.1, le=1.0, description="Maximum correlation between positions"
    )
    volatility_preference: str = Field(
        default="medium", description="Volatility preference (low, medium, high)"
    )

    class Config:
        schema_extra = {
            "example": {
                "risk_tolerance": "moderate",
                "max_portfolio_risk": 0.02,
                "max_position_size": 0.05,
                "max_drawdown": 0.15,
                "stop_loss_percentage": 0.10,
                "take_profit_percentage": 0.20,
                "correlation_threshold": 0.7,
                "volatility_preference": "medium",
            }
        }


class RiskProfileResponse(BaseModel):
    """Response model for risk profile"""

    user_id: int = Field(..., description="User ID")
    risk_tolerance: str = Field(..., description="Risk tolerance")
    max_portfolio_risk: float = Field(
        ..., description="Maximum portfolio risk percentage"
    )
    max_position_size: float = Field(
        ..., description="Maximum position size percentage"
    )
    max_drawdown: float = Field(..., description="Maximum acceptable drawdown")
    stop_loss_percentage: float = Field(..., description="Default stop loss percentage")
    take_profit_percentage: float = Field(
        ..., description="Default take profit percentage"
    )
    correlation_threshold: float = Field(
        ..., description="Maximum correlation between positions"
    )
    volatility_preference: str = Field(..., description="Volatility preference")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "risk_tolerance": "moderate",
                "max_portfolio_risk": 0.02,
                "max_position_size": 0.05,
                "max_drawdown": 0.15,
                "stop_loss_percentage": 0.10,
                "take_profit_percentage": 0.20,
                "correlation_threshold": 0.7,
                "volatility_preference": "medium",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        }


class PositionRiskRequest(BaseModel):
    """Request model for calculating position risk"""

    market_id: int = Field(..., description="Market ID")
    position_size: float = Field(..., gt=0, description="Position size")
    entry_price: float = Field(..., gt=0, description="Entry price")
    current_price: float = Field(..., gt=0, description="Current price")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "position_size": 100.0,
                "entry_price": 50.0,
                "current_price": 52.0,
            }
        }


class PositionRiskResponse(BaseModel):
    """Response model for position risk"""

    position_id: str = Field(..., description="Position ID")
    user_id: int = Field(..., description="User ID")
    market_id: int = Field(..., description="Market ID")
    position_size: float = Field(..., description="Position size")
    entry_price: float = Field(..., description="Entry price")
    current_price: float = Field(..., description="Current price")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    risk_amount: float = Field(..., description="Risk amount")
    risk_percentage: float = Field(..., description="Risk percentage")
    var_95: float = Field(..., description="Value at Risk (95% confidence)")
    var_99: float = Field(..., description="Value at Risk (99% confidence)")
    expected_shortfall: float = Field(..., description="Expected shortfall")
    beta: float = Field(..., description="Market beta")
    volatility: float = Field(..., description="Volatility")
    correlation_score: float = Field(..., description="Correlation score")
    risk_score: float = Field(..., description="Overall risk score (0-100)")
    last_updated: datetime = Field(..., description="Last update timestamp")

    class Config:
        schema_extra = {
            "example": {
                "position_id": "pos_1_1_20240115_103000",
                "user_id": 1,
                "market_id": 1,
                "position_size": 100.0,
                "entry_price": 50.0,
                "current_price": 52.0,
                "unrealized_pnl": 200.0,
                "risk_amount": 5000.0,
                "risk_percentage": 0.05,
                "var_95": -250.0,
                "var_99": -400.0,
                "expected_shortfall": -300.0,
                "beta": 1.2,
                "volatility": 0.25,
                "correlation_score": 0.3,
                "risk_score": 45.5,
                "last_updated": "2024-01-15T10:30:00Z",
            }
        }


class PortfolioRiskResponse(BaseModel):
    """Response model for portfolio risk"""

    user_id: int = Field(..., description="User ID")
    total_value: float = Field(..., description="Total portfolio value")
    total_risk: float = Field(..., description="Total portfolio risk")
    portfolio_var_95: float = Field(..., description="Portfolio VaR (95% confidence)")
    portfolio_var_99: float = Field(..., description="Portfolio VaR (99% confidence)")
    expected_shortfall: float = Field(..., description="Expected shortfall")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    current_drawdown: float = Field(..., description="Current drawdown")
    diversification_score: float = Field(..., description="Diversification score")
    concentration_risk: float = Field(..., description="Concentration risk")
    correlation_risk: float = Field(..., description="Correlation risk")
    volatility_risk: float = Field(..., description="Volatility risk")
    overall_risk_score: float = Field(..., description="Overall risk score (0-100)")
    risk_level: str = Field(..., description="Risk level (low, medium, high, critical)")
    recommendations: List[str] = Field(
        ..., description="Risk management recommendations"
    )
    last_updated: datetime = Field(..., description="Last update timestamp")

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "total_value": 100000.0,
                "total_risk": 2000.0,
                "portfolio_var_95": -1500.0,
                "portfolio_var_99": -2500.0,
                "expected_shortfall": -1800.0,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.5,
                "max_drawdown": 0.12,
                "current_drawdown": 0.08,
                "diversification_score": 0.75,
                "concentration_risk": 0.3,
                "correlation_risk": 0.4,
                "volatility_risk": 0.22,
                "overall_risk_score": 45.5,
                "risk_level": "medium",
                "recommendations": [
                    "Consider diversifying across more markets",
                    "Portfolio risk is well-managed",
                ],
                "last_updated": "2024-01-15T10:30:00Z",
            }
        }


class PositionLimitCheckResponse(BaseModel):
    """Response model for position limit check"""

    allowed: bool = Field(..., description="Whether position is allowed")
    reason: str = Field(..., description="Reason for decision")
    max_allowed: Optional[float] = Field(
        None, description="Maximum allowed position size"
    )
    max_allowed_risk: Optional[float] = Field(
        None, description="Maximum allowed portfolio risk"
    )
    timestamp: datetime = Field(..., description="Check timestamp")

    class Config:
        schema_extra = {
            "example": {
                "allowed": True,
                "reason": "Position within risk limits",
                "max_allowed": 5000.0,
                "max_allowed_risk": 2000.0,
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class RiskAlertResponse(BaseModel):
    """Response model for risk alert"""

    alert_id: str = Field(..., description="Alert ID")
    user_id: int = Field(..., description="User ID")
    alert_type: str = Field(..., description="Alert type")
    severity: str = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    current_value: float = Field(..., description="Current value")
    threshold_value: float = Field(..., description="Threshold value")
    position_id: Optional[str] = Field(None, description="Related position ID")
    market_id: Optional[int] = Field(None, description="Related market ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    resolved: bool = Field(..., description="Whether alert is resolved")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")

    class Config:
        schema_extra = {
            "example": {
                "alert_id": "alert_1_drawdown_20240115_103000",
                "user_id": 1,
                "alert_type": "drawdown",
                "severity": "high",
                "message": "Current drawdown 18% exceeds limit 15%",
                "current_value": 0.18,
                "threshold_value": 0.15,
                "position_id": None,
                "market_id": None,
                "created_at": "2024-01-15T10:30:00Z",
                "resolved": False,
                "resolved_at": None,
            }
        }


class RiskDashboardResponse(BaseModel):
    """Response model for risk dashboard"""

    user_id: int = Field(..., description="User ID")
    risk_profile: Dict[str, Any] = Field(..., description="Risk profile")
    portfolio_risk: Dict[str, Any] = Field(..., description="Portfolio risk metrics")
    position_risks: List[Dict[str, Any]] = Field(
        ..., description="Position risk metrics"
    )
    risk_alerts: List[Dict[str, Any]] = Field(..., description="Active risk alerts")
    risk_metrics: Dict[str, Any] = Field(..., description="Risk metrics summary")
    recommendations: List[str] = Field(
        ..., description="Risk management recommendations"
    )
    last_updated: str = Field(..., description="Last update timestamp")

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "risk_profile": {
                    "risk_tolerance": "moderate",
                    "max_portfolio_risk": 0.02,
                    "max_position_size": 0.05,
                    "max_drawdown": 0.15,
                },
                "portfolio_risk": {
                    "total_value": 100000.0,
                    "portfolio_var_95": -1500.0,
                    "overall_risk_score": 45.5,
                    "risk_level": "medium",
                },
                "position_risks": [
                    {
                        "position_id": "pos_1_1_20240115_103000",
                        "market_id": 1,
                        "risk_score": 45.5,
                        "var_95": -250.0,
                    }
                ],
                "risk_alerts": [],
                "risk_metrics": {
                    "total_positions": 5,
                    "high_risk_positions": 1,
                    "active_alerts": 0,
                },
                "recommendations": ["Portfolio risk is well-managed"],
                "last_updated": "2024-01-15T10:30:00Z",
            }
        }


class VaRAnalyticsResponse(BaseModel):
    """Response model for VaR analytics"""

    user_id: int = Field(..., description="User ID")
    confidence_level: float = Field(..., description="VaR confidence level")
    time_horizon_days: int = Field(..., description="Time horizon in days")
    var_value: float = Field(..., description="VaR value")
    var_percentage: float = Field(..., description="VaR as percentage of portfolio")
    expected_shortfall: float = Field(..., description="Expected shortfall")
    portfolio_value: float = Field(..., description="Portfolio value")
    risk_level: str = Field(..., description="Risk level")
    timestamp: str = Field(..., description="Analysis timestamp")

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "confidence_level": 0.95,
                "time_horizon_days": 1,
                "var_value": -1500.0,
                "var_percentage": 1.5,
                "expected_shortfall": -1800.0,
                "portfolio_value": 100000.0,
                "risk_level": "medium",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class DrawdownAnalyticsResponse(BaseModel):
    """Response model for drawdown analytics"""

    user_id: int = Field(..., description="User ID")
    period_days: int = Field(..., description="Analysis period in days")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    current_drawdown: float = Field(..., description="Current drawdown")
    max_drawdown_percentage: float = Field(
        ..., description="Maximum drawdown percentage"
    )
    current_drawdown_percentage: float = Field(
        ..., description="Current drawdown percentage"
    )
    drawdown_limit: float = Field(..., description="Drawdown limit")
    drawdown_status: str = Field(..., description="Drawdown status")
    portfolio_value: float = Field(..., description="Portfolio value")
    timestamp: str = Field(..., description="Analysis timestamp")

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "period_days": 30,
                "max_drawdown": 0.12,
                "current_drawdown": 0.08,
                "max_drawdown_percentage": 12.0,
                "current_drawdown_percentage": 8.0,
                "drawdown_limit": 0.15,
                "drawdown_status": "within_limits",
                "portfolio_value": 100000.0,
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class CorrelationAnalyticsResponse(BaseModel):
    """Response model for correlation analytics"""

    user_id: int = Field(..., description="User ID")
    correlation_risk: float = Field(..., description="Correlation risk")
    correlation_percentage: float = Field(..., description="Correlation percentage")
    correlation_threshold: float = Field(..., description="Correlation threshold")
    correlation_status: str = Field(..., description="Correlation status")
    diversification_score: float = Field(..., description="Diversification score")
    diversification_percentage: float = Field(
        ..., description="Diversification percentage"
    )
    recommendations: List[str] = Field(..., description="Recommendations")
    timestamp: str = Field(..., description="Analysis timestamp")

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "correlation_risk": 0.4,
                "correlation_percentage": 40.0,
                "correlation_threshold": 0.7,
                "correlation_status": "low",
                "diversification_score": 0.75,
                "diversification_percentage": 75.0,
                "recommendations": ["Portfolio is well diversified"],
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class VolatilityAnalyticsResponse(BaseModel):
    """Response model for volatility analytics"""

    user_id: int = Field(..., description="User ID")
    volatility_risk: float = Field(..., description="Volatility risk")
    volatility_percentage: float = Field(..., description="Volatility percentage")
    annualized_volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    risk_adjusted_return: float = Field(..., description="Risk-adjusted return")
    volatility_status: str = Field(..., description="Volatility status")
    timestamp: str = Field(..., description="Analysis timestamp")

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "volatility_risk": 0.22,
                "volatility_percentage": 22.0,
                "annualized_volatility": 0.35,
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.5,
                "risk_adjusted_return": 1.2,
                "volatility_status": "medium",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class RiskSimulationRequest(BaseModel):
    """Request model for risk simulation"""

    market_id: int = Field(..., description="Market ID")
    position_size: float = Field(..., gt=0, description="Position size")
    entry_price: float = Field(..., gt=0, description="Entry price")
    scenarios: int = Field(
        default=1000, ge=100, le=10000, description="Number of simulation scenarios"
    )

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "position_size": 100.0,
                "entry_price": 50.0,
                "scenarios": 1000,
            }
        }


class RiskSimulationResponse(BaseModel):
    """Response model for risk simulation"""

    market_id: int = Field(..., description="Market ID")
    position_size: float = Field(..., description="Position size")
    entry_price: float = Field(..., description="Entry price")
    scenarios: int = Field(..., description="Number of scenarios")
    simulation_results: Dict[str, Any] = Field(..., description="Simulation results")
    price_distribution: Dict[str, Any] = Field(..., description="Price distribution")
    timestamp: str = Field(..., description="Simulation timestamp")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "position_size": 100.0,
                "entry_price": 50.0,
                "scenarios": 1000,
                "simulation_results": {
                    "var_95": -250.0,
                    "var_99": -400.0,
                    "expected_shortfall": -300.0,
                    "probability_of_loss": 0.45,
                    "max_profit": 500.0,
                    "max_loss": -800.0,
                    "average_pnl": 50.0,
                    "pnl_volatility": 200.0,
                },
                "price_distribution": {
                    "min_price": 45.0,
                    "max_price": 58.0,
                    "average_price": 50.5,
                    "price_volatility": 2.5,
                },
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }


class RiskManagementHealthResponse(BaseModel):
    """Response model for risk management health check"""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: str = Field(..., description="Health check timestamp")
    features: List[str] = Field(..., description="Available features")
    metrics: Dict[str, Any] = Field(..., description="Service metrics")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "risk_management",
                "timestamp": "2024-01-15T10:30:00Z",
                "features": [
                    "risk_profiles",
                    "position_risk_calculation",
                    "portfolio_risk_assessment",
                    "var_analytics",
                    "drawdown_monitoring",
                    "correlation_analysis",
                    "volatility_tracking",
                    "risk_alerts",
                    "position_limit_checks",
                    "websocket_alerts",
                ],
                "metrics": {
                    "active_risk_profiles": 1500,
                    "total_position_risks": 2500,
                    "active_portfolio_risks": 1500,
                    "active_alerts": 50,
                },
            }
        }


class RiskManagementStatsResponse(BaseModel):
    """Response model for risk management statistics"""

    total_risk_profiles: int = Field(..., description="Total number of risk profiles")
    total_position_risks: int = Field(..., description="Total number of position risks")
    total_portfolio_risks: int = Field(
        ..., description="Total number of portfolio risks"
    )
    total_alerts: int = Field(..., description="Total number of alerts")
    active_alerts: int = Field(..., description="Number of active alerts")
    resolved_alerts: int = Field(..., description="Number of resolved alerts")
    risk_level_distribution: Dict[str, int] = Field(
        ..., description="Risk level distribution"
    )
    alert_severity_distribution: Dict[str, int] = Field(
        ..., description="Alert severity distribution"
    )
    average_risk_scores: Dict[str, float] = Field(
        ..., description="Average risk scores"
    )

    class Config:
        schema_extra = {
            "example": {
                "total_risk_profiles": 1500,
                "total_position_risks": 2500,
                "total_portfolio_risks": 1500,
                "total_alerts": 200,
                "active_alerts": 50,
                "resolved_alerts": 150,
                "risk_level_distribution": {
                    "low": 500,
                    "medium": 800,
                    "high": 150,
                    "critical": 50,
                },
                "alert_severity_distribution": {
                    "low": 100,
                    "medium": 50,
                    "high": 30,
                    "critical": 20,
                },
                "average_risk_scores": {"position_risk": 45.5, "portfolio_risk": 42.3},
            }
        }


class WebSocketRiskAlertMessage(BaseModel):
    """Base model for WebSocket risk alert messages"""

    type: str = Field(..., description="Message type")
    timestamp: str = Field(..., description="Message timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")


class WebSocketUserAlertSubscription(BaseModel):
    """Request model for WebSocket user alert subscription"""

    type: str = Field("subscribe_user_alerts", description="Message type")
    user_id: int = Field(..., description="User ID to subscribe to")


class WebSocketPortfolioRiskRequest(BaseModel):
    """Request model for WebSocket portfolio risk updates"""

    type: str = Field("get_portfolio_risk", description="Message type")
    user_id: int = Field(..., description="User ID")
