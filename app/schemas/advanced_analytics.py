"""
Pydantic schemas for Advanced Analytics API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime


class CorrelationAnalysisRequest(BaseModel):
    """Request model for correlation analysis"""

    market_ids: List[int] = Field(..., description="List of market IDs to analyze")
    correlation_type: str = Field(
        default="pearson", description="Type of correlation (pearson, spearman)"
    )

    class Config:
        schema_extra = {
            "example": {"market_ids": [1, 2, 3, 4, 5], "correlation_type": "pearson"}
        }


class CorrelationAnalysisResponse(BaseModel):
    """Response model for correlation analysis"""

    market_id_1: int = Field(..., description="First market ID")
    market_id_2: int = Field(..., description="Second market ID")
    correlation_coefficient: float = Field(
        ..., ge=-1.0, le=1.0, description="Correlation coefficient"
    )
    correlation_type: str = Field(..., description="Type of correlation used")
    p_value: float = Field(
        ..., ge=0.0, le=1.0, description="Statistical significance p-value"
    )
    significance_level: str = Field(
        ..., description="Significance level (high, medium, low)"
    )
    sample_size: int = Field(..., description="Number of data points used")
    analysis_date: datetime = Field(..., description="When analysis was performed")

    class Config:
        schema_extra = {
            "example": {
                "market_id_1": 1,
                "market_id_2": 2,
                "correlation_coefficient": 0.75,
                "correlation_type": "pearson",
                "p_value": 0.001,
                "significance_level": "high",
                "sample_size": 30,
                "analysis_date": "2024-01-15T10:30:00Z",
            }
        }


class ClusteringRequest(BaseModel):
    """Request model for market clustering"""

    market_ids: List[int] = Field(..., description="List of market IDs to cluster")
    n_clusters: int = Field(
        default=3, ge=2, le=10, description="Number of clusters to create"
    )

    class Config:
        schema_extra = {
            "example": {"market_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "n_clusters": 3}
        }


class ClusteringResponse(BaseModel):
    """Response model for market clustering"""

    cluster_id: int = Field(..., description="Cluster identifier")
    market_ids: List[int] = Field(..., description="Markets in this cluster")
    cluster_size: int = Field(..., description="Number of markets in cluster")
    cluster_characteristics: Dict[str, Any] = Field(
        ..., description="Cluster characteristics"
    )
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Average similarity within cluster"
    )

    class Config:
        schema_extra = {
            "example": {
                "cluster_id": 0,
                "market_ids": [1, 2, 3],
                "cluster_size": 3,
                "cluster_characteristics": {
                    "cluster_type": "high_activity",
                    "avg_volume": 150000.0,
                    "avg_participants": 200,
                    "avg_volatility": 0.3,
                    "avg_sentiment": 0.6,
                },
                "similarity_score": 0.85,
            }
        }


class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection"""

    market_id: int = Field(..., description="Market ID where anomaly was detected")
    anomaly_type: str = Field(
        ..., description="Type of anomaly (price_spike, volume_surge, sentiment_shift)"
    )
    severity: str = Field(
        ..., description="Anomaly severity (low, medium, high, critical)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    description: str = Field(..., description="Description of the anomaly")
    detected_at: datetime = Field(..., description="When anomaly was detected")
    historical_context: Dict[str, Any] = Field(
        ..., description="Historical context and statistics"
    )

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "anomaly_type": "price_spike",
                "severity": "high",
                "confidence": 0.85,
                "description": "Price change of 0.15 (z-score: 3.2)",
                "detected_at": "2024-01-15T10:30:00Z",
                "historical_context": {
                    "mean_change": 0.02,
                    "std_change": 0.04,
                    "z_score": 3.2,
                    "price_history_length": 30,
                },
            }
        }


class ForecastingRequest(BaseModel):
    """Request model for market forecasting"""

    market_ids: List[int] = Field(..., description="List of market IDs to forecast")
    horizon: str = Field(
        default="24h", description="Forecast horizon (1h, 24h, 7d, 30d)"
    )

    class Config:
        schema_extra = {"example": {"market_ids": [1, 2, 3, 4, 5], "horizon": "24h"}}


class ForecastingResponse(BaseModel):
    """Response model for market forecasting"""

    market_id: int = Field(..., description="Market ID")
    forecast_horizon: str = Field(..., description="Forecast time horizon")
    predicted_price: float = Field(..., description="Predicted price")
    confidence_interval: Tuple[float, float] = Field(
        ..., description="Confidence interval (lower, upper)"
    )
    trend_direction: str = Field(
        ..., description="Trend direction (bullish, bearish, neutral)"
    )
    key_factors: List[str] = Field(
        ..., description="Key factors influencing the forecast"
    )
    risk_assessment: str = Field(..., description="Risk assessment (low, medium, high)")
    forecast_date: datetime = Field(..., description="When forecast was generated")

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "forecast_horizon": "24h",
                "predicted_price": 0.58,
                "confidence_interval": [0.52, 0.64],
                "trend_direction": "bullish",
                "key_factors": ["strong_price_momentum", "increasing_volume"],
                "risk_assessment": "medium",
                "forecast_date": "2024-01-15T10:30:00Z",
            }
        }


class MarketAnalysisSummary(BaseModel):
    """Response model for comprehensive market analysis"""

    market_id: int = Field(..., description="Market ID")
    analysis_date: datetime = Field(..., description="When analysis was performed")
    correlations: List[CorrelationAnalysisResponse] = Field(
        ..., description="Market correlations"
    )
    anomalies: List[AnomalyDetectionResponse] = Field(
        ..., description="Detected anomalies"
    )
    forecasts: List[ForecastingResponse] = Field(..., description="Market forecasts")
    analysis_summary: Dict[str, Any] = Field(
        ..., description="Summary of analysis results"
    )

    class Config:
        schema_extra = {
            "example": {
                "market_id": 1,
                "analysis_date": "2024-01-15T10:30:00Z",
                "correlations": [],
                "anomalies": [],
                "forecasts": [],
                "analysis_summary": {
                    "total_correlations": 5,
                    "significant_correlations": 3,
                    "total_anomalies": 1,
                    "critical_anomalies": 0,
                    "forecast_available": True,
                    "overall_risk_level": "medium",
                },
            }
        }


class PortfolioAnalysisRequest(BaseModel):
    """Request model for portfolio analysis"""

    user_id: int = Field(..., description="User ID for portfolio analysis")
    include_correlations: bool = Field(
        default=True, description="Include correlation analysis"
    )
    include_clustering: bool = Field(
        default=True, description="Include clustering analysis"
    )
    include_anomalies: bool = Field(
        default=True, description="Include anomaly detection"
    )
    include_forecasts: bool = Field(
        default=True, description="Include trend forecasting"
    )

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1,
                "include_correlations": True,
                "include_clustering": True,
                "include_anomalies": True,
                "include_forecasts": True,
            }
        }


class MarketComparisonRequest(BaseModel):
    """Request model for market comparison"""

    market_ids: List[int] = Field(
        ..., min_items=2, description="List of market IDs to compare"
    )
    analysis_types: List[str] = Field(
        default=["correlations", "clustering", "anomalies", "forecasts"],
        description="Types of analysis to perform",
    )

    class Config:
        schema_extra = {
            "example": {
                "market_ids": [1, 2, 3, 4, 5],
                "analysis_types": [
                    "correlations",
                    "clustering",
                    "anomalies",
                    "forecasts",
                ],
            }
        }


class AdvancedAnalyticsHealthResponse(BaseModel):
    """Response model for advanced analytics health check"""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: datetime = Field(..., description="Health check timestamp")
    features: List[str] = Field(..., description="Available features")
    dependencies: List[str] = Field(..., description="Required dependencies")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "service": "advanced_analytics",
                "timestamp": "2024-01-15T10:30:00Z",
                "features": [
                    "correlation_analysis",
                    "market_clustering",
                    "anomaly_detection",
                    "trend_forecasting",
                    "portfolio_analysis",
                    "market_comparison",
                    "websocket_updates",
                ],
                "dependencies": ["numpy", "pandas", "scipy", "scikit-learn", "redis"],
            }
        }


class WebSocketAdvancedAnalyticsMessage(BaseModel):
    """Base model for WebSocket advanced analytics messages"""

    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(..., description="Message timestamp")
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")


class WebSocketCorrelationRequest(BaseModel):
    """Request model for WebSocket correlation analysis"""

    type: str = Field("request_correlation_analysis", description="Message type")
    market_ids: List[int] = Field(
        ..., description="Market IDs for correlation analysis"
    )


class WebSocketAnomalyRequest(BaseModel):
    """Request model for WebSocket anomaly detection"""

    type: str = Field("request_anomaly_detection", description="Message type")
    market_ids: List[int] = Field(..., description="Market IDs for anomaly detection")


class WebSocketAdvancedSubscriptionRequest(BaseModel):
    """Request model for WebSocket advanced analytics subscription"""

    type: str = Field("subscribe_advanced_updates", description="Message type")
    features: List[str] = Field(
        default=["correlations", "anomalies", "forecasts"],
        description="Features to subscribe to",
    )


class PortfolioMetricsResponse(BaseModel):
    """Response model for portfolio metrics"""

    diversification_score: float = Field(
        ..., ge=0.0, le=1.0, description="Portfolio diversification score"
    )
    correlation_risk: float = Field(
        ..., ge=0.0, le=1.0, description="Correlation risk score"
    )
    anomaly_risk: float = Field(..., ge=0.0, le=1.0, description="Anomaly risk score")
    forecast_optimism: float = Field(
        ..., ge=0.0, le=1.0, description="Forecast optimism score"
    )
    overall_portfolio_health: str = Field(
        ..., description="Overall portfolio health rating"
    )

    class Config:
        schema_extra = {
            "example": {
                "diversification_score": 0.75,
                "correlation_risk": 0.25,
                "anomaly_risk": 0.10,
                "forecast_optimism": 0.60,
                "overall_portfolio_health": "good",
            }
        }


class MarketComparisonMatrixResponse(BaseModel):
    """Response model for market comparison matrix"""

    market_ids: List[int] = Field(..., description="List of compared market IDs")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        ..., description="Correlation matrix"
    )
    clustering_results: List[Dict[str, Any]] = Field(
        ..., description="Clustering results"
    )
    anomaly_summary: Dict[str, Any] = Field(..., description="Anomaly summary")
    forecast_summary: Dict[str, Any] = Field(..., description="Forecast summary")
    comparison_metrics: Dict[str, Dict[str, Any]] = Field(
        ..., description="Comparison metrics"
    )
    insights: List[str] = Field(..., description="Generated insights")

    class Config:
        schema_extra = {
            "example": {
                "market_ids": [1, 2, 3, 4, 5],
                "correlation_matrix": {
                    "1": {"1": 1.0, "2": 0.75, "3": 0.45, "4": 0.30, "5": 0.20},
                    "2": {"1": 0.75, "2": 1.0, "3": 0.60, "4": 0.40, "5": 0.25},
                },
                "clustering_results": [
                    {
                        "cluster_id": 0,
                        "market_ids": [1, 2],
                        "cluster_type": "high_activity",
                    }
                ],
                "anomaly_summary": {
                    "total_anomalies": 2,
                    "anomalies_by_market": {"1": 1, "2": 1},
                    "anomalies_by_type": {"price_spike": 2},
                },
                "forecast_summary": {
                    "forecasts_by_market": {},
                    "trend_distribution": {"bullish": 3, "bearish": 1, "neutral": 1},
                },
                "comparison_metrics": {
                    "1": {
                        "anomaly_count": 1,
                        "critical_anomalies": 0,
                        "forecast_trend": "bullish",
                        "forecast_risk": "medium",
                    }
                },
                "insights": [
                    "Found 3 strong correlations between markets",
                    "Markets are grouped into 2 distinct clusters",
                    "Detected 2 anomalies across 2 markets",
                ],
            }
        }
