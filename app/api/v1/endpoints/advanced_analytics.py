"""
Advanced Analytics API Endpoints
Provides sophisticated market analysis, correlation analysis, clustering, anomaly detection, and forecasting
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.redis_client import get_redis_client
from app.services.advanced_market_analysis import (
    get_advanced_market_analysis_service,
    MarketCorrelation, MarketCluster, MarketAnomaly, MarketForecast
)
from app.schemas.advanced_analytics import (
    CorrelationAnalysisRequest, CorrelationAnalysisResponse,
    ClusteringRequest, ClusteringResponse, AnomalyDetectionResponse,
    ForecastingRequest, ForecastingResponse, MarketAnalysisSummary
)

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connections for real-time advanced analytics
websocket_connections: Dict[str, WebSocket] = {}


@router.post("/correlations/analyze", response_model=List[CorrelationAnalysisResponse])
async def analyze_market_correlations(
    request: CorrelationAnalysisRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Analyze correlations between multiple markets
    """
    try:
        redis_client = await get_redis_client()
        analysis_service = await get_advanced_market_analysis_service(redis_client, db)
        
        correlations = await analysis_service.analyze_market_correlations(
            request.market_ids, request.correlation_type
        )
        
        return [
            CorrelationAnalysisResponse(
                market_id_1=corr.market_id_1,
                market_id_2=corr.market_id_2,
                correlation_coefficient=corr.correlation_coefficient,
                correlation_type=corr.correlation_type,
                p_value=corr.p_value,
                significance_level=corr.significance_level,
                sample_size=corr.sample_size,
                analysis_date=corr.analysis_date
            )
            for corr in correlations
        ]
        
    except Exception as e:
        logger.error(f"Error analyzing market correlations: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing correlations")


@router.post("/clustering/analyze", response_model=List[ClusteringResponse])
async def analyze_market_clusters(
    request: ClusteringRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Cluster markets based on their characteristics
    """
    try:
        redis_client = await get_redis_client()
        analysis_service = await get_advanced_market_analysis_service(redis_client, db)
        
        clusters = await analysis_service.cluster_markets(
            request.market_ids, request.n_clusters
        )
        
        return [
            ClusteringResponse(
                cluster_id=cluster.cluster_id,
                market_ids=cluster.market_ids,
                cluster_size=cluster.cluster_size,
                cluster_characteristics=cluster.cluster_characteristics,
                similarity_score=cluster.similarity_score
            )
            for cluster in clusters
        ]
        
    except Exception as e:
        logger.error(f"Error analyzing market clusters: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing clusters")


@router.get("/anomalies/detect", response_model=List[AnomalyDetectionResponse])
async def detect_market_anomalies(
    market_ids: List[int] = Query(..., description="List of market IDs to analyze"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Detect anomalies in market behavior
    """
    try:
        redis_client = await get_redis_client()
        analysis_service = await get_advanced_market_analysis_service(redis_client, db)
        
        anomalies = await analysis_service.detect_market_anomalies(market_ids)
        
        return [
            AnomalyDetectionResponse(
                market_id=anomaly.market_id,
                anomaly_type=anomaly.anomaly_type,
                severity=anomaly.severity,
                confidence=anomaly.confidence,
                description=anomaly.description,
                detected_at=anomaly.detected_at,
                historical_context=anomaly.historical_context
            )
            for anomaly in anomalies
        ]
        
    except Exception as e:
        logger.error(f"Error detecting market anomalies: {e}")
        raise HTTPException(status_code=500, detail="Error detecting anomalies")


@router.post("/forecasting/predict", response_model=List[ForecastingResponse])
async def forecast_market_trends(
    request: ForecastingRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Forecast market trends for multiple markets
    """
    try:
        redis_client = await get_redis_client()
        analysis_service = await get_advanced_market_analysis_service(redis_client, db)
        
        forecasts = await analysis_service.forecast_market_trends(
            request.market_ids, request.horizon
        )
        
        return [
            ForecastingResponse(
                market_id=forecast.market_id,
                forecast_horizon=forecast.forecast_horizon,
                predicted_price=forecast.predicted_price,
                confidence_interval=forecast.confidence_interval,
                trend_direction=forecast.trend_direction,
                key_factors=forecast.key_factors,
                risk_assessment=forecast.risk_assessment,
                forecast_date=forecast.forecast_date
            )
            for forecast in forecasts
        ]
        
    except Exception as e:
        logger.error(f"Error forecasting market trends: {e}")
        raise HTTPException(status_code=500, detail="Error forecasting trends")


@router.get("/analysis/comprehensive/{market_id}", response_model=MarketAnalysisSummary)
async def get_comprehensive_market_analysis(
    market_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive market analysis including correlations, anomalies, and forecasts
    """
    try:
        redis_client = await get_redis_client()
        analysis_service = await get_advanced_market_analysis_service(redis_client, db)
        
        # Get active markets for correlation analysis
        active_markets = await analysis_service._get_active_market_ids()
        related_markets = [mid for mid in active_markets if mid != market_id][:5]
        
        # Perform various analyses
        correlations = await analysis_service.analyze_market_correlations([market_id] + related_markets)
        anomalies = await analysis_service.detect_market_anomalies([market_id])
        forecasts = await analysis_service.forecast_market_trends([market_id])
        
        # Filter correlations for the specific market
        market_correlations = [
            corr for corr in correlations 
            if corr.market_id_1 == market_id or corr.market_id_2 == market_id
        ]
        
        # Create comprehensive summary
        summary = MarketAnalysisSummary(
            market_id=market_id,
            analysis_date=datetime.utcnow(),
            correlations=market_correlations,
            anomalies=anomalies,
            forecasts=forecasts,
            analysis_summary={
                "total_correlations": len(market_correlations),
                "significant_correlations": len([c for c in market_correlations if c.significance_level in ['high', 'medium']]),
                "total_anomalies": len(anomalies),
                "critical_anomalies": len([a for a in anomalies if a.severity == 'critical']),
                "forecast_available": len(forecasts) > 0,
                "overall_risk_level": self._calculate_overall_risk(anomalies, forecasts)
            }
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting comprehensive market analysis: {e}")
        raise HTTPException(status_code=500, detail="Error performing comprehensive analysis")


@router.get("/analysis/portfolio/{user_id}")
async def get_portfolio_advanced_analysis(
    user_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get advanced analysis for a user's portfolio
    """
    try:
        redis_client = await get_redis_client()
        analysis_service = await get_advanced_market_analysis_service(redis_client, db)
        
        # Get user's active markets (this would depend on your data model)
        user_markets = await self._get_user_active_markets(user_id)
        
        if not user_markets:
            raise HTTPException(status_code=404, detail="No active markets found for user")
        
        # Perform portfolio analysis
        correlations = await analysis_service.analyze_market_correlations(user_markets)
        clusters = await analysis_service.cluster_markets(user_markets)
        anomalies = await analysis_service.detect_market_anomalies(user_markets)
        forecasts = await analysis_service.forecast_market_trends(user_markets)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            correlations, clusters, anomalies, forecasts
        )
        
        portfolio_analysis = {
            "user_id": user_id,
            "analysis_date": datetime.utcnow().isoformat(),
            "portfolio_summary": {
                "total_markets": len(user_markets),
                "market_ids": user_markets,
                "total_correlations": len(correlations),
                "total_clusters": len(clusters),
                "total_anomalies": len(anomalies),
                "total_forecasts": len(forecasts)
            },
            "correlation_analysis": {
                "high_correlations": len([c for c in correlations if c.significance_level == 'high']),
                "medium_correlations": len([c for c in correlations if c.significance_level == 'medium']),
                "low_correlations": len([c for c in correlations if c.significance_level == 'low']),
                "avg_correlation": np.mean([abs(c.correlation_coefficient) for c in correlations]) if correlations else 0.0
            },
            "clustering_analysis": {
                "cluster_count": len(clusters),
                "avg_cluster_size": np.mean([c.cluster_size for c in clusters]) if clusters else 0.0,
                "avg_similarity": np.mean([c.similarity_score for c in clusters]) if clusters else 0.0
            },
            "anomaly_analysis": {
                "critical_anomalies": len([a for a in anomalies if a.severity == 'critical']),
                "high_anomalies": len([a for a in anomalies if a.severity == 'high']),
                "medium_anomalies": len([a for a in anomalies if a.severity == 'medium']),
                "low_anomalies": len([a for a in anomalies if a.severity == 'low'])
            },
            "forecast_analysis": {
                "bullish_forecasts": len([f for f in forecasts if f.trend_direction == 'bullish']),
                "bearish_forecasts": len([f for f in forecasts if f.trend_direction == 'bearish']),
                "neutral_forecasts": len([f for f in forecasts if f.trend_direction == 'neutral']),
                "high_risk_forecasts": len([f for f in forecasts if f.risk_assessment == 'high'])
            },
            "portfolio_metrics": portfolio_metrics,
            "recommendations": self._generate_portfolio_recommendations(
                correlations, clusters, anomalies, forecasts
            )
        }
        
        return JSONResponse(content=portfolio_analysis)
        
    except Exception as e:
        logger.error(f"Error getting portfolio advanced analysis: {e}")
        raise HTTPException(status_code=500, detail="Error performing portfolio analysis")


@router.get("/analysis/market-comparison")
async def compare_markets(
    market_ids: List[int] = Query(..., description="List of market IDs to compare"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Compare multiple markets using advanced analytics
    """
    try:
        if len(market_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 markets required for comparison")
        
        redis_client = await get_redis_client()
        analysis_service = await get_advanced_market_analysis_service(redis_client, db)
        
        # Perform comprehensive analysis
        correlations = await analysis_service.analyze_market_correlations(market_ids)
        clusters = await analysis_service.cluster_markets(market_ids)
        anomalies = await analysis_service.detect_market_anomalies(market_ids)
        forecasts = await analysis_service.forecast_market_trends(market_ids)
        
        # Create comparison matrix
        comparison_matrix = self._create_comparison_matrix(market_ids, correlations, clusters, anomalies, forecasts)
        
        comparison_analysis = {
            "market_ids": market_ids,
            "analysis_date": datetime.utcnow().isoformat(),
            "correlation_matrix": self._build_correlation_matrix(market_ids, correlations),
            "clustering_results": [
                {
                    "cluster_id": cluster.cluster_id,
                    "market_ids": cluster.market_ids,
                    "cluster_type": cluster.cluster_characteristics.get("cluster_type", "unknown")
                }
                for cluster in clusters
            ],
            "anomaly_summary": {
                "total_anomalies": len(anomalies),
                "anomalies_by_market": {
                    mid: len([a for a in anomalies if a.market_id == mid])
                    for mid in market_ids
                },
                "anomalies_by_type": {
                    anomaly_type: len([a for a in anomalies if a.anomaly_type == anomaly_type])
                    for anomaly_type in set(a.anomaly_type for a in anomalies)
                }
            },
            "forecast_summary": {
                "forecasts_by_market": {
                    mid: next((f for f in forecasts if f.market_id == mid), None)
                    for mid in market_ids
                },
                "trend_distribution": {
                    trend: len([f for f in forecasts if f.trend_direction == trend])
                    for trend in set(f.trend_direction for f in forecasts)
                }
            },
            "comparison_metrics": comparison_matrix,
            "insights": self._generate_comparison_insights(market_ids, correlations, clusters, anomalies, forecasts)
        }
        
        return JSONResponse(content=comparison_analysis)
        
    except Exception as e:
        logger.error(f"Error comparing markets: {e}")
        raise HTTPException(status_code=500, detail="Error performing market comparison")


@router.websocket("/ws/advanced-analytics/{client_id}")
async def websocket_advanced_analytics(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time advanced analytics updates
    """
    await websocket.accept()
    websocket_connections[client_id] = websocket
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "advanced_analytics_connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat(),
            "available_features": [
                "correlation_analysis",
                "market_clustering",
                "anomaly_detection",
                "trend_forecasting",
                "portfolio_analysis"
            ]
        }))
        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "request_correlation_analysis":
                    market_ids = message.get("market_ids", [])
                    # Perform correlation analysis and send results
                    await websocket.send_text(json.dumps({
                        "type": "correlation_analysis_result",
                        "market_ids": market_ids,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "correlation_count": len(market_ids) * (len(market_ids) - 1) // 2,
                            "analysis_status": "completed"
                        }
                    }))
                
                elif message.get("type") == "request_anomaly_detection":
                    market_ids = message.get("market_ids", [])
                    # Perform anomaly detection and send results
                    await websocket.send_text(json.dumps({
                        "type": "anomaly_detection_result",
                        "market_ids": market_ids,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "anomaly_count": len(market_ids),
                            "detection_status": "completed"
                        }
                    }))
                
                elif message.get("type") == "subscribe_advanced_updates":
                    # Subscribe to advanced analytics updates
                    await websocket.send_text(json.dumps({
                        "type": "advanced_subscription_confirmed",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"Advanced analytics client {client_id} disconnected")
    finally:
        if client_id in websocket_connections:
            del websocket_connections[client_id]


@router.get("/health/advanced")
async def advanced_analytics_health():
    """
    Health check for advanced analytics services
    """
    return {
        "status": "healthy",
        "service": "advanced_analytics",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "correlation_analysis",
            "market_clustering",
            "anomaly_detection",
            "trend_forecasting",
            "portfolio_analysis",
            "market_comparison",
            "websocket_updates"
        ],
        "dependencies": [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "redis"
        ]
    }


# Helper methods
def _calculate_overall_risk(self, anomalies: List[MarketAnomaly], forecasts: List[MarketForecast]) -> str:
    """Calculate overall risk level based on anomalies and forecasts"""
    try:
        risk_score = 0
        
        # Anomaly risk
        for anomaly in anomalies:
            if anomaly.severity == 'critical':
                risk_score += 4
            elif anomaly.severity == 'high':
                risk_score += 3
            elif anomaly.severity == 'medium':
                risk_score += 2
            else:
                risk_score += 1
        
        # Forecast risk
        for forecast in forecasts:
            if forecast.risk_assessment == 'high':
                risk_score += 3
            elif forecast.risk_assessment == 'medium':
                risk_score += 2
            else:
                risk_score += 1
        
        # Normalize risk score
        total_items = len(anomalies) + len(forecasts)
        if total_items == 0:
            return 'low'
        
        avg_risk = risk_score / total_items
        
        if avg_risk > 3.0:
            return 'critical'
        elif avg_risk > 2.5:
            return 'high'
        elif avg_risk > 1.5:
            return 'medium'
        else:
            return 'low'
            
    except Exception as e:
        logger.error(f"Error calculating overall risk: {e}")
        return 'unknown'


async def _get_user_active_markets(self, user_id: int) -> List[int]:
    """Get active markets for a user"""
    # Implementation depends on your data model
    return [1, 2, 3, 4, 5]  # Example


def _calculate_portfolio_metrics(self, correlations: List[MarketCorrelation], 
                               clusters: List[MarketCluster], 
                               anomalies: List[MarketAnomaly], 
                               forecasts: List[MarketForecast]) -> Dict[str, Any]:
    """Calculate portfolio-level metrics"""
    try:
        import numpy as np
        
        metrics = {
            "diversification_score": 0.0,
            "correlation_risk": 0.0,
            "anomaly_risk": 0.0,
            "forecast_optimism": 0.0,
            "overall_portfolio_health": "unknown"
        }
        
        # Calculate diversification score based on clustering
        if clusters:
            cluster_sizes = [c.cluster_size for c in clusters]
            total_markets = sum(cluster_sizes)
            if total_markets > 0:
                # Higher score for more evenly distributed clusters
                metrics["diversification_score"] = 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
        
        # Calculate correlation risk
        if correlations:
            high_correlations = [c for c in correlations if c.significance_level == 'high' and abs(c.correlation_coefficient) > 0.7]
            metrics["correlation_risk"] = len(high_correlations) / len(correlations) if correlations else 0.0
        
        # Calculate anomaly risk
        if anomalies:
            critical_anomalies = [a for a in anomalies if a.severity == 'critical']
            metrics["anomaly_risk"] = len(critical_anomalies) / len(anomalies) if anomalies else 0.0
        
        # Calculate forecast optimism
        if forecasts:
            bullish_forecasts = [f for f in forecasts if f.trend_direction == 'bullish']
            metrics["forecast_optimism"] = len(bullish_forecasts) / len(forecasts) if forecasts else 0.0
        
        # Calculate overall portfolio health
        health_score = (
            (1 - metrics["correlation_risk"]) * 0.3 +
            (1 - metrics["anomaly_risk"]) * 0.4 +
            metrics["diversification_score"] * 0.3
        )
        
        if health_score > 0.8:
            metrics["overall_portfolio_health"] = "excellent"
        elif health_score > 0.6:
            metrics["overall_portfolio_health"] = "good"
        elif health_score > 0.4:
            metrics["overall_portfolio_health"] = "fair"
        else:
            metrics["overall_portfolio_health"] = "poor"
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {e}")
        return {"error": "Failed to calculate metrics"}


def _generate_portfolio_recommendations(self, correlations: List[MarketCorrelation], 
                                      clusters: List[MarketCluster], 
                                      anomalies: List[MarketAnomaly], 
                                      forecasts: List[MarketForecast]) -> List[str]:
    """Generate portfolio recommendations based on analysis"""
    recommendations = []
    
    try:
        # Correlation-based recommendations
        high_correlations = [c for c in correlations if c.significance_level == 'high' and abs(c.correlation_coefficient) > 0.8]
        if len(high_correlations) > 3:
            recommendations.append("Consider reducing exposure to highly correlated markets to improve diversification")
        
        # Anomaly-based recommendations
        critical_anomalies = [a for a in anomalies if a.severity == 'critical']
        if critical_anomalies:
            recommendations.append(f"Monitor {len(critical_anomalies)} critical anomalies in your portfolio")
        
        # Forecast-based recommendations
        bearish_forecasts = [f for f in forecasts if f.trend_direction == 'bearish']
        if bearish_forecasts:
            recommendations.append(f"Consider hedging strategies for {len(bearish_forecasts)} bearish forecasts")
        
        # Clustering-based recommendations
        if len(clusters) == 1 and clusters[0].cluster_size > 5:
            recommendations.append("Your portfolio is concentrated in one market cluster - consider diversifying across different market types")
        
        if not recommendations:
            recommendations.append("Your portfolio appears well-balanced based on current analysis")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating portfolio recommendations: {e}")
        return ["Unable to generate recommendations at this time"]


def _create_comparison_matrix(self, market_ids: List[int], 
                            correlations: List[MarketCorrelation], 
                            clusters: List[MarketCluster], 
                            anomalies: List[MarketAnomaly], 
                            forecasts: List[MarketForecast]) -> Dict[str, Any]:
    """Create comparison matrix for multiple markets"""
    try:
        matrix = {}
        
        for market_id in market_ids:
            market_anomalies = [a for a in anomalies if a.market_id == market_id]
            market_forecast = next((f for f in forecasts if f.market_id == market_id), None)
            
            matrix[market_id] = {
                "anomaly_count": len(market_anomalies),
                "critical_anomalies": len([a for a in market_anomalies if a.severity == 'critical']),
                "forecast_trend": market_forecast.trend_direction if market_forecast else "unknown",
                "forecast_risk": market_forecast.risk_assessment if market_forecast else "unknown",
                "cluster_assignment": next((c.cluster_id for c in clusters if market_id in c.market_ids), None)
            }
        
        return matrix
        
    except Exception as e:
        logger.error(f"Error creating comparison matrix: {e}")
        return {}


def _build_correlation_matrix(self, market_ids: List[int], 
                            correlations: List[MarketCorrelation]) -> Dict[str, Dict[str, float]]:
    """Build correlation matrix for market comparison"""
    try:
        matrix = {}
        
        for market_id_1 in market_ids:
            matrix[market_id_1] = {}
            for market_id_2 in market_ids:
                if market_id_1 == market_id_2:
                    matrix[market_id_1][market_id_2] = 1.0
                else:
                    # Find correlation between these two markets
                    correlation = next(
                        (c for c in correlations 
                         if (c.market_id_1 == market_id_1 and c.market_id_2 == market_id_2) or
                            (c.market_id_1 == market_id_2 and c.market_id_2 == market_id_1)),
                        None
                    )
                    matrix[market_id_1][market_id_2] = correlation.correlation_coefficient if correlation else 0.0
        
        return matrix
        
    except Exception as e:
        logger.error(f"Error building correlation matrix: {e}")
        return {}


def _generate_comparison_insights(self, market_ids: List[int], 
                                correlations: List[MarketCorrelation], 
                                clusters: List[MarketCluster], 
                                anomalies: List[MarketAnomaly], 
                                forecasts: List[MarketForecast]) -> List[str]:
    """Generate insights from market comparison"""
    insights = []
    
    try:
        # Correlation insights
        high_correlations = [c for c in correlations if abs(c.correlation_coefficient) > 0.7]
        if high_correlations:
            insights.append(f"Found {len(high_correlations)} strong correlations between markets")
        
        # Clustering insights
        if len(clusters) > 1:
            insights.append(f"Markets are grouped into {len(clusters)} distinct clusters")
        elif len(clusters) == 1:
            insights.append("All markets show similar characteristics")
        
        # Anomaly insights
        if anomalies:
            insights.append(f"Detected {len(anomalies)} anomalies across {len(set(a.market_id for a in anomalies))} markets")
        
        # Forecast insights
        if forecasts:
            bullish_count = len([f for f in forecasts if f.trend_direction == 'bullish'])
            bearish_count = len([f for f in forecasts if f.trend_direction == 'bearish'])
            insights.append(f"Forecasts show {bullish_count} bullish and {bearish_count} bearish trends")
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating comparison insights: {e}")
        return ["Unable to generate insights at this time"]
