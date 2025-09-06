"""
Advanced Market Analysis Service
Provides sophisticated market insights, correlation analysis, and predictive modeling
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json

logger = logging.getLogger(__name__)


@dataclass
class MarketCorrelation:
    """Market correlation analysis result"""

    market_id_1: int
    market_id_2: int
    correlation_coefficient: float
    correlation_type: str  # 'pearson', 'spearman'
    p_value: float
    significance_level: str  # 'high', 'medium', 'low'
    sample_size: int
    analysis_date: datetime


@dataclass
class MarketCluster:
    """Market clustering result"""

    cluster_id: int
    market_ids: List[int]
    cluster_center: List[float]
    cluster_size: int
    cluster_characteristics: Dict[str, Any]
    similarity_score: float


@dataclass
class MarketAnomaly:
    """Market anomaly detection result"""

    market_id: int
    anomaly_type: str  # 'price_spike', 'volume_surge', 'sentiment_shift'
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    description: str
    detected_at: datetime
    historical_context: Dict[str, Any]


@dataclass
class MarketForecast:
    """Market forecasting result"""

    market_id: int
    forecast_horizon: str  # '1h', '24h', '7d', '30d'
    predicted_price: float
    confidence_interval: Tuple[float, float]
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    key_factors: List[str]
    risk_assessment: str
    forecast_date: datetime


class AdvancedMarketAnalysisService:
    """Advanced market analysis service for sophisticated insights"""

    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.correlation_cache: Dict[str, MarketCorrelation] = {}
        self.cluster_cache: Dict[str, MarketCluster] = {}
        self.anomaly_cache: Dict[str, MarketAnomaly] = {}

    async def initialize(self):
        """Initialize the advanced market analysis service"""
        logger.info("Initializing Advanced Market Analysis Service")

        # Start background analysis tasks
        asyncio.create_task(self._periodic_correlation_analysis())
        asyncio.create_task(self._periodic_clustering_analysis())
        asyncio.create_task(self._periodic_anomaly_detection())

        logger.info("Advanced Market Analysis Service initialized successfully")

    async def analyze_market_correlations(
        self, market_ids: List[int], correlation_type: str = "pearson"
    ) -> List[MarketCorrelation]:
        """Analyze correlations between multiple markets"""
        try:
            correlations = []

            # Get market data for all markets
            market_data = await self._get_market_data_batch(market_ids)
            if not market_data:
                return correlations

            # Calculate pairwise correlations
            for i, market_id_1 in enumerate(market_ids):
                for market_id_2 in market_ids[i + 1 :]:
                    correlation = await self._calculate_correlation(
                        market_id_1, market_id_2, market_data, correlation_type
                    )
                    if correlation:
                        correlations.append(correlation)
                        await self._cache_correlation(correlation)

            return correlations

        except Exception as e:
            logger.error(f"Error analyzing market correlations: {e}")
            return []

    async def _calculate_correlation(
        self,
        market_id_1: int,
        market_id_2: int,
        market_data: Dict,
        correlation_type: str,
    ) -> Optional[MarketCorrelation]:
        """Calculate correlation between two markets"""
        try:
            # Get price data for both markets
            prices_1 = await self._get_price_history(market_id_1, days=30)
            prices_2 = await self._get_price_history(market_id_2, days=30)

            if len(prices_1) < 10 or len(prices_2) < 10:
                return None

            # Align data lengths
            min_length = min(len(prices_1), len(prices_2))
            prices_1 = prices_1[-min_length:]
            prices_2 = prices_2[-min_length:]

            # Calculate correlation
            if correlation_type == "pearson":
                corr_coef, p_value = pearsonr(prices_1, prices_2)
            elif correlation_type == "spearman":
                corr_coef, p_value = spearmanr(prices_1, prices_2)
            else:
                return None

            # Determine significance level
            if p_value < 0.001:
                significance = "high"
            elif p_value < 0.05:
                significance = "medium"
            else:
                significance = "low"

            return MarketCorrelation(
                market_id_1=market_id_1,
                market_id_2=market_id_2,
                correlation_coefficient=corr_coef,
                correlation_type=correlation_type,
                p_value=p_value,
                significance_level=significance,
                sample_size=min_length,
                analysis_date=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return None

    async def cluster_markets(
        self, market_ids: List[int], n_clusters: int = 3
    ) -> List[MarketCluster]:
        """Cluster markets based on their characteristics"""
        try:
            # Get market features for clustering
            market_features = await self._get_market_features_batch(market_ids)
            if not market_features:
                return []

            # Prepare feature matrix
            feature_matrix = []
            valid_market_ids = []

            for market_id in market_ids:
                if market_id in market_features:
                    features = market_features[market_id]
                    feature_vector = [
                        features.get("total_volume", 0),
                        features.get("participant_count", 0),
                        features.get("price_volatility", 0),
                        features.get("volume_trend", 0),
                        features.get("social_sentiment", 0),
                        features.get("liquidity_score", 0),
                    ]
                    feature_matrix.append(feature_vector)
                    valid_market_ids.append(market_id)

            if len(feature_matrix) < n_clusters:
                return []

            # Normalize features
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(feature_matrix_scaled)

            # Create cluster results
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_market_ids = [
                    valid_market_ids[i]
                    for i in range(len(cluster_labels))
                    if cluster_labels[i] == cluster_id
                ]

                if cluster_market_ids:
                    cluster_center = kmeans.cluster_centers_[cluster_id].tolist()
                    cluster_characteristics = self._analyze_cluster_characteristics(
                        cluster_market_ids, market_features
                    )

                    # Calculate similarity score
                    similarity_score = self._calculate_cluster_similarity(
                        cluster_market_ids, market_features
                    )

                    cluster = MarketCluster(
                        cluster_id=cluster_id,
                        market_ids=cluster_market_ids,
                        cluster_center=cluster_center,
                        cluster_size=len(cluster_market_ids),
                        cluster_characteristics=cluster_characteristics,
                        similarity_score=similarity_score,
                    )
                    clusters.append(cluster)
                    await self._cache_cluster(cluster)

            return clusters

        except Exception as e:
            logger.error(f"Error clustering markets: {e}")
            return []

    def _analyze_cluster_characteristics(
        self, market_ids: List[int], market_features: Dict
    ) -> Dict[str, Any]:
        """Analyze characteristics of a market cluster"""
        try:
            cluster_features = [
                market_features[mid] for mid in market_ids if mid in market_features
            ]

            if not cluster_features:
                return {}

            # Calculate average characteristics
            avg_volume = np.mean([f.get("total_volume", 0) for f in cluster_features])
            avg_participants = np.mean(
                [f.get("participant_count", 0) for f in cluster_features]
            )
            avg_volatility = np.mean(
                [f.get("price_volatility", 0) for f in cluster_features]
            )
            avg_sentiment = np.mean(
                [f.get("social_sentiment", 0) for f in cluster_features]
            )

            # Determine cluster type
            if avg_volume > 100000 and avg_participants > 100:
                cluster_type = "high_activity"
            elif avg_volatility > 0.5:
                cluster_type = "high_volatility"
            elif avg_sentiment > 0.7:
                cluster_type = "positive_sentiment"
            else:
                cluster_type = "standard"

            return {
                "cluster_type": cluster_type,
                "avg_volume": avg_volume,
                "avg_participants": avg_participants,
                "avg_volatility": avg_volatility,
                "avg_sentiment": avg_sentiment,
                "market_count": len(market_ids),
            }

        except Exception as e:
            logger.error(f"Error analyzing cluster characteristics: {e}")
            return {}

    def _calculate_cluster_similarity(
        self, market_ids: List[int], market_features: Dict
    ) -> float:
        """Calculate similarity score within a cluster"""
        try:
            if len(market_ids) < 2:
                return 1.0

            # Calculate pairwise similarities
            similarities = []
            for i, mid1 in enumerate(market_ids):
                for mid2 in market_ids[i + 1 :]:
                    if mid1 in market_features and mid2 in market_features:
                        similarity = self._calculate_market_similarity(
                            market_features[mid1], market_features[mid2]
                        )
                        similarities.append(similarity)

            return np.mean(similarities) if similarities else 0.0

        except Exception as e:
            logger.error(f"Error calculating cluster similarity: {e}")
            return 0.0

    def _calculate_market_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two markets based on features"""
        try:
            # Normalize and compare key features
            features = [
                "total_volume",
                "participant_count",
                "price_volatility",
                "social_sentiment",
            ]
            similarities = []

            for feature in features:
                val1 = features1.get(feature, 0)
                val2 = features2.get(feature, 0)

                if val1 == 0 and val2 == 0:
                    similarity = 1.0
                elif val1 == 0 or val2 == 0:
                    similarity = 0.0
                else:
                    similarity = 1 - abs(val1 - val2) / max(val1, val2)

                similarities.append(similarity)

            return np.mean(similarities)

        except Exception as e:
            logger.error(f"Error calculating market similarity: {e}")
            return 0.0

    async def detect_market_anomalies(
        self, market_ids: List[int]
    ) -> List[MarketAnomaly]:
        """Detect anomalies in market behavior"""
        try:
            anomalies = []

            for market_id in market_ids:
                market_anomalies = await self._detect_market_anomalies(market_id)
                anomalies.extend(market_anomalies)

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting market anomalies: {e}")
            return []

    async def _detect_market_anomalies(self, market_id: int) -> List[MarketAnomaly]:
        """Detect anomalies for a specific market"""
        try:
            anomalies = []

            # Get market data
            market_data = await self._get_market_data(market_id)
            if not market_data:
                return anomalies

            # Get historical data
            price_history = await self._get_price_history(market_id, days=30)
            volume_history = await self._get_volume_history(market_id, days=30)

            if len(price_history) < 10:
                return anomalies

            # Detect price spikes
            price_anomaly = self._detect_price_anomaly(market_id, price_history)
            if price_anomaly:
                anomalies.append(price_anomaly)

            # Detect volume surges
            volume_anomaly = self._detect_volume_anomaly(market_id, volume_history)
            if volume_anomaly:
                anomalies.append(volume_anomaly)

            # Detect sentiment shifts
            sentiment_anomaly = await self._detect_sentiment_anomaly(market_id)
            if sentiment_anomaly:
                anomalies.append(sentiment_anomaly)

            # Cache anomalies
            for anomaly in anomalies:
                await self._cache_anomaly(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Error detecting anomalies for market {market_id}: {e}")
            return []

    def _detect_price_anomaly(
        self, market_id: int, price_history: List[float]
    ) -> Optional[MarketAnomaly]:
        """Detect price anomalies using statistical methods"""
        try:
            if len(price_history) < 10:
                return None

            # Calculate price changes
            price_changes = np.diff(price_history)

            # Calculate z-scores
            mean_change = np.mean(price_changes)
            std_change = np.std(price_changes)

            if std_change == 0:
                return None

            latest_change = price_changes[-1]
            z_score = abs(latest_change - mean_change) / std_change

            # Detect anomaly if z-score > 2.5
            if z_score > 2.5:
                severity = (
                    "critical"
                    if z_score > 4.0
                    else "high" if z_score > 3.0 else "medium"
                )

                return MarketAnomaly(
                    market_id=market_id,
                    anomaly_type="price_spike",
                    severity=severity,
                    confidence=min(z_score / 5.0, 1.0),
                    description=f"Price change of {latest_change:.4f} (z-score: {z_score:.2f})",
                    detected_at=datetime.utcnow(),
                    historical_context={
                        "mean_change": mean_change,
                        "std_change": std_change,
                        "z_score": z_score,
                        "price_history_length": len(price_history),
                    },
                )

            return None

        except Exception as e:
            logger.error(f"Error detecting price anomaly: {e}")
            return None

    def _detect_volume_anomaly(
        self, market_id: int, volume_history: List[float]
    ) -> Optional[MarketAnomaly]:
        """Detect volume anomalies"""
        try:
            if len(volume_history) < 10:
                return None

            # Calculate volume changes
            volume_changes = np.diff(volume_history)

            # Calculate z-scores
            mean_change = np.mean(volume_changes)
            std_change = np.std(volume_changes)

            if std_change == 0:
                return None

            latest_change = volume_changes[-1]
            z_score = abs(latest_change - mean_change) / std_change

            # Detect anomaly if z-score > 2.0
            if z_score > 2.0:
                severity = (
                    "critical"
                    if z_score > 4.0
                    else "high" if z_score > 3.0 else "medium"
                )

                return MarketAnomaly(
                    market_id=market_id,
                    anomaly_type="volume_surge",
                    severity=severity,
                    confidence=min(z_score / 5.0, 1.0),
                    description=f"Volume change of {latest_change:.2f} (z-score: {z_score:.2f})",
                    detected_at=datetime.utcnow(),
                    historical_context={
                        "mean_change": mean_change,
                        "std_change": std_change,
                        "z_score": z_score,
                        "volume_history_length": len(volume_history),
                    },
                )

            return None

        except Exception as e:
            logger.error(f"Error detecting volume anomaly: {e}")
            return None

    async def _detect_sentiment_anomaly(
        self, market_id: int
    ) -> Optional[MarketAnomaly]:
        """Detect sentiment anomalies"""
        try:
            # Get recent sentiment data
            recent_sentiment = await self._get_recent_sentiment(market_id)
            historical_sentiment = await self._get_historical_sentiment(market_id)

            if not recent_sentiment or not historical_sentiment:
                return None

            # Calculate sentiment change
            sentiment_change = abs(recent_sentiment - np.mean(historical_sentiment))
            sentiment_std = np.std(historical_sentiment)

            if sentiment_std == 0:
                return None

            z_score = sentiment_change / sentiment_std

            # Detect anomaly if z-score > 2.0
            if z_score > 2.0:
                severity = "high" if z_score > 3.0 else "medium"

                return MarketAnomaly(
                    market_id=market_id,
                    anomaly_type="sentiment_shift",
                    severity=severity,
                    confidence=min(z_score / 4.0, 1.0),
                    description=f"Sentiment shift of {sentiment_change:.3f} (z-score: {z_score:.2f})",
                    detected_at=datetime.utcnow(),
                    historical_context={
                        "recent_sentiment": recent_sentiment,
                        "historical_mean": np.mean(historical_sentiment),
                        "sentiment_std": sentiment_std,
                        "z_score": z_score,
                    },
                )

            return None

        except Exception as e:
            logger.error(f"Error detecting sentiment anomaly: {e}")
            return None

    async def forecast_market_trends(
        self, market_ids: List[int], horizon: str = "24h"
    ) -> List[MarketForecast]:
        """Forecast market trends for multiple markets"""
        try:
            forecasts = []

            for market_id in market_ids:
                forecast = await self._forecast_market_trend(market_id, horizon)
                if forecast:
                    forecasts.append(forecast)

            return forecasts

        except Exception as e:
            logger.error(f"Error forecasting market trends: {e}")
            return []

    async def _forecast_market_trend(
        self, market_id: int, horizon: str
    ) -> Optional[MarketForecast]:
        """Forecast trend for a specific market"""
        try:
            # Get market data
            price_history = await self._get_price_history(market_id, days=30)
            volume_history = await self._get_volume_history(market_id, days=30)

            if len(price_history) < 10:
                return None

            # Calculate trend indicators
            current_price = price_history[-1]
            price_trend = self._calculate_price_trend(price_history)
            volume_trend = self._calculate_volume_trend(volume_history)

            # Simple forecasting model (in production, use more sophisticated models)
            forecast_price = current_price * (1 + price_trend)

            # Calculate confidence interval
            price_volatility = np.std(price_history[-10:]) / np.mean(
                price_history[-10:]
            )
            confidence_margin = price_volatility * 2  # 95% confidence interval
            confidence_interval = (
                forecast_price * (1 - confidence_margin),
                forecast_price * (1 + confidence_margin),
            )

            # Determine trend direction
            if price_trend > 0.05:
                trend_direction = "bullish"
            elif price_trend < -0.05:
                trend_direction = "bearish"
            else:
                trend_direction = "neutral"

            # Identify key factors
            key_factors = []
            if abs(price_trend) > 0.1:
                key_factors.append("strong_price_momentum")
            if volume_trend > 0.2:
                key_factors.append("increasing_volume")
            if price_volatility > 0.3:
                key_factors.append("high_volatility")

            # Risk assessment
            if price_volatility > 0.5:
                risk_assessment = "high"
            elif price_volatility > 0.2:
                risk_assessment = "medium"
            else:
                risk_assessment = "low"

            return MarketForecast(
                market_id=market_id,
                forecast_horizon=horizon,
                predicted_price=forecast_price,
                confidence_interval=confidence_interval,
                trend_direction=trend_direction,
                key_factors=key_factors,
                risk_assessment=risk_assessment,
                forecast_date=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error forecasting market trend: {e}")
            return None

    def _calculate_price_trend(self, price_history: List[float]) -> float:
        """Calculate price trend using linear regression"""
        try:
            if len(price_history) < 5:
                return 0.0

            x = np.arange(len(price_history))
            y = np.array(price_history)

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Normalize trend by current price
            current_price = price_history[-1]
            normalized_trend = slope / current_price if current_price > 0 else 0.0

            return normalized_trend

        except Exception as e:
            logger.error(f"Error calculating price trend: {e}")
            return 0.0

    def _calculate_volume_trend(self, volume_history: List[float]) -> float:
        """Calculate volume trend"""
        try:
            if len(volume_history) < 5:
                return 0.0

            recent_avg = np.mean(volume_history[-5:])
            older_avg = np.mean(volume_history[-10:-5])

            if older_avg == 0:
                return 0.0

            return (recent_avg - older_avg) / older_avg

        except Exception as e:
            logger.error(f"Error calculating volume trend: {e}")
            return 0.0

    async def _periodic_correlation_analysis(self):
        """Periodically analyze market correlations"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Get active markets
                active_markets = await self._get_active_market_ids()
                if len(active_markets) > 1:
                    await self.analyze_market_correlations(active_markets[:10])

            except Exception as e:
                logger.error(f"Error in periodic correlation analysis: {e}")

    async def _periodic_clustering_analysis(self):
        """Periodically perform market clustering"""
        while True:
            try:
                await asyncio.sleep(7200)  # Run every 2 hours

                # Get active markets
                active_markets = await self._get_active_market_ids()
                if len(active_markets) > 3:
                    await self.cluster_markets(active_markets[:20])

            except Exception as e:
                logger.error(f"Error in periodic clustering analysis: {e}")

    async def _periodic_anomaly_detection(self):
        """Periodically detect market anomalies"""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes

                # Get active markets
                active_markets = await self._get_active_market_ids()
                if active_markets:
                    await self.detect_market_anomalies(active_markets[:10])

            except Exception as e:
                logger.error(f"Error in periodic anomaly detection: {e}")

    # Caching methods
    async def _cache_correlation(self, correlation: MarketCorrelation):
        """Cache correlation result"""
        cache_key = f"correlation:{correlation.market_id_1}:{correlation.market_id_2}"
        await self.redis.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps(
                {
                    "correlation_coefficient": correlation.correlation_coefficient,
                    "correlation_type": correlation.correlation_type,
                    "p_value": correlation.p_value,
                    "significance_level": correlation.significance_level,
                    "analysis_date": correlation.analysis_date.isoformat(),
                }
            ),
        )

    async def _cache_cluster(self, cluster: MarketCluster):
        """Cache cluster result"""
        cache_key = f"cluster:{cluster.cluster_id}"
        await self.redis.setex(
            cache_key,
            7200,  # 2 hours TTL
            json.dumps(
                {
                    "market_ids": cluster.market_ids,
                    "cluster_size": cluster.cluster_size,
                    "cluster_characteristics": cluster.cluster_characteristics,
                    "similarity_score": cluster.similarity_score,
                }
            ),
        )

    async def _cache_anomaly(self, anomaly: MarketAnomaly):
        """Cache anomaly result"""
        cache_key = f"anomaly:{anomaly.market_id}:{anomaly.anomaly_type}"
        await self.redis.setex(
            cache_key,
            1800,  # 30 minutes TTL
            json.dumps(
                {
                    "severity": anomaly.severity,
                    "confidence": anomaly.confidence,
                    "description": anomaly.description,
                    "detected_at": anomaly.detected_at.isoformat(),
                    "historical_context": anomaly.historical_context,
                }
            ),
        )

    # Data retrieval methods (implementations depend on your data models)
    async def _get_market_data_batch(self, market_ids: List[int]) -> Dict[int, Dict]:
        """Get market data for multiple markets"""
        # Implementation depends on your database models
        return {mid: {"id": mid, "title": f"Market {mid}"} for mid in market_ids}

    async def _get_market_features_batch(
        self, market_ids: List[int]
    ) -> Dict[int, Dict]:
        """Get market features for multiple markets"""
        # Implementation depends on your database models
        return {
            mid: {
                "total_volume": 100000.0 + mid * 1000,
                "participant_count": 100 + mid * 10,
                "price_volatility": 0.2 + (mid % 3) * 0.1,
                "volume_trend": 0.1 + (mid % 2) * 0.05,
                "social_sentiment": 0.5 + (mid % 5) * 0.1,
                "liquidity_score": 0.7 + (mid % 3) * 0.1,
            }
            for mid in market_ids
        }

    async def _get_market_data(self, market_id: int) -> Optional[Dict]:
        """Get market data for a specific market"""
        # Implementation depends on your database models
        return {"id": market_id, "title": f"Market {market_id}"}

    async def _get_price_history(self, market_id: int, days: int = 30) -> List[float]:
        """Get price history for a market"""
        # Implementation depends on your database models
        return [
            0.5 + 0.1 * np.sin(i / 10) + 0.05 * np.random.random() for i in range(days)
        ]

    async def _get_volume_history(self, market_id: int, days: int = 30) -> List[float]:
        """Get volume history for a market"""
        # Implementation depends on your database models
        return [1000.0 + 200 * np.random.random() for _ in range(days)]

    async def _get_recent_sentiment(self, market_id: int) -> Optional[float]:
        """Get recent sentiment for a market"""
        # Implementation depends on your database models
        return 0.6 + 0.2 * np.random.random()

    async def _get_historical_sentiment(self, market_id: int) -> List[float]:
        """Get historical sentiment for a market"""
        # Implementation depends on your database models
        return [0.5 + 0.2 * np.random.random() for _ in range(30)]

    async def _get_active_market_ids(self) -> List[int]:
        """Get list of active market IDs"""
        # Implementation depends on your database models
        return list(range(1, 21))


# Factory function
async def get_advanced_market_analysis_service(
    redis_client: redis.Redis, db_session: Session
) -> AdvancedMarketAnalysisService:
    """Get advanced market analysis service instance"""
    service = AdvancedMarketAnalysisService(redis_client, db_session)
    await service.initialize()
    return service
