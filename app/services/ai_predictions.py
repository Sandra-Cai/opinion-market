"""
Advanced AI Prediction Service for Opinion Market
Uses machine learning to predict market outcomes and provide trading insights
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import redis.asyncio as redis
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of an AI prediction"""

    market_id: int
    predicted_outcome: str
    confidence: float
    probability_a: float
    probability_b: float
    features_used: List[str]
    model_version: str
    prediction_time: datetime
    validity_duration: timedelta
    recommendation: str
    risk_level: str


@dataclass
class MarketFeatures:
    """Features extracted from market data"""

    market_id: int
    total_volume: float
    participant_count: int
    days_remaining: int
    price_volatility: float
    volume_trend: float
    social_sentiment: float
    news_sentiment: float
    historical_accuracy: float
    category_risk: float
    liquidity_score: float
    momentum_indicator: float


class AIPredictionService:
    """Advanced AI prediction service for market outcomes"""

    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            "total_volume",
            "participant_count",
            "days_remaining",
            "price_volatility",
            "volume_trend",
            "social_sentiment",
            "news_sentiment",
            "historical_accuracy",
            "category_risk",
            "liquidity_score",
            "momentum_indicator",
        ]

    async def initialize(self):
        """Initialize the AI prediction service"""
        logger.info("Initializing AI Prediction Service")

        # Load pre-trained models
        await self._load_models()

        # Start background tasks
        asyncio.create_task(self._update_models_periodically())
        asyncio.create_task(self._collect_market_data())

        logger.info("AI Prediction Service initialized successfully")

    async def _load_models(self):
        """Load pre-trained machine learning models"""
        try:
            # Load classification model for outcome prediction
            self.models["classifier"] = joblib.load("models/market_classifier.pkl")
            self.scalers["classifier"] = joblib.load("models/market_scaler.pkl")

            # Load regression model for confidence scoring
            self.models["regressor"] = joblib.load("models/confidence_regressor.pkl")
            self.scalers["regressor"] = joblib.load("models/confidence_scaler.pkl")

            logger.info("Pre-trained models loaded successfully")
        except FileNotFoundError:
            logger.warning("Pre-trained models not found, will train new models")
            await self._train_models()

    async def _train_models(self):
        """Train new machine learning models"""
        logger.info("Training new AI models...")

        # Get historical market data
        historical_data = await self._get_historical_market_data()

        if len(historical_data) < 100:
            logger.warning("Insufficient historical data for training")
            return

        # Prepare features and labels
        X, y_class, y_conf = self._prepare_training_data(historical_data)

        # Split data
        X_train, X_test, y_class_train, y_class_test, y_conf_train, y_conf_test = (
            train_test_split(X, y_class, y_conf, test_size=0.2, random_state=42)
        )

        # Train classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_class_train)

        # Train regressor for confidence
        regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_train, y_conf_train)

        # Evaluate models
        class_pred = classifier.predict(X_test)
        conf_pred = regressor.predict(X_test)

        accuracy = accuracy_score(y_class_test, class_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_class_test, class_pred, average="weighted"
        )

        logger.info(
            f"Model training completed - Accuracy: {accuracy:.3f}, F1: {f1:.3f}"
        )

        # Save models
        self.models["classifier"] = classifier
        self.models["regressor"] = regressor

        # Save scalers
        scaler_class = StandardScaler()
        scaler_conf = StandardScaler()
        scaler_class.fit(X_train)
        scaler_conf.fit(X_train)

        self.scalers["classifier"] = scaler_class
        self.scalers["regressor"] = scaler_conf

        # Save to disk
        joblib.dump(classifier, "models/market_classifier.pkl")
        joblib.dump(regressor, "models/market_regressor.pkl")
        joblib.dump(scaler_class, "models/market_scaler.pkl")
        joblib.dump(scaler_conf, "models/confidence_scaler.pkl")

    async def predict_market_outcome(
        self, market_id: int
    ) -> Optional[PredictionResult]:
        """Predict the outcome of a specific market"""
        try:
            # Get market features
            features = await self._extract_market_features(market_id)
            if not features:
                return None

            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)

            # Make prediction
            if "classifier" not in self.models:
                logger.warning("Models not loaded, cannot make prediction")
                return None

            # Scale features
            scaled_features = self.scalers["classifier"].transform([feature_vector])

            # Predict outcome
            outcome_prediction = self.models["classifier"].predict(scaled_features)[0]
            outcome_probabilities = self.models["classifier"].predict_proba(
                scaled_features
            )[0]

            # Predict confidence
            confidence_scaled = self.scalers["regressor"].transform([feature_vector])
            confidence = self.models["regressor"].predict(confidence_scaled)[0]
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            # Generate recommendation
            recommendation = self._generate_recommendation(
                outcome_prediction, confidence, features
            )

            # Determine risk level
            risk_level = self._calculate_risk_level(features, confidence)

            result = PredictionResult(
                market_id=market_id,
                predicted_outcome=outcome_prediction,
                confidence=confidence,
                probability_a=outcome_probabilities[0],
                probability_b=outcome_probabilities[1],
                features_used=self.feature_columns,
                model_version="v1.0.0",
                prediction_time=datetime.utcnow(),
                validity_duration=timedelta(hours=24),
                recommendation=recommendation,
                risk_level=risk_level,
            )

            # Cache prediction
            await self._cache_prediction(market_id, result)

            return result

        except Exception as e:
            logger.error(f"Error predicting market outcome: {e}")
            return None

    async def _extract_market_features(
        self, market_id: int
    ) -> Optional[MarketFeatures]:
        """Extract features from market data"""
        try:
            # Get market data from database
            market_data = await self._get_market_data(market_id)
            if not market_data:
                return None

            # Calculate features
            total_volume = market_data.get("total_volume", 0.0)
            participant_count = market_data.get("participant_count", 0)
            days_remaining = market_data.get("days_remaining", 0)

            # Calculate price volatility
            price_history = await self._get_price_history(market_id)
            price_volatility = self._calculate_volatility(price_history)

            # Calculate volume trend
            volume_trend = await self._calculate_volume_trend(market_id)

            # Get sentiment data
            social_sentiment = await self._get_social_sentiment(market_id)
            news_sentiment = await self._get_news_sentiment(market_id)

            # Calculate historical accuracy
            historical_accuracy = await self._calculate_historical_accuracy(market_id)

            # Calculate category risk
            category_risk = self._calculate_category_risk(
                market_data.get("category", "")
            )

            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(market_data)

            # Calculate momentum indicator
            momentum_indicator = self._calculate_momentum_indicator(price_history)

            return MarketFeatures(
                market_id=market_id,
                total_volume=total_volume,
                participant_count=participant_count,
                days_remaining=days_remaining,
                price_volatility=price_volatility,
                volume_trend=volume_trend,
                social_sentiment=social_sentiment,
                news_sentiment=news_sentiment,
                historical_accuracy=historical_accuracy,
                category_risk=category_risk,
                liquidity_score=liquidity_score,
                momentum_indicator=momentum_indicator,
            )

        except Exception as e:
            logger.error(f"Error extracting market features: {e}")
            return None

    def _prepare_feature_vector(self, features: MarketFeatures) -> List[float]:
        """Prepare feature vector for model input"""
        return [
            features.total_volume,
            features.participant_count,
            features.days_remaining,
            features.price_volatility,
            features.volume_trend,
            features.social_sentiment,
            features.news_sentiment,
            features.historical_accuracy,
            features.category_risk,
            features.liquidity_score,
            features.momentum_indicator,
        ]

    def _generate_recommendation(
        self, outcome: str, confidence: float, features: MarketFeatures
    ) -> str:
        """Generate trading recommendation based on prediction"""
        if confidence < 0.3:
            return "Low confidence prediction - consider waiting for more data"
        elif confidence < 0.6:
            return f"Moderate confidence - consider small position in {outcome}"
        elif confidence < 0.8:
            return f"High confidence - recommended position in {outcome}"
        else:
            return f"Very high confidence - strong recommendation for {outcome}"

    def _calculate_risk_level(self, features: MarketFeatures, confidence: float) -> str:
        """Calculate risk level for the prediction"""
        risk_score = 0

        # High volatility increases risk
        if features.price_volatility > 0.5:
            risk_score += 2
        elif features.price_volatility > 0.3:
            risk_score += 1

        # Low liquidity increases risk
        if features.liquidity_score < 0.3:
            risk_score += 2
        elif features.liquidity_score < 0.6:
            risk_score += 1

        # Low confidence increases risk
        if confidence < 0.5:
            risk_score += 2
        elif confidence < 0.7:
            risk_score += 1

        # Category risk
        risk_score += features.category_risk

        if risk_score >= 5:
            return "HIGH"
        elif risk_score >= 3:
            return "MEDIUM"
        else:
            return "LOW"

    async def _cache_prediction(self, market_id: int, prediction: PredictionResult):
        """Cache prediction result"""
        cache_key = f"ai_prediction:{market_id}"
        cache_data = {
            "predicted_outcome": prediction.predicted_outcome,
            "confidence": prediction.confidence,
            "probability_a": prediction.probability_a,
            "probability_b": prediction.probability_b,
            "recommendation": prediction.recommendation,
            "risk_level": prediction.risk_level,
            "prediction_time": prediction.prediction_time.isoformat(),
            "validity_duration": prediction.validity_duration.total_seconds(),
        }

        await self.redis.setex(
            cache_key,
            int(prediction.validity_duration.total_seconds()),
            str(cache_data),
        )

    async def get_cached_prediction(self, market_id: int) -> Optional[PredictionResult]:
        """Get cached prediction if available and valid"""
        cache_key = f"ai_prediction:{market_id}"
        cached_data = await self.redis.get(cache_key)

        if cached_data:
            try:
                import json

                data = json.loads(cached_data)  # Safe JSON parsing instead of eval()
                return PredictionResult(
                    market_id=market_id,
                    predicted_outcome=data["predicted_outcome"],
                    confidence=data["confidence"],
                    probability_a=data["probability_a"],
                    probability_b=data["probability_b"],
                    features_used=self.feature_columns,
                    model_version="v1.0.0",
                    prediction_time=datetime.fromisoformat(data["prediction_time"]),
                    validity_duration=timedelta(seconds=data["validity_duration"]),
                    recommendation=data["recommendation"],
                    risk_level=data["risk_level"],
                )
            except Exception as e:
                logger.error(f"Error parsing cached prediction: {e}")

        return None

    async def _update_models_periodically(self):
        """Periodically update models with new data"""
        while True:
            try:
                await asyncio.sleep(24 * 60 * 60)  # Update daily
                logger.info("Updating AI models with new data...")
                await self._train_models()
            except Exception as e:
                logger.error(f"Error updating models: {e}")

    async def _collect_market_data(self):
        """Collect market data for model training"""
        while True:
            try:
                await asyncio.sleep(60 * 60)  # Collect hourly
                # Implementation for collecting market data
                pass
            except Exception as e:
                logger.error(f"Error collecting market data: {e}")

    # Helper methods (implementations would depend on your data models)
    async def _get_market_data(self, market_id: int) -> Optional[Dict]:
        """Get market data from database"""
        # Implementation depends on your database models
        return {
            "total_volume": 100000.0,
            "participant_count": 150,
            "days_remaining": 30,
            "category": "technology",
        }

    async def _get_price_history(self, market_id: int) -> List[float]:
        """Get price history for market"""
        # Implementation depends on your data structure
        return [0.5, 0.52, 0.48, 0.55, 0.53]

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)

    async def _calculate_volume_trend(self, market_id: int) -> float:
        """Calculate volume trend"""
        # Implementation depends on your data
        return 0.1

    async def _get_social_sentiment(self, market_id: int) -> float:
        """Get social media sentiment"""
        # Implementation depends on your sentiment analysis
        return 0.6

    async def _get_news_sentiment(self, market_id: int) -> float:
        """Get news sentiment"""
        # Implementation depends on your sentiment analysis
        return 0.7

    async def _calculate_historical_accuracy(self, market_id: int) -> float:
        """Calculate historical prediction accuracy"""
        # Implementation depends on your historical data
        return 0.75

    def _calculate_category_risk(self, category: str) -> float:
        """Calculate risk based on market category"""
        risk_map = {
            "technology": 0.3,
            "finance": 0.4,
            "politics": 0.6,
            "sports": 0.2,
            "entertainment": 0.3,
        }
        return risk_map.get(category.lower(), 0.5)

    def _calculate_liquidity_score(self, market_data: Dict) -> float:
        """Calculate liquidity score"""
        # Implementation depends on your liquidity metrics
        return 0.8

    def _calculate_momentum_indicator(self, prices: List[float]) -> float:
        """Calculate momentum indicator"""
        if len(prices) < 2:
            return 0.0
        return (prices[-1] - prices[0]) / prices[0]

    async def _get_historical_market_data(self) -> List[Dict]:
        """Get historical market data for training"""
        # Implementation depends on your database
        return []

    def _prepare_training_data(
        self, historical_data: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for models"""
        # Implementation depends on your data structure
        X = np.random.rand(100, len(self.feature_columns))
        y_class = np.random.choice(["A", "B"], 100)
        y_conf = np.random.rand(100)
        return X, y_class, y_conf


# Factory function
async def get_ai_prediction_service(
    redis_client: redis.Redis, db_session: Session
) -> AIPredictionService:
    """Get AI prediction service instance"""
    service = AIPredictionService(redis_client, db_session)
    await service.initialize()
    return service
