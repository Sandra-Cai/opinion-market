"""
AI Analytics Service
Provides machine learning models, predictive analytics, and intelligent insights
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis as redis_sync
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class AIModel:
    """AI model information"""

    model_id: str
    model_name: str
    model_type: str  # 'regression', 'classification', 'clustering', 'deep_learning'
    algorithm: str
    version: str
    training_data_size: int
    accuracy_score: float
    last_trained: datetime
    is_active: bool
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    created_at: datetime
    last_updated: datetime


@dataclass
class Prediction:
    """AI prediction result"""

    prediction_id: str
    model_id: str
    input_data: Dict[str, Any]
    prediction_value: Union[float, int, str]
    confidence_score: float
    prediction_type: str  # 'price', 'trend', 'risk', 'sentiment'
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class TrainingJob:
    """AI model training job"""

    job_id: str
    model_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float
    start_time: datetime
    end_time: Optional[datetime]
    training_metrics: Dict[str, Any]
    error_message: Optional[str]
    created_at: datetime
    last_updated: datetime


@dataclass
class FeatureSet:
    """Feature set for AI models"""

    feature_set_id: str
    feature_set_name: str
    features: List[str]
    feature_types: Dict[str, str]
    data_sources: List[str]
    last_updated: datetime
    created_at: datetime


class AIAnalyticsService:
    """Comprehensive AI analytics service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.models: Dict[str, AIModel] = {}
        self.predictions: Dict[str, List[Prediction]] = defaultdict(list)
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.feature_sets: Dict[str, FeatureSet] = {}

        # Model storage
        self.trained_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_encoders: Dict[str, Any] = {}

        # Performance tracking
        self.model_performance: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.prediction_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # AI specific data
        self.market_sentiment: Dict[str, Dict[str, float]] = {}
        self.risk_metrics: Dict[str, Dict[str, float]] = {}
        self.trend_analysis: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Initialize the AI analytics service"""
        logger.info("Initializing AI Analytics Service")

        # Load existing models and data
        await self._load_models()
        await self._load_feature_sets()
        await self._initialize_default_models()

        # Start background tasks
        asyncio.create_task(self._update_model_performance())
        asyncio.create_task(self._monitor_model_health())
        asyncio.create_task(self._update_sentiment_analysis())

        logger.info("AI Analytics Service initialized successfully")

    async def create_ai_model(
        self,
        model_name: str,
        model_type: str,
        algorithm: str,
        hyperparameters: Dict[str, Any] = None,
    ) -> AIModel:
        """Create a new AI model"""
        try:
            model_id = (
                f"ai_model_{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )

            if hyperparameters is None:
                hyperparameters = self._get_default_hyperparameters(algorithm)

            model = AIModel(
                model_id=model_id,
                model_name=model_name,
                model_type=model_type,
                algorithm=algorithm,
                version="1.0.0",
                training_data_size=0,
                accuracy_score=0.0,
                last_trained=datetime.utcnow(),
                is_active=False,
                hyperparameters=hyperparameters,
                feature_importance={},
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.models[model_id] = model
            await self._cache_ai_model(model)

            logger.info(f"Created AI model {model_name}")
            return model

        except Exception as e:
            logger.error(f"Error creating AI model: {e}")
            raise

    async def train_model(
        self,
        model_id: str,
        training_data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
    ) -> TrainingJob:
        """Train an AI model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")

            model_info = self.models[model_id]

            # Create training job
            job_id = (
                f"training_job_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )
            training_job = TrainingJob(
                job_id=job_id,
                model_id=model_id,
                status="running",
                progress=0.0,
                start_time=datetime.utcnow(),
                end_time=None,
                training_metrics={},
                error_message=None,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.training_jobs[job_id] = training_job
            await self._cache_training_job(training_job)

            # Start training in background
            asyncio.create_task(
                self._train_model_async(
                    training_job, training_data, target_column, test_size
                )
            )

            logger.info(f"Started training job {job_id} for model {model_id}")
            return training_job

        except Exception as e:
            logger.error(f"Error starting model training: {e}")
            raise

    async def make_prediction(
        self, model_id: str, input_data: Dict[str, Any], prediction_type: str = "price"
    ) -> Prediction:
        """Make a prediction using a trained model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")

            if model_id not in self.trained_models:
                raise ValueError(f"Model {model_id} is not trained")

            model_info = self.models[model_id]
            trained_model = self.trained_models[model_id]
            scaler = self.scalers.get(model_id)

            # Prepare input data
            features = self._prepare_features(input_data, model_info)

            if scaler:
                features = scaler.transform(features.reshape(1, -1))

            # Make prediction
            if model_info.model_type == "classification":
                prediction_value = trained_model.predict(features)[0]
                confidence_score = (
                    np.max(trained_model.predict_proba(features))
                    if hasattr(trained_model, "predict_proba")
                    else 0.5
                )
            else:
                prediction_value = float(trained_model.predict(features)[0])
                confidence_score = 0.8  # Default confidence for regression

            # Create prediction record
            prediction_id = (
                f"prediction_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )
            prediction = Prediction(
                prediction_id=prediction_id,
                model_id=model_id,
                input_data=input_data,
                prediction_value=prediction_value,
                confidence_score=confidence_score,
                prediction_type=prediction_type,
                timestamp=datetime.utcnow(),
                metadata={
                    "model_type": model_info.model_type,
                    "algorithm": model_info.algorithm,
                    "version": model_info.version,
                },
            )

            self.predictions[model_id].append(prediction)
            await self._cache_prediction(prediction)

            logger.info(f"Made prediction {prediction_id} using model {model_id}")
            return prediction

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    async def get_market_sentiment(self, market_id: str) -> Dict[str, Any]:
        """Get market sentiment analysis"""
        try:
            sentiment_data = self.market_sentiment.get(market_id, {})

            if not sentiment_data:
                # Generate sentiment data
                sentiment_data = await self._generate_sentiment_analysis(market_id)
                self.market_sentiment[market_id] = sentiment_data

            return {
                "market_id": market_id,
                "sentiment_score": sentiment_data.get("sentiment_score", 0.0),
                "sentiment_label": sentiment_data.get("sentiment_label", "neutral"),
                "confidence": sentiment_data.get("confidence", 0.0),
                "factors": sentiment_data.get("factors", {}),
                "last_updated": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            raise

    async def get_risk_analysis(self, market_id: str) -> Dict[str, Any]:
        """Get comprehensive risk analysis"""
        try:
            risk_data = self.risk_metrics.get(market_id, {})

            if not risk_data:
                # Generate risk metrics
                risk_data = await self._generate_risk_analysis(market_id)
                self.risk_metrics[market_id] = risk_data

            return {
                "market_id": market_id,
                "volatility": risk_data.get("volatility", 0.0),
                "var_95": risk_data.get("var_95", 0.0),
                "expected_shortfall": risk_data.get("expected_shortfall", 0.0),
                "max_drawdown": risk_data.get("max_drawdown", 0.0),
                "sharpe_ratio": risk_data.get("sharpe_ratio", 0.0),
                "risk_level": risk_data.get("risk_level", "medium"),
                "last_updated": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting risk analysis: {e}")
            raise

    async def get_trend_analysis(self, market_id: str) -> Dict[str, Any]:
        """Get trend analysis and forecasting"""
        try:
            trend_data = self.trend_analysis.get(market_id, {})

            if not trend_data:
                # Generate trend analysis
                trend_data = await self._generate_trend_analysis(market_id)
                self.trend_analysis[market_id] = trend_data

            return {
                "market_id": market_id,
                "trend_direction": trend_data.get("trend_direction", "neutral"),
                "trend_strength": trend_data.get("trend_strength", 0.0),
                "support_levels": trend_data.get("support_levels", []),
                "resistance_levels": trend_data.get("resistance_levels", []),
                "forecast": trend_data.get("forecast", {}),
                "confidence": trend_data.get("confidence", 0.0),
                "last_updated": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting trend analysis: {e}")
            raise

    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")

            model_info = self.models[model_id]
            performance_history = self.model_performance.get(model_id, [])

            return {
                "model_id": model_id,
                "model_name": model_info.model_name,
                "algorithm": model_info.algorithm,
                "current_accuracy": model_info.accuracy_score,
                "training_data_size": model_info.training_data_size,
                "last_trained": model_info.last_trained.isoformat(),
                "performance_history": list(performance_history),
                "feature_importance": model_info.feature_importance,
                "is_active": model_info.is_active,
                "last_updated": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            raise

    def _get_default_hyperparameters(self, algorithm: str) -> Dict[str, Any]:
        """Get default hyperparameters for an algorithm"""
        defaults = {
            "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            },
            "xgboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            },
            "lightgbm": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            },
            "neural_network": {
                "hidden_layer_sizes": (100, 50),
                "activation": "relu",
                "solver": "adam",
                "max_iter": 1000,
                "random_state": 42,
            },
        }
        return defaults.get(algorithm, {})

    async def _train_model_async(
        self,
        training_job: TrainingJob,
        training_data: pd.DataFrame,
        target_column: str,
        test_size: float,
    ):
        """Train model asynchronously"""
        try:
            model_id = training_job.model_id
            model_info = self.models[model_id]

            # Update progress
            training_job.progress = 10.0
            await self._cache_training_job(training_job)

            # Prepare data
            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            training_job.progress = 30.0
            await self._cache_training_job(training_job)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            training_job.progress = 50.0
            await self._cache_training_job(training_job)

            # Train model
            model = self._create_model_instance(
                model_info.algorithm, model_info.hyperparameters
            )
            model.fit(X_train_scaled, y_train)

            training_job.progress = 80.0
            await self._cache_training_job(training_job)

            # Evaluate model
            if model_info.model_type == "classification":
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
            else:
                y_pred = model.predict(X_test_scaled)
                accuracy = 1.0 / (1.0 + mean_squared_error(y_test, y_pred))

            # Update model
            model_info.accuracy_score = accuracy
            model_info.training_data_size = len(training_data)
            model_info.last_trained = datetime.utcnow()
            model_info.is_active = True

            # Store trained model and scaler
            self.trained_models[model_id] = model
            self.scalers[model_id] = scaler

            # Calculate feature importance
            if hasattr(model, "feature_importances_"):
                feature_names = X.columns.tolist()
                importances = model.feature_importances_
                model_info.feature_importance = dict(zip(feature_names, importances))

            # Update training job
            training_job.status = "completed"
            training_job.progress = 100.0
            training_job.end_time = datetime.utcnow()
            training_job.training_metrics = {
                "accuracy": accuracy,
                "test_size": len(X_test),
                "train_size": len(X_train),
            }

            await self._cache_training_job(training_job)
            await self._cache_ai_model(model_info)

            logger.info(
                f"Completed training job {training_job.job_id} for model {model_id}"
            )

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            training_job.status = "failed"
            training_job.error_message = str(e)
            training_job.end_time = datetime.utcnow()
            await self._cache_training_job(training_job)

    def _create_model_instance(self, algorithm: str, hyperparameters: Dict[str, Any]):
        """Create model instance based on algorithm"""
        if algorithm == "random_forest":
            return RandomForestRegressor(**hyperparameters)
        elif algorithm == "gradient_boosting":
            return GradientBoostingRegressor(**hyperparameters)
        elif algorithm == "xgboost":
            return xgb.XGBRegressor(**hyperparameters)
        elif algorithm == "lightgbm":
            return lgb.LGBMRegressor(**hyperparameters)
        elif algorithm == "neural_network":
            return MLPRegressor(**hyperparameters)
        elif algorithm == "linear_regression":
            return LinearRegression(**hyperparameters)
        else:
            return RandomForestRegressor()

    def _prepare_features(
        self, input_data: Dict[str, Any], model_info: AIModel
    ) -> np.ndarray:
        """Prepare features for prediction"""
        # This is a simplified feature preparation
        # In practice, you would have more sophisticated feature engineering
        features = []
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            else:
                features.append(0.0)

        return np.array(features)

    async def _generate_sentiment_analysis(self, market_id: str) -> Dict[str, Any]:
        """Generate market sentiment analysis"""
        # Simulate sentiment analysis
        sentiment_score = np.random.normal(0, 0.3)
        sentiment_score = max(-1, min(1, sentiment_score))

        if sentiment_score > 0.3:
            sentiment_label = "bullish"
        elif sentiment_score < -0.3:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"

        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "confidence": abs(sentiment_score),
            "factors": {
                "news_sentiment": np.random.normal(0, 0.2),
                "social_sentiment": np.random.normal(0, 0.2),
                "technical_sentiment": np.random.normal(0, 0.2),
            },
        }

    async def _generate_risk_analysis(self, market_id: str) -> Dict[str, Any]:
        """Generate risk analysis metrics"""
        # Simulate risk metrics
        volatility = np.random.uniform(0.1, 0.5)
        var_95 = np.random.uniform(0.02, 0.1)
        expected_shortfall = var_95 * 1.5
        max_drawdown = np.random.uniform(0.05, 0.25)
        sharpe_ratio = np.random.uniform(-1, 2)

        # Determine risk level
        if volatility > 0.4 or var_95 > 0.08:
            risk_level = "high"
        elif volatility > 0.2 or var_95 > 0.04:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "volatility": volatility,
            "var_95": var_95,
            "expected_shortfall": expected_shortfall,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "risk_level": risk_level,
        }

    async def _generate_trend_analysis(self, market_id: str) -> Dict[str, Any]:
        """Generate trend analysis"""
        # Simulate trend analysis
        trend_direction = np.random.choice(["bullish", "bearish", "neutral"])
        trend_strength = np.random.uniform(0.3, 0.9)

        support_levels = [np.random.uniform(0.8, 0.95) for _ in range(3)]
        resistance_levels = [np.random.uniform(1.05, 1.2) for _ in range(3)]

        forecast = {
            "1d": np.random.uniform(0.95, 1.05),
            "1w": np.random.uniform(0.9, 1.1),
            "1m": np.random.uniform(0.8, 1.2),
        }

        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "support_levels": sorted(support_levels),
            "resistance_levels": sorted(resistance_levels),
            "forecast": forecast,
            "confidence": trend_strength,
        }

    # Background tasks
    async def _update_model_performance(self):
        """Update model performance metrics"""
        while True:
            try:
                # Update performance metrics for all models
                for model_id, model_info in self.models.items():
                    if model_info.is_active:
                        # Simulate performance updates
                        performance_change = np.random.normal(0, 0.01)
                        new_accuracy = max(
                            0, min(1, model_info.accuracy_score + performance_change)
                        )

                        if abs(new_accuracy - model_info.accuracy_score) > 0.001:
                            model_info.accuracy_score = new_accuracy
                            model_info.last_updated = datetime.utcnow()

                            # Store performance history
                            self.model_performance[model_id].append(
                                {
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "accuracy": new_accuracy,
                                }
                            )

                            await self._cache_ai_model(model_info)

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error updating model performance: {e}")
                await asyncio.sleep(600)

    async def _monitor_model_health(self):
        """Monitor model health and performance"""
        while True:
            try:
                # Monitor model health (in practice, this would check model drift, etc.)
                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error monitoring model health: {e}")
                await asyncio.sleep(7200)

    async def _update_sentiment_analysis(self):
        """Update sentiment analysis"""
        while True:
            try:
                # Update sentiment analysis (in practice, this would fetch new data)
                await asyncio.sleep(1800)  # Update every 30 minutes

            except Exception as e:
                logger.error(f"Error updating sentiment analysis: {e}")
                await asyncio.sleep(3600)

    # Helper methods
    async def _load_models(self):
        """Load AI models from database"""
        pass

    async def _load_feature_sets(self):
        """Load feature sets from database"""
        pass

    async def _initialize_default_models(self):
        """Initialize default AI models"""
        pass

    # Caching methods
    async def _cache_ai_model(self, model: AIModel):
        """Cache AI model"""
        try:
            cache_key = f"ai_model:{model.model_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "model_name": model.model_name,
                        "model_type": model.model_type,
                        "algorithm": model.algorithm,
                        "version": model.version,
                        "training_data_size": model.training_data_size,
                        "accuracy_score": model.accuracy_score,
                        "last_trained": model.last_trained.isoformat(),
                        "is_active": model.is_active,
                        "hyperparameters": model.hyperparameters,
                        "feature_importance": model.feature_importance,
                        "created_at": model.created_at.isoformat(),
                        "last_updated": model.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching AI model: {e}")

    async def _cache_prediction(self, prediction: Prediction):
        """Cache prediction"""
        try:
            cache_key = f"prediction:{prediction.prediction_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "model_id": prediction.model_id,
                        "input_data": prediction.input_data,
                        "prediction_value": prediction.prediction_value,
                        "confidence_score": prediction.confidence_score,
                        "prediction_type": prediction.prediction_type,
                        "timestamp": prediction.timestamp.isoformat(),
                        "metadata": prediction.metadata,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching prediction: {e}")

    async def _cache_training_job(self, job: TrainingJob):
        """Cache training job"""
        try:
            cache_key = f"training_job:{job.job_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "model_id": job.model_id,
                        "status": job.status,
                        "progress": job.progress,
                        "start_time": job.start_time.isoformat(),
                        "end_time": job.end_time.isoformat() if job.end_time else None,
                        "training_metrics": job.training_metrics,
                        "error_message": job.error_message,
                        "created_at": job.created_at.isoformat(),
                        "last_updated": job.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching training job: {e}")


# Factory function
async def get_ai_analytics_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> AIAnalyticsService:
    """Get AI analytics service instance"""
    service = AIAnalyticsService(redis_client, db_session)
    await service.initialize()
    return service
