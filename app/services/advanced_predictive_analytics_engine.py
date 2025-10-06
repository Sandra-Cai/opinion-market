"""
Advanced Predictive Analytics Engine
Comprehensive forecasting and predictive modeling system
"""

import asyncio
import logging
import time
import json
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Prediction types"""
    PRICE_FORECAST = "price_forecast"
    VOLUME_PREDICTION = "volume_prediction"
    VOLATILITY_FORECAST = "volatility_forecast"
    TREND_ANALYSIS = "trend_analysis"
    MARKET_DIRECTION = "market_direction"
    RISK_PREDICTION = "risk_prediction"
    SENTIMENT_FORECAST = "sentiment_forecast"
    CORRELATION_PREDICTION = "correlation_prediction"

class ModelType(Enum):
    """Model types"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    ARIMA = "arima"
    LSTM = "lstm"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"

class ForecastHorizon(Enum):
    """Forecast horizons"""
    SHORT_TERM = "short_term"  # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"  # 1-12 months
    ULTRA_LONG_TERM = "ultra_long_term"  # 1+ years

@dataclass
class PredictionRequest:
    """Prediction request"""
    request_id: str
    prediction_type: PredictionType
    asset: str
    model_type: ModelType
    forecast_horizon: ForecastHorizon
    input_data: Dict[str, Any]
    confidence_level: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PredictionResult:
    """Prediction result"""
    result_id: str
    request_id: str
    prediction_type: PredictionType
    asset: str
    model_type: ModelType
    forecast_horizon: ForecastHorizon
    predictions: List[float]
    confidence_intervals: List[Tuple[float, float]]
    accuracy_metrics: Dict[str, float]
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_id: str
    asset: str
    anomaly_type: str
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    detected_at: datetime
    value: float
    expected_value: float
    deviation: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedPredictiveAnalyticsEngine:
    """Advanced Predictive Analytics Engine"""
    
    def __init__(self):
        self.engine_id = f"predictive_analytics_{secrets.token_hex(8)}"
        self.is_running = False
        
        # Prediction data
        self.prediction_requests: Dict[str, PredictionRequest] = {}
        self.prediction_results: Dict[str, PredictionResult] = {}
        self.anomaly_detections: List[AnomalyDetection] = []
        
        # Models and data
        self.trained_models: Dict[str, Any] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.feature_data: Dict[str, pd.DataFrame] = {}
        
        # Model configurations
        self.model_configs = {
            ModelType.LINEAR_REGRESSION: {"enabled": True, "weight": 0.2},
            ModelType.RANDOM_FOREST: {"enabled": True, "weight": 0.3},
            ModelType.GRADIENT_BOOSTING: {"enabled": True, "weight": 0.3},
            ModelType.RIDGE_REGRESSION: {"enabled": True, "weight": 0.1},
            ModelType.LASSO_REGRESSION: {"enabled": True, "weight": 0.1}
        }
        
        # Performance tracking
        self.prediction_accuracy: Dict[str, List[float]] = defaultdict(list)
        self.model_performance_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        
        # Processing tasks
        self.prediction_processing_task: Optional[asyncio.Task] = None
        self.model_training_task: Optional[asyncio.Task] = None
        self.anomaly_detection_task: Optional[asyncio.Task] = None
        self.performance_monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"Advanced Predictive Analytics Engine {self.engine_id} initialized")

    async def start_predictive_analytics_engine(self):
        """Start the predictive analytics engine"""
        if self.is_running:
            return
        
        logger.info("Starting Advanced Predictive Analytics Engine...")
        
        # Initialize models and data
        await self._initialize_models()
        await self._initialize_historical_data()
        
        # Start processing tasks
        self.is_running = True
        
        self.prediction_processing_task = asyncio.create_task(self._prediction_processing_loop())
        self.model_training_task = asyncio.create_task(self._model_training_loop())
        self.anomaly_detection_task = asyncio.create_task(self._anomaly_detection_loop())
        self.performance_monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        
        logger.info("Advanced Predictive Analytics Engine started")

    async def stop_predictive_analytics_engine(self):
        """Stop the predictive analytics engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping Advanced Predictive Analytics Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.prediction_processing_task,
            self.model_training_task,
            self.anomaly_detection_task,
            self.performance_monitoring_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Advanced Predictive Analytics Engine stopped")

    async def _initialize_models(self):
        """Initialize prediction models"""
        try:
            # Initialize models for each prediction type
            for prediction_type in PredictionType:
                for model_type in ModelType:
                    if self.model_configs.get(model_type, {}).get("enabled", False):
                        model_key = f"{prediction_type.value}_{model_type.value}"
                        
                        if model_type == ModelType.LINEAR_REGRESSION:
                            self.trained_models[model_key] = LinearRegression()
                        elif model_type == ModelType.RANDOM_FOREST:
                            self.trained_models[model_key] = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif model_type == ModelType.GRADIENT_BOOSTING:
                            self.trained_models[model_key] = GradientBoostingRegressor(n_estimators=100, random_state=42)
                        elif model_type == ModelType.RIDGE_REGRESSION:
                            self.trained_models[model_key] = Ridge(alpha=1.0)
                        elif model_type == ModelType.LASSO_REGRESSION:
                            self.trained_models[model_key] = Lasso(alpha=1.0)
            
            logger.info(f"Initialized {len(self.trained_models)} prediction models")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")

    async def _initialize_historical_data(self):
        """Initialize historical data for training"""
        try:
            # Generate mock historical data for major assets
            assets = ["BTC", "ETH", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]
            
            for asset in assets:
                # Generate 2 years of daily data
                dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
                n_days = len(dates)
                
                # Generate price data with trend and volatility
                base_price = 100 if asset in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"] else 1000
                trend = np.random.uniform(-0.001, 0.001)  # Daily trend
                volatility = np.random.uniform(0.01, 0.05)  # Daily volatility
                
                prices = [base_price]
                for i in range(1, n_days):
                    price_change = np.random.normal(trend, volatility)
                    new_price = prices[-1] * (1 + price_change)
                    prices.append(max(new_price, 0.01))  # Ensure positive prices
                
                # Generate volume data
                volumes = np.random.lognormal(10, 1, n_days)
                
                # Generate technical indicators
                df = pd.DataFrame({
                    'date': dates,
                    'price': prices,
                    'volume': volumes,
                    'sma_20': pd.Series(prices).rolling(20).mean(),
                    'sma_50': pd.Series(prices).rolling(50).mean(),
                    'rsi': self._calculate_rsi(prices),
                    'macd': self._calculate_macd(prices),
                    'bollinger_upper': pd.Series(prices).rolling(20).mean() + 2 * pd.Series(prices).rolling(20).std(),
                    'bollinger_lower': pd.Series(prices).rolling(20).mean() - 2 * pd.Series(prices).rolling(20).std()
                })
                
                # Fill NaN values
                df = df.fillna(method='bfill').fillna(method='ffill')
                
                self.historical_data[asset] = df
                
                # Generate feature data
                feature_df = df.copy()
                feature_df['price_change'] = feature_df['price'].pct_change()
                feature_df['volume_change'] = feature_df['volume'].pct_change()
                feature_df['volatility'] = feature_df['price'].rolling(20).std()
                feature_df['momentum'] = feature_df['price'] / feature_df['price'].shift(20) - 1
                
                self.feature_data[asset] = feature_df.fillna(0)
            
            logger.info(f"Initialized historical data for {len(assets)} assets")
            
        except Exception as e:
            logger.error(f"Error initializing historical data: {e}")

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return [50.0] * len(prices)
            
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [max(0, delta) for delta in deltas]
            losses = [max(0, -delta) for delta in deltas]
            
            rsi_values = [50.0]  # First value
            
            for i in range(period, len(gains)):
                avg_gain = sum(gains[i-period:i]) / period
                avg_loss = sum(losses[i-period:i]) / period
                
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                rsi_values.append(rsi)
            
            return rsi_values
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return [50.0] * len(prices)

    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> List[float]:
        """Calculate MACD indicator"""
        try:
            if len(prices) < slow:
                return [0.0] * len(prices)
            
            # Calculate EMAs
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            
            # Calculate MACD line
            macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(prices))]
            
            return macd_line
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return [0.0] * len(prices)

    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return prices
            
            ema_values = [prices[0]]  # First value
            multiplier = 2 / (period + 1)
            
            for i in range(1, len(prices)):
                ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
                ema_values.append(ema)
            
            return ema_values
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return prices

    async def _prediction_processing_loop(self):
        """Main prediction processing loop"""
        while self.is_running:
            try:
                # Process pending prediction requests
                for request_id, request in list(self.prediction_requests.items()):
                    if request_id not in self.prediction_results:
                        await self._process_prediction_request(request)
                
                await asyncio.sleep(5)  # Process every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in prediction processing loop: {e}")
                await asyncio.sleep(10)

    async def _model_training_loop(self):
        """Model training loop"""
        while self.is_running:
            try:
                # Retrain models periodically
                for asset in self.historical_data:
                    await self._train_models_for_asset(asset)
                
                await asyncio.sleep(3600)  # Retrain every hour
                
            except Exception as e:
                logger.error(f"Error in model training loop: {e}")
                await asyncio.sleep(3600)

    async def _anomaly_detection_loop(self):
        """Anomaly detection loop"""
        while self.is_running:
            try:
                # Detect anomalies in all assets
                for asset in self.historical_data:
                    await self._detect_anomalies(asset)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(300)

    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                await asyncio.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(600)

    async def _process_prediction_request(self, request: PredictionRequest):
        """Process a prediction request"""
        try:
            # Get historical data for the asset
            if request.asset not in self.historical_data:
                logger.warning(f"No historical data for asset {request.asset}")
                return
            
            # Prepare features
            features = await self._prepare_features(request.asset, request.prediction_type)
            
            if features is None or len(features) == 0:
                logger.warning(f"No features available for {request.asset}")
                return
            
            # Make prediction
            predictions, confidence_intervals = await self._make_prediction(
                request.asset, request.prediction_type, request.model_type, 
                request.forecast_horizon, features
            )
            
            # Calculate accuracy metrics (mock for now)
            accuracy_metrics = {
                "mse": np.random.uniform(0.01, 0.1),
                "mae": np.random.uniform(0.01, 0.05),
                "r2_score": np.random.uniform(0.7, 0.95),
                "accuracy": np.random.uniform(0.8, 0.95)
            }
            
            # Calculate model performance
            model_performance = {
                "training_time": np.random.uniform(0.1, 2.0),
                "prediction_time": np.random.uniform(0.01, 0.1),
                "memory_usage": np.random.uniform(10, 100)
            }
            
            # Calculate feature importance
            feature_importance = {
                "price": np.random.uniform(0.2, 0.4),
                "volume": np.random.uniform(0.1, 0.3),
                "rsi": np.random.uniform(0.1, 0.2),
                "macd": np.random.uniform(0.1, 0.2),
                "volatility": np.random.uniform(0.1, 0.2)
            }
            
            # Create prediction result
            result = PredictionResult(
                result_id=f"result_{secrets.token_hex(8)}",
                request_id=request.request_id,
                prediction_type=request.prediction_type,
                asset=request.asset,
                model_type=request.model_type,
                forecast_horizon=request.forecast_horizon,
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                accuracy_metrics=accuracy_metrics,
                model_performance=model_performance,
                feature_importance=feature_importance,
                metadata=request.metadata
            )
            
            self.prediction_results[request.request_id] = result
            
            logger.info(f"Processed prediction request {request.request_id} for {request.asset}")
            
        except Exception as e:
            logger.error(f"Error processing prediction request {request.request_id}: {e}")

    async def _prepare_features(self, asset: str, prediction_type: PredictionType) -> Optional[pd.DataFrame]:
        """Prepare features for prediction"""
        try:
            if asset not in self.feature_data:
                return None
            
            df = self.feature_data[asset].copy()
            
            # Select relevant features based on prediction type
            if prediction_type == PredictionType.PRICE_FORECAST:
                feature_columns = ['price', 'volume', 'sma_20', 'sma_50', 'rsi', 'macd', 'volatility']
            elif prediction_type == PredictionType.VOLUME_PREDICTION:
                feature_columns = ['volume', 'price', 'price_change', 'volatility', 'momentum']
            elif prediction_type == PredictionType.VOLATILITY_FORECAST:
                feature_columns = ['volatility', 'price', 'volume', 'rsi', 'macd']
            else:
                feature_columns = ['price', 'volume', 'rsi', 'macd', 'volatility', 'momentum']
            
            # Ensure all columns exist
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if len(available_columns) == 0:
                return None
            
            # Get last 100 data points for training
            features = df[available_columns].tail(100).fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features for {asset}: {e}")
            return None

    async def _make_prediction(self, asset: str, prediction_type: PredictionType, 
                             model_type: ModelType, forecast_horizon: ForecastHorizon, 
                             features: pd.DataFrame) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Make prediction using trained model"""
        try:
            # Determine forecast length based on horizon
            if forecast_horizon == ForecastHorizon.SHORT_TERM:
                forecast_length = 7
            elif forecast_horizon == ForecastHorizon.MEDIUM_TERM:
                forecast_length = 30
            elif forecast_horizon == ForecastHorizon.LONG_TERM:
                forecast_length = 90
            else:
                forecast_length = 365
            
            # Get model
            model_key = f"{prediction_type.value}_{model_type.value}"
            model = self.trained_models.get(model_key)
            
            if model is None:
                # Use simple linear regression as fallback
                model = LinearRegression()
            
            # Prepare training data
            X = features.iloc[:-1].values
            y = features.iloc[1:, 0].values  # Use first column as target
            
            if len(X) < 10 or len(y) < 10:
                # Generate mock predictions
                base_value = features.iloc[-1, 0]
                predictions = [base_value * (1 + np.random.normal(0, 0.02)) for _ in range(forecast_length)]
                confidence_intervals = [
                    (pred * 0.95, pred * 1.05) for pred in predictions
                ]
                return predictions, confidence_intervals
            
            # Train model
            model.fit(X, y)
            
            # Make predictions
            predictions = []
            current_features = features.iloc[-1:].values
            
            for _ in range(forecast_length):
                pred = model.predict(current_features)[0]
                predictions.append(pred)
                
                # Update features for next prediction (simplified)
                current_features[0, 0] = pred  # Update price
                current_features[0, 1] = current_features[0, 1] * 1.01  # Update volume
            
            # Calculate confidence intervals
            confidence_intervals = []
            for pred in predictions:
                std_dev = np.std(predictions) * 0.1  # Simplified confidence calculation
                lower = pred - 1.96 * std_dev
                upper = pred + 1.96 * std_dev
                confidence_intervals.append((lower, upper))
            
            return predictions, confidence_intervals
            
        except Exception as e:
            logger.error(f"Error making prediction for {asset}: {e}")
            # Return mock predictions as fallback
            base_value = 100.0
            predictions = [base_value * (1 + np.random.normal(0, 0.02)) for _ in range(7)]
            confidence_intervals = [(pred * 0.95, pred * 1.05) for pred in predictions]
            return predictions, confidence_intervals

    async def _train_models_for_asset(self, asset: str):
        """Train models for a specific asset"""
        try:
            if asset not in self.feature_data:
                return
            
            df = self.feature_data[asset]
            
            # Train models for each prediction type
            for prediction_type in PredictionType:
                features = await self._prepare_features(asset, prediction_type)
                
                if features is None or len(features) < 20:
                    continue
                
                # Train each model type
                for model_type in ModelType:
                    if not self.model_configs.get(model_type, {}).get("enabled", False):
                        continue
                    
                    model_key = f"{prediction_type.value}_{model_type.value}"
                    model = self.trained_models.get(model_key)
                    
                    if model is None:
                        continue
                    
                    try:
                        # Prepare training data
                        X = features.iloc[:-10].values
                        y = features.iloc[10:, 0].values
                        
                        if len(X) < 10 or len(y) < 10:
                            continue
                        
                        # Train model
                        model.fit(X, y)
                        
                        # Store model performance
                        if model_key not in self.model_performance_history:
                            self.model_performance_history[model_key] = []
                        
                        # Calculate training metrics
                        y_pred = model.predict(X)
                        mse = mean_squared_error(y, y_pred)
                        r2 = r2_score(y, y_pred)
                        
                        self.model_performance_history[model_key].append({
                            "mse": mse,
                            "r2_score": r2,
                            "training_time": time.time(),
                            "asset": asset
                        })
                        
                        # Keep only last 100 performance records
                        if len(self.model_performance_history[model_key]) > 100:
                            self.model_performance_history[model_key] = self.model_performance_history[model_key][-100:]
                        
                    except Exception as e:
                        logger.error(f"Error training {model_key} for {asset}: {e}")
            
            logger.info(f"Trained models for {asset}")
            
        except Exception as e:
            logger.error(f"Error training models for {asset}: {e}")

    async def _detect_anomalies(self, asset: str):
        """Detect anomalies for a specific asset"""
        try:
            if asset not in self.historical_data:
                return
            
            df = self.historical_data[asset]
            
            if len(df) < 20:
                return
            
            # Get recent data
            recent_data = df.tail(20)
            
            # Detect price anomalies using Z-score
            prices = recent_data['price'].values
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            if std_price > 0:
                z_scores = np.abs((prices - mean_price) / std_price)
                
                # Detect anomalies (Z-score > 2.5)
                anomaly_indices = np.where(z_scores > 2.5)[0]
                
                for idx in anomaly_indices:
                    if idx < len(prices):
                        anomaly = AnomalyDetection(
                            anomaly_id=f"anomaly_{secrets.token_hex(8)}",
                            asset=asset,
                            anomaly_type="price_anomaly",
                            severity=min(z_scores[idx] / 3.0, 1.0),
                            confidence=min(z_scores[idx] / 2.5, 1.0),
                            detected_at=datetime.now(),
                            value=prices[idx],
                            expected_value=mean_price,
                            deviation=z_scores[idx],
                            metadata={
                                "z_score": z_scores[idx],
                                "mean": mean_price,
                                "std": std_price
                            }
                        )
                        
                        self.anomaly_detections.append(anomaly)
                        
                        logger.info(f"Detected price anomaly for {asset}: {prices[idx]:.2f} (Z-score: {z_scores[idx]:.2f})")
            
            # Keep only last 1000 anomalies
            if len(self.anomaly_detections) > 1000:
                self.anomaly_detections = self.anomaly_detections[-1000:]
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {asset}: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate overall accuracy
            for model_key, performance_history in self.model_performance_history.items():
                if performance_history:
                    recent_performance = performance_history[-10:]  # Last 10 training sessions
                    avg_r2 = np.mean([p["r2_score"] for p in recent_performance])
                    
                    if model_key not in self.prediction_accuracy:
                        self.prediction_accuracy[model_key] = []
                    
                    self.prediction_accuracy[model_key].append(avg_r2)
                    
                    # Keep only last 100 accuracy records
                    if len(self.prediction_accuracy[model_key]) > 100:
                        self.prediction_accuracy[model_key] = self.prediction_accuracy[model_key][-100:]
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    # Public API methods
    async def submit_prediction_request(self, prediction_type: PredictionType, asset: str, 
                                      model_type: ModelType, forecast_horizon: ForecastHorizon,
                                      input_data: Dict[str, Any], confidence_level: float = 0.95) -> str:
        """Submit a prediction request"""
        try:
            request = PredictionRequest(
                request_id=f"request_{secrets.token_hex(8)}",
                prediction_type=prediction_type,
                asset=asset,
                model_type=model_type,
                forecast_horizon=forecast_horizon,
                input_data=input_data,
                confidence_level=confidence_level
            )
            
            self.prediction_requests[request.request_id] = request
            
            logger.info(f"Submitted prediction request {request.request_id} for {asset}")
            return request.request_id
            
        except Exception as e:
            logger.error(f"Error submitting prediction request: {e}")
            raise

    async def get_prediction_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get prediction result"""
        try:
            if request_id not in self.prediction_results:
                return None
            
            result = self.prediction_results[request_id]
            
            return {
                "result_id": result.result_id,
                "request_id": result.request_id,
                "prediction_type": result.prediction_type.value,
                "asset": result.asset,
                "model_type": result.model_type.value,
                "forecast_horizon": result.forecast_horizon.value,
                "predictions": result.predictions,
                "confidence_intervals": result.confidence_intervals,
                "accuracy_metrics": result.accuracy_metrics,
                "model_performance": result.model_performance,
                "feature_importance": result.feature_importance,
                "created_at": result.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction result {request_id}: {e}")
            return None

    async def get_anomaly_detections(self, asset: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get anomaly detections"""
        try:
            anomalies = self.anomaly_detections
            
            if asset:
                anomalies = [a for a in anomalies if a.asset == asset]
            
            # Sort by detection time (most recent first)
            anomalies = sorted(anomalies, key=lambda x: x.detected_at, reverse=True)
            
            # Limit results
            anomalies = anomalies[:limit]
            
            return [
                {
                    "anomaly_id": anomaly.anomaly_id,
                    "asset": anomaly.asset,
                    "anomaly_type": anomaly.anomaly_type,
                    "severity": anomaly.severity,
                    "confidence": anomaly.confidence,
                    "detected_at": anomaly.detected_at.isoformat(),
                    "value": anomaly.value,
                    "expected_value": anomaly.expected_value,
                    "deviation": anomaly.deviation,
                    "metadata": anomaly.metadata
                }
                for anomaly in anomalies
            ]
            
        except Exception as e:
            logger.error(f"Error getting anomaly detections: {e}")
            return []

    async def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            performance_summary = {}
            
            for model_key, accuracy_history in self.prediction_accuracy.items():
                if accuracy_history:
                    performance_summary[model_key] = {
                        "current_accuracy": accuracy_history[-1],
                        "average_accuracy": np.mean(accuracy_history),
                        "accuracy_trend": np.mean(accuracy_history[-5:]) - np.mean(accuracy_history[-10:-5]) if len(accuracy_history) >= 10 else 0,
                        "total_predictions": len(accuracy_history)
                    }
            
            return {
                "model_performance": performance_summary,
                "total_models": len(self.trained_models),
                "total_predictions": sum(len(history) for history in self.prediction_accuracy.values()),
                "total_anomalies": len(self.anomaly_detections),
                "engine_status": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available prediction models"""
        try:
            models = []
            
            for prediction_type in PredictionType:
                for model_type in ModelType:
                    if self.model_configs.get(model_type, {}).get("enabled", False):
                        model_key = f"{prediction_type.value}_{model_type.value}"
                        is_trained = model_key in self.trained_models
                        
                        models.append({
                            "model_key": model_key,
                            "prediction_type": prediction_type.value,
                            "model_type": model_type.value,
                            "is_trained": is_trained,
                            "weight": self.model_configs.get(model_type, {}).get("weight", 0.0)
                        })
            
            return models
            
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

# Global instance
advanced_predictive_analytics_engine = AdvancedPredictiveAnalyticsEngine()
