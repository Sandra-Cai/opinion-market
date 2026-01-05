"""
Advanced Analytics Engine with Machine Learning
Comprehensive analytics with predictive modeling and insights
"""

import asyncio
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import logging

# Machine Learning imports (with fallbacks)
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsInsight:
    """Analytics insight data structure"""
    insight_type: str
    title: str
    description: str
    confidence: float
    impact: str  # 'low', 'medium', 'high'
    recommendations: List[str] = field(default_factory=list)
    data_points: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PredictionResult:
    """Prediction result data structure"""
    metric_name: str
    predicted_value: float
    confidence: float
    time_horizon: int  # minutes
    trend: str
    factors: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    metric_name: str
    anomaly_score: float
    is_anomaly: bool
    expected_value: float
    actual_value: float
    severity: str  # 'low', 'medium', 'high'
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedAnalyticsEngine:
    """Advanced analytics engine with machine learning capabilities"""
    
    def __init__(self):
        self.data_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.insights: List[AnalyticsInsight] = []
        self.predictions: List[PredictionResult] = []
        self.anomalies: List[AnomalyDetection] = []
        
        # Analytics configuration
        self.config = {
            "prediction_horizons": [15, 30, 60, 120],  # minutes
            "anomaly_threshold": 2.5,  # standard deviations
            "min_data_points": 50,
            "model_retrain_interval": 3600,  # seconds
            "insight_confidence_threshold": 0.7
        }
        
        # Performance tracking
        self.performance_metrics = {
            "predictions_generated": 0,
            "insights_generated": 0,
            "anomalies_detected": 0,
            "model_accuracy": {}
        }
        
        self.analytics_active = False
        self.analytics_task: Optional[asyncio.Task] = None
        
    async def start_analytics(self):
        """Start the analytics engine"""
        if self.analytics_active:
            logger.warning("Analytics engine already active")
            return
            
        self.analytics_active = True
        self.analytics_task = asyncio.create_task(self._analytics_loop())
        logger.info("Advanced analytics engine started")
        
    async def stop_analytics(self):
        """Stop the analytics engine"""
        self.analytics_active = False
        if self.analytics_task:
            self.analytics_task.cancel()
            try:
                await self.analytics_task
            except asyncio.CancelledError:
                pass
        logger.info("Advanced analytics engine stopped")
        
    async def _analytics_loop(self):
        """Main analytics processing loop"""
        while self.analytics_active:
            try:
                # Generate predictions
                await self._generate_predictions()
                
                # Detect anomalies
                await self._detect_anomalies()
                
                # Generate insights
                await self._generate_insights()
                
                # Retrain models if needed
                await self._retrain_models()
                
                # Wait before next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}")
                await asyncio.sleep(600)  # Wait longer on error
                
    def add_data_point(self, metric_name: str, value: float, timestamp: datetime = None, metadata: Dict[str, Any] = None):
        """Add a data point for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
            
        data_point = {
            "value": value,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        self.data_history[metric_name].append(data_point)
        
    async def _generate_predictions(self):
        """Generate predictions using machine learning models"""
        if not ML_AVAILABLE:
            return
            
        try:
            for metric_name, history in self.data_history.items():
                if len(history) < self.config["min_data_points"]:
                    continue
                    
                # Prepare data for modeling
                X, y = self._prepare_prediction_data(metric_name)
                if X is None or len(X) < 20:
                    continue
                    
                # Train or update model
                model = await self._get_or_train_model(metric_name, X, y)
                if model is None:
                    continue
                    
                # Generate predictions for different horizons
                for horizon in self.config["prediction_horizons"]:
                    prediction = await self._predict_future_value(
                        model, metric_name, horizon, X, y
                    )
                    if prediction:
                        self.predictions.append(prediction)
                        self.performance_metrics["predictions_generated"] += 1
                        
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            
    def _prepare_prediction_data(self, metric_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare data for prediction modeling"""
        try:
            history = list(self.data_history[metric_name])
            if len(history) < 20:
                return None, None
                
            # Create time series features
            values = [dp["value"] for dp in history]
            timestamps = [dp["timestamp"].timestamp() for dp in history]
            
            # Create feature matrix
            X = []
            y = []
            
            window_size = 10
            for i in range(window_size, len(values)):
                # Features: previous values, time features, trends
                features = []
                
                # Previous values
                features.extend(values[i-window_size:i])
                
                # Time features
                current_time = timestamps[i]
                features.append(current_time % 86400)  # Time of day
                features.append(current_time % 604800)  # Day of week
                
                # Trend features
                recent_values = values[i-5:i]
                if len(recent_values) >= 2:
                    trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                    features.append(trend)
                else:
                    features.append(0)
                    
                # Moving averages
                if len(values[i-10:i]) >= 5:
                    ma_5 = statistics.mean(values[i-5:i])
                    ma_10 = statistics.mean(values[i-10:i])
                    features.extend([ma_5, ma_10, ma_5 - ma_10])
                else:
                    features.extend([0, 0, 0])
                    
                X.append(features)
                y.append(values[i])
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            return None, None
            
    async def _get_or_train_model(self, metric_name: str, X: np.ndarray, y: np.ndarray) -> Optional[Any]:
        """Get existing model or train a new one"""
        try:
            if metric_name in self.models:
                return self.models[metric_name]
                
            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble model
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and scaler
            self.models[metric_name] = model
            self.scalers[metric_name] = scaler
            self.performance_metrics["model_accuracy"][metric_name] = r2
            
            logger.info(f"Trained model for {metric_name}: R² = {r2:.3f}, MSE = {mse:.3f}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training model for {metric_name}: {e}")
            return None
            
    async def _predict_future_value(self, model: Any, metric_name: str, horizon: int, 
                                  X: np.ndarray, y: np.ndarray) -> Optional[PredictionResult]:
        """Predict future value for a specific horizon"""
        try:
            # Use the most recent data point as base
            latest_features = X[-1].copy()
            
            # Adjust time features for future prediction
            current_time = time.time()
            future_time = current_time + (horizon * 60)
            latest_features[-3] = future_time % 86400  # Time of day
            latest_features[-2] = future_time % 604800  # Day of week
            
            # Scale features
            if metric_name in self.scalers:
                latest_features_scaled = self.scalers[metric_name].transform([latest_features])
                predicted_value = model.predict(latest_features_scaled)[0]
            else:
                predicted_value = model.predict([latest_features])[0]
                
            # Calculate confidence based on model accuracy
            confidence = self.performance_metrics["model_accuracy"].get(metric_name, 0.5)
            
            # Determine trend
            recent_values = y[-10:] if len(y) >= 10 else y
            if len(recent_values) >= 2:
                current_trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                if current_trend > 0.01:
                    trend = "increasing"
                elif current_trend < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"
                
            # Calculate factor importance
            if hasattr(model, 'feature_importances_'):
                factors = {
                    "historical_values": float(np.mean(model.feature_importances_[:10])),
                    "time_features": float(np.mean(model.feature_importances_[10:12])),
                    "trend": float(model.feature_importances_[12]) if len(model.feature_importances_) > 12 else 0.0
                }
            else:
                factors = {"historical_values": 0.7, "time_features": 0.2, "trend": 0.1}
                
            return PredictionResult(
                metric_name=metric_name,
                predicted_value=float(predicted_value),
                confidence=float(confidence),
                time_horizon=horizon,
                trend=trend,
                factors=factors
            )
            
        except Exception as e:
            logger.error(f"Error predicting future value: {e}")
            return None
            
    async def _detect_anomalies(self):
        """Detect anomalies in the data"""
        try:
            for metric_name, history in self.data_history.items():
                if len(history) < 20:
                    continue
                    
                recent_data = list(history)[-50:]  # Last 50 points
                values = [dp["value"] for dp in recent_data]
                
                if len(values) < 10:
                    continue
                    
                # Calculate statistical thresholds
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else 0
                
                if std_val == 0:
                    continue
                    
                # Check for anomalies in recent values
                for i, data_point in enumerate(recent_data[-5:]):  # Check last 5 points
                    value = data_point["value"]
                    z_score = abs((value - mean_val) / std_val)
                    
                    if z_score > self.config["anomaly_threshold"]:
                        # Determine severity
                        if z_score > 4:
                            severity = "high"
                        elif z_score > 3:
                            severity = "medium"
                        else:
                            severity = "low"
                            
                        anomaly = AnomalyDetection(
                            metric_name=metric_name,
                            anomaly_score=float(z_score),
                            is_anomaly=True,
                            expected_value=float(mean_val),
                            actual_value=float(value),
                            severity=severity,
                            description=f"Value {value:.2f} is {z_score:.1f} standard deviations from mean {mean_val:.2f}",
                            timestamp=data_point["timestamp"]
                        )
                        
                        self.anomalies.append(anomaly)
                        self.performance_metrics["anomalies_detected"] += 1
                        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            
    async def _generate_insights(self):
        """Generate analytical insights"""
        try:
            insights = []
            
            # Performance insights
            if "performance_score" in self.data_history:
                recent_scores = [dp["value"] for dp in list(self.data_history["performance_score"])[-10:]]
                if recent_scores:
                    avg_score = statistics.mean(recent_scores)
                    if avg_score > 90:
                        insights.append(AnalyticsInsight(
                            insight_type="performance",
                            title="Excellent System Performance",
                            description=f"System performance score averaging {avg_score:.1f}/100",
                            confidence=0.95,
                            impact="high",
                            recommendations=["Continue current optimization strategies", "Monitor for any degradation"]
                        ))
                    elif avg_score < 70:
                        insights.append(AnalyticsInsight(
                            insight_type="performance",
                            title="Performance Needs Attention",
                            description=f"System performance score averaging {avg_score:.1f}/100",
                            confidence=0.9,
                            impact="high",
                            recommendations=["Review system resources", "Check for bottlenecks", "Consider scaling"]
                        ))
            
            # Cache insights
            if "cache_hit_rate" in self.data_history:
                recent_hit_rates = [dp["value"] for dp in list(self.data_history["cache_hit_rate"])[-10:]]
                if recent_hit_rates:
                    avg_hit_rate = statistics.mean(recent_hit_rates)
                    if avg_hit_rate > 85:
                        insights.append(AnalyticsInsight(
                            insight_type="cache",
                            title="Optimal Cache Performance",
                            description=f"Cache hit rate averaging {avg_hit_rate:.1f}%",
                            confidence=0.9,
                            impact="medium",
                            recommendations=["Cache is performing well", "Consider increasing cache size for even better performance"]
                        ))
                    elif avg_hit_rate < 60:
                        insights.append(AnalyticsInsight(
                            insight_type="cache",
                            title="Cache Optimization Needed",
                            description=f"Cache hit rate averaging {avg_hit_rate:.1f}%",
                            confidence=0.85,
                            impact="medium",
                            recommendations=["Increase cache size", "Review cache eviction policies", "Optimize cache keys"]
                        ))
            
            # Resource utilization insights
            if "cpu_usage" in self.data_history and "memory_usage" in self.data_history:
                recent_cpu = [dp["value"] for dp in list(self.data_history["cpu_usage"])[-10:]]
                recent_memory = [dp["value"] for dp in list(self.data_history["memory_usage"])[-10:]]
                
                if recent_cpu and recent_memory:
                    avg_cpu = statistics.mean(recent_cpu)
                    avg_memory = statistics.mean(recent_memory)
                    
                    if avg_cpu > 80 and avg_memory > 80:
                        insights.append(AnalyticsInsight(
                            insight_type="resources",
                            title="High Resource Utilization",
                            description=f"CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%",
                            confidence=0.9,
                            impact="high",
                            recommendations=["Consider scaling resources", "Optimize application code", "Review resource allocation"]
                        ))
                    elif avg_cpu < 30 and avg_memory < 50:
                        insights.append(AnalyticsInsight(
                            insight_type="resources",
                            title="Low Resource Utilization",
                            description=f"CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%",
                            confidence=0.8,
                            impact="low",
                            recommendations=["System can handle more load", "Consider cost optimization", "Monitor for scaling opportunities"]
                        ))
            
            # Add insights to collection
            for insight in insights:
                if insight.confidence >= self.config["insight_confidence_threshold"]:
                    self.insights.append(insight)
                    self.performance_metrics["insights_generated"] += 1
                    
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            
    async def _retrain_models(self):
        """Retrain models periodically"""
        try:
            current_time = time.time()
            
            for metric_name, model in self.models.items():
                # Check if model needs retraining
                if hasattr(model, '_last_trained'):
                    if current_time - model._last_trained < self.config["model_retrain_interval"]:
                        continue
                        
                # Retrain model
                X, y = self._prepare_prediction_data(metric_name)
                if X is not None and len(X) >= 20:
                    await self._get_or_train_model(metric_name, X, y)
                    model._last_trained = current_time
                    
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        try:
            # Get recent predictions
            recent_predictions = [p for p in self.predictions if 
                                (datetime.now() - p.created_at).total_seconds() < 3600]
            
            # Get recent insights
            recent_insights = [i for i in self.insights if 
                             (datetime.now() - i.created_at).total_seconds() < 3600]
            
            # Get recent anomalies
            recent_anomalies = [a for a in self.anomalies if 
                              (datetime.now() - a.timestamp).total_seconds() < 3600]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "analytics_active": self.analytics_active,
                "data_points_collected": sum(len(history) for history in self.data_history.values()),
                "models_trained": len(self.models),
                "recent_predictions": len(recent_predictions),
                "recent_insights": len(recent_insights),
                "recent_anomalies": len(recent_anomalies),
                "performance_metrics": self.performance_metrics,
                "ml_available": ML_AVAILABLE,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {"error": str(e)}


# Global instance
advanced_analytics_engine = AdvancedAnalyticsEngine()


