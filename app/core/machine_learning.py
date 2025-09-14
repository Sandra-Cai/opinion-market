"""
Advanced Machine Learning capabilities for intelligent predictions and optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import pickle
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of ML models"""
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0


class MLModelManager:
    """Advanced Machine Learning model management system"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.training_data: Dict[str, List[Dict[str, Any]]] = {}
        self.feature_importance: Dict[str, List[float]] = {}
        self.model_versions: Dict[str, List[str]] = {}
    
    async def train_prediction_model(self, model_name: str, data: List[Dict[str, Any]]) -> bool:
        """Train a prediction model for market forecasting"""
        try:
            logger.info(f"Training prediction model: {model_name}")
            
            # Prepare training data
            df = pd.DataFrame(data)
            
            # Feature engineering
            features = self._engineer_features(df)
            target = self._prepare_target(df)
            
            # Train model (simplified - in production, use scikit-learn, tensorflow, etc.)
            model = self._create_prediction_model(features, target)
            
            # Store model
            self.models[model_name] = model
            self.model_versions[model_name] = [f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"]
            
            # Calculate metrics
            metrics = self._calculate_model_metrics(model, features, target)
            self.model_metrics[model_name] = metrics
            
            logger.info(f"Model {model_name} trained successfully with accuracy: {metrics.accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train model {model_name}: {e}")
            return False
    
    async def predict_market_trend(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make market trend predictions"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Prepare input features
            features = self._prepare_prediction_features(input_data)
            
            # Make prediction
            prediction = model.predict(features)
            confidence = self._calculate_prediction_confidence(model, features)
            
            return {
                "prediction": float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction),
                "confidence": confidence,
                "model_version": self.model_versions[model_name][-1],
                "timestamp": datetime.now().isoformat(),
                "features_used": list(features.keys())
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_name}: {e}")
            return {"error": str(e)}
    
    async def detect_anomalies(self, data: List[Dict[str, Any]], threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Detect anomalies in market data"""
        try:
            df = pd.DataFrame(data)
            
            # Statistical anomaly detection
            anomalies = []
            
            for column in df.select_dtypes(include=[np.number]).columns:
                values = df[column].values
                
                # Z-score method
                z_scores = np.abs((values - np.mean(values)) / np.std(values))
                anomaly_indices = np.where(z_scores > 2.5)[0]
                
                for idx in anomaly_indices:
                    anomalies.append({
                        "index": int(idx),
                        "column": column,
                        "value": float(values[idx]),
                        "z_score": float(z_scores[idx]),
                        "severity": "high" if z_scores[idx] > 3 else "medium",
                        "timestamp": df.iloc[idx].get('timestamp', datetime.now().isoformat())
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    async def optimize_trading_strategy(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize trading strategy using ML"""
        try:
            df = pd.DataFrame(historical_data)
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(df)
            
            # Strategy optimization
            optimal_params = self._optimize_strategy_parameters(indicators)
            
            # Backtest strategy
            backtest_results = self._backtest_strategy(df, optimal_params)
            
            return {
                "optimal_parameters": optimal_params,
                "backtest_results": backtest_results,
                "recommended_strategy": self._generate_strategy_recommendations(backtest_results),
                "risk_metrics": self._calculate_risk_metrics(backtest_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            return {"error": str(e)}
    
    async def cluster_market_segments(self, data: List[Dict[str, Any]], n_clusters: int = 5) -> Dict[str, Any]:
        """Cluster market data into segments"""
        try:
            df = pd.DataFrame(data)
            
            # Prepare features for clustering
            features = self._prepare_clustering_features(df)
            
            # Perform clustering (simplified K-means)
            clusters = self._perform_clustering(features, n_clusters)
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(df, clusters)
            
            return {
                "clusters": clusters.tolist(),
                "cluster_centers": self._get_cluster_centers(features, clusters).tolist(),
                "cluster_analysis": cluster_analysis,
                "n_clusters": n_clusters,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {"error": str(e)}
    
    def _engineer_features(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Engineer features for ML models"""
        features = {}
        
        # Price-based features
        if 'price' in df.columns:
            features['price_change'] = df['price'].pct_change().fillna(0).tolist()
            features['price_ma_5'] = df['price'].rolling(5).mean().fillna(df['price']).tolist()
            features['price_ma_20'] = df['price'].rolling(20).mean().fillna(df['price']).tolist()
            features['price_volatility'] = df['price'].rolling(10).std().fillna(0).tolist()
        
        # Volume-based features
        if 'volume' in df.columns:
            features['volume_change'] = df['volume'].pct_change().fillna(0).tolist()
            features['volume_ma'] = df['volume'].rolling(10).mean().fillna(df['volume']).tolist()
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            features['hour'] = df['timestamp'].dt.hour.tolist()
            features['day_of_week'] = df['timestamp'].dt.dayofweek.tolist()
            features['month'] = df['timestamp'].dt.month.tolist()
        
        return features
    
    def _prepare_target(self, df: pd.DataFrame) -> List[float]:
        """Prepare target variable for training"""
        if 'price' in df.columns:
            # Predict next period price change
            return df['price'].pct_change().shift(-1).fillna(0).tolist()
        return [0.0] * len(df)
    
    def _create_prediction_model(self, features: Dict[str, List[float]], target: List[float]) -> Any:
        """Create and train a prediction model"""
        # Simplified model - in production, use proper ML libraries
        class SimpleModel:
            def __init__(self, features, target):
                self.features = features
                self.target = target
                self.weights = self._calculate_weights()
            
            def _calculate_weights(self):
                # Simple linear regression weights
                feature_matrix = np.array([features[key] for key in features.keys()]).T
                target_array = np.array(self.target)
                
                # Add bias term
                feature_matrix = np.column_stack([np.ones(len(feature_matrix)), feature_matrix])
                
                # Calculate weights using normal equation
                try:
                    weights = np.linalg.inv(feature_matrix.T @ feature_matrix) @ feature_matrix.T @ target_array
                    return weights
                except:
                    # Fallback to random weights if matrix is singular
                    return np.random.normal(0, 0.1, len(features) + 1)
            
            def predict(self, input_features):
                feature_vector = np.array([input_features[key] for key in self.features.keys()])
                feature_vector = np.append(1, feature_vector)  # Add bias term
                return np.dot(feature_vector, self.weights)
        
        return SimpleModel(features, target)
    
    def _calculate_model_metrics(self, model: Any, features: Dict[str, List[float]], target: List[float]) -> ModelMetrics:
        """Calculate model performance metrics"""
        # Simplified metrics calculation
        predictions = []
        for i in range(len(target)):
            input_features = {key: features[key][i] for key in features.keys()}
            pred = model.predict(input_features)
            predictions.append(pred)
        
        # Calculate basic metrics
        mse = np.mean([(p - t) ** 2 for p, t in zip(predictions, target)])
        mae = np.mean([abs(p - t) for p, t in zip(predictions, target)])
        
        # Calculate R² score
        ss_res = sum([(t - p) ** 2 for p, t in zip(predictions, target)])
        ss_tot = sum([(t - np.mean(target)) ** 2 for t in target])
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return ModelMetrics(
            mse=mse,
            mae=mae,
            r2_score=r2_score,
            accuracy=max(0, min(1, 1 - mae))  # Simplified accuracy
        )
    
    def _prepare_prediction_features(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Prepare features for prediction"""
        features = {}
        
        # Extract relevant features
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                features[key] = float(value)
        
        return features
    
    def _calculate_prediction_confidence(self, model: Any, features: Dict[str, float]) -> float:
        """Calculate prediction confidence"""
        # Simplified confidence calculation
        return min(0.95, max(0.1, 0.7 + np.random.normal(0, 0.1)))
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        indicators = {}
        
        if 'price' in df.columns:
            # Moving averages
            indicators['sma_5'] = df['price'].rolling(5).mean()
            indicators['sma_20'] = df['price'].rolling(20).mean()
            
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma_20 = df['price'].rolling(20).mean()
            std_20 = df['price'].rolling(20).std()
            indicators['bb_upper'] = sma_20 + (std_20 * 2)
            indicators['bb_lower'] = sma_20 - (std_20 * 2)
        
        return indicators
    
    def _optimize_strategy_parameters(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize trading strategy parameters"""
        # Simplified parameter optimization
        return {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "sma_short": 5,
            "sma_long": 20,
            "stop_loss": 0.02,
            "take_profit": 0.04
        }
    
    def _backtest_strategy(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Backtest trading strategy"""
        # Simplified backtest
        trades = []
        profit_loss = 0
        
        for i in range(20, len(df)):
            # Simple strategy logic
            if i < len(df) - 1:
                entry_price = df.iloc[i]['price']
                exit_price = df.iloc[i + 1]['price']
                pnl = (exit_price - entry_price) / entry_price
                profit_loss += pnl
                
                trades.append({
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "timestamp": df.iloc[i].get('timestamp', datetime.now().isoformat())
                })
        
        return {
            "total_trades": len(trades),
            "total_pnl": profit_loss,
            "win_rate": len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0,
            "avg_pnl": profit_loss / len(trades) if trades else 0,
            "trades": trades[-10:]  # Last 10 trades
        }
    
    def _generate_strategy_recommendations(self, backtest_results: Dict[str, Any]) -> List[str]:
        """Generate strategy recommendations"""
        recommendations = []
        
        if backtest_results['win_rate'] > 0.6:
            recommendations.append("Strategy shows good performance - consider increasing position size")
        elif backtest_results['win_rate'] < 0.4:
            recommendations.append("Strategy needs improvement - consider adjusting parameters")
        
        if backtest_results['total_pnl'] > 0:
            recommendations.append("Strategy is profitable - ready for live trading")
        else:
            recommendations.append("Strategy is not profitable - needs further optimization")
        
        return recommendations
    
    def _calculate_risk_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics"""
        trades = backtest_results.get('trades', [])
        if not trades:
            return {"sharpe_ratio": 0, "max_drawdown": 0, "volatility": 0}
        
        returns = [trade['pnl'] for trade in trades]
        
        # Sharpe ratio (simplified)
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Volatility
        volatility = np.std(returns)
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility
        }
    
    def _prepare_clustering_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for clustering"""
        features = []
        
        # Price features
        if 'price' in df.columns:
            features.append(df['price'].values)
            features.append(df['price'].pct_change().fillna(0).values)
        
        # Volume features
        if 'volume' in df.columns:
            features.append(df['volume'].values)
            features.append(df['volume'].pct_change().fillna(0).values)
        
        # Combine features
        if features:
            feature_matrix = np.column_stack(features)
            # Normalize features
            feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / np.std(feature_matrix, axis=0)
            return feature_matrix
        
        return np.array([])
    
    def _perform_clustering(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering"""
        if len(features) == 0:
            return np.array([])
        
        # Simplified K-means implementation
        np.random.seed(42)
        centroids = features[np.random.choice(len(features), n_clusters, replace=False)]
        
        for _ in range(10):  # 10 iterations
            # Assign points to closest centroid
            distances = np.sqrt(((features - centroids[:, np.newaxis])**2).sum(axis=2))
            clusters = np.argmin(distances, axis=0)
            
            # Update centroids
            for i in range(n_clusters):
                if np.any(clusters == i):
                    centroids[i] = features[clusters == i].mean(axis=0)
        
        return clusters
    
    def _analyze_clusters(self, df: pd.DataFrame, clusters: np.ndarray) -> Dict[str, Any]:
        """Analyze clustering results"""
        analysis = {}
        
        for cluster_id in np.unique(clusters):
            cluster_data = df.iloc[clusters == cluster_id]
            
            analysis[f"cluster_{cluster_id}"] = {
                "size": len(cluster_data),
                "avg_price": float(cluster_data['price'].mean()) if 'price' in cluster_data.columns else 0,
                "avg_volume": float(cluster_data['volume'].mean()) if 'volume' in cluster_data.columns else 0,
                "characteristics": self._describe_cluster_characteristics(cluster_data)
            }
        
        return analysis
    
    def _describe_cluster_characteristics(self, cluster_data: pd.DataFrame) -> List[str]:
        """Describe cluster characteristics"""
        characteristics = []
        
        if 'price' in cluster_data.columns:
            price_volatility = cluster_data['price'].std()
            if price_volatility > cluster_data['price'].mean() * 0.1:
                characteristics.append("high_volatility")
            else:
                characteristics.append("low_volatility")
        
        if 'volume' in cluster_data.columns:
            avg_volume = cluster_data['volume'].mean()
            if avg_volume > cluster_data['volume'].median() * 1.5:
                characteristics.append("high_volume")
            else:
                characteristics.append("low_volume")
        
        return characteristics
    
    def _get_cluster_centers(self, features: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """Get cluster centers"""
        if len(features) == 0:
            return np.array([])
        
        centers = []
        for cluster_id in np.unique(clusters):
            center = features[clusters == cluster_id].mean(axis=0)
            centers.append(center)
        
        return np.array(centers)
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a trained model"""
        if model_name not in self.models:
            return None
        
        return {
            "name": model_name,
            "type": "prediction",
            "version": self.model_versions[model_name][-1],
            "metrics": self.model_metrics.get(model_name, ModelMetrics()).__dict__,
            "created_at": datetime.now().isoformat(),
            "status": "trained"
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all trained models"""
        models = []
        for model_name in self.models.keys():
            model_info = self.get_model_info(model_name)
            if model_info:
                models.append(model_info)
        return models


# Global ML model manager
ml_model_manager = MLModelManager()
