import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import asyncio
from dataclasses import dataclass

from app.core.database import SessionLocal
from app.models.market import Market, MarketStatus
from app.models.trade import Trade
from app.models.user import User
from app.services.market_data_feed import get_market_data_feed

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Result of a market prediction"""
    market_id: int
    predicted_price_a: float
    predicted_price_b: float
    confidence: float
    prediction_horizon: str  # 1h, 24h, 7d
    features_used: List[str]
    model_type: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class UserBehaviorProfile:
    """User behavior analysis profile"""
    user_id: int
    risk_tolerance: float  # 0-1 scale
    trading_frequency: float
    preferred_market_categories: List[str]
    average_trade_size: float
    win_rate: float
    holding_period: float  # average days
    sentiment_bias: float  # -1 to 1
    volatility_preference: float
    metadata: Dict[str, Any]

@dataclass
class TradingRecommendation:
    """Automated trading recommendation"""
    user_id: int
    market_id: int
    recommendation_type: str  # buy, sell, hold
    confidence: float
    reasoning: str
    expected_return: float
    risk_level: str  # low, medium, high
    time_horizon: str  # short, medium, long
    timestamp: datetime
    metadata: Dict[str, Any]

class MachineLearningService:
    """Comprehensive machine learning service for prediction markets"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.user_profiles = {}
        self.prediction_cache = {}
        self.model_performance = {}
        self.feature_importance = {}
        
        # Model configurations
        self.model_configs = {
            'price_prediction': {
                'type': 'gradient_boosting',
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6
                }
            },
            'volume_prediction': {
                'type': 'random_forest',
                'params': {
                    'n_estimators': 50,
                    'max_depth': 10
                }
            },
            'user_behavior': {
                'type': 'linear_regression',
                'params': {}
            }
        }
    
    async def initialize(self):
        """Initialize the ML service"""
        await self._load_models()
        await self._start_model_training_loop()
        logger.info("Machine Learning service initialized")
    
    async def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load models from disk if they exist
            for model_name in self.model_configs.keys():
                model_path = f"models/{model_name}_model.pkl"
                scaler_path = f"models/{model_name}_scaler.pkl"
                
                try:
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                    logger.info(f"Loaded model: {model_name}")
                except FileNotFoundError:
                    logger.info(f"Model not found: {model_name}, will train new model")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def _start_model_training_loop(self):
        """Start periodic model training"""
        asyncio.create_task(self._model_training_loop())
    
    async def _model_training_loop(self):
        """Periodic model training loop"""
        while True:
            try:
                await self._train_all_models()
                await asyncio.sleep(3600)  # Train every hour
            except Exception as e:
                logger.error(f"Error in model training loop: {e}")
                await asyncio.sleep(3600)
    
    async def _train_all_models(self):
        """Train all ML models"""
        try:
            # Train price prediction model
            await self._train_price_prediction_model()
            
            # Train volume prediction model
            await self._train_volume_prediction_model()
            
            # Train user behavior model
            await self._train_user_behavior_model()
            
            logger.info("All models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    async def _train_price_prediction_model(self):
        """Train price prediction model"""
        try:
            # Get historical market data
            market_data = await self._get_historical_market_data()
            
            if len(market_data) < 100:  # Need sufficient data
                logger.warning("Insufficient data for price prediction model training")
                return
            
            # Prepare features and targets
            features, targets = self._prepare_price_prediction_data(market_data)
            
            if len(features) == 0:
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            config = self.model_configs['price_prediction']
            if config['type'] == 'gradient_boosting':
                model = GradientBoostingRegressor(**config['params'])
            else:
                model = RandomForestRegressor(**config['params'])
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model and performance
            self.models['price_prediction'] = model
            self.scalers['price_prediction'] = scaler
            self.model_performance['price_prediction'] = {
                'mse': mse,
                'r2': r2,
                'last_trained': datetime.utcnow()
            }
            
            # Save model
            joblib.dump(model, 'models/price_prediction_model.pkl')
            joblib.dump(scaler, 'models/price_prediction_scaler.pkl')
            
            logger.info(f"Price prediction model trained - MSE: {mse:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"Error training price prediction model: {e}")
    
    async def _train_volume_prediction_model(self):
        """Train volume prediction model"""
        try:
            # Get historical volume data
            volume_data = await self._get_historical_volume_data()
            
            if len(volume_data) < 50:
                return
            
            # Prepare features and targets
            features, targets = self._prepare_volume_prediction_data(volume_data)
            
            if len(features) == 0:
                return
            
            # Split and scale data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            config = self.model_configs['volume_prediction']
            model = RandomForestRegressor(**config['params'])
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store model
            self.models['volume_prediction'] = model
            self.scalers['volume_prediction'] = scaler
            self.model_performance['volume_prediction'] = {
                'mse': mse,
                'r2': r2,
                'last_trained': datetime.utcnow()
            }
            
            # Save model
            joblib.dump(model, 'models/volume_prediction_model.pkl')
            joblib.dump(scaler, 'models/volume_prediction_scaler.pkl')
            
            logger.info(f"Volume prediction model trained - MSE: {mse:.4f}, R²: {r2:.4f}")
            
        except Exception as e:
            logger.error(f"Error training volume prediction model: {e}")
    
    async def _train_user_behavior_model(self):
        """Train user behavior analysis model"""
        try:
            # Get user behavior data
            user_data = await self._get_user_behavior_data()
            
            if len(user_data) < 20:
                return
            
            # Prepare features
            features, targets = self._prepare_user_behavior_data(user_data)
            
            if len(features) == 0:
                return
            
            # Train model
            config = self.model_configs['user_behavior']
            model = LinearRegression(**config['params'])
            model.fit(features, targets)
            
            # Store model
            self.models['user_behavior'] = model
            self.model_performance['user_behavior'] = {
                'last_trained': datetime.utcnow()
            }
            
            # Save model
            joblib.dump(model, 'models/user_behavior_model.pkl')
            
            logger.info("User behavior model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training user behavior model: {e}")
    
    async def predict_market_price(self, market_id: int, horizon: str = "24h") -> Optional[PredictionResult]:
        """Predict market price for a specific horizon"""
        try:
            # Check cache first
            cache_key = f"{market_id}_{horizon}"
            if cache_key in self.prediction_cache:
                cached_pred = self.prediction_cache[cache_key]
                if (datetime.utcnow() - cached_pred.timestamp).seconds < 300:  # 5 minutes
                    return cached_pred
            
            # Get current market data
            market_data = await self._get_market_features(market_id)
            
            if not market_data:
                return None
            
            # Make prediction
            if 'price_prediction' not in self.models:
                return None
            
            model = self.models['price_prediction']
            scaler = self.scalers['price_prediction']
            
            # Scale features
            features_scaled = scaler.transform([market_data])
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            
            # Calculate confidence based on model performance
            confidence = self._calculate_prediction_confidence(market_id, horizon)
            
            # Create result
            result = PredictionResult(
                market_id=market_id,
                predicted_price_a=prediction,
                predicted_price_b=1.0 - prediction,
                confidence=confidence,
                prediction_horizon=horizon,
                features_used=list(market_data.keys()),
                model_type='gradient_boosting',
                timestamp=datetime.utcnow(),
                metadata={
                    'model_performance': self.model_performance.get('price_prediction', {}),
                    'feature_importance': self._get_feature_importance()
                }
            )
            
            # Cache result
            self.prediction_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting market price: {e}")
            return None
    
    async def analyze_user_behavior(self, user_id: int) -> Optional[UserBehaviorProfile]:
        """Analyze user behavior and create profile"""
        try:
            # Check cache
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
                if (datetime.utcnow() - profile.timestamp).days < 1:  # Cache for 1 day
                    return profile
            
            # Get user trading data
            user_data = await self._get_user_trading_data(user_id)
            
            if not user_data:
                return None
            
            # Calculate behavior metrics
            risk_tolerance = self._calculate_risk_tolerance(user_data)
            trading_frequency = self._calculate_trading_frequency(user_data)
            preferred_categories = self._get_preferred_categories(user_data)
            avg_trade_size = self._calculate_average_trade_size(user_data)
            win_rate = self._calculate_win_rate(user_data)
            holding_period = self._calculate_holding_period(user_data)
            sentiment_bias = self._calculate_sentiment_bias(user_data)
            volatility_preference = self._calculate_volatility_preference(user_data)
            
            # Create profile
            profile = UserBehaviorProfile(
                user_id=user_id,
                risk_tolerance=risk_tolerance,
                trading_frequency=trading_frequency,
                preferred_market_categories=preferred_categories,
                average_trade_size=avg_trade_size,
                win_rate=win_rate,
                holding_period=holding_period,
                sentiment_bias=sentiment_bias,
                volatility_preference=volatility_preference,
                metadata={
                    'total_trades': len(user_data['trades']),
                    'total_volume': user_data['total_volume'],
                    'analysis_date': datetime.utcnow().isoformat()
                }
            )
            
            # Cache profile
            self.user_profiles[user_id] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior: {e}")
            return None
    
    async def generate_trading_recommendation(self, user_id: int, market_id: int) -> Optional[TradingRecommendation]:
        """Generate personalized trading recommendation"""
        try:
            # Get user profile
            user_profile = await self.analyze_user_behavior(user_id)
            if not user_profile:
                return None
            
            # Get market prediction
            prediction = await self.predict_market_price(market_id, "24h")
            if not prediction:
                return None
            
            # Get current market data
            market_data_feed = get_market_data_feed()
            current_data = await market_data_feed.get_market_data(market_id)
            if not current_data:
                return None
            
            # Generate recommendation based on user profile and prediction
            recommendation = self._generate_recommendation(
                user_profile, prediction, current_data
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating trading recommendation: {e}")
            return None
    
    def _generate_recommendation(self, user_profile: UserBehaviorProfile, 
                               prediction: PredictionResult, 
                               current_data: Any) -> TradingRecommendation:
        """Generate trading recommendation based on profile and prediction"""
        current_price_a = current_data.price_a
        predicted_price_a = prediction.predicted_price_a
        
        # Calculate expected return
        price_change = (predicted_price_a - current_price_a) / current_price_a
        expected_return = price_change * 100
        
        # Determine recommendation type
        if abs(expected_return) < 2:  # Less than 2% expected return
            recommendation_type = "hold"
            confidence = 0.3
        elif expected_return > 5 and user_profile.risk_tolerance > 0.5:
            recommendation_type = "buy"
            confidence = min(prediction.confidence * 0.8, 0.9)
        elif expected_return < -5 and user_profile.risk_tolerance > 0.3:
            recommendation_type = "sell"
            confidence = min(prediction.confidence * 0.8, 0.9)
        else:
            recommendation_type = "hold"
            confidence = 0.5
        
        # Determine risk level
        if abs(expected_return) < 3:
            risk_level = "low"
        elif abs(expected_return) < 8:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        # Determine time horizon
        if user_profile.holding_period < 1:
            time_horizon = "short"
        elif user_profile.holding_period < 7:
            time_horizon = "medium"
        else:
            time_horizon = "long"
        
        # Generate reasoning
        reasoning = self._generate_recommendation_reasoning(
            recommendation_type, expected_return, user_profile, prediction
        )
        
        return TradingRecommendation(
            user_id=user_profile.user_id,
            market_id=prediction.market_id,
            recommendation_type=recommendation_type,
            confidence=confidence,
            reasoning=reasoning,
            expected_return=expected_return,
            risk_level=risk_level,
            time_horizon=time_horizon,
            timestamp=datetime.utcnow(),
            metadata={
                'prediction_confidence': prediction.confidence,
                'user_risk_tolerance': user_profile.risk_tolerance,
                'market_volatility': current_data.volatility
            }
        )
    
    def _generate_recommendation_reasoning(self, recommendation_type: str, 
                                         expected_return: float, 
                                         user_profile: UserBehaviorProfile,
                                         prediction: PredictionResult) -> str:
        """Generate human-readable reasoning for recommendation"""
        if recommendation_type == "buy":
            return f"Market predicted to increase by {expected_return:.1f}% with {prediction.confidence:.1%} confidence. " \
                   f"Your risk tolerance ({user_profile.risk_tolerance:.1%}) supports this position."
        elif recommendation_type == "sell":
            return f"Market predicted to decrease by {abs(expected_return):.1f}% with {prediction.confidence:.1%} confidence. " \
                   f"Consider reducing exposure given the expected decline."
        else:
            return f"Expected return of {expected_return:.1f}% is below threshold for action. " \
                   f"Monitor for better opportunities or consider other markets."
    
    async def _get_historical_market_data(self) -> List[Dict]:
        """Get historical market data for training"""
        try:
            db = SessionLocal()
            
            # Get trades from last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            trades = db.query(Trade).filter(
                Trade.created_at >= thirty_days_ago
            ).order_by(Trade.created_at).all()
            
            # Group by market and time periods
            market_data = {}
            for trade in trades:
                market_id = trade.market_id
                hour = trade.created_at.replace(minute=0, second=0, microsecond=0)
                
                if market_id not in market_data:
                    market_data[market_id] = {}
                
                if hour not in market_data[market_id]:
                    market_data[market_id][hour] = {
                        'trades': [],
                        'volume': 0,
                        'avg_price': 0
                    }
                
                market_data[market_id][hour]['trades'].append(trade)
                market_data[market_id][hour]['volume'] += trade.total_value
                market_data[market_id][hour]['avg_price'] = (
                    (market_data[market_id][hour]['avg_price'] * (len(market_data[market_id][hour]['trades']) - 1) + 
                     trade.price_per_share) / len(market_data[market_id][hour]['trades'])
                )
            
            db.close()
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting historical market data: {e}")
            return []
    
    def _prepare_price_prediction_data(self, market_data: Dict) -> Tuple[List, List]:
        """Prepare features and targets for price prediction"""
        features = []
        targets = []
        
        for market_id, hourly_data in market_data.items():
            hours = sorted(hourly_data.keys())
            
            for i in range(1, len(hours)):
                # Features from previous hour
                prev_hour = hours[i-1]
                current_hour = hours[i]
                
                prev_data = hourly_data[prev_hour]
                current_data = hourly_data[current_hour]
                
                # Create features
                feature_vector = [
                    prev_data['volume'],
                    prev_data['avg_price'],
                    len(prev_data['trades']),
                    (current_hour - prev_hour).total_seconds() / 3600  # hours difference
                ]
                
                # Target is current hour's average price
                target = current_data['avg_price']
                
                features.append(feature_vector)
                targets.append(target)
        
        return features, targets
    
    async def _get_user_trading_data(self, user_id: int) -> Optional[Dict]:
        """Get user's trading history"""
        try:
            db = SessionLocal()
            
            # Get user's trades
            trades = db.query(Trade).filter(
                Trade.user_id == user_id
            ).order_by(Trade.created_at).all()
            
            if not trades:
                return None
            
            # Get user's markets
            markets = db.query(Market).filter(
                Market.id.in_([t.market_id for t in trades])
            ).all()
            
            db.close()
            
            return {
                'trades': trades,
                'markets': markets,
                'total_volume': sum(t.total_value for t in trades)
            }
            
        except Exception as e:
            logger.error(f"Error getting user trading data: {e}")
            return None
    
    def _calculate_risk_tolerance(self, user_data: Dict) -> float:
        """Calculate user's risk tolerance"""
        trades = user_data['trades']
        
        if len(trades) < 5:
            return 0.5  # Default moderate risk
        
        # Calculate average trade size relative to total volume
        avg_trade_size = sum(t.total_value for t in trades) / len(trades)
        total_volume = user_data['total_volume']
        
        # Risk tolerance based on trade size concentration
        risk_tolerance = min(avg_trade_size / total_volume * 10, 1.0)
        
        return risk_tolerance
    
    def _calculate_trading_frequency(self, user_data: Dict) -> float:
        """Calculate user's trading frequency"""
        trades = user_data['trades']
        
        if len(trades) < 2:
            return 0.0
        
        # Calculate trades per day
        first_trade = trades[0].created_at
        last_trade = trades[-1].created_at
        days_active = (last_trade - first_trade).days + 1
        
        return len(trades) / days_active
    
    def _get_preferred_categories(self, user_data: Dict) -> List[str]:
        """Get user's preferred market categories"""
        markets = user_data['markets']
        category_counts = {}
        
        for market in markets:
            category = market.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Return top 3 categories
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        return [cat for cat, count in sorted_categories[:3]]
    
    def _calculate_average_trade_size(self, user_data: Dict) -> float:
        """Calculate average trade size"""
        trades = user_data['trades']
        return sum(t.total_value for t in trades) / len(trades) if trades else 0.0
    
    def _calculate_win_rate(self, user_data: Dict) -> float:
        """Calculate user's win rate"""
        trades = user_data['trades']
        
        if len(trades) < 5:
            return 0.5  # Default 50% win rate
        
        # Simplified win rate calculation
        # In a real implementation, you'd compare trade prices to market resolution
        return 0.6  # Placeholder
    
    def _calculate_holding_period(self, user_data: Dict) -> float:
        """Calculate average holding period in days"""
        trades = user_data['trades']
        
        if len(trades) < 2:
            return 1.0  # Default 1 day
        
        # Calculate average time between trades
        intervals = []
        for i in range(1, len(trades)):
            interval = (trades[i].created_at - trades[i-1].created_at).days
            intervals.append(interval)
        
        return sum(intervals) / len(intervals) if intervals else 1.0
    
    def _calculate_sentiment_bias(self, user_data: Dict) -> float:
        """Calculate user's sentiment bias (-1 to 1)"""
        trades = user_data['trades']
        
        if len(trades) < 5:
            return 0.0  # Neutral
        
        # Analyze trade patterns for sentiment
        # This is a simplified implementation
        return 0.1  # Slightly positive bias
    
    def _calculate_volatility_preference(self, user_data: Dict) -> float:
        """Calculate user's volatility preference"""
        trades = user_data['trades']
        
        if len(trades) < 5:
            return 0.5  # Moderate preference
        
        # Analyze trade timing and market conditions
        # This is a simplified implementation
        return 0.6  # Slightly high volatility preference
    
    def _calculate_prediction_confidence(self, market_id: int, horizon: str) -> float:
        """Calculate prediction confidence based on model performance"""
        if 'price_prediction' not in self.model_performance:
            return 0.5
        
        performance = self.model_performance['price_prediction']
        r2 = performance.get('r2', 0.5)
        
        # Adjust confidence based on horizon
        horizon_multipliers = {
            '1h': 0.9,
            '24h': 0.7,
            '7d': 0.5
        }
        
        multiplier = horizon_multipliers.get(horizon, 0.7)
        return min(r2 * multiplier, 0.95)
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model"""
        if 'price_prediction' not in self.models:
            return {}
        
        model = self.models['price_prediction']
        if hasattr(model, 'feature_importances_'):
            feature_names = ['volume', 'price', 'trade_count', 'time_diff']
            return dict(zip(feature_names, model.feature_importances_))
        
        return {}
    
    async def _get_market_features(self, market_id: int) -> Optional[Dict]:
        """Get current market features for prediction"""
        try:
            db = SessionLocal()
            
            # Get recent trades
            recent_trades = db.query(Trade).filter(
                Trade.market_id == market_id
            ).order_by(Trade.created_at.desc()).limit(10).all()
            
            if not recent_trades:
                return None
            
            # Calculate features
            total_volume = sum(t.total_value for t in recent_trades)
            avg_price = sum(t.price_per_share for t in recent_trades) / len(recent_trades)
            trade_count = len(recent_trades)
            
            # Time difference from first to last trade
            time_diff = (recent_trades[0].created_at - recent_trades[-1].created_at).total_seconds() / 3600
            
            db.close()
            
            return {
                'volume': total_volume,
                'price': avg_price,
                'trade_count': trade_count,
                'time_diff': time_diff
            }
            
        except Exception as e:
            logger.error(f"Error getting market features: {e}")
            return None
    
    async def _get_historical_volume_data(self) -> List[Dict]:
        """Get historical volume data for training"""
        # Similar to market data but focused on volume patterns
        return []
    
    def _prepare_volume_prediction_data(self, volume_data: List[Dict]) -> Tuple[List, List]:
        """Prepare features and targets for volume prediction"""
        # Similar to price prediction but for volume
        return [], []
    
    async def _get_user_behavior_data(self) -> List[Dict]:
        """Get user behavior data for training"""
        # Get user profiles and behavior patterns
        return []
    
    def _prepare_user_behavior_data(self, user_data: List[Dict]) -> Tuple[List, List]:
        """Prepare features and targets for user behavior prediction"""
        # Prepare user behavior features
        return [], []

# Global ML service instance
ml_service = MachineLearningService()

def get_ml_service() -> MachineLearningService:
    """Get the global ML service instance"""
    return ml_service
