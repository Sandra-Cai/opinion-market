"""
Machine Learning Service for Opinion Market
Handles ML models, predictions, and AI-powered features
"""

import asyncio
import json
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
import os

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc
from fastapi import HTTPException, status

from app.core.database import get_db, get_redis_client
from app.core.config import settings
from app.core.cache import cache
from app.core.logging import log_system_metric
from app.models.user import User
from app.models.market import Market, MarketCategory
from app.models.trade import Trade, TradeType


class MLModelType(str, Enum):
    """Types of ML models"""
    PRICE_PREDICTION = "price_prediction"
    MARKET_TREND = "market_trend"
    USER_BEHAVIOR = "user_behavior"
    RISK_ASSESSMENT = "risk_assessment"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    FRAUD_DETECTION = "fraud_detection"


@dataclass
class MLPrediction:
    """ML prediction data structure"""
    prediction_id: str
    model_type: MLModelType
    input_data: Dict[str, Any]
    prediction: Any
    confidence: float
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class MLModel:
    """ML model data structure"""
    model_id: str
    model_type: MLModelType
    version: str
    accuracy: float
    created_at: datetime
    last_trained: datetime
    parameters: Dict[str, Any]
    is_active: bool


class MLService:
    """Service for machine learning operations"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.cache_ttl = 600  # 10 minutes
        self.models = {}
        self.model_path = settings.ML_MODEL_PATH
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Create model directory if it doesn't exist
            os.makedirs(self.model_path, exist_ok=True)
            
            # Load existing models
            self._load_models()
            
            # Initialize default models if none exist
            if not self.models:
                self._create_default_models()
            
            log_system_metric("ml_models_initialized", 1, {
                "model_count": len(self.models),
                "model_types": list(self.models.keys())
            })
            
        except Exception as e:
            log_system_metric("ml_initialization_error", 1, {"error": str(e)})
    
    def _load_models(self):
        """Load existing ML models from disk"""
        try:
            for model_type in MLModelType:
                model_file = os.path.join(self.model_path, f"{model_type.value}.pkl")
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                        self.models[model_type] = model
                        
        except Exception as e:
            log_system_metric("ml_model_loading_error", 1, {"error": str(e)})
    
    def _create_default_models(self):
        """Create default ML models"""
        try:
            # Create simple price prediction model
            self.models[MLModelType.PRICE_PREDICTION] = self._create_price_prediction_model()
            
            # Create market trend model
            self.models[MLModelType.MARKET_TREND] = self._create_market_trend_model()
            
            # Create user behavior model
            self.models[MLModelType.USER_BEHAVIOR] = self._create_user_behavior_model()
            
            # Create risk assessment model
            self.models[MLModelType.RISK_ASSESSMENT] = self._create_risk_assessment_model()
            
            # Save models
            self._save_models()
            
        except Exception as e:
            log_system_metric("ml_default_models_error", 1, {"error": str(e)})
    
    def _create_price_prediction_model(self):
        """Create a simple price prediction model"""
        # This is a simplified model - in reality, you'd use proper ML libraries
        return {
            "model_type": MLModelType.PRICE_PREDICTION,
            "version": "1.0.0",
            "accuracy": 0.75,
            "created_at": datetime.utcnow(),
            "last_trained": datetime.utcnow(),
            "parameters": {
                "learning_rate": 0.01,
                "epochs": 100,
                "hidden_layers": [64, 32]
            },
            "is_active": True
        }
    
    def _create_market_trend_model(self):
        """Create a market trend prediction model"""
        return {
            "model_type": MLModelType.MARKET_TREND,
            "version": "1.0.0",
            "accuracy": 0.68,
            "created_at": datetime.utcnow(),
            "last_trained": datetime.utcnow(),
            "parameters": {
                "window_size": 30,
                "features": ["volume", "price_change", "trade_count"]
            },
            "is_active": True
        }
    
    def _create_user_behavior_model(self):
        """Create a user behavior prediction model"""
        return {
            "model_type": MLModelType.USER_BEHAVIOR,
            "version": "1.0.0",
            "accuracy": 0.82,
            "created_at": datetime.utcnow(),
            "last_trained": datetime.utcnow(),
            "parameters": {
                "features": ["trade_frequency", "avg_trade_size", "success_rate"]
            },
            "is_active": True
        }
    
    def _create_risk_assessment_model(self):
        """Create a risk assessment model"""
        return {
            "model_type": MLModelType.RISK_ASSESSMENT,
            "version": "1.0.0",
            "accuracy": 0.79,
            "created_at": datetime.utcnow(),
            "last_trained": datetime.utcnow(),
            "parameters": {
                "risk_factors": ["volatility", "liquidity", "market_age"]
            },
            "is_active": True
        }
    
    def _save_models(self):
        """Save ML models to disk"""
        try:
            for model_type, model in self.models.items():
                model_file = os.path.join(self.model_path, f"{model_type.value}.pkl")
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                    
        except Exception as e:
            log_system_metric("ml_model_saving_error", 1, {"error": str(e)})
    
    async def predict_price(self, market_id: int, db: Session) -> MLPrediction:
        """Predict future price for a market"""
        try:
            # Get market data
            market = db.query(Market).filter(Market.id == market_id).first()
            if not market:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Market not found"
                )
            
            # Get recent trades
            recent_trades = (
                db.query(Trade)
                .filter(Trade.market_id == market_id)
                .order_by(desc(Trade.created_at))
                .limit(100)
                .all()
            )
            
            # Prepare input data
            input_data = {
                "market_id": market_id,
                "current_price_a": market.price_a,
                "current_price_b": market.price_b,
                "volume_24h": market.volume_24h,
                "trending_score": market.trending_score,
                "trade_count": len(recent_trades),
                "avg_trade_size": sum(t.total_value for t in recent_trades) / len(recent_trades) if recent_trades else 0
            }
            
            # Make prediction (simplified)
            prediction = self._make_price_prediction(input_data)
            
            # Create prediction object
            ml_prediction = MLPrediction(
                prediction_id=str(uuid.uuid4()),
                model_type=MLModelType.PRICE_PREDICTION,
                input_data=input_data,
                prediction=prediction,
                confidence=0.75,  # Mock confidence
                created_at=datetime.utcnow(),
                metadata={
                    "model_version": self.models[MLModelType.PRICE_PREDICTION]["version"],
                    "market_id": market_id
                }
            )
            
            # Cache prediction
            cache_key = f"price_prediction:{market_id}"
            cache.set(cache_key, ml_prediction, ttl=settings.ML_PREDICTION_CACHE_TTL)
            
            # Log prediction
            log_system_metric("price_prediction_made", 1, {
                "market_id": market_id,
                "prediction_id": ml_prediction.prediction_id,
                "confidence": ml_prediction.confidence
            })
            
            return ml_prediction
            
        except Exception as e:
            log_system_metric("price_prediction_error", 1, {"error": str(e)})
            raise
    
    async def predict_market_trend(self, market_id: int, db: Session) -> MLPrediction:
        """Predict market trend"""
        try:
            # Get market data
            market = db.query(Market).filter(Market.id == market_id).first()
            if not market:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Market not found"
                )
            
            # Prepare input data
            input_data = {
                "market_id": market_id,
                "volume_trend": market.volume_24h / max(market.volume_total, 1),
                "price_volatility": abs(market.price_a - market.price_b),
                "trending_score": market.trending_score,
                "market_age": (datetime.utcnow() - market.created_at).days
            }
            
            # Make prediction (simplified)
            prediction = self._make_trend_prediction(input_data)
            
            # Create prediction object
            ml_prediction = MLPrediction(
                prediction_id=str(uuid.uuid4()),
                model_type=MLModelType.MARKET_TREND,
                input_data=input_data,
                prediction=prediction,
                confidence=0.68,  # Mock confidence
                created_at=datetime.utcnow(),
                metadata={
                    "model_version": self.models[MLModelType.MARKET_TREND]["version"],
                    "market_id": market_id
                }
            )
            
            # Cache prediction
            cache_key = f"trend_prediction:{market_id}"
            cache.set(cache_key, ml_prediction, ttl=settings.ML_PREDICTION_CACHE_TTL)
            
            return ml_prediction
            
        except Exception as e:
            log_system_metric("trend_prediction_error", 1, {"error": str(e)})
            raise
    
    async def predict_user_behavior(self, user_id: int, db: Session) -> MLPrediction:
        """Predict user behavior"""
        try:
            # Get user data
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Get user trades
            user_trades = db.query(Trade).filter(Trade.user_id == user_id).all()
            
            # Prepare input data
            input_data = {
                "user_id": user_id,
                "total_trades": len(user_trades),
                "avg_trade_size": sum(t.total_value for t in user_trades) / len(user_trades) if user_trades else 0,
                "success_rate": len([t for t in user_trades if t.status == "executed"]) / len(user_trades) if user_trades else 0,
                "trading_frequency": len(user_trades) / max(1, (datetime.utcnow() - user.created_at).days),
                "preferred_categories": self._get_user_preferred_categories(user_id, db)
            }
            
            # Make prediction (simplified)
            prediction = self._make_behavior_prediction(input_data)
            
            # Create prediction object
            ml_prediction = MLPrediction(
                prediction_id=str(uuid.uuid4()),
                model_type=MLModelType.USER_BEHAVIOR,
                input_data=input_data,
                prediction=prediction,
                confidence=0.82,  # Mock confidence
                created_at=datetime.utcnow(),
                metadata={
                    "model_version": self.models[MLModelType.USER_BEHAVIOR]["version"],
                    "user_id": user_id
                }
            )
            
            # Cache prediction
            cache_key = f"behavior_prediction:{user_id}"
            cache.set(cache_key, ml_prediction, ttl=settings.ML_PREDICTION_CACHE_TTL)
            
            return ml_prediction
            
        except Exception as e:
            log_system_metric("behavior_prediction_error", 1, {"error": str(e)})
            raise
    
    async def assess_risk(self, market_id: int, db: Session) -> MLPrediction:
        """Assess market risk"""
        try:
            # Get market data
            market = db.query(Market).filter(Market.id == market_id).first()
            if not market:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Market not found"
                )
            
            # Get market trades
            market_trades = db.query(Trade).filter(Trade.market_id == market_id).all()
            
            # Prepare input data
            input_data = {
                "market_id": market_id,
                "volatility": self._calculate_volatility(market_trades),
                "liquidity": market.volume_total,
                "market_age": (datetime.utcnow() - market.created_at).days,
                "trade_count": len(market_trades),
                "price_spread": abs(market.price_a - market.price_b)
            }
            
            # Make prediction (simplified)
            prediction = self._make_risk_assessment(input_data)
            
            # Create prediction object
            ml_prediction = MLPrediction(
                prediction_id=str(uuid.uuid4()),
                model_type=MLModelType.RISK_ASSESSMENT,
                input_data=input_data,
                prediction=prediction,
                confidence=0.79,  # Mock confidence
                created_at=datetime.utcnow(),
                metadata={
                    "model_version": self.models[MLModelType.RISK_ASSESSMENT]["version"],
                    "market_id": market_id
                }
            )
            
            # Cache prediction
            cache_key = f"risk_assessment:{market_id}"
            cache.set(cache_key, ml_prediction, ttl=settings.ML_PREDICTION_CACHE_TTL)
            
            return ml_prediction
            
        except Exception as e:
            log_system_metric("risk_assessment_error", 1, {"error": str(e)})
            raise
    
    def _make_price_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make price prediction (simplified)"""
        # This is a mock prediction - in reality, you'd use proper ML models
        current_price_a = input_data["current_price_a"]
        current_price_b = input_data["current_price_b"]
        
        # Simple trend-based prediction
        volume_factor = min(input_data["volume_24h"] / 1000, 1.0)
        trend_factor = input_data["trending_score"] / 100.0
        
        # Predict next price (simplified)
        predicted_price_a = current_price_a + (trend_factor * 0.01 * volume_factor)
        predicted_price_b = current_price_b - (trend_factor * 0.01 * volume_factor)
        
        # Ensure prices stay within bounds
        predicted_price_a = max(0.0, min(1.0, predicted_price_a))
        predicted_price_b = max(0.0, min(1.0, predicted_price_b))
        
        return {
            "predicted_price_a": predicted_price_a,
            "predicted_price_b": predicted_price_b,
            "price_change_a": predicted_price_a - current_price_a,
            "price_change_b": predicted_price_b - current_price_b,
            "direction": "up" if predicted_price_a > current_price_a else "down"
        }
    
    def _make_trend_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make trend prediction (simplified)"""
        # This is a mock prediction
        volume_trend = input_data["volume_trend"]
        trending_score = input_data["trending_score"]
        
        # Simple trend calculation
        if volume_trend > 0.1 and trending_score > 50:
            trend = "bullish"
        elif volume_trend < -0.1 and trending_score < 30:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {
            "trend": trend,
            "strength": min(trending_score / 100.0, 1.0),
            "confidence": 0.68
        }
    
    def _make_behavior_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make behavior prediction (simplified)"""
        # This is a mock prediction
        trading_frequency = input_data["trading_frequency"]
        success_rate = input_data["success_rate"]
        
        # Simple behavior classification
        if trading_frequency > 5 and success_rate > 0.7:
            behavior_type = "aggressive_trader"
        elif trading_frequency > 2 and success_rate > 0.5:
            behavior_type = "moderate_trader"
        else:
            behavior_type = "conservative_trader"
        
        return {
            "behavior_type": behavior_type,
            "predicted_activity": min(trading_frequency * 1.1, 10.0),
            "risk_tolerance": "high" if trading_frequency > 5 else "medium" if trading_frequency > 2 else "low"
        }
    
    def _make_risk_assessment(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make risk assessment (simplified)"""
        # This is a mock assessment
        volatility = input_data["volatility"]
        liquidity = input_data["liquidity"]
        market_age = input_data["market_age"]
        
        # Simple risk calculation
        risk_score = (volatility * 0.4) + ((1 - min(liquidity / 10000, 1)) * 0.3) + ((1 - min(market_age / 30, 1)) * 0.3)
        
        if risk_score > 0.7:
            risk_level = "high"
        elif risk_score > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "factors": {
                "volatility": volatility,
                "liquidity": liquidity,
                "market_age": market_age
            }
        }
    
    def _calculate_volatility(self, trades: List[Trade]) -> float:
        """Calculate price volatility from trades"""
        if len(trades) < 2:
            return 0.0
        
        prices = [t.price_per_share for t in trades]
        mean_price = sum(prices) / len(prices)
        
        variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
        return variance ** 0.5
    
    def _get_user_preferred_categories(self, user_id: int, db: Session) -> List[str]:
        """Get user's preferred market categories"""
        try:
            # Get user's most traded categories
            category_stats = (
                db.query(Market.category, func.count(Trade.id))
                .join(Trade, Market.id == Trade.market_id)
                .filter(Trade.user_id == user_id)
                .group_by(Market.category)
                .order_by(desc(func.count(Trade.id)))
                .limit(3)
                .all()
            )
            
            return [category.value for category, count in category_stats]
        except Exception:
            return []
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about all ML models"""
        try:
            model_info = {}
            for model_type, model in self.models.items():
                model_info[model_type.value] = {
                    "version": model["version"],
                    "accuracy": model["accuracy"],
                    "created_at": model["created_at"].isoformat(),
                    "last_trained": model["last_trained"].isoformat(),
                    "is_active": model["is_active"]
                }
            
            return {
                "total_models": len(self.models),
                "models": model_info
            }
            
        except Exception as e:
            log_system_metric("ml_model_info_error", 1, {"error": str(e)})
            raise


# Global ML service instance
ml_service = MLService()
