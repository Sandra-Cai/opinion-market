"""
AI Insights Engine
Comprehensive AI-powered insights and recommendations system
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import secrets
import numpy as np
import hashlib
import pickle

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Insight type enumeration"""
    MARKET_TREND = "market_trend"
    USER_BEHAVIOR = "user_behavior"
    PRICE_PREDICTION = "price_prediction"
    RISK_ASSESSMENT = "risk_assessment"
    OPPORTUNITY_DETECTION = "opportunity_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"


class RecommendationType(Enum):
    """Recommendation type enumeration"""
    TRADING_ACTION = "trading_action"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    RISK_MANAGEMENT = "risk_management"
    MARKET_TIMING = "market_timing"
    ASSET_SELECTION = "asset_selection"
    POSITION_SIZING = "position_sizing"
    DIVERSIFICATION = "diversification"
    REBALANCING = "rebalancing"


class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class Insight:
    """Insight data structure"""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    confidence_level: ConfidenceLevel
    confidence_score: float
    data_points: List[Dict[str, Any]]
    generated_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    impact_score: float = 0.0


@dataclass
class Recommendation:
    """Recommendation data structure"""
    recommendation_id: str
    recommendation_type: RecommendationType
    title: str
    description: str
    action: str
    confidence_level: ConfidenceLevel
    confidence_score: float
    expected_return: float
    risk_level: str
    time_horizon: str
    generated_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    supporting_insights: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class AIModel:
    """AI model data structure"""
    model_id: str
    model_name: str
    model_type: str
    version: str
    accuracy: float
    last_trained: datetime
    training_data_size: int
    features: List[str]
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    status: str = "active"


class AIInsightsEngine:
    """AI Insights Engine for comprehensive AI-powered insights and recommendations"""
    
    def __init__(self):
        self.insights: Dict[str, Insight] = {}
        self.recommendations: Dict[str, Recommendation] = {}
        self.ai_models: Dict[str, AIModel] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.market_data: Dict[str, List[Dict[str, Any]]] = {}
        self.patterns: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "ai_insights_enabled": True,
            "real_time_insights_enabled": True,
            "recommendation_engine_enabled": True,
            "pattern_recognition_enabled": True,
            "sentiment_analysis_enabled": True,
            "anomaly_detection_enabled": True,
            "insight_generation_interval": 300,  # 5 minutes
            "recommendation_update_interval": 600,  # 10 minutes
            "model_retraining_interval": 86400,  # 24 hours
            "insight_retention_days": 30,
            "recommendation_retention_days": 7,
            "confidence_threshold": 0.7,
            "max_insights_per_user": 100,
            "max_recommendations_per_user": 50,
            "personalization_enabled": True,
            "explainability_enabled": True
        }
        
        # AI model configurations
        self.model_configs = {
            "market_trend": {
                "model_type": "time_series",
                "features": ["price", "volume", "sentiment", "news"],
                "algorithm": "LSTM",
                "lookback_window": 30,
                "prediction_horizon": 7
            },
            "sentiment_analysis": {
                "model_type": "nlp",
                "features": ["text", "context", "source"],
                "algorithm": "BERT",
                "max_length": 512,
                "batch_size": 32
            },
            "anomaly_detection": {
                "model_type": "unsupervised",
                "features": ["price", "volume", "volatility"],
                "algorithm": "IsolationForest",
                "contamination": 0.1,
                "n_estimators": 100
            },
            "risk_assessment": {
                "model_type": "classification",
                "features": ["volatility", "correlation", "liquidity"],
                "algorithm": "RandomForest",
                "n_estimators": 200,
                "max_depth": 10
            }
        }
        
        # Insight templates
        self.insight_templates = {
            InsightType.MARKET_TREND: {
                "title_template": "Market Trend Analysis: {asset}",
                "description_template": "Based on recent price movements and volume analysis, {asset} shows {trend_direction} trend with {confidence} confidence.",
                "required_data": ["price_data", "volume_data", "technical_indicators"]
            },
            InsightType.USER_BEHAVIOR: {
                "title_template": "User Behavior Pattern: {pattern_type}",
                "description_template": "Analysis of user trading patterns reveals {pattern_description} with {frequency} frequency.",
                "required_data": ["trading_history", "user_actions", "session_data"]
            },
            InsightType.PRICE_PREDICTION: {
                "title_template": "Price Prediction: {asset}",
                "description_template": "AI model predicts {asset} price will be {predicted_price} in {time_horizon} with {confidence} confidence.",
                "required_data": ["historical_prices", "market_indicators", "external_factors"]
            },
            InsightType.RISK_ASSESSMENT: {
                "title_template": "Risk Assessment: {portfolio_or_asset}",
                "description_template": "Risk analysis indicates {risk_level} risk level for {portfolio_or_asset} based on {risk_factors}.",
                "required_data": ["volatility_data", "correlation_matrix", "liquidity_metrics"]
            }
        }
        
        # Recommendation templates
        self.recommendation_templates = {
            RecommendationType.TRADING_ACTION: {
                "title_template": "Trading Recommendation: {action} {asset}",
                "description_template": "Based on market analysis, recommend {action} {asset} at {price} with {confidence} confidence.",
                "action_template": "{action} {quantity} {asset} at {price}"
            },
            RecommendationType.PORTFOLIO_OPTIMIZATION: {
                "title_template": "Portfolio Optimization",
                "description_template": "Portfolio analysis suggests rebalancing to optimize risk-return profile.",
                "action_template": "Rebalance portfolio: {rebalancing_actions}"
            },
            RecommendationType.RISK_MANAGEMENT: {
                "title_template": "Risk Management Alert",
                "description_template": "Risk metrics indicate elevated risk levels. Consider risk mitigation strategies.",
                "action_template": "Implement risk management: {risk_actions}"
            }
        }
        
        # Monitoring
        self.ai_insights_active = False
        self.ai_insights_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.ai_insights_stats = {
            "insights_generated": 0,
            "recommendations_generated": 0,
            "models_trained": 0,
            "patterns_detected": 0,
            "anomalies_detected": 0,
            "sentiment_analyses": 0,
            "user_profiles_updated": 0,
            "accuracy_improvements": 0
        }
        
    async def start_ai_insights_engine(self):
        """Start the AI insights engine"""
        if self.ai_insights_active:
            logger.warning("AI insights engine already active")
            return
            
        self.ai_insights_active = True
        self.ai_insights_task = asyncio.create_task(self._ai_insights_processing_loop())
        
        # Initialize AI models
        await self._initialize_ai_models()
        
        logger.info("AI Insights Engine started")
        
    async def stop_ai_insights_engine(self):
        """Stop the AI insights engine"""
        self.ai_insights_active = False
        if self.ai_insights_task:
            self.ai_insights_task.cancel()
            try:
                await self.ai_insights_task
            except asyncio.CancelledError:
                pass
        logger.info("AI Insights Engine stopped")
        
    async def _ai_insights_processing_loop(self):
        """Main AI insights processing loop"""
        while self.ai_insights_active:
            try:
                # Generate real-time insights
                if self.config["real_time_insights_enabled"]:
                    await self._generate_real_time_insights()
                    
                # Update recommendations
                if self.config["recommendation_engine_enabled"]:
                    await self._update_recommendations()
                    
                # Detect patterns
                if self.config["pattern_recognition_enabled"]:
                    await self._detect_patterns()
                    
                # Analyze sentiment
                if self.config["sentiment_analysis_enabled"]:
                    await self._analyze_sentiment()
                    
                # Detect anomalies
                if self.config["anomaly_detection_enabled"]:
                    await self._detect_anomalies()
                    
                # Update user profiles
                if self.config["personalization_enabled"]:
                    await self._update_user_profiles()
                    
                # Clean up old insights and recommendations
                await self._cleanup_old_data()
                
                # Wait before next cycle
                await asyncio.sleep(self.config["insight_generation_interval"])
                
            except Exception as e:
                logger.error(f"Error in AI insights processing loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _initialize_ai_models(self):
        """Initialize AI models"""
        try:
            for model_name, config in self.model_configs.items():
                model_id = f"model_{model_name}_{int(time.time())}"
                
                model = AIModel(
                    model_id=model_id,
                    model_name=model_name,
                    model_type=config["model_type"],
                    version="1.0.0",
                    accuracy=0.85 + (secrets.randbelow(15) / 100.0),  # Mock accuracy
                    last_trained=datetime.now(),
                    training_data_size=10000 + secrets.randbelow(50000),
                    features=config["features"],
                    hyperparameters=config,
                    performance_metrics={
                        "precision": 0.82 + (secrets.randbelow(18) / 100.0),
                        "recall": 0.80 + (secrets.randbelow(20) / 100.0),
                        "f1_score": 0.81 + (secrets.randbelow(19) / 100.0)
                    }
                )
                
                self.ai_models[model_id] = model
                
            logger.info(f"Initialized {len(self.ai_models)} AI models")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {e}")
            
    async def generate_insight(self, insight_type: InsightType, data: Dict[str, Any], user_id: Optional[str] = None) -> Insight:
        """Generate AI-powered insight"""
        try:
            insight_id = f"insight_{int(time.time())}_{secrets.token_hex(4)}"
            
            # Get template
            template = self.insight_templates.get(insight_type, {})
            
            # Generate insight content
            title, description, confidence_score = await self._generate_insight_content(insight_type, data, template)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(confidence_score)
            
            # Create insight
            insight = Insight(
                insight_id=insight_id,
                insight_type=insight_type,
                title=title,
                description=description,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                data_points=data.get("data_points", []),
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=self.config["insight_retention_days"]),
                metadata=data.get("metadata", {}),
                tags=data.get("tags", []),
                impact_score=self._calculate_impact_score(insight_type, confidence_score, data)
            )
            
            # Store insight
            self.insights[insight_id] = insight
            
            # Update user profile if provided
            if user_id:
                await self._update_user_insight_profile(user_id, insight)
                
            self.ai_insights_stats["insights_generated"] += 1
            
            logger.info(f"AI insight generated: {insight_id} - {insight_type.value}")
            return insight
            
        except Exception as e:
            logger.error(f"Error generating insight: {e}")
            raise
            
    async def generate_recommendation(self, recommendation_type: RecommendationType, data: Dict[str, Any], user_id: Optional[str] = None) -> Recommendation:
        """Generate AI-powered recommendation"""
        try:
            recommendation_id = f"rec_{int(time.time())}_{secrets.token_hex(4)}"
            
            # Get template
            template = self.recommendation_templates.get(recommendation_type, {})
            
            # Generate recommendation content
            title, description, action, confidence_score = await self._generate_recommendation_content(recommendation_type, data, template)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(confidence_score)
            
            # Create recommendation
            recommendation = Recommendation(
                recommendation_id=recommendation_id,
                recommendation_type=recommendation_type,
                title=title,
                description=description,
                action=action,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                expected_return=data.get("expected_return", 0.0),
                risk_level=data.get("risk_level", "medium"),
                time_horizon=data.get("time_horizon", "short"),
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=self.config["recommendation_retention_days"]),
                metadata=data.get("metadata", {}),
                supporting_insights=data.get("supporting_insights", []),
                prerequisites=data.get("prerequisites", [])
            )
            
            # Store recommendation
            self.recommendations[recommendation_id] = recommendation
            
            # Update user profile if provided
            if user_id:
                await self._update_user_recommendation_profile(user_id, recommendation)
                
            self.ai_insights_stats["recommendations_generated"] += 1
            
            logger.info(f"AI recommendation generated: {recommendation_id} - {recommendation_type.value}")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            raise
            
    async def _generate_insight_content(self, insight_type: InsightType, data: Dict[str, Any], template: Dict[str, str]) -> Tuple[str, str, float]:
        """Generate insight content using AI models"""
        try:
            # Simulate AI model inference
            confidence_score = 0.7 + (secrets.randbelow(30) / 100.0)  # 0.7-1.0
            
            # Generate title and description based on type
            if insight_type == InsightType.MARKET_TREND:
                asset = data.get("asset", "Unknown Asset")
                trend_direction = "bullish" if secrets.randbelow(2) else "bearish"
                title = f"Market Trend Analysis: {asset}"
                description = f"Based on recent price movements and volume analysis, {asset} shows {trend_direction} trend with {confidence_score:.1%} confidence."
                
            elif insight_type == InsightType.USER_BEHAVIOR:
                pattern_type = data.get("pattern_type", "Trading Pattern")
                pattern_description = "increased trading frequency during market volatility"
                frequency = "high"
                title = f"User Behavior Pattern: {pattern_type}"
                description = f"Analysis of user trading patterns reveals {pattern_description} with {frequency} frequency."
                
            elif insight_type == InsightType.PRICE_PREDICTION:
                asset = data.get("asset", "Unknown Asset")
                predicted_price = data.get("predicted_price", 100.0)
                time_horizon = data.get("time_horizon", "7 days")
                title = f"Price Prediction: {asset}"
                description = f"AI model predicts {asset} price will be ${predicted_price:.2f} in {time_horizon} with {confidence_score:.1%} confidence."
                
            elif insight_type == InsightType.RISK_ASSESSMENT:
                portfolio_or_asset = data.get("portfolio_or_asset", "Portfolio")
                risk_level = data.get("risk_level", "medium")
                risk_factors = "volatility and correlation analysis"
                title = f"Risk Assessment: {portfolio_or_asset}"
                description = f"Risk analysis indicates {risk_level} risk level for {portfolio_or_asset} based on {risk_factors}."
                
            else:
                title = f"AI Insight: {insight_type.value}"
                description = f"AI-generated insight for {insight_type.value} with {confidence_score:.1%} confidence."
                
            return title, description, confidence_score
            
        except Exception as e:
            logger.error(f"Error generating insight content: {e}")
            return "AI Insight", "Generated insight", 0.5
            
    async def _generate_recommendation_content(self, recommendation_type: RecommendationType, data: Dict[str, Any], template: Dict[str, str]) -> Tuple[str, str, str, float]:
        """Generate recommendation content using AI models"""
        try:
            # Simulate AI model inference
            confidence_score = 0.6 + (secrets.randbelow(40) / 100.0)  # 0.6-1.0
            
            # Generate content based on type
            if recommendation_type == RecommendationType.TRADING_ACTION:
                action = data.get("action", "BUY")
                asset = data.get("asset", "Unknown Asset")
                price = data.get("price", 100.0)
                quantity = data.get("quantity", 1)
                title = f"Trading Recommendation: {action} {asset}"
                description = f"Based on market analysis, recommend {action} {asset} at ${price:.2f} with {confidence_score:.1%} confidence."
                action_text = f"{action} {quantity} {asset} at ${price:.2f}"
                
            elif recommendation_type == RecommendationType.PORTFOLIO_OPTIMIZATION:
                rebalancing_actions = data.get("rebalancing_actions", "Reduce tech exposure, increase bonds")
                title = "Portfolio Optimization"
                description = "Portfolio analysis suggests rebalancing to optimize risk-return profile."
                action_text = f"Rebalance portfolio: {rebalancing_actions}"
                
            elif recommendation_type == RecommendationType.RISK_MANAGEMENT:
                risk_actions = data.get("risk_actions", "Implement stop-loss orders")
                title = "Risk Management Alert"
                description = "Risk metrics indicate elevated risk levels. Consider risk mitigation strategies."
                action_text = f"Implement risk management: {risk_actions}"
                
            else:
                title = f"AI Recommendation: {recommendation_type.value}"
                description = f"AI-generated recommendation for {recommendation_type.value} with {confidence_score:.1%} confidence."
                action_text = f"Consider {recommendation_type.value.lower()}"
                
            return title, description, action_text, confidence_score
            
        except Exception as e:
            logger.error(f"Error generating recommendation content: {e}")
            return "AI Recommendation", "Generated recommendation", "Consider action", 0.5
            
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level from score"""
        if confidence_score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
            
    def _calculate_impact_score(self, insight_type: InsightType, confidence_score: float, data: Dict[str, Any]) -> float:
        """Calculate impact score for insight"""
        try:
            # Base impact score
            base_impact = confidence_score
            
            # Adjust based on insight type
            type_multipliers = {
                InsightType.MARKET_TREND: 1.2,
                InsightType.PRICE_PREDICTION: 1.5,
                InsightType.RISK_ASSESSMENT: 1.3,
                InsightType.ANOMALY_DETECTION: 1.4,
                InsightType.OPPORTUNITY_DETECTION: 1.6
            }
            
            multiplier = type_multipliers.get(insight_type, 1.0)
            
            # Adjust based on data quality
            data_quality = data.get("data_quality", 0.8)
            
            impact_score = base_impact * multiplier * data_quality
            
            return min(impact_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating impact score: {e}")
            return 0.5
            
    async def _generate_real_time_insights(self):
        """Generate real-time insights"""
        try:
            # Generate market trend insights
            market_data = await self._get_market_data()
            if market_data:
                for asset, data in market_data.items():
                    insight_data = {
                        "asset": asset,
                        "data_points": data,
                        "metadata": {"source": "real_time", "timestamp": datetime.now().isoformat()}
                    }
                    await self.generate_insight(InsightType.MARKET_TREND, insight_data)
                    
        except Exception as e:
            logger.error(f"Error generating real-time insights: {e}")
            
    async def _update_recommendations(self):
        """Update recommendations based on latest insights"""
        try:
            # Get recent insights
            recent_insights = [
                insight for insight in self.insights.values()
                if (datetime.now() - insight.generated_at).total_seconds() < 3600  # Last hour
            ]
            
            # Generate recommendations based on insights
            for insight in recent_insights:
                if insight.confidence_score >= self.config["confidence_threshold"]:
                    recommendation_data = {
                        "supporting_insights": [insight.insight_id],
                        "metadata": {"based_on_insight": insight.insight_id}
                    }
                    
                    # Determine recommendation type based on insight type
                    if insight.insight_type == InsightType.MARKET_TREND:
                        await self.generate_recommendation(RecommendationType.TRADING_ACTION, recommendation_data)
                    elif insight.insight_type == InsightType.RISK_ASSESSMENT:
                        await self.generate_recommendation(RecommendationType.RISK_MANAGEMENT, recommendation_data)
                        
        except Exception as e:
            logger.error(f"Error updating recommendations: {e}")
            
    async def _detect_patterns(self):
        """Detect patterns in data"""
        try:
            # Simulate pattern detection
            pattern_data = {
                "pattern_type": "Price Momentum",
                "pattern_description": "strong upward momentum with increasing volume",
                "frequency": "high",
                "confidence": 0.85
            }
            
            await self.generate_insight(InsightType.PATTERN_RECOGNITION, pattern_data)
            self.ai_insights_stats["patterns_detected"] += 1
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            
    async def _analyze_sentiment(self):
        """Analyze market sentiment"""
        try:
            # Simulate sentiment analysis
            sentiment_data = {
                "sentiment_score": 0.7 + (secrets.randbelow(30) / 100.0),
                "sentiment_type": "positive" if secrets.randbelow(2) else "negative",
                "data_points": ["news_articles", "social_media", "market_indicators"]
            }
            
            await self.generate_insight(InsightType.SENTIMENT_ANALYSIS, sentiment_data)
            self.ai_insights_stats["sentiment_analyses"] += 1
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            
    async def _detect_anomalies(self):
        """Detect anomalies in data"""
        try:
            # Simulate anomaly detection
            anomaly_data = {
                "anomaly_type": "Price Spike",
                "severity": "medium",
                "affected_assets": ["BTC", "ETH"],
                "confidence": 0.8
            }
            
            await self.generate_insight(InsightType.ANOMALY_DETECTION, anomaly_data)
            self.ai_insights_stats["anomalies_detected"] += 1
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            
    async def _update_user_profiles(self):
        """Update user profiles for personalization"""
        try:
            # Simulate user profile updates
            for user_id in list(self.user_profiles.keys())[:10]:  # Update first 10 users
                profile = self.user_profiles[user_id]
                profile["last_updated"] = datetime.now().isoformat()
                profile["insights_count"] = profile.get("insights_count", 0) + 1
                
            self.ai_insights_stats["user_profiles_updated"] += 1
            
        except Exception as e:
            logger.error(f"Error updating user profiles: {e}")
            
    async def _update_user_insight_profile(self, user_id: str, insight: Insight):
        """Update user profile with insight"""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "user_id": user_id,
                    "insights": [],
                    "preferences": {},
                    "created_at": datetime.now().isoformat()
                }
                
            profile = self.user_profiles[user_id]
            profile["insights"].append(insight.insight_id)
            profile["last_insight"] = insight.generated_at.isoformat()
            
        except Exception as e:
            logger.error(f"Error updating user insight profile: {e}")
            
    async def _update_user_recommendation_profile(self, user_id: str, recommendation: Recommendation):
        """Update user profile with recommendation"""
        try:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    "user_id": user_id,
                    "recommendations": [],
                    "preferences": {},
                    "created_at": datetime.now().isoformat()
                }
                
            profile = self.user_profiles[user_id]
            profile["recommendations"].append(recommendation.recommendation_id)
            profile["last_recommendation"] = recommendation.generated_at.isoformat()
            
        except Exception as e:
            logger.error(f"Error updating user recommendation profile: {e}")
            
    async def _get_market_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get market data for insights"""
        try:
            # Simulate market data
            assets = ["BTC", "ETH", "AAPL", "GOOGL", "TSLA"]
            market_data = {}
            
            for asset in assets:
                market_data[asset] = [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "price": 100.0 + (secrets.randbelow(1000) / 10.0),
                        "volume": 1000000 + secrets.randbelow(5000000),
                        "change": (secrets.randbelow(200) - 100) / 100.0
                    }
                ]
                
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {}
            
    async def _cleanup_old_data(self):
        """Clean up old insights and recommendations"""
        try:
            current_time = datetime.now()
            
            # Clean up expired insights
            expired_insights = [
                insight_id for insight_id, insight in self.insights.items()
                if insight.expires_at < current_time
            ]
            
            for insight_id in expired_insights:
                del self.insights[insight_id]
                
            # Clean up expired recommendations
            expired_recommendations = [
                rec_id for rec_id, rec in self.recommendations.items()
                if rec.expires_at < current_time
            ]
            
            for rec_id in expired_recommendations:
                del self.recommendations[rec_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    def get_ai_insights_summary(self) -> Dict[str, Any]:
        """Get comprehensive AI insights summary"""
        try:
            # Calculate insights by type
            insights_by_type = defaultdict(int)
            for insight in self.insights.values():
                insights_by_type[insight.insight_type.value] += 1
                
            # Calculate recommendations by type
            recommendations_by_type = defaultdict(int)
            for rec in self.recommendations.values():
                recommendations_by_type[rec.recommendation_type.value] += 1
                
            # Calculate confidence distribution
            confidence_distribution = defaultdict(int)
            for insight in self.insights.values():
                confidence_distribution[insight.confidence_level.value] += 1
                
            # Calculate model statistics
            model_stats = {}
            for model_id, model in self.ai_models.items():
                model_stats[model_id] = {
                    "name": model.model_name,
                    "type": model.model_type,
                    "accuracy": model.accuracy,
                    "status": model.status,
                    "last_trained": model.last_trained.isoformat()
                }
                
            return {
                "timestamp": datetime.now().isoformat(),
                "ai_insights_active": self.ai_insights_active,
                "total_insights": len(self.insights),
                "total_recommendations": len(self.recommendations),
                "total_models": len(self.ai_models),
                "total_users": len(self.user_profiles),
                "insights_by_type": dict(insights_by_type),
                "recommendations_by_type": dict(recommendations_by_type),
                "confidence_distribution": dict(confidence_distribution),
                "model_stats": model_stats,
                "stats": self.ai_insights_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting AI insights summary: {e}")
            return {"error": str(e)}


# Global instance
ai_insights_engine = AIInsightsEngine()
