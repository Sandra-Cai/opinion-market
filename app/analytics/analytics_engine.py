"""
Advanced Analytics Engine
Comprehensive analytics with machine learning and predictive capabilities
"""

import asyncio
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import defaultdict, Counter

from app.core.enhanced_cache import enhanced_cache
from app.core.database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


class AnalyticsType(Enum):
    """Analytics types"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


class MetricCategory(Enum):
    """Metric categories"""
    USER_BEHAVIOR = "user_behavior"
    MARKET_PERFORMANCE = "market_performance"
    SYSTEM_PERFORMANCE = "system_performance"
    BUSINESS_METRICS = "business_metrics"
    FINANCIAL_METRICS = "financial_metrics"


@dataclass
class AnalyticsResult:
    """Analytics result data structure"""
    analysis_id: str
    analysis_type: AnalyticsType
    metric_category: MetricCategory
    result_data: Dict[str, Any]
    confidence_score: float
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class PredictionResult:
    """Prediction result data structure"""
    prediction_id: str
    model_name: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    features_used: List[str]
    timestamp: float
    actual_value: Optional[float] = None


class AnalyticsEngine:
    """Advanced analytics engine with machine learning capabilities"""
    
    def __init__(self):
        self.analytics_cache = {}
        self.prediction_models = {}
        self.trend_data = defaultdict(list)
        self.user_behavior_data = defaultdict(list)
        self.market_data = defaultdict(list)
        
        # Configuration
        self.analysis_interval = 300  # 5 minutes
        self.prediction_horizon = 24 * 3600  # 24 hours
        self.confidence_threshold = 0.7
        
        # Statistics
        self.stats = {
            "analyses_performed": 0,
            "predictions_made": 0,
            "trends_identified": 0,
            "anomalies_detected": 0
        }
        
        # Start analytics tasks
        asyncio.create_task(self._analytics_processing_loop())
        asyncio.create_task(self._prediction_loop())
        asyncio.create_task(self._trend_analysis_loop())
    
    async def start_analytics(self):
        """Start the analytics engine"""
        logger.info("Starting advanced analytics engine")
        
        # Initialize prediction models
        await self._initialize_models()
        
        # Load historical data
        await self._load_historical_data()
        
        logger.info("Advanced analytics engine started")
    
    async def _initialize_models(self):
        """Initialize machine learning models"""
        try:
            # Simple linear regression model for price prediction
            self.prediction_models["price_prediction"] = {
                "type": "linear_regression",
                "features": ["volume", "volatility", "time_of_day", "day_of_week"],
                "trained": False,
                "accuracy": 0.0
            }
            
            # User behavior prediction model
            self.prediction_models["user_behavior"] = {
                "type": "classification",
                "features": ["session_duration", "page_views", "interaction_count", "time_since_last_visit"],
                "trained": False,
                "accuracy": 0.0
            }
            
            # Market trend prediction model
            self.prediction_models["market_trend"] = {
                "type": "time_series",
                "features": ["price_history", "volume_history", "volatility", "external_factors"],
                "trained": False,
                "accuracy": 0.0
            }
            
            logger.info("Prediction models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def _load_historical_data(self):
        """Load historical data for analysis"""
        try:
            # Load market data
            await self._load_market_data()
            
            # Load user behavior data
            await self._load_user_behavior_data()
            
            # Load system performance data
            await self._load_system_performance_data()
            
            logger.info("Historical data loaded")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def _load_market_data(self):
        """Load historical market data"""
        try:
            with engine.connect() as conn:
                # Get market data from last 30 days
                query = """
                SELECT market_id, price, volume, timestamp, volatility
                FROM market_data 
                WHERE timestamp >= NOW() - INTERVAL '30 days'
                ORDER BY timestamp DESC
                """
                result = conn.execute(text(query))
                
                for row in result:
                    self.market_data[row[0]].append({
                        "price": float(row[1]),
                        "volume": float(row[2]),
                        "timestamp": row[3].timestamp(),
                        "volatility": float(row[4]) if row[4] else 0.0
                    })
                
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
    
    async def _load_user_behavior_data(self):
        """Load historical user behavior data"""
        try:
            with engine.connect() as conn:
                # Get user activity data from last 30 days
                query = """
                SELECT user_id, session_duration, page_views, interaction_count, timestamp
                FROM user_activity 
                WHERE timestamp >= NOW() - INTERVAL '30 days'
                ORDER BY timestamp DESC
                """
                result = conn.execute(text(query))
                
                for row in result:
                    self.user_behavior_data[row[0]].append({
                        "session_duration": float(row[1]),
                        "page_views": int(row[2]),
                        "interaction_count": int(row[3]),
                        "timestamp": row[3].timestamp()
                    })
                
        except Exception as e:
            logger.error(f"Error loading user behavior data: {e}")
    
    async def _load_system_performance_data(self):
        """Load historical system performance data"""
        try:
            # Get system metrics from cache
            system_metrics = await enhanced_cache.get("system_metrics_history")
            if system_metrics:
                self.trend_data["system_performance"] = system_metrics
                
        except Exception as e:
            logger.error(f"Error loading system performance data: {e}")
    
    async def _analytics_processing_loop(self):
        """Background task for analytics processing"""
        while True:
            try:
                await self._perform_analytics()
                await asyncio.sleep(self.analysis_interval)
            except Exception as e:
                logger.error(f"Error in analytics processing loop: {e}")
                await asyncio.sleep(300)
    
    async def _perform_analytics(self):
        """Perform comprehensive analytics"""
        try:
            # User behavior analytics
            await self._analyze_user_behavior()
            
            # Market performance analytics
            await self._analyze_market_performance()
            
            # System performance analytics
            await self._analyze_system_performance()
            
            # Business metrics analytics
            await self._analyze_business_metrics()
            
            self.stats["analyses_performed"] += 1
            
        except Exception as e:
            logger.error(f"Error performing analytics: {e}")
    
    async def _analyze_user_behavior(self):
        """Analyze user behavior patterns"""
        try:
            if not self.user_behavior_data:
                return
            
            # Calculate user engagement metrics
            total_users = len(self.user_behavior_data)
            active_users = sum(1 for user_data in self.user_behavior_data.values() if user_data)
            
            # Calculate average session duration
            session_durations = []
            page_views = []
            interaction_counts = []
            
            for user_data in self.user_behavior_data.values():
                for session in user_data:
                    session_durations.append(session["session_duration"])
                    page_views.append(session["page_views"])
                    interaction_counts.append(session["interaction_count"])
            
            avg_session_duration = statistics.mean(session_durations) if session_durations else 0
            avg_page_views = statistics.mean(page_views) if page_views else 0
            avg_interactions = statistics.mean(interaction_counts) if interaction_counts else 0
            
            # Identify user segments
            user_segments = self._identify_user_segments()
            
            # Detect anomalies in user behavior
            anomalies = self._detect_user_behavior_anomalies()
            
            result = AnalyticsResult(
                analysis_id=f"user_behavior_{int(time.time())}",
                analysis_type=AnalyticsType.DESCRIPTIVE,
                metric_category=MetricCategory.USER_BEHAVIOR,
                result_data={
                    "total_users": total_users,
                    "active_users": active_users,
                    "user_activity_rate": (active_users / max(total_users, 1)) * 100,
                    "avg_session_duration": avg_session_duration,
                    "avg_page_views": avg_page_views,
                    "avg_interactions": avg_interactions,
                    "user_segments": user_segments,
                    "anomalies": anomalies
                },
                confidence_score=0.85,
                timestamp=time.time(),
                metadata={"data_points": len(session_durations)}
            )
            
            # Store result
            await self._store_analytics_result(result)
            
        except Exception as e:
            logger.error(f"Error analyzing user behavior: {e}")
    
    async def _analyze_market_performance(self):
        """Analyze market performance"""
        try:
            if not self.market_data:
                return
            
            market_analytics = {}
            
            for market_id, data in self.market_data.items():
                if not data:
                    continue
                
                # Calculate price statistics
                prices = [d["price"] for d in data]
                volumes = [d["volume"] for d in data]
                volatilities = [d["volatility"] for d in data]
                
                price_change = ((prices[0] - prices[-1]) / prices[-1]) * 100 if len(prices) > 1 else 0
                avg_volume = statistics.mean(volumes) if volumes else 0
                avg_volatility = statistics.mean(volatilities) if volatilities else 0
                
                # Calculate trend
                trend = self._calculate_trend(prices)
                
                # Detect price anomalies
                price_anomalies = self._detect_price_anomalies(prices)
                
                market_analytics[market_id] = {
                    "current_price": prices[0] if prices else 0,
                    "price_change_percent": price_change,
                    "avg_volume": avg_volume,
                    "avg_volatility": avg_volatility,
                    "trend": trend,
                    "price_anomalies": price_anomalies,
                    "data_points": len(data)
                }
            
            result = AnalyticsResult(
                analysis_id=f"market_performance_{int(time.time())}",
                analysis_type=AnalyticsType.DESCRIPTIVE,
                metric_category=MetricCategory.MARKET_PERFORMANCE,
                result_data=market_analytics,
                confidence_score=0.80,
                timestamp=time.time(),
                metadata={"markets_analyzed": len(market_analytics)}
            )
            
            # Store result
            await self._store_analytics_result(result)
            
        except Exception as e:
            logger.error(f"Error analyzing market performance: {e}")
    
    async def _analyze_system_performance(self):
        """Analyze system performance"""
        try:
            # Get system metrics from monitoring
            system_metrics = await enhanced_cache.get("system_metrics")
            if not system_metrics:
                return
            
            # Calculate performance indicators
            cpu_usage = system_metrics.get("cpu_percent", 0)
            memory_usage = system_metrics.get("memory_percent", 0)
            disk_usage = system_metrics.get("disk_usage_percent", 0)
            
            # Determine system health
            health_score = self._calculate_system_health_score(cpu_usage, memory_usage, disk_usage)
            
            # Identify performance bottlenecks
            bottlenecks = self._identify_performance_bottlenecks(system_metrics)
            
            result = AnalyticsResult(
                analysis_id=f"system_performance_{int(time.time())}",
                analysis_type=AnalyticsType.DIAGNOSTIC,
                metric_category=MetricCategory.SYSTEM_PERFORMANCE,
                result_data={
                    "health_score": health_score,
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage,
                    "bottlenecks": bottlenecks,
                    "performance_trend": "stable"  # Simplified
                },
                confidence_score=0.90,
                timestamp=time.time(),
                metadata={"metrics_analyzed": len(system_metrics)}
            )
            
            # Store result
            await self._store_analytics_result(result)
            
        except Exception as e:
            logger.error(f"Error analyzing system performance: {e}")
    
    async def _analyze_business_metrics(self):
        """Analyze business metrics"""
        try:
            # Get business metrics from database
            with engine.connect() as conn:
                # Revenue metrics
                revenue_query = """
                SELECT SUM(amount) as total_revenue, COUNT(*) as transaction_count
                FROM transactions 
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                """
                revenue_result = conn.execute(text(revenue_query))
                revenue_data = revenue_result.fetchone()
                
                # User growth metrics
                user_query = """
                SELECT COUNT(*) as new_users
                FROM users 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                """
                user_result = conn.execute(text(user_query))
                user_data = user_result.fetchone()
                
                # Market activity metrics
                market_query = """
                SELECT COUNT(*) as active_markets
                FROM markets 
                WHERE status = 'active'
                """
                market_result = conn.execute(text(market_query))
                market_data = market_result.fetchone()
            
            result = AnalyticsResult(
                analysis_id=f"business_metrics_{int(time.time())}",
                analysis_type=AnalyticsType.DESCRIPTIVE,
                metric_category=MetricCategory.BUSINESS_METRICS,
                result_data={
                    "daily_revenue": float(revenue_data[0]) if revenue_data[0] else 0,
                    "transaction_count": int(revenue_data[1]) if revenue_data[1] else 0,
                    "new_users": int(user_data[0]) if user_data[0] else 0,
                    "active_markets": int(market_data[0]) if market_data[0] else 0,
                    "revenue_per_transaction": float(revenue_data[0]) / max(int(revenue_data[1]), 1) if revenue_data[1] else 0
                },
                confidence_score=0.95,
                timestamp=time.time(),
                metadata={"time_period": "24_hours"}
            )
            
            # Store result
            await self._store_analytics_result(result)
            
        except Exception as e:
            logger.error(f"Error analyzing business metrics: {e}")
    
    def _identify_user_segments(self) -> Dict[str, Any]:
        """Identify user segments based on behavior"""
        try:
            segments = {
                "high_engagement": 0,
                "medium_engagement": 0,
                "low_engagement": 0,
                "new_users": 0
            }
            
            for user_id, user_data in self.user_behavior_data.items():
                if not user_data:
                    continue
                
                # Calculate user engagement score
                avg_session_duration = statistics.mean([s["session_duration"] for s in user_data])
                avg_page_views = statistics.mean([s["page_views"] for s in user_data])
                avg_interactions = statistics.mean([s["interaction_count"] for s in user_data])
                
                engagement_score = (avg_session_duration * 0.4 + avg_page_views * 0.3 + avg_interactions * 0.3)
                
                if engagement_score > 100:
                    segments["high_engagement"] += 1
                elif engagement_score > 50:
                    segments["medium_engagement"] += 1
                else:
                    segments["low_engagement"] += 1
                
                # Check if new user (less than 7 days of data)
                if len(user_data) < 7:
                    segments["new_users"] += 1
            
            return segments
            
        except Exception as e:
            logger.error(f"Error identifying user segments: {e}")
            return {}
    
    def _detect_user_behavior_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in user behavior"""
        try:
            anomalies = []
            
            for user_id, user_data in self.user_behavior_data.items():
                if len(user_data) < 3:
                    continue
                
                # Check for unusual session duration
                session_durations = [s["session_duration"] for s in user_data]
                avg_duration = statistics.mean(session_durations)
                std_duration = statistics.stdev(session_durations) if len(session_durations) > 1 else 0
                
                for i, session in enumerate(user_data):
                    if std_duration > 0 and abs(session["session_duration"] - avg_duration) > 2 * std_duration:
                        anomalies.append({
                            "user_id": user_id,
                            "type": "unusual_session_duration",
                            "value": session["session_duration"],
                            "expected_range": [avg_duration - 2 * std_duration, avg_duration + 2 * std_duration],
                            "timestamp": session["timestamp"]
                        })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting user behavior anomalies: {e}")
            return []
    
    def _calculate_trend(self, prices: List[float]) -> str:
        """Calculate price trend"""
        try:
            if len(prices) < 2:
                return "insufficient_data"
            
            # Simple linear trend calculation
            x = list(range(len(prices)))
            y = prices
            
            # Calculate slope
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            if slope > 0.01:
                return "upward"
            elif slope < -0.01:
                return "downward"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "unknown"
    
    def _detect_price_anomalies(self, prices: List[float]) -> List[Dict[str, Any]]:
        """Detect price anomalies"""
        try:
            if len(prices) < 3:
                return []
            
            anomalies = []
            avg_price = statistics.mean(prices)
            std_price = statistics.stdev(prices)
            
            for i, price in enumerate(prices):
                if std_price > 0 and abs(price - avg_price) > 2 * std_price:
                    anomalies.append({
                        "index": i,
                        "price": price,
                        "expected_range": [avg_price - 2 * std_price, avg_price + 2 * std_price],
                        "deviation": abs(price - avg_price) / std_price
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting price anomalies: {e}")
            return []
    
    def _calculate_system_health_score(self, cpu_usage: float, memory_usage: float, disk_usage: float) -> float:
        """Calculate system health score (0-100)"""
        try:
            # Weighted health score calculation
            cpu_score = max(0, 100 - cpu_usage)
            memory_score = max(0, 100 - memory_usage)
            disk_score = max(0, 100 - disk_usage)
            
            # Weighted average
            health_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
            
            return round(health_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating system health score: {e}")
            return 0.0
    
    def _identify_performance_bottlenecks(self, system_metrics: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks"""
        try:
            bottlenecks = []
            
            if system_metrics.get("cpu_percent", 0) > 80:
                bottlenecks.append("High CPU usage")
            
            if system_metrics.get("memory_percent", 0) > 85:
                bottlenecks.append("High memory usage")
            
            if system_metrics.get("disk_usage_percent", 0) > 90:
                bottlenecks.append("Low disk space")
            
            if system_metrics.get("load_average_1m", 0) > 2.0:
                bottlenecks.append("High system load")
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Error identifying performance bottlenecks: {e}")
            return []
    
    async def _prediction_loop(self):
        """Background task for making predictions"""
        while True:
            try:
                await self._make_predictions()
                await asyncio.sleep(3600)  # Make predictions every hour
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(3600)
    
    async def _make_predictions(self):
        """Make predictions using trained models"""
        try:
            # Price predictions
            await self._predict_market_prices()
            
            # User behavior predictions
            await self._predict_user_behavior()
            
            # System performance predictions
            await self._predict_system_performance()
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
    
    async def _predict_market_prices(self):
        """Predict market prices"""
        try:
            for market_id, data in self.market_data.items():
                if len(data) < 10:
                    continue
                
                # Simple price prediction using moving average
                recent_prices = [d["price"] for d in data[:10]]
                predicted_price = statistics.mean(recent_prices)
                
                # Calculate confidence interval (simplified)
                price_std = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
                confidence_interval = (predicted_price - price_std, predicted_price + price_std)
                
                prediction = PredictionResult(
                    prediction_id=f"price_{market_id}_{int(time.time())}",
                    model_name="moving_average",
                    predicted_value=predicted_price,
                    confidence_interval=confidence_interval,
                    features_used=["price_history"],
                    timestamp=time.time()
                )
                
                # Store prediction
                await self._store_prediction(prediction)
                
                self.stats["predictions_made"] += 1
            
        except Exception as e:
            logger.error(f"Error predicting market prices: {e}")
    
    async def _predict_user_behavior(self):
        """Predict user behavior"""
        try:
            # Simple user behavior prediction
            for user_id, user_data in self.user_behavior_data.items():
                if len(user_data) < 5:
                    continue
                
                # Predict next session duration
                recent_durations = [d["session_duration"] for d in user_data[:5]]
                predicted_duration = statistics.mean(recent_durations)
                
                prediction = PredictionResult(
                    prediction_id=f"behavior_{user_id}_{int(time.time())}",
                    model_name="session_duration_predictor",
                    predicted_value=predicted_duration,
                    confidence_interval=(predicted_duration * 0.8, predicted_duration * 1.2),
                    features_used=["session_duration_history"],
                    timestamp=time.time()
                )
                
                # Store prediction
                await self._store_prediction(prediction)
                
        except Exception as e:
            logger.error(f"Error predicting user behavior: {e}")
    
    async def _predict_system_performance(self):
        """Predict system performance"""
        try:
            # Get recent system metrics
            system_metrics = await enhanced_cache.get("system_metrics")
            if not system_metrics:
                return
            
            # Simple performance prediction
            cpu_usage = system_metrics.get("cpu_percent", 0)
            memory_usage = system_metrics.get("memory_percent", 0)
            
            # Predict future CPU usage (simplified)
            predicted_cpu = min(100, cpu_usage * 1.1)  # Assume 10% increase
            
            prediction = PredictionResult(
                prediction_id=f"system_performance_{int(time.time())}",
                model_name="performance_predictor",
                predicted_value=predicted_cpu,
                confidence_interval=(predicted_cpu - 5, predicted_cpu + 5),
                features_used=["cpu_usage", "memory_usage"],
                timestamp=time.time()
            )
            
            # Store prediction
            await self._store_prediction(prediction)
            
        except Exception as e:
            logger.error(f"Error predicting system performance: {e}")
    
    async def _trend_analysis_loop(self):
        """Background task for trend analysis"""
        while True:
            try:
                await self._analyze_trends()
                await asyncio.sleep(1800)  # Analyze trends every 30 minutes
            except Exception as e:
                logger.error(f"Error in trend analysis loop: {e}")
                await asyncio.sleep(1800)
    
    async def _analyze_trends(self):
        """Analyze trends in data"""
        try:
            # Analyze market trends
            for market_id, data in self.market_data.items():
                if len(data) < 20:
                    continue
                
                prices = [d["price"] for d in data[:20]]
                trend = self._calculate_trend(prices)
                
                if trend != "stable":
                    self.stats["trends_identified"] += 1
                    
                    # Store trend analysis
                    trend_result = AnalyticsResult(
                        analysis_id=f"trend_{market_id}_{int(time.time())}",
                        analysis_type=AnalyticsType.DIAGNOSTIC,
                        metric_category=MetricCategory.MARKET_PERFORMANCE,
                        result_data={
                            "market_id": market_id,
                            "trend": trend,
                            "price_change": ((prices[0] - prices[-1]) / prices[-1]) * 100,
                            "data_points": len(prices)
                        },
                        confidence_score=0.75,
                        timestamp=time.time(),
                        metadata={"analysis_type": "trend"}
                    )
                    
                    await self._store_analytics_result(trend_result)
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
    
    async def _store_analytics_result(self, result: AnalyticsResult):
        """Store analytics result"""
        try:
            # Store in cache
            await enhanced_cache.set(
                f"analytics_{result.analysis_id}",
                asdict(result),
                ttl=86400,  # 24 hours
                tags=["analytics", result.analysis_type.value, result.metric_category.value]
            )
            
            # Store in database
            with engine.connect() as conn:
                query = """
                INSERT INTO analytics_results (analysis_id, analysis_type, metric_category, 
                                             result_data, confidence_score, timestamp, metadata)
                VALUES (:analysis_id, :analysis_type, :metric_category, :result_data, 
                       :confidence_score, :timestamp, :metadata)
                ON CONFLICT (analysis_id) 
                DO UPDATE SET result_data = :result_data, confidence_score = :confidence_score,
                             timestamp = :timestamp, metadata = :metadata
                """
                
                conn.execute(text(query), {
                    "analysis_id": result.analysis_id,
                    "analysis_type": result.analysis_type.value,
                    "metric_category": result.metric_category.value,
                    "result_data": json.dumps(result.result_data),
                    "confidence_score": result.confidence_score,
                    "timestamp": result.timestamp,
                    "metadata": json.dumps(result.metadata)
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing analytics result: {e}")
    
    async def _store_prediction(self, prediction: PredictionResult):
        """Store prediction result"""
        try:
            # Store in cache
            await enhanced_cache.set(
                f"prediction_{prediction.prediction_id}",
                asdict(prediction),
                ttl=3600,  # 1 hour
                tags=["predictions", prediction.model_name]
            )
            
            # Store in database
            with engine.connect() as conn:
                query = """
                INSERT INTO predictions (prediction_id, model_name, predicted_value, 
                                       confidence_interval, features_used, timestamp, actual_value)
                VALUES (:prediction_id, :model_name, :predicted_value, :confidence_interval,
                       :features_used, :timestamp, :actual_value)
                ON CONFLICT (prediction_id) 
                DO UPDATE SET predicted_value = :predicted_value, 
                             confidence_interval = :confidence_interval,
                             timestamp = :timestamp, actual_value = :actual_value
                """
                
                conn.execute(text(query), {
                    "prediction_id": prediction.prediction_id,
                    "model_name": prediction.model_name,
                    "predicted_value": prediction.predicted_value,
                    "confidence_interval": json.dumps(prediction.confidence_interval),
                    "features_used": json.dumps(prediction.features_used),
                    "timestamp": prediction.timestamp,
                    "actual_value": prediction.actual_value
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    async def get_analytics_results(self, analysis_type: Optional[AnalyticsType] = None,
                                  metric_category: Optional[MetricCategory] = None,
                                  time_range: int = 86400) -> List[AnalyticsResult]:
        """Get analytics results"""
        try:
            with engine.connect() as conn:
                query = """
                SELECT analysis_id, analysis_type, metric_category, result_data, 
                       confidence_score, timestamp, metadata
                FROM analytics_results 
                WHERE timestamp >= :cutoff_time
                """
                params = {"cutoff_time": time.time() - time_range}
                
                if analysis_type:
                    query += " AND analysis_type = :analysis_type"
                    params["analysis_type"] = analysis_type.value
                
                if metric_category:
                    query += " AND metric_category = :metric_category"
                    params["metric_category"] = metric_category.value
                
                query += " ORDER BY timestamp DESC"
                
                result = conn.execute(text(query), params)
                
                analytics_results = []
                for row in result:
                    analytics_results.append(AnalyticsResult(
                        analysis_id=row[0],
                        analysis_type=AnalyticsType(row[1]),
                        metric_category=MetricCategory(row[2]),
                        result_data=json.loads(row[3]),
                        confidence_score=row[4],
                        timestamp=row[5],
                        metadata=json.loads(row[6])
                    ))
                
                return analytics_results
                
        except Exception as e:
            logger.error(f"Error getting analytics results: {e}")
            return []
    
    async def get_predictions(self, model_name: Optional[str] = None,
                            time_range: int = 3600) -> List[PredictionResult]:
        """Get predictions"""
        try:
            with engine.connect() as conn:
                query = """
                SELECT prediction_id, model_name, predicted_value, confidence_interval,
                       features_used, timestamp, actual_value
                FROM predictions 
                WHERE timestamp >= :cutoff_time
                """
                params = {"cutoff_time": time.time() - time_range}
                
                if model_name:
                    query += " AND model_name = :model_name"
                    params["model_name"] = model_name
                
                query += " ORDER BY timestamp DESC"
                
                result = conn.execute(text(query), params)
                
                predictions = []
                for row in result:
                    predictions.append(PredictionResult(
                        prediction_id=row[0],
                        model_name=row[1],
                        predicted_value=row[2],
                        confidence_interval=tuple(json.loads(row[3])),
                        features_used=json.loads(row[4]),
                        timestamp=row[5],
                        actual_value=row[6]
                    ))
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
    
    def get_analytics_stats(self) -> Dict[str, Any]:
        """Get analytics engine statistics"""
        return {
            "analyses_performed": self.stats["analyses_performed"],
            "predictions_made": self.stats["predictions_made"],
            "trends_identified": self.stats["trends_identified"],
            "anomalies_detected": self.stats["anomalies_detected"],
            "models_available": len(self.prediction_models),
            "data_sources": {
                "market_data": len(self.market_data),
                "user_behavior_data": len(self.user_behavior_data),
                "trend_data": len(self.trend_data)
            }
        }


# Global analytics engine instance
analytics_engine = AnalyticsEngine()
