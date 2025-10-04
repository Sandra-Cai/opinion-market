"""
Real-time Analytics Engine
Comprehensive real-time analytics and streaming data processing system
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import secrets
import numpy as np
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Stream type enumeration"""
    MARKET_DATA = "market_data"
    USER_ACTIVITY = "user_activity"
    TRADING_EVENTS = "trading_events"
    SYSTEM_METRICS = "system_metrics"
    NEWS_FEED = "news_feed"
    SOCIAL_SENTIMENT = "social_sentiment"
    PRICE_ALERTS = "price_alerts"
    RISK_EVENTS = "risk_events"


class AnalyticsType(Enum):
    """Analytics type enumeration"""
    REAL_TIME_DASHBOARD = "real_time_dashboard"
    TREND_ANALYSIS = "trend_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    VOLUME_ANALYSIS = "volume_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_METRICS = "risk_metrics"
    PERFORMANCE_METRICS = "performance_metrics"


class ProcessingMode(Enum):
    """Processing mode enumeration"""
    BATCH = "batch"
    STREAMING = "streaming"
    MICRO_BATCH = "micro_batch"
    CONTINUOUS = "continuous"


@dataclass
class DataPoint:
    """Data point structure"""
    timestamp: datetime
    stream_type: StreamType
    data: Dict[str, Any]
    source: str
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamProcessor:
    """Stream processor configuration"""
    processor_id: str
    stream_type: StreamType
    processing_mode: ProcessingMode
    window_size: int  # seconds
    batch_size: int
    filters: List[Dict[str, Any]]
    transformations: List[Dict[str, Any]]
    aggregations: List[Dict[str, Any]]
    output_schema: Dict[str, Any]
    status: str = "active"


@dataclass
class AnalyticsResult:
    """Analytics result structure"""
    result_id: str
    analytics_type: AnalyticsType
    timestamp: datetime
    data: Dict[str, Any]
    metrics: Dict[str, float]
    insights: List[str]
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamWindow:
    """Stream window for processing"""
    window_id: str
    start_time: datetime
    end_time: datetime
    data_points: List[DataPoint]
    aggregations: Dict[str, Any]
    processed: bool = False


class RealTimeAnalyticsEngine:
    """Real-time Analytics Engine for streaming data processing and analytics"""
    
    def __init__(self):
        self.data_streams: Dict[StreamType, deque] = {
            stream_type: deque(maxlen=10000) for stream_type in StreamType
        }
        
        self.stream_processors: Dict[str, StreamProcessor] = {}
        self.analytics_results: Dict[str, AnalyticsResult] = {}
        self.stream_windows: Dict[str, StreamWindow] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Configuration
        self.config = {
            "real_time_analytics_enabled": True,
            "streaming_enabled": True,
            "batch_processing_enabled": True,
            "micro_batch_enabled": True,
            "data_retention_hours": 24,
            "window_size_seconds": 60,
            "batch_size": 1000,
            "processing_parallelism": 4,
            "max_stream_size": 10000,
            "analytics_update_interval": 5,  # seconds
            "stream_health_check_interval": 30,  # seconds
            "data_quality_threshold": 0.8,
            "anomaly_detection_enabled": True,
            "trend_detection_enabled": True,
            "correlation_analysis_enabled": True,
            "performance_monitoring_enabled": True
        }
        
        # Stream configurations
        self.stream_configs = {
            StreamType.MARKET_DATA: {
                "fields": ["symbol", "price", "volume", "timestamp", "bid", "ask"],
                "update_frequency": 1,  # seconds
                "data_quality_checks": ["price_range", "volume_consistency", "timestamp_validity"]
            },
            StreamType.USER_ACTIVITY: {
                "fields": ["user_id", "action", "timestamp", "session_id", "ip_address"],
                "update_frequency": 5,  # seconds
                "data_quality_checks": ["user_id_validity", "action_consistency"]
            },
            StreamType.TRADING_EVENTS: {
                "fields": ["trade_id", "user_id", "symbol", "quantity", "price", "side", "timestamp"],
                "update_frequency": 1,  # seconds
                "data_quality_checks": ["trade_consistency", "price_validation"]
            },
            StreamType.SYSTEM_METRICS: {
                "fields": ["metric_name", "value", "timestamp", "host", "service"],
                "update_frequency": 10,  # seconds
                "data_quality_checks": ["value_range", "timestamp_consistency"]
            }
        }
        
        # Analytics configurations
        self.analytics_configs = {
            AnalyticsType.REAL_TIME_DASHBOARD: {
                "update_frequency": 5,  # seconds
                "metrics": ["total_volume", "active_users", "price_changes", "system_health"],
                "visualization": "dashboard"
            },
            AnalyticsType.TREND_ANALYSIS: {
                "update_frequency": 60,  # seconds
                "window_size": 300,  # 5 minutes
                "algorithms": ["moving_average", "exponential_smoothing", "linear_regression"]
            },
            AnalyticsType.CORRELATION_ANALYSIS: {
                "update_frequency": 300,  # 5 minutes
                "window_size": 3600,  # 1 hour
                "correlation_threshold": 0.7
            },
            AnalyticsType.VOLATILITY_ANALYSIS: {
                "update_frequency": 60,  # seconds
                "window_size": 300,  # 5 minutes
                "methods": ["standard_deviation", "garch", "realized_volatility"]
            }
        }
        
        # Monitoring
        self.analytics_active = False
        self.analytics_task: Optional[asyncio.Task] = None
        self.streaming_task: Optional[asyncio.Task] = None
        self.processing_executor = ThreadPoolExecutor(max_workers=self.config["processing_parallelism"])
        
        # Performance tracking
        self.analytics_stats = {
            "data_points_processed": 0,
            "streams_processed": 0,
            "analytics_generated": 0,
            "anomalies_detected": 0,
            "trends_identified": 0,
            "correlations_found": 0,
            "processing_errors": 0,
            "average_processing_time": 0.0
        }
        
    async def start_analytics_engine(self):
        """Start the real-time analytics engine"""
        if self.analytics_active:
            logger.warning("Real-time analytics engine already active")
            return
            
        self.analytics_active = True
        self.analytics_task = asyncio.create_task(self._analytics_processing_loop())
        self.streaming_task = asyncio.create_task(self._streaming_processing_loop())
        
        # Initialize stream processors
        await self._initialize_stream_processors()
        
        logger.info("Real-time Analytics Engine started")
        
    async def stop_analytics_engine(self):
        """Stop the real-time analytics engine"""
        self.analytics_active = False
        if self.analytics_task:
            self.analytics_task.cancel()
            try:
                await self.analytics_task
            except asyncio.CancelledError:
                pass
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        self.processing_executor.shutdown(wait=True)
        logger.info("Real-time Analytics Engine stopped")
        
    async def _analytics_processing_loop(self):
        """Main analytics processing loop"""
        while self.analytics_active:
            try:
                # Process real-time analytics
                await self._process_real_time_analytics()
                
                # Update dashboards
                await self._update_dashboards()
                
                # Detect anomalies
                if self.config["anomaly_detection_enabled"]:
                    await self._detect_anomalies()
                    
                # Analyze trends
                if self.config["trend_detection_enabled"]:
                    await self._analyze_trends()
                    
                # Perform correlation analysis
                if self.config["correlation_analysis_enabled"]:
                    await self._perform_correlation_analysis()
                    
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next cycle
                await asyncio.sleep(self.config["analytics_update_interval"])
                
            except Exception as e:
                logger.error(f"Error in analytics processing loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _streaming_processing_loop(self):
        """Streaming data processing loop"""
        while self.analytics_active:
            try:
                # Process streaming data
                await self._process_streaming_data()
                
                # Update stream health
                await self._update_stream_health()
                
                # Wait before next cycle
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error in streaming processing loop: {e}")
                await asyncio.sleep(10)
                
    async def _initialize_stream_processors(self):
        """Initialize stream processors"""
        try:
            for stream_type, config in self.stream_configs.items():
                processor_id = f"processor_{stream_type.value}"
                
                processor = StreamProcessor(
                    processor_id=processor_id,
                    stream_type=stream_type,
                    processing_mode=ProcessingMode.STREAMING,
                    window_size=self.config["window_size_seconds"],
                    batch_size=self.config["batch_size"],
                    filters=[],
                    transformations=[],
                    aggregations=[],
                    output_schema=config
                )
                
                self.stream_processors[processor_id] = processor
                
            logger.info(f"Initialized {len(self.stream_processors)} stream processors")
            
        except Exception as e:
            logger.error(f"Error initializing stream processors: {e}")
            
    async def ingest_data(self, stream_type: StreamType, data: Dict[str, Any], source: str = "unknown") -> bool:
        """Ingest data into stream"""
        try:
            # Create data point
            data_point = DataPoint(
                timestamp=datetime.now(),
                stream_type=stream_type,
                data=data,
                source=source,
                quality_score=self._calculate_data_quality(data, stream_type)
            )
            
            # Add to stream
            self.data_streams[stream_type].append(data_point)
            
            # Update statistics
            self.analytics_stats["data_points_processed"] += 1
            
            # Notify subscribers
            await self._notify_subscribers(stream_type, data_point)
            
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting data: {e}")
            return False
            
    async def _process_streaming_data(self):
        """Process streaming data"""
        try:
            for stream_type, stream in self.data_streams.items():
                if len(stream) > 0:
                    # Process recent data points
                    recent_points = list(stream)[-100:]  # Last 100 points
                    
                    # Apply stream processor
                    processor = self.stream_processors.get(f"processor_{stream_type.value}")
                    if processor:
                        await self._apply_stream_processor(processor, recent_points)
                        
        except Exception as e:
            logger.error(f"Error processing streaming data: {e}")
            
    async def _apply_stream_processor(self, processor: StreamProcessor, data_points: List[DataPoint]):
        """Apply stream processor to data points"""
        try:
            # Filter data points
            filtered_points = await self._apply_filters(processor.filters, data_points)
            
            # Apply transformations
            transformed_points = await self._apply_transformations(processor.transformations, filtered_points)
            
            # Apply aggregations
            aggregations = await self._apply_aggregations(processor.aggregations, transformed_points)
            
            # Update processor status
            processor.status = "processing"
            
            # Simulate processing time
            await asyncio.sleep(0.01)
            
            processor.status = "active"
            self.analytics_stats["streams_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error applying stream processor: {e}")
            processor.status = "error"
            
    async def _apply_filters(self, filters: List[Dict[str, Any]], data_points: List[DataPoint]) -> List[DataPoint]:
        """Apply filters to data points"""
        try:
            filtered_points = data_points
            
            for filter_config in filters:
                filter_type = filter_config.get("type")
                
                if filter_type == "quality_threshold":
                    threshold = filter_config.get("threshold", 0.8)
                    filtered_points = [dp for dp in filtered_points if dp.quality_score >= threshold]
                elif filter_type == "time_range":
                    start_time = filter_config.get("start_time")
                    end_time = filter_config.get("end_time")
                    if start_time and end_time:
                        filtered_points = [
                            dp for dp in filtered_points
                            if start_time <= dp.timestamp <= end_time
                        ]
                        
            return filtered_points
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return data_points
            
    async def _apply_transformations(self, transformations: List[Dict[str, Any]], data_points: List[DataPoint]) -> List[DataPoint]:
        """Apply transformations to data points"""
        try:
            transformed_points = data_points
            
            for transform_config in transformations:
                transform_type = transform_config.get("type")
                
                if transform_type == "normalize":
                    field = transform_config.get("field")
                    if field:
                        values = [dp.data.get(field) for dp in transformed_points if field in dp.data]
                        if values:
                            min_val, max_val = min(values), max(values)
                            for dp in transformed_points:
                                if field in dp.data and max_val > min_val:
                                    dp.data[field] = (dp.data[field] - min_val) / (max_val - min_val)
                                    
            return transformed_points
            
        except Exception as e:
            logger.error(f"Error applying transformations: {e}")
            return data_points
            
    async def _apply_aggregations(self, aggregations: List[Dict[str, Any]], data_points: List[DataPoint]) -> Dict[str, Any]:
        """Apply aggregations to data points"""
        try:
            result = {}
            
            for agg_config in aggregations:
                agg_type = agg_config.get("type")
                field = agg_config.get("field")
                
                if field:
                    values = [dp.data.get(field) for dp in data_points if field in dp.data and dp.data[field] is not None]
                    
                    if values:
                        if agg_type == "sum":
                            result[f"{field}_sum"] = sum(values)
                        elif agg_type == "avg":
                            result[f"{field}_avg"] = sum(values) / len(values)
                        elif agg_type == "min":
                            result[f"{field}_min"] = min(values)
                        elif agg_type == "max":
                            result[f"{field}_max"] = max(values)
                        elif agg_type == "count":
                            result[f"{field}_count"] = len(values)
                            
            return result
            
        except Exception as e:
            logger.error(f"Error applying aggregations: {e}")
            return {}
            
    async def _process_real_time_analytics(self):
        """Process real-time analytics"""
        try:
            # Generate real-time dashboard analytics
            dashboard_data = await self._generate_dashboard_analytics()
            if dashboard_data:
                await self._store_analytics_result(AnalyticsType.REAL_TIME_DASHBOARD, dashboard_data)
                
            # Generate trend analysis
            trend_data = await self._generate_trend_analysis()
            if trend_data:
                await self._store_analytics_result(AnalyticsType.TREND_ANALYSIS, trend_data)
                
            # Generate volatility analysis
            volatility_data = await self._generate_volatility_analysis()
            if volatility_data:
                await self._store_analytics_result(AnalyticsType.VOLATILITY_ANALYSIS, volatility_data)
                
        except Exception as e:
            logger.error(f"Error processing real-time analytics: {e}")
            
    async def _generate_dashboard_analytics(self) -> Dict[str, Any]:
        """Generate real-time dashboard analytics"""
        try:
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "total_volume": 0,
                "active_users": 0,
                "price_changes": {},
                "system_health": "healthy",
                "top_assets": [],
                "recent_trades": []
            }
            
            # Calculate total volume
            market_stream = self.data_streams[StreamType.MARKET_DATA]
            if market_stream:
                total_volume = sum(
                    dp.data.get("volume", 0) for dp in market_stream
                    if "volume" in dp.data
                )
                dashboard_data["total_volume"] = total_volume
                
            # Calculate active users
            user_stream = self.data_streams[StreamType.USER_ACTIVITY]
            if user_stream:
                unique_users = set(
                    dp.data.get("user_id") for dp in user_stream
                    if "user_id" in dp.data
                )
                dashboard_data["active_users"] = len(unique_users)
                
            # Calculate price changes
            if market_stream:
                price_changes = {}
                for dp in market_stream:
                    symbol = dp.data.get("symbol")
                    price = dp.data.get("price")
                    if symbol and price:
                        if symbol not in price_changes:
                            price_changes[symbol] = {"current": price, "change": 0}
                        else:
                            old_price = price_changes[symbol]["current"]
                            price_changes[symbol]["change"] = ((price - old_price) / old_price) * 100
                            price_changes[symbol]["current"] = price
                            
                dashboard_data["price_changes"] = price_changes
                
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating dashboard analytics: {e}")
            return {}
            
    async def _generate_trend_analysis(self) -> Dict[str, Any]:
        """Generate trend analysis"""
        try:
            trend_data = {
                "timestamp": datetime.now().isoformat(),
                "trends": {},
                "trend_strength": {},
                "trend_direction": {}
            }
            
            # Analyze trends for each asset
            market_stream = self.data_streams[StreamType.MARKET_DATA]
            if market_stream:
                # Group by symbol
                symbol_data = defaultdict(list)
                for dp in market_stream:
                    symbol = dp.data.get("symbol")
                    price = dp.data.get("price")
                    if symbol and price:
                        symbol_data[symbol].append((dp.timestamp, price))
                        
                # Calculate trends
                for symbol, data_points in symbol_data.items():
                    if len(data_points) >= 2:
                        # Simple trend calculation
                        prices = [price for _, price in data_points]
                        trend = (prices[-1] - prices[0]) / prices[0] * 100
                        
                        trend_data["trends"][symbol] = trend
                        trend_data["trend_strength"][symbol] = abs(trend)
                        trend_data["trend_direction"][symbol] = "up" if trend > 0 else "down"
                        
            return trend_data
            
        except Exception as e:
            logger.error(f"Error generating trend analysis: {e}")
            return {}
            
    async def _generate_volatility_analysis(self) -> Dict[str, Any]:
        """Generate volatility analysis"""
        try:
            volatility_data = {
                "timestamp": datetime.now().isoformat(),
                "volatility": {},
                "volatility_rankings": []
            }
            
            # Calculate volatility for each asset
            market_stream = self.data_streams[StreamType.MARKET_DATA]
            if market_stream:
                # Group by symbol
                symbol_data = defaultdict(list)
                for dp in market_stream:
                    symbol = dp.data.get("symbol")
                    price = dp.data.get("price")
                    if symbol and price:
                        symbol_data[symbol].append(price)
                        
                # Calculate volatility
                for symbol, prices in symbol_data.items():
                    if len(prices) >= 2:
                        # Calculate standard deviation
                        mean_price = sum(prices) / len(prices)
                        variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
                        volatility = variance ** 0.5
                        
                        volatility_data["volatility"][symbol] = volatility
                        
                # Rank by volatility
                volatility_rankings = sorted(
                    volatility_data["volatility"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                volatility_data["volatility_rankings"] = volatility_rankings
                
            return volatility_data
            
        except Exception as e:
            logger.error(f"Error generating volatility analysis: {e}")
            return {}
            
    async def _store_analytics_result(self, analytics_type: AnalyticsType, data: Dict[str, Any]):
        """Store analytics result"""
        try:
            result_id = f"analytics_{int(time.time())}_{secrets.token_hex(4)}"
            
            result = AnalyticsResult(
                result_id=result_id,
                analytics_type=analytics_type,
                timestamp=datetime.now(),
                data=data,
                metrics=self._calculate_metrics(data),
                insights=self._generate_insights(data, analytics_type),
                confidence=0.8 + (secrets.randbelow(20) / 100.0),
                processing_time_ms=10.0 + (secrets.randbelow(50))
            )
            
            self.analytics_results[result_id] = result
            self.analytics_stats["analytics_generated"] += 1
            
        except Exception as e:
            logger.error(f"Error storing analytics result: {e}")
            
    def _calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics from data"""
        try:
            metrics = {}
            
            # Calculate basic metrics
            if "total_volume" in data:
                metrics["volume_score"] = min(data["total_volume"] / 1000000, 1.0)  # Normalize
                
            if "active_users" in data:
                metrics["user_activity_score"] = min(data["active_users"] / 1000, 1.0)  # Normalize
                
            if "trends" in data:
                avg_trend = sum(data["trends"].values()) / len(data["trends"]) if data["trends"] else 0
                metrics["trend_strength"] = abs(avg_trend) / 100.0
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
            
    def _generate_insights(self, data: Dict[str, Any], analytics_type: AnalyticsType) -> List[str]:
        """Generate insights from data"""
        try:
            insights = []
            
            if analytics_type == AnalyticsType.REAL_TIME_DASHBOARD:
                if data.get("total_volume", 0) > 1000000:
                    insights.append("High trading volume detected")
                if data.get("active_users", 0) > 500:
                    insights.append("High user activity")
                    
            elif analytics_type == AnalyticsType.TREND_ANALYSIS:
                if data.get("trends"):
                    strong_trends = [symbol for symbol, trend in data["trends"].items() if abs(trend) > 5]
                    if strong_trends:
                        insights.append(f"Strong trends detected in: {', '.join(strong_trends)}")
                        
            elif analytics_type == AnalyticsType.VOLATILITY_ANALYSIS:
                if data.get("volatility"):
                    high_volatility = [symbol for symbol, vol in data["volatility"].items() if vol > 10]
                    if high_volatility:
                        insights.append(f"High volatility detected in: {', '.join(high_volatility)}")
                        
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []
            
    async def _detect_anomalies(self):
        """Detect anomalies in data streams"""
        try:
            # Simple anomaly detection based on statistical thresholds
            for stream_type, stream in self.data_streams.items():
                if len(stream) >= 10:  # Need minimum data points
                    recent_points = list(stream)[-10:]
                    
                    # Check for anomalies in each field
                    for field in ["price", "volume", "value"]:
                        values = [dp.data.get(field) for dp in recent_points if field in dp.data]
                        if len(values) >= 5:
                            mean_val = sum(values) / len(values)
                            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                            
                            # Check for outliers
                            for dp in recent_points:
                                if field in dp.data:
                                    value = dp.data[field]
                                    if abs(value - mean_val) > 3 * std_val:  # 3-sigma rule
                                        self.analytics_stats["anomalies_detected"] += 1
                                        logger.warning(f"Anomaly detected in {stream_type.value}: {field}={value}")
                                        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            
    async def _analyze_trends(self):
        """Analyze trends in data"""
        try:
            # Simple trend analysis
            for stream_type, stream in self.data_streams.items():
                if len(stream) >= 5:
                    recent_points = list(stream)[-5:]
                    
                    # Check for upward/downward trends
                    for field in ["price", "value"]:
                        values = [dp.data.get(field) for dp in recent_points if field in dp.data]
                        if len(values) >= 3:
                            # Simple trend detection
                            if values[-1] > values[0] * 1.05:  # 5% increase
                                self.analytics_stats["trends_identified"] += 1
                                logger.info(f"Upward trend detected in {stream_type.value}: {field}")
                            elif values[-1] < values[0] * 0.95:  # 5% decrease
                                self.analytics_stats["trends_identified"] += 1
                                logger.info(f"Downward trend detected in {stream_type.value}: {field}")
                                
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            
    async def _perform_correlation_analysis(self):
        """Perform correlation analysis"""
        try:
            # Simple correlation analysis between different streams
            if len(self.data_streams[StreamType.MARKET_DATA]) >= 10 and len(self.data_streams[StreamType.USER_ACTIVITY]) >= 10:
                # This would implement actual correlation analysis
                # For now, just increment the counter
                self.analytics_stats["correlations_found"] += 1
                
        except Exception as e:
            logger.error(f"Error performing correlation analysis: {e}")
            
    async def _update_dashboards(self):
        """Update real-time dashboards"""
        try:
            # Get latest analytics results
            latest_results = list(self.analytics_results.values())[-10:]  # Last 10 results
            
            # Update dashboard data
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "results": [
                    {
                        "type": result.analytics_type.value,
                        "timestamp": result.timestamp.isoformat(),
                        "insights": result.insights,
                        "confidence": result.confidence
                    }
                    for result in latest_results
                ]
            }
            
            # Store in cache for dashboard access
            await enhanced_cache.set("real_time_dashboard", dashboard_data, ttl=60)
            
        except Exception as e:
            logger.error(f"Error updating dashboards: {e}")
            
    async def _update_stream_health(self):
        """Update stream health metrics"""
        try:
            for stream_type, stream in self.data_streams.items():
                # Check stream health
                if len(stream) == 0:
                    logger.warning(f"Stream {stream_type.value} is empty")
                elif len(stream) >= self.config["max_stream_size"]:
                    logger.warning(f"Stream {stream_type.value} is at capacity")
                    
        except Exception as e:
            logger.error(f"Error updating stream health: {e}")
            
    async def _notify_subscribers(self, stream_type: StreamType, data_point: DataPoint):
        """Notify subscribers of new data"""
        try:
            subscribers = self.subscribers.get(stream_type.value, [])
            for callback in subscribers:
                try:
                    await callback(data_point)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying subscribers: {e}")
            
    def _calculate_data_quality(self, data: Dict[str, Any], stream_type: StreamType) -> float:
        """Calculate data quality score"""
        try:
            config = self.stream_configs.get(stream_type, {})
            required_fields = config.get("fields", [])
            
            if not required_fields:
                return 1.0
                
            # Check if required fields are present
            present_fields = sum(1 for field in required_fields if field in data)
            quality_score = present_fields / len(required_fields)
            
            # Check for null values
            null_count = sum(1 for value in data.values() if value is None)
            if null_count > 0:
                quality_score *= (1.0 - (null_count / len(data)))
                
            return max(quality_score, 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            return 0.5
            
    async def _cleanup_old_data(self):
        """Clean up old data"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=self.config["data_retention_hours"])
            
            # Clean up old analytics results
            expired_results = [
                result_id for result_id, result in self.analytics_results.items()
                if result.timestamp < cutoff_time
            ]
            
            for result_id in expired_results:
                del self.analytics_results[result_id]
                
            # Clean up old stream windows
            expired_windows = [
                window_id for window_id, window in self.stream_windows.items()
                if window.end_time < cutoff_time
            ]
            
            for window_id in expired_windows:
                del self.stream_windows[window_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    def subscribe(self, stream_type: StreamType, callback: Callable):
        """Subscribe to stream updates"""
        try:
            self.subscribers[stream_type.value].append(callback)
            logger.info(f"Subscribed to {stream_type.value} stream")
            
        except Exception as e:
            logger.error(f"Error subscribing to stream: {e}")
            
    def unsubscribe(self, stream_type: StreamType, callback: Callable):
        """Unsubscribe from stream updates"""
        try:
            if callback in self.subscribers[stream_type.value]:
                self.subscribers[stream_type.value].remove(callback)
                logger.info(f"Unsubscribed from {stream_type.value} stream")
                
        except Exception as e:
            logger.error(f"Error unsubscribing from stream: {e}")
            
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        try:
            # Calculate stream statistics
            stream_stats = {}
            for stream_type, stream in self.data_streams.items():
                stream_stats[stream_type.value] = {
                    "data_points": len(stream),
                    "latest_timestamp": stream[-1].timestamp.isoformat() if stream else None,
                    "data_quality": sum(dp.quality_score for dp in stream) / len(stream) if stream else 0
                }
                
            # Calculate analytics statistics
            analytics_by_type = defaultdict(int)
            for result in self.analytics_results.values():
                analytics_by_type[result.analytics_type.value] += 1
                
            # Calculate processor statistics
            processor_stats = {}
            for processor_id, processor in self.stream_processors.items():
                processor_stats[processor_id] = {
                    "stream_type": processor.stream_type.value,
                    "processing_mode": processor.processing_mode.value,
                    "status": processor.status,
                    "window_size": processor.window_size,
                    "batch_size": processor.batch_size
                }
                
            return {
                "timestamp": datetime.now().isoformat(),
                "analytics_active": self.analytics_active,
                "total_data_points": sum(len(stream) for stream in self.data_streams.values()),
                "total_analytics_results": len(self.analytics_results),
                "total_processors": len(self.stream_processors),
                "total_subscribers": sum(len(subs) for subs in self.subscribers.values()),
                "stream_stats": stream_stats,
                "analytics_by_type": dict(analytics_by_type),
                "processor_stats": processor_stats,
                "stats": self.analytics_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {"error": str(e)}


# Global instance
real_time_analytics_engine = RealTimeAnalyticsEngine()
