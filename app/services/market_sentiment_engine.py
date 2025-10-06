"""
Market Sentiment Engine
Real-time market sentiment analysis from multiple sources
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
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class SentimentSource(Enum):
    """Sentiment data sources"""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    REDDIT = "reddit"
    TWITTER = "twitter"
    FORUMS = "forums"
    ANALYST_REPORTS = "analyst_reports"
    PRICE_ACTION = "price_action"
    VOLUME = "volume"
    OPTIONS_FLOW = "options_flow"
    INSIDER_TRADING = "insider_trading"

class SentimentType(Enum):
    """Sentiment types"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VERY_BULLISH = "very_bullish"
    VERY_BEARISH = "very_bearish"

class SentimentConfidence(Enum):
    """Sentiment confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class SentimentData:
    """Sentiment data point"""
    data_id: str
    source: SentimentSource
    asset: str
    sentiment_type: SentimentType
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SentimentAggregate:
    """Aggregated sentiment for an asset"""
    asset: str
    overall_sentiment: SentimentType
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source_breakdown: Dict[SentimentSource, float] = field(default_factory=dict)
    sentiment_distribution: Dict[SentimentType, int] = field(default_factory=dict)
    volume: int = 0
    trend: str = "stable"  # rising, falling, stable
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SentimentAlert:
    """Sentiment alert"""
    alert_id: str
    asset: str
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    sentiment_score: float
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class MarketSentimentEngine:
    """Advanced Market Sentiment Engine"""
    
    def __init__(self):
        self.engine_id = f"sentiment_engine_{secrets.token_hex(8)}"
        self.is_running = False
        
        # Sentiment data
        self.sentiment_data: Dict[str, List[SentimentData]] = defaultdict(list)
        self.sentiment_aggregates: Dict[str, SentimentAggregate] = {}
        self.sentiment_alerts: List[SentimentAlert] = []
        
        # Sentiment analysis models
        self.sentiment_keywords = {
            SentimentType.VERY_BULLISH: [
                "moon", "rocket", "explosive", "breakthrough", "revolutionary",
                "game changer", "massive", "huge", "incredible", "amazing"
            ],
            SentimentType.BULLISH: [
                "bullish", "positive", "up", "rise", "gain", "profit", "success",
                "good", "great", "excellent", "strong", "buy", "long"
            ],
            SentimentType.NEUTRAL: [
                "stable", "neutral", "hold", "wait", "monitor", "watch",
                "steady", "unchanged", "flat", "sideways"
            ],
            SentimentType.BEARISH: [
                "bearish", "negative", "down", "fall", "drop", "loss", "decline",
                "bad", "weak", "sell", "short", "crash", "dump"
            ],
            SentimentType.VERY_BEARISH: [
                "crash", "collapse", "disaster", "terrible", "awful", "horrible",
                "nightmare", "doomed", "dead", "worthless", "scam"
            ]
        }
        
        # Sentiment thresholds
        self.sentiment_thresholds = {
            SentimentType.VERY_BULLISH: 0.7,
            SentimentType.BULLISH: 0.3,
            SentimentType.NEUTRAL: 0.0,
            SentimentType.BEARISH: -0.3,
            SentimentType.VERY_BEARISH: -0.7
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "extreme_bullish": 0.8,
            "extreme_bearish": -0.8,
            "sentiment_shift": 0.5,  # Change in sentiment score
            "volume_spike": 3.0  # 3x normal volume
        }
        
        # Processing tasks
        self.sentiment_processing_task: Optional[asyncio.Task] = None
        self.aggregation_task: Optional[asyncio.Task] = None
        self.alert_monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.sentiment_history: Dict[str, List[float]] = defaultdict(list)
        self.accuracy_metrics: Dict[str, float] = {}
        
        logger.info(f"Market Sentiment Engine {self.engine_id} initialized")

    async def start_market_sentiment_engine(self):
        """Start the market sentiment engine"""
        if self.is_running:
            return
        
        logger.info("Starting Market Sentiment Engine...")
        
        # Initialize sentiment models
        await self._initialize_sentiment_models()
        
        # Start processing tasks
        self.is_running = True
        
        self.sentiment_processing_task = asyncio.create_task(self._sentiment_processing_loop())
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.alert_monitoring_task = asyncio.create_task(self._alert_monitoring_loop())
        
        logger.info("Market Sentiment Engine started")

    async def stop_market_sentiment_engine(self):
        """Stop the market sentiment engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping Market Sentiment Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.sentiment_processing_task,
            self.aggregation_task,
            self.alert_monitoring_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Market Sentiment Engine stopped")

    async def _initialize_sentiment_models(self):
        """Initialize sentiment analysis models"""
        # In a real implementation, this would load pre-trained models
        # For now, we'll use keyword-based sentiment analysis
        
        # Initialize sentiment aggregates for major assets
        major_assets = ["BTC", "ETH", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]
        
        for asset in major_assets:
            self.sentiment_aggregates[asset] = SentimentAggregate(
                asset=asset,
                overall_sentiment=SentimentType.NEUTRAL,
                sentiment_score=0.0,
                confidence=0.5,
                volume=0
            )
        
        logger.info(f"Initialized sentiment models for {len(major_assets)} assets")

    async def _sentiment_processing_loop(self):
        """Main sentiment processing loop"""
        while self.is_running:
            try:
                # Process sentiment data from various sources
                await self._process_news_sentiment()
                await self._process_social_media_sentiment()
                await self._process_price_action_sentiment()
                await self._process_volume_sentiment()
                
                await asyncio.sleep(30)  # Process every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in sentiment processing loop: {e}")
                await asyncio.sleep(60)

    async def _aggregation_loop(self):
        """Sentiment aggregation loop"""
        while self.is_running:
            try:
                # Aggregate sentiment for all assets
                for asset in self.sentiment_aggregates:
                    await self._aggregate_sentiment(asset)
                
                await asyncio.sleep(60)  # Aggregate every minute
                
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(60)

    async def _alert_monitoring_loop(self):
        """Alert monitoring loop"""
        while self.is_running:
            try:
                # Check for sentiment alerts
                await self._check_sentiment_alerts()
                
                await asyncio.sleep(120)  # Check alerts every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(120)

    async def _process_news_sentiment(self):
        """Process news sentiment"""
        try:
            # Mock news sentiment processing
            assets = ["BTC", "ETH", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            
            for asset in assets:
                # Generate mock news sentiment
                sentiment_score = np.random.uniform(-0.8, 0.8)
                confidence = np.random.uniform(0.6, 0.9)
                
                # Mock news headlines
                headlines = [
                    f"{asset} shows strong performance in latest trading session",
                    f"Analysts upgrade {asset} price target following earnings",
                    f"{asset} faces regulatory challenges in key markets",
                    f"Market volatility impacts {asset} trading volume",
                    f"{asset} demonstrates resilience amid market uncertainty"
                ]
                
                headline = np.random.choice(headlines)
                
                sentiment_data = SentimentData(
                    data_id=f"news_{secrets.token_hex(8)}",
                    source=SentimentSource.NEWS,
                    asset=asset,
                    sentiment_type=self._classify_sentiment(sentiment_score),
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    text=headline,
                    metadata={
                        "source_url": f"https://news.example.com/{asset.lower()}-news",
                        "publisher": "Financial News",
                        "category": "market_analysis"
                    }
                )
                
                self.sentiment_data[asset].append(sentiment_data)
                
                # Keep only last 1000 sentiment data points per asset
                if len(self.sentiment_data[asset]) > 1000:
                    self.sentiment_data[asset] = self.sentiment_data[asset][-1000:]
            
        except Exception as e:
            logger.error(f"Error processing news sentiment: {e}")

    async def _process_social_media_sentiment(self):
        """Process social media sentiment"""
        try:
            # Mock social media sentiment processing
            assets = ["BTC", "ETH", "AAPL", "TSLA"]
            
            for asset in assets:
                # Generate multiple social media posts
                for _ in range(np.random.randint(5, 15)):
                    sentiment_score = np.random.uniform(-1.0, 1.0)
                    confidence = np.random.uniform(0.4, 0.8)
                    
                    # Mock social media posts
                    posts = [
                        f"Just bought more {asset}! 🚀",
                        f"{asset} is going to the moon! 🌙",
                        f"Selling all my {asset} positions",
                        f"{asset} looking bearish today",
                        f"Holding {asset} for the long term",
                        f"{asset} price action is confusing",
                        f"Bullish on {asset} fundamentals",
                        f"{asset} technical analysis shows weakness"
                    ]
                    
                    post = np.random.choice(posts)
                    
                    sentiment_data = SentimentData(
                        data_id=f"social_{secrets.token_hex(8)}",
                        source=SentimentSource.SOCIAL_MEDIA,
                        asset=asset,
                        sentiment_type=self._classify_sentiment(sentiment_score),
                        sentiment_score=sentiment_score,
                        confidence=confidence,
                        text=post,
                        metadata={
                            "platform": "twitter",
                            "user_followers": np.random.randint(100, 10000),
                            "engagement": np.random.randint(10, 1000)
                        }
                    )
                    
                    self.sentiment_data[asset].append(sentiment_data)
            
        except Exception as e:
            logger.error(f"Error processing social media sentiment: {e}")

    async def _process_price_action_sentiment(self):
        """Process price action sentiment"""
        try:
            # Mock price action sentiment
            assets = ["BTC", "ETH", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            
            for asset in assets:
                # Generate price action sentiment based on mock price movements
                price_change = np.random.uniform(-0.1, 0.1)  # -10% to +10%
                
                if price_change > 0.05:
                    sentiment_score = 0.7
                    confidence = 0.8
                elif price_change > 0.02:
                    sentiment_score = 0.4
                    confidence = 0.7
                elif price_change < -0.05:
                    sentiment_score = -0.7
                    confidence = 0.8
                elif price_change < -0.02:
                    sentiment_score = -0.4
                    confidence = 0.7
                else:
                    sentiment_score = 0.0
                    confidence = 0.6
                
                sentiment_data = SentimentData(
                    data_id=f"price_{secrets.token_hex(8)}",
                    source=SentimentSource.PRICE_ACTION,
                    asset=asset,
                    sentiment_type=self._classify_sentiment(sentiment_score),
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    metadata={
                        "price_change": price_change,
                        "volume": np.random.uniform(1000000, 10000000),
                        "volatility": np.random.uniform(0.1, 0.5)
                    }
                )
                
                self.sentiment_data[asset].append(sentiment_data)
            
        except Exception as e:
            logger.error(f"Error processing price action sentiment: {e}")

    async def _process_volume_sentiment(self):
        """Process volume sentiment"""
        try:
            # Mock volume sentiment
            assets = ["BTC", "ETH", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            
            for asset in assets:
                # Generate volume-based sentiment
                volume_ratio = np.random.uniform(0.5, 3.0)  # 0.5x to 3x normal volume
                
                if volume_ratio > 2.0:
                    sentiment_score = 0.6  # High volume often indicates strong sentiment
                    confidence = 0.7
                elif volume_ratio < 0.7:
                    sentiment_score = -0.3  # Low volume indicates weak sentiment
                    confidence = 0.6
                else:
                    sentiment_score = 0.0
                    confidence = 0.5
                
                sentiment_data = SentimentData(
                    data_id=f"volume_{secrets.token_hex(8)}",
                    source=SentimentSource.VOLUME,
                    asset=asset,
                    sentiment_type=self._classify_sentiment(sentiment_score),
                    sentiment_score=sentiment_score,
                    confidence=confidence,
                    metadata={
                        "volume_ratio": volume_ratio,
                        "normal_volume": np.random.uniform(1000000, 5000000),
                        "current_volume": volume_ratio * np.random.uniform(1000000, 5000000)
                    }
                )
                
                self.sentiment_data[asset].append(sentiment_data)
            
        except Exception as e:
            logger.error(f"Error processing volume sentiment: {e}")

    def _classify_sentiment(self, sentiment_score: float) -> SentimentType:
        """Classify sentiment score into sentiment type"""
        if sentiment_score >= self.sentiment_thresholds[SentimentType.VERY_BULLISH]:
            return SentimentType.VERY_BULLISH
        elif sentiment_score >= self.sentiment_thresholds[SentimentType.BULLISH]:
            return SentimentType.BULLISH
        elif sentiment_score <= self.sentiment_thresholds[SentimentType.VERY_BEARISH]:
            return SentimentType.VERY_BEARISH
        elif sentiment_score <= self.sentiment_thresholds[SentimentType.BEARISH]:
            return SentimentType.BEARISH
        else:
            return SentimentType.NEUTRAL

    async def _aggregate_sentiment(self, asset: str):
        """Aggregate sentiment for a specific asset"""
        try:
            if asset not in self.sentiment_data or len(self.sentiment_data[asset]) == 0:
                return
            
            # Get recent sentiment data (last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_data = [
                data for data in self.sentiment_data[asset]
                if data.created_at >= cutoff_time
            ]
            
            if len(recent_data) == 0:
                return
            
            # Calculate weighted average sentiment score
            total_weight = 0
            weighted_sentiment = 0
            
            source_scores = defaultdict(list)
            sentiment_counts = Counter()
            
            for data in recent_data:
                weight = data.confidence
                weighted_sentiment += data.sentiment_score * weight
                total_weight += weight
                
                source_scores[data.source].append(data.sentiment_score)
                sentiment_counts[data.sentiment_type] += 1
            
            if total_weight > 0:
                overall_sentiment_score = weighted_sentiment / total_weight
                overall_confidence = min(total_weight / len(recent_data), 1.0)
            else:
                overall_sentiment_score = 0.0
                overall_confidence = 0.0
            
            # Calculate source breakdown
            source_breakdown = {}
            for source, scores in source_scores.items():
                if scores:
                    source_breakdown[source] = np.mean(scores)
            
            # Determine trend
            if asset in self.sentiment_history:
                if len(self.sentiment_history[asset]) > 0:
                    previous_score = self.sentiment_history[asset][-1]
                    if overall_sentiment_score > previous_score + 0.1:
                        trend = "rising"
                    elif overall_sentiment_score < previous_score - 0.1:
                        trend = "falling"
                    else:
                        trend = "stable"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            # Update sentiment history
            self.sentiment_history[asset].append(overall_sentiment_score)
            if len(self.sentiment_history[asset]) > 100:
                self.sentiment_history[asset] = self.sentiment_history[asset][-100:]
            
            # Update aggregate
            self.sentiment_aggregates[asset] = SentimentAggregate(
                asset=asset,
                overall_sentiment=self._classify_sentiment(overall_sentiment_score),
                sentiment_score=overall_sentiment_score,
                confidence=overall_confidence,
                source_breakdown=source_breakdown,
                sentiment_distribution=dict(sentiment_counts),
                volume=len(recent_data),
                trend=trend,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error aggregating sentiment for {asset}: {e}")

    async def _check_sentiment_alerts(self):
        """Check for sentiment alerts"""
        try:
            for asset, aggregate in self.sentiment_aggregates.items():
                # Check for extreme sentiment
                if aggregate.sentiment_score >= self.alert_thresholds["extreme_bullish"]:
                    await self._create_alert(
                        asset=asset,
                        alert_type="extreme_bullish",
                        severity="high",
                        message=f"Extreme bullish sentiment detected for {asset}",
                        sentiment_score=aggregate.sentiment_score,
                        threshold=self.alert_thresholds["extreme_bullish"]
                    )
                
                elif aggregate.sentiment_score <= self.alert_thresholds["extreme_bearish"]:
                    await self._create_alert(
                        asset=asset,
                        alert_type="extreme_bearish",
                        severity="high",
                        message=f"Extreme bearish sentiment detected for {asset}",
                        sentiment_score=aggregate.sentiment_score,
                        threshold=self.alert_thresholds["extreme_bearish"]
                    )
                
                # Check for sentiment shift
                if len(self.sentiment_history[asset]) >= 2:
                    recent_change = abs(
                        self.sentiment_history[asset][-1] - self.sentiment_history[asset][-2]
                    )
                    
                    if recent_change >= self.alert_thresholds["sentiment_shift"]:
                        await self._create_alert(
                            asset=asset,
                            alert_type="sentiment_shift",
                            severity="medium",
                            message=f"Significant sentiment shift detected for {asset}",
                            sentiment_score=aggregate.sentiment_score,
                            threshold=self.alert_thresholds["sentiment_shift"]
                        )
                
                # Check for volume spike
                if aggregate.volume > 0:
                    # Calculate average volume (simplified)
                    avg_volume = 50  # Mock average volume
                    if aggregate.volume >= avg_volume * self.alert_thresholds["volume_spike"]:
                        await self._create_alert(
                            asset=asset,
                            alert_type="volume_spike",
                            severity="medium",
                            message=f"Volume spike detected for {asset}",
                            sentiment_score=aggregate.sentiment_score,
                            threshold=self.alert_thresholds["volume_spike"]
                        )
            
        except Exception as e:
            logger.error(f"Error checking sentiment alerts: {e}")

    async def _create_alert(self, asset: str, alert_type: str, severity: str, 
                          message: str, sentiment_score: float, threshold: float):
        """Create a sentiment alert"""
        try:
            alert = SentimentAlert(
                alert_id=f"alert_{secrets.token_hex(8)}",
                asset=asset,
                alert_type=alert_type,
                severity=severity,
                message=message,
                sentiment_score=sentiment_score,
                threshold=threshold,
                metadata={
                    "engine_id": self.engine_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.sentiment_alerts.append(alert)
            
            # Keep only last 100 alerts
            if len(self.sentiment_alerts) > 100:
                self.sentiment_alerts = self.sentiment_alerts[-100:]
            
            logger.info(f"Sentiment alert created: {alert_type} for {asset} - {severity}")
            
        except Exception as e:
            logger.error(f"Error creating sentiment alert: {e}")

    # Public API methods
    async def get_sentiment(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get sentiment for a specific asset"""
        try:
            if asset not in self.sentiment_aggregates:
                return None
            
            aggregate = self.sentiment_aggregates[asset]
            
            return {
                "asset": aggregate.asset,
                "overall_sentiment": aggregate.overall_sentiment.value,
                "sentiment_score": aggregate.sentiment_score,
                "confidence": aggregate.confidence,
                "source_breakdown": {
                    source.value: score for source, score in aggregate.source_breakdown.items()
                },
                "sentiment_distribution": {
                    sentiment.value: count for sentiment, count in aggregate.sentiment_distribution.items()
                },
                "volume": aggregate.volume,
                "trend": aggregate.trend,
                "last_updated": aggregate.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {asset}: {e}")
            return None

    async def get_all_sentiments(self) -> List[Dict[str, Any]]:
        """Get sentiment for all assets"""
        try:
            sentiments = []
            for asset in self.sentiment_aggregates:
                sentiment_data = await self.get_sentiment(asset)
                if sentiment_data:
                    sentiments.append(sentiment_data)
            
            return sentiments
            
        except Exception as e:
            logger.error(f"Error getting all sentiments: {e}")
            return []

    async def get_sentiment_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent sentiment alerts"""
        try:
            recent_alerts = sorted(
                self.sentiment_alerts,
                key=lambda x: x.created_at,
                reverse=True
            )[:limit]
            
            return [
                {
                    "alert_id": alert.alert_id,
                    "asset": alert.asset,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "sentiment_score": alert.sentiment_score,
                    "threshold": alert.threshold,
                    "created_at": alert.created_at.isoformat()
                }
                for alert in recent_alerts
            ]
            
        except Exception as e:
            logger.error(f"Error getting sentiment alerts: {e}")
            return []

    async def get_sentiment_history(self, asset: str, hours: int = 24) -> List[float]:
        """Get sentiment history for an asset"""
        try:
            if asset not in self.sentiment_history:
                return []
            
            # Return last N hours of sentiment data
            cutoff_time = datetime.now() - timedelta(hours=hours)
            history = self.sentiment_history[asset]
            
            # In a real implementation, we would filter by timestamp
            # For now, return the last portion of history
            return history[-min(len(history), hours):]
            
        except Exception as e:
            logger.error(f"Error getting sentiment history for {asset}: {e}")
            return []

    async def add_sentiment_data(self, source: SentimentSource, asset: str, 
                               text: str, sentiment_score: Optional[float] = None,
                               confidence: float = 0.5) -> str:
        """Add sentiment data manually"""
        try:
            # Analyze text sentiment if not provided
            if sentiment_score is None:
                sentiment_score = await self._analyze_text_sentiment(text)
            
            sentiment_data = SentimentData(
                data_id=f"manual_{secrets.token_hex(8)}",
                source=source,
                asset=asset,
                sentiment_type=self._classify_sentiment(sentiment_score),
                sentiment_score=sentiment_score,
                confidence=confidence,
                text=text,
                metadata={"manual": True}
            )
            
            self.sentiment_data[asset].append(sentiment_data)
            
            return sentiment_data.data_id
            
        except Exception as e:
            logger.error(f"Error adding sentiment data: {e}")
            raise

    async def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze text sentiment using keyword matching"""
        try:
            text_lower = text.lower()
            sentiment_scores = []
            
            for sentiment_type, keywords in self.sentiment_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        if sentiment_type == SentimentType.VERY_BULLISH:
                            sentiment_scores.append(0.8)
                        elif sentiment_type == SentimentType.BULLISH:
                            sentiment_scores.append(0.4)
                        elif sentiment_type == SentimentType.NEUTRAL:
                            sentiment_scores.append(0.0)
                        elif sentiment_type == SentimentType.BEARISH:
                            sentiment_scores.append(-0.4)
                        elif sentiment_type == SentimentType.VERY_BEARISH:
                            sentiment_scores.append(-0.8)
            
            if sentiment_scores:
                return np.mean(sentiment_scores)
            else:
                return 0.0  # Neutral if no keywords found
                
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return 0.0

    async def get_sentiment_metrics(self) -> Dict[str, Any]:
        """Get sentiment engine metrics"""
        try:
            total_data_points = sum(len(data_list) for data_list in self.sentiment_data.values())
            total_alerts = len(self.sentiment_alerts)
            
            # Calculate accuracy metrics (simplified)
            accuracy = np.random.uniform(0.7, 0.9)  # Mock accuracy
            
            return {
                "engine_id": self.engine_id,
                "is_running": self.is_running,
                "total_assets": len(self.sentiment_aggregates),
                "total_data_points": total_data_points,
                "total_alerts": total_alerts,
                "accuracy": accuracy,
                "uptime": "active" if self.is_running else "stopped",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment metrics: {e}")
            return {}

# Global instance
market_sentiment_engine = MarketSentimentEngine()
