"""
Business Intelligence Engine
Comprehensive data analytics and business intelligence system
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import statistics
import json
import pandas as pd
import numpy as np
from enum import Enum

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class TimeGranularity(Enum):
    """Time granularity enumeration"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass
class BusinessMetric:
    """Business metric data structure"""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BusinessInsight:
    """Business insight data structure"""
    insight_id: str
    title: str
    description: str
    insight_type: str
    confidence: float
    impact: str
    recommendations: List[str]
    data_points: List[Dict[str, Any]]
    generated_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class BusinessReport:
    """Business report data structure"""
    report_id: str
    title: str
    description: str
    report_type: str
    data: Dict[str, Any]
    insights: List[BusinessInsight]
    generated_at: datetime
    period_start: datetime
    period_end: datetime


class BusinessIntelligenceEngine:
    """Business Intelligence Engine for comprehensive data analytics"""
    
    def __init__(self):
        self.metrics: List[BusinessMetric] = []
        self.insights: List[BusinessInsight] = []
        self.reports: List[BusinessReport] = []
        
        # Data storage
        self.metric_data: Dict[str, List[BusinessMetric]] = defaultdict(list)
        self.aggregated_data: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "retention_days": 90,
            "aggregation_interval": 3600,  # 1 hour
            "insight_generation_interval": 1800,  # 30 minutes
            "report_generation_interval": 86400,  # 24 hours
            "max_metrics_per_type": 10000,
            "confidence_threshold": 0.7,
            "anomaly_threshold": 2.0  # Standard deviations
        }
        
        # Business metrics configuration
        self.business_metrics = {
            "user_engagement": {
                "name": "User Engagement",
                "description": "Measures user activity and engagement",
                "metrics": ["daily_active_users", "session_duration", "page_views", "bounce_rate"]
            },
            "revenue": {
                "name": "Revenue Metrics",
                "description": "Financial performance indicators",
                "metrics": ["total_revenue", "revenue_per_user", "conversion_rate", "average_order_value"]
            },
            "operational": {
                "name": "Operational Metrics",
                "description": "System and operational performance",
                "metrics": ["response_time", "error_rate", "throughput", "availability"]
            },
            "market": {
                "name": "Market Metrics",
                "description": "Market and competitive analysis",
                "metrics": ["market_share", "customer_satisfaction", "churn_rate", "growth_rate"]
            }
        }
        
        # Monitoring
        self.bi_active = False
        self.bi_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.bi_stats = {
            "metrics_collected": 0,
            "insights_generated": 0,
            "reports_generated": 0,
            "anomalies_detected": 0,
            "trends_identified": 0
        }
        
    async def start_bi_engine(self):
        """Start the business intelligence engine"""
        if self.bi_active:
            logger.warning("BI Engine already active")
            return
            
        self.bi_active = True
        self.bi_task = asyncio.create_task(self._bi_processing_loop())
        logger.info("Business Intelligence Engine started")
        
    async def stop_bi_engine(self):
        """Stop the business intelligence engine"""
        self.bi_active = False
        if self.bi_task:
            self.bi_task.cancel()
            try:
                await self.bi_task
            except asyncio.CancelledError:
                pass
        logger.info("Business Intelligence Engine stopped")
        
    async def _bi_processing_loop(self):
        """Main BI processing loop"""
        while self.bi_active:
            try:
                # Aggregate metrics
                await self._aggregate_metrics()
                
                # Generate insights
                await self._generate_insights()
                
                # Detect anomalies
                await self._detect_anomalies()
                
                # Generate reports
                await self._generate_reports()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next cycle
                await asyncio.sleep(self.config["insight_generation_interval"])
                
            except Exception as e:
                logger.error(f"Error in BI processing loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def collect_metric(self, metric: BusinessMetric):
        """Collect a business metric"""
        try:
            # Add metric to storage
            self.metrics.append(metric)
            self.metric_data[metric.metric_id].append(metric)
            
            # Update statistics
            self.bi_stats["metrics_collected"] += 1
            
            # Store in cache for real-time access
            await enhanced_cache.set(
                f"metric_{metric.metric_id}_{int(metric.timestamp.timestamp())}",
                metric,
                ttl=86400  # 24 hours
            )
            
            logger.debug(f"Metric collected: {metric.name} = {metric.value}")
            
        except Exception as e:
            logger.error(f"Error collecting metric: {e}")
            
    async def collect_metrics_batch(self, metrics: List[BusinessMetric]):
        """Collect multiple metrics in batch"""
        try:
            for metric in metrics:
                await self.collect_metric(metric)
                
        except Exception as e:
            logger.error(f"Error collecting metrics batch: {e}")
            
    async def _aggregate_metrics(self):
        """Aggregate metrics by time periods"""
        try:
            current_time = datetime.now()
            
            # Aggregate by different time granularities
            for granularity in [TimeGranularity.HOUR, TimeGranularity.DAY]:
                await self._aggregate_by_granularity(granularity, current_time)
                
        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")
            
    async def _aggregate_by_granularity(self, granularity: TimeGranularity, current_time: datetime):
        """Aggregate metrics by specific time granularity"""
        try:
            # Calculate time window
            if granularity == TimeGranularity.HOUR:
                window_start = current_time.replace(minute=0, second=0, microsecond=0)
                window_end = window_start + timedelta(hours=1)
            elif granularity == TimeGranularity.DAY:
                window_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                window_end = window_start + timedelta(days=1)
            else:
                return
                
            # Group metrics by type
            metrics_by_type = defaultdict(list)
            for metric in self.metrics:
                if window_start <= metric.timestamp < window_end:
                    metrics_by_type[metric.metric_id].append(metric)
                    
            # Calculate aggregations
            for metric_id, metrics in metrics_by_type.items():
                if not metrics:
                    continue
                    
                values = [m.value for m in metrics]
                
                aggregation = {
                    "metric_id": metric_id,
                    "granularity": granularity.value,
                    "period_start": window_start,
                    "period_end": window_end,
                    "count": len(values),
                    "sum": sum(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                }
                
                # Store aggregation
                key = f"agg_{metric_id}_{granularity.value}_{int(window_start.timestamp())}"
                await enhanced_cache.set(key, aggregation, ttl=86400 * 7)  # 7 days
                
        except Exception as e:
            logger.error(f"Error aggregating by granularity: {e}")
            
    async def _generate_insights(self):
        """Generate business insights from metrics"""
        try:
            # Analyze trends
            trends = await self._analyze_trends()
            
            # Analyze patterns
            patterns = await self._analyze_patterns()
            
            # Analyze correlations
            correlations = await self._analyze_correlations()
            
            # Generate insights from analysis
            for trend in trends:
                await self._create_insight_from_trend(trend)
                
            for pattern in patterns:
                await self._create_insight_from_pattern(pattern)
                
            for correlation in correlations:
                await self._create_insight_from_correlation(correlation)
                
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            
    async def _analyze_trends(self) -> List[Dict[str, Any]]:
        """Analyze trends in metrics"""
        try:
            trends = []
            current_time = datetime.now()
            
            # Analyze trends for each metric type
            for metric_id in self.metric_data.keys():
                metrics = self.metric_data[metric_id]
                if len(metrics) < 10:  # Need enough data points
                    continue
                    
                # Get recent metrics (last 7 days)
                recent_metrics = [
                    m for m in metrics
                    if (current_time - m.timestamp).total_seconds() < 604800
                ]
                
                if len(recent_metrics) < 10:
                    continue
                    
                # Calculate trend
                values = [m.value for m in recent_metrics]
                timestamps = [m.timestamp.timestamp() for m in recent_metrics]
                
                # Simple linear regression for trend
                if len(values) > 1:
                    slope = self._calculate_slope(timestamps, values)
                    trend_strength = abs(slope)
                    
                    if trend_strength > 0.1:  # Significant trend
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        
                        trends.append({
                            "metric_id": metric_id,
                            "direction": trend_direction,
                            "strength": trend_strength,
                            "data_points": len(recent_metrics),
                            "period": "7_days"
                        })
                        
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return []
            
    async def _analyze_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in metrics"""
        try:
            patterns = []
            current_time = datetime.now()
            
            # Analyze daily patterns
            for metric_id in self.metric_data.keys():
                metrics = self.metric_data[metric_id]
                if len(metrics) < 24:  # Need at least 24 hours of data
                    continue
                    
                # Group by hour of day
                hourly_values = defaultdict(list)
                for metric in metrics:
                    if (current_time - metric.timestamp).total_seconds() < 86400 * 7:  # Last 7 days
                        hour = metric.timestamp.hour
                        hourly_values[hour].append(metric.value)
                        
                # Calculate hourly averages
                hourly_averages = {}
                for hour, values in hourly_values.items():
                    if values:
                        hourly_averages[hour] = statistics.mean(values)
                        
                # Detect peak hours
                if hourly_averages:
                    max_hour = max(hourly_averages.keys(), key=lambda h: hourly_averages[h])
                    min_hour = min(hourly_averages.keys(), key=lambda h: hourly_averages[h])
                    
                    if hourly_averages[max_hour] > hourly_averages[min_hour] * 1.5:  # Significant difference
                        patterns.append({
                            "metric_id": metric_id,
                            "pattern_type": "daily_cycle",
                            "peak_hour": max_hour,
                            "low_hour": min_hour,
                            "peak_value": hourly_averages[max_hour],
                            "low_value": hourly_averages[min_hour]
                        })
                        
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return []
            
    async def _analyze_correlations(self) -> List[Dict[str, Any]]:
        """Analyze correlations between metrics"""
        try:
            correlations = []
            current_time = datetime.now()
            
            # Get recent metrics for correlation analysis
            recent_metrics = [
                m for m in self.metrics
                if (current_time - m.timestamp).total_seconds() < 86400 * 7  # Last 7 days
            ]
            
            if len(recent_metrics) < 20:
                return correlations
                
            # Group metrics by timestamp (within 1 hour)
            time_groups = defaultdict(list)
            for metric in recent_metrics:
                time_key = int(metric.timestamp.timestamp() // 3600) * 3600
                time_groups[time_key].append(metric)
                
            # Calculate correlations between metric types
            metric_types = list(self.metric_data.keys())
            for i, metric_type1 in enumerate(metric_types):
                for metric_type2 in metric_types[i+1:]:
                    correlation = await self._calculate_correlation(
                        metric_type1, metric_type2, time_groups
                    )
                    
                    if correlation and abs(correlation["correlation"]) > 0.5:  # Strong correlation
                        correlations.append(correlation)
                        
            return correlations
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return []
            
    async def _calculate_correlation(self, metric_type1: str, metric_type2: str, time_groups: Dict[int, List[BusinessMetric]]) -> Optional[Dict[str, Any]]:
        """Calculate correlation between two metric types"""
        try:
            values1 = []
            values2 = []
            
            for time_key, metrics in time_groups.items():
                metric1_values = [m.value for m in metrics if m.metric_id == metric_type1]
                metric2_values = [m.value for m in metrics if m.metric_id == metric_type2]
                
                if metric1_values and metric2_values:
                    values1.append(statistics.mean(metric1_values))
                    values2.append(statistics.mean(metric2_values))
                    
            if len(values1) < 5:  # Need enough data points
                return None
                
            # Calculate Pearson correlation
            correlation = np.corrcoef(values1, values2)[0, 1]
            
            if not np.isnan(correlation):
                return {
                    "metric_type1": metric_type1,
                    "metric_type2": metric_type2,
                    "correlation": correlation,
                    "data_points": len(values1)
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return None
            
    def _calculate_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate slope of linear regression"""
        try:
            n = len(x_values)
            if n < 2:
                return 0
                
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(y_values)
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator == 0:
                return 0
                
            return numerator / denominator
            
        except Exception as e:
            logger.error(f"Error calculating slope: {e}")
            return 0
            
    async def _create_insight_from_trend(self, trend: Dict[str, Any]):
        """Create insight from trend analysis"""
        try:
            insight = BusinessInsight(
                insight_id=f"trend_{int(time.time())}_{trend['metric_id']}",
                title=f"Trend Detected: {trend['metric_id']}",
                description=f"Metric {trend['metric_id']} shows {trend['direction']} trend with strength {trend['strength']:.2f}",
                insight_type="trend",
                confidence=min(trend['strength'] * 2, 1.0),
                impact="medium",
                recommendations=[
                    f"Monitor {trend['metric_id']} closely",
                    f"Investigate factors causing {trend['direction']} trend",
                    "Consider adjusting strategy based on trend"
                ],
                data_points=[trend],
                generated_at=datetime.now()
            )
            
            self.insights.append(insight)
            self.bi_stats["insights_generated"] += 1
            self.bi_stats["trends_identified"] += 1
            
        except Exception as e:
            logger.error(f"Error creating insight from trend: {e}")
            
    async def _create_insight_from_pattern(self, pattern: Dict[str, Any]):
        """Create insight from pattern analysis"""
        try:
            insight = BusinessInsight(
                insight_id=f"pattern_{int(time.time())}_{pattern['metric_id']}",
                title=f"Pattern Detected: {pattern['metric_id']}",
                description=f"Metric {pattern['metric_id']} shows daily pattern with peak at hour {pattern['peak_hour']}",
                insight_type="pattern",
                confidence=0.8,
                impact="medium",
                recommendations=[
                    f"Optimize operations for hour {pattern['peak_hour']}",
                    "Consider load balancing based on pattern",
                    "Plan resource allocation accordingly"
                ],
                data_points=[pattern],
                generated_at=datetime.now()
            )
            
            self.insights.append(insight)
            self.bi_stats["insights_generated"] += 1
            
        except Exception as e:
            logger.error(f"Error creating insight from pattern: {e}")
            
    async def _create_insight_from_correlation(self, correlation: Dict[str, Any]):
        """Create insight from correlation analysis"""
        try:
            correlation_strength = "strong" if abs(correlation["correlation"]) > 0.7 else "moderate"
            correlation_direction = "positive" if correlation["correlation"] > 0 else "negative"
            
            insight = BusinessInsight(
                insight_id=f"correlation_{int(time.time())}_{correlation['metric_type1']}_{correlation['metric_type2']}",
                title=f"Correlation Found: {correlation['metric_type1']} & {correlation['metric_type2']}",
                description=f"{correlation_strength} {correlation_direction} correlation between {correlation['metric_type1']} and {correlation['metric_type2']}",
                insight_type="correlation",
                confidence=abs(correlation["correlation"]),
                impact="high",
                recommendations=[
                    f"Investigate relationship between {correlation['metric_type1']} and {correlation['metric_type2']}",
                    "Consider optimizing both metrics together",
                    "Use correlation for predictive modeling"
                ],
                data_points=[correlation],
                generated_at=datetime.now()
            )
            
            self.insights.append(insight)
            self.bi_stats["insights_generated"] += 1
            
        except Exception as e:
            logger.error(f"Error creating insight from correlation: {e}")
            
    async def _detect_anomalies(self):
        """Detect anomalies in metrics"""
        try:
            current_time = datetime.now()
            
            for metric_id in self.metric_data.keys():
                metrics = self.metric_data[metric_id]
                if len(metrics) < 20:  # Need enough data for anomaly detection
                    continue
                    
                # Get recent metrics
                recent_metrics = [
                    m for m in metrics
                    if (current_time - m.timestamp).total_seconds() < 86400  # Last 24 hours
                ]
                
                if len(recent_metrics) < 10:
                    continue
                    
                # Calculate baseline (mean and std dev of historical data)
                historical_metrics = [
                    m for m in metrics
                    if (current_time - m.timestamp).total_seconds() > 86400  # Older than 24 hours
                ]
                
                if len(historical_metrics) < 10:
                    continue
                    
                historical_values = [m.value for m in historical_metrics]
                baseline_mean = statistics.mean(historical_values)
                baseline_std = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                
                # Detect anomalies in recent metrics
                for metric in recent_metrics:
                    z_score = abs(metric.value - baseline_mean) / baseline_std if baseline_std > 0 else 0
                    
                    if z_score > self.config["anomaly_threshold"]:
                        await self._create_anomaly_insight(metric, z_score, baseline_mean, baseline_std)
                        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            
    async def _create_anomaly_insight(self, metric: BusinessMetric, z_score: float, baseline_mean: float, baseline_std: float):
        """Create insight for detected anomaly"""
        try:
            anomaly_type = "high" if metric.value > baseline_mean else "low"
            
            insight = BusinessInsight(
                insight_id=f"anomaly_{int(time.time())}_{metric.metric_id}",
                title=f"Anomaly Detected: {metric.metric_id}",
                description=f"Metric {metric.metric_id} shows {anomaly_type} anomaly with value {metric.value:.2f} (Z-score: {z_score:.2f})",
                insight_type="anomaly",
                confidence=min(z_score / 3.0, 1.0),  # Cap confidence at 1.0
                impact="high",
                recommendations=[
                    f"Investigate cause of {anomaly_type} anomaly",
                    "Check for system issues or external factors",
                    "Monitor metric closely for resolution"
                ],
                data_points=[{
                    "metric": metric,
                    "z_score": z_score,
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std
                }],
                generated_at=datetime.now()
            )
            
            self.insights.append(insight)
            self.bi_stats["insights_generated"] += 1
            self.bi_stats["anomalies_detected"] += 1
            
        except Exception as e:
            logger.error(f"Error creating anomaly insight: {e}")
            
    async def _generate_reports(self):
        """Generate business reports"""
        try:
            current_time = datetime.now()
            
            # Generate daily report
            if current_time.hour == 0:  # At midnight
                await self._generate_daily_report(current_time)
                
            # Generate weekly report
            if current_time.weekday() == 0 and current_time.hour == 0:  # Monday at midnight
                await self._generate_weekly_report(current_time)
                
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            
    async def _generate_daily_report(self, current_time: datetime):
        """Generate daily business report"""
        try:
            # Calculate report period
            report_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            report_end = report_start + timedelta(days=1)
            
            # Get metrics for the day
            daily_metrics = [
                m for m in self.metrics
                if report_start <= m.timestamp < report_end
            ]
            
            # Get insights for the day
            daily_insights = [
                i for i in self.insights
                if report_start <= i.generated_at < report_end
            ]
            
            # Generate report data
            report_data = {
                "summary": {
                    "total_metrics": len(daily_metrics),
                    "total_insights": len(daily_insights),
                    "anomalies_detected": len([i for i in daily_insights if i.insight_type == "anomaly"]),
                    "trends_identified": len([i for i in daily_insights if i.insight_type == "trend"])
                },
                "metrics_by_type": self._group_metrics_by_type(daily_metrics),
                "insights_by_type": self._group_insights_by_type(daily_insights),
                "top_insights": sorted(daily_insights, key=lambda x: x.confidence, reverse=True)[:5]
            }
            
            # Create report
            report = BusinessReport(
                report_id=f"daily_{int(report_start.timestamp())}",
                title=f"Daily Business Report - {report_start.strftime('%Y-%m-%d')}",
                description="Daily summary of business metrics and insights",
                report_type="daily",
                data=report_data,
                insights=daily_insights,
                generated_at=current_time,
                period_start=report_start,
                period_end=report_end
            )
            
            self.reports.append(report)
            self.bi_stats["reports_generated"] += 1
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            
    async def _generate_weekly_report(self, current_time: datetime):
        """Generate weekly business report"""
        try:
            # Calculate report period
            report_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            report_end = report_start + timedelta(weeks=1)
            
            # Get metrics for the week
            weekly_metrics = [
                m for m in self.metrics
                if report_start <= m.timestamp < report_end
            ]
            
            # Get insights for the week
            weekly_insights = [
                i for i in self.insights
                if report_start <= i.generated_at < report_end
            ]
            
            # Generate report data
            report_data = {
                "summary": {
                    "total_metrics": len(weekly_metrics),
                    "total_insights": len(weekly_insights),
                    "anomalies_detected": len([i for i in weekly_insights if i.insight_type == "anomaly"]),
                    "trends_identified": len([i for i in weekly_insights if i.insight_type == "trend"])
                },
                "metrics_by_type": self._group_metrics_by_type(weekly_metrics),
                "insights_by_type": self._group_insights_by_type(weekly_insights),
                "top_insights": sorted(weekly_insights, key=lambda x: x.confidence, reverse=True)[:10],
                "weekly_trends": await self._analyze_weekly_trends(weekly_metrics)
            }
            
            # Create report
            report = BusinessReport(
                report_id=f"weekly_{int(report_start.timestamp())}",
                title=f"Weekly Business Report - Week of {report_start.strftime('%Y-%m-%d')}",
                description="Weekly summary of business metrics and insights",
                report_type="weekly",
                data=report_data,
                insights=weekly_insights,
                generated_at=current_time,
                period_start=report_start,
                period_end=report_end
            )
            
            self.reports.append(report)
            self.bi_stats["reports_generated"] += 1
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            
    def _group_metrics_by_type(self, metrics: List[BusinessMetric]) -> Dict[str, int]:
        """Group metrics by type"""
        return dict(Counter(m.metric_id for m in metrics))
        
    def _group_insights_by_type(self, insights: List[BusinessInsight]) -> Dict[str, int]:
        """Group insights by type"""
        return dict(Counter(i.insight_type for i in insights))
        
    async def _analyze_weekly_trends(self, metrics: List[BusinessMetric]) -> List[Dict[str, Any]]:
        """Analyze weekly trends"""
        try:
            trends = []
            metrics_by_type = defaultdict(list)
            
            # Group metrics by type
            for metric in metrics:
                metrics_by_type[metric.metric_id].append(metric)
                
            # Analyze trends for each type
            for metric_id, type_metrics in metrics_by_type.items():
                if len(type_metrics) < 7:  # Need at least 7 data points
                    continue
                    
                # Sort by timestamp
                type_metrics.sort(key=lambda x: x.timestamp)
                
                # Calculate trend
                values = [m.value for m in type_metrics]
                timestamps = [m.timestamp.timestamp() for m in type_metrics]
                
                slope = self._calculate_slope(timestamps, values)
                trend_strength = abs(slope)
                
                if trend_strength > 0.05:  # Significant trend
                    trend_direction = "increasing" if slope > 0 else "decreasing"
                    
                    trends.append({
                        "metric_id": metric_id,
                        "direction": trend_direction,
                        "strength": trend_strength,
                        "data_points": len(type_metrics)
                    })
                    
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing weekly trends: {e}")
            return []
            
    async def _cleanup_old_data(self):
        """Clean up old data"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=self.config["retention_days"])
            
            # Clean up old metrics
            self.metrics = [
                m for m in self.metrics
                if m.timestamp > cutoff_time
            ]
            
            # Clean up old insights
            self.insights = [
                i for i in self.insights
                if i.generated_at > cutoff_time
            ]
            
            # Clean up old reports
            self.reports = [
                r for r in self.reports
                if r.generated_at > cutoff_time
            ]
            
            # Clean up metric data
            for metric_id in list(self.metric_data.keys()):
                self.metric_data[metric_id] = [
                    m for m in self.metric_data[metric_id]
                    if m.timestamp > cutoff_time
                ]
                
                # Remove empty metric data
                if not self.metric_data[metric_id]:
                    del self.metric_data[metric_id]
                    
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    def get_bi_summary(self) -> Dict[str, Any]:
        """Get comprehensive BI summary"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "bi_active": self.bi_active,
                "total_metrics": len(self.metrics),
                "total_insights": len(self.insights),
                "total_reports": len(self.reports),
                "metrics_by_type": len(self.metric_data),
                "stats": self.bi_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting BI summary: {e}")
            return {"error": str(e)}


# Global instance
business_intelligence_engine = BusinessIntelligenceEngine()
