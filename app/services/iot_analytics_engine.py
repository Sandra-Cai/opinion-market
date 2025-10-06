"""
IoT Analytics Engine
Advanced IoT data analytics, pattern recognition, and predictive insights
"""

import asyncio
import logging
import time
import json
import hashlib
import secrets
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AnalyticsType(Enum):
    """Analytics types"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    PREDICTIVE = "predictive"
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PRESCRIPTIVE = "prescriptive"

class InsightType(Enum):
    """Insight types"""
    ANOMALY = "anomaly"
    TREND = "trend"
    PATTERN = "pattern"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"
    ALERT = "alert"
    STREAMING = "streaming"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnalyticsJob:
    """Analytics job"""
    job_id: str
    job_type: AnalyticsType
    device_ids: List[str]
    sensor_types: List[str]
    parameters: Dict[str, Any]
    status: str  # "pending", "running", "completed", "failed"
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataInsight:
    """Data insight"""
    insight_id: str
    device_id: str
    sensor_type: str
    insight_type: InsightType
    title: str
    description: str
    confidence: float
    severity: AlertSeverity
    timestamp: datetime
    data_points: List[float]
    trend: str
    anomaly_score: Optional[float] = None
    correlation_coefficient: Optional[float] = None
    prediction_accuracy: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictiveModel:
    """Predictive model"""
    model_id: str
    model_type: str
    device_id: str
    sensor_type: str
    algorithm: str
    accuracy: float
    training_data_size: int
    last_trained: datetime
    is_active: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataPattern:
    """Data pattern"""
    pattern_id: str
    device_id: str
    sensor_type: str
    pattern_type: str
    frequency: int
    duration: float
    confidence: float
    first_detected: datetime
    last_detected: datetime
    pattern_data: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

class IoTAnalyticsEngine:
    """Advanced IoT Analytics Engine"""
    
    def __init__(self):
        self.engine_id = f"iot_analytics_{secrets.token_hex(8)}"
        self.is_running = False
        
        # Analytics data
        self.analytics_jobs: List[AnalyticsJob] = []
        self.data_insights: List[DataInsight] = []
        self.predictive_models: Dict[str, PredictiveModel] = {}
        self.data_patterns: List[DataPattern] = []
        
        # Analytics configurations
        self.analytics_configs = {
            "real_time_processing_interval": 10,  # seconds
            "batch_processing_interval": 3600,    # 1 hour
            "streaming_window_size": 1000,        # data points
            "anomaly_detection_threshold": 0.8,
            "pattern_detection_min_frequency": 3,
            "prediction_horizon": 24,             # hours
            "model_retraining_interval": 86400,   # 24 hours
            "insight_retention_days": 30
        }
        
        # Algorithm configurations
        self.algorithm_configs = {
            "anomaly_detection": {
                "isolation_forest": {"contamination": 0.1},
                "one_class_svm": {"nu": 0.1},
                "local_outlier_factor": {"n_neighbors": 20}
            },
            "trend_analysis": {
                "linear_regression": {"fit_intercept": True},
                "polynomial_regression": {"degree": 2},
                "exponential_smoothing": {"alpha": 0.3}
            },
            "pattern_recognition": {
                "fourier_transform": {"n_fft": 1024},
                "wavelet_transform": {"wavelet": "db4"},
                "autocorrelation": {"max_lags": 100}
            },
            "prediction": {
                "arima": {"order": (1, 1, 1)},
                "lstm": {"units": 50, "epochs": 100},
                "random_forest": {"n_estimators": 100}
            }
        }
        
        # Processing tasks
        self.real_time_analytics_task: Optional[asyncio.Task] = None
        self.batch_analytics_task: Optional[asyncio.Task] = None
        self.streaming_analytics_task: Optional[asyncio.Task] = None
        self.model_training_task: Optional[asyncio.Task] = None
        self.insight_generation_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.analytics_performance: Dict[str, List[float]] = {}
        self.model_performance: Dict[str, float] = {}
        self.insight_accuracy: Dict[str, float] = {}
        
        logger.info(f"IoT Analytics Engine {self.engine_id} initialized")

    async def start_analytics_engine(self):
        """Start the analytics engine"""
        if self.is_running:
            return
        
        logger.info("Starting IoT Analytics Engine...")
        
        # Initialize analytics data
        await self._initialize_predictive_models()
        await self._initialize_data_patterns()
        
        # Start processing tasks
        self.is_running = True
        
        self.real_time_analytics_task = asyncio.create_task(self._real_time_analytics_loop())
        self.batch_analytics_task = asyncio.create_task(self._batch_analytics_loop())
        self.streaming_analytics_task = asyncio.create_task(self._streaming_analytics_loop())
        self.model_training_task = asyncio.create_task(self._model_training_loop())
        self.insight_generation_task = asyncio.create_task(self._insight_generation_loop())
        
        logger.info("IoT Analytics Engine started")

    async def stop_analytics_engine(self):
        """Stop the analytics engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping IoT Analytics Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.real_time_analytics_task,
            self.batch_analytics_task,
            self.streaming_analytics_task,
            self.model_training_task,
            self.insight_generation_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("IoT Analytics Engine stopped")

    async def _initialize_predictive_models(self):
        """Initialize predictive models"""
        try:
            # Create mock predictive models
            model_types = ["temperature_prediction", "humidity_prediction", "pressure_prediction", 
                          "motion_prediction", "sound_prediction", "air_quality_prediction"]
            algorithms = ["arima", "lstm", "random_forest", "linear_regression", "polynomial_regression"]
            
            for i in range(20):  # Generate 20 mock models
                model = PredictiveModel(
                    model_id=f"model_{secrets.token_hex(8)}",
                    model_type=secrets.choice(model_types),
                    device_id=f"device_{secrets.token_hex(8)}",
                    sensor_type=secrets.choice(["temperature", "humidity", "pressure", "motion", "sound"]),
                    algorithm=secrets.choice(algorithms),
                    accuracy=random.uniform(0.7, 0.95),
                    training_data_size=secrets.randbelow(10000) + 1000,
                    last_trained=datetime.now() - timedelta(days=secrets.randbelow(30))
                )
                
                self.predictive_models[model.model_id] = model
            
            logger.info(f"Initialized {len(self.predictive_models)} predictive models")
            
        except Exception as e:
            logger.error(f"Error initializing predictive models: {e}")

    async def _initialize_data_patterns(self):
        """Initialize data patterns"""
        try:
            # Create mock data patterns
            pattern_types = ["daily_cycle", "weekly_cycle", "seasonal", "anomaly", "trend", "correlation"]
            
            for i in range(50):  # Generate 50 mock patterns
                pattern = DataPattern(
                    pattern_id=f"pattern_{secrets.token_hex(8)}",
                    device_id=f"device_{secrets.token_hex(8)}",
                    sensor_type=secrets.choice(["temperature", "humidity", "pressure", "motion", "sound"]),
                    pattern_type=secrets.choice(pattern_types),
                    frequency=secrets.randbelow(100) + 1,
                    duration=random.uniform(1, 24),  # hours
                    confidence=random.uniform(0.6, 0.95),
                    first_detected=datetime.now() - timedelta(days=secrets.randbelow(30)),
                    last_detected=datetime.now() - timedelta(hours=secrets.randbelow(24)),
                    pattern_data=[random.uniform(0, 100) for _ in range(100)]
                )
                
                self.data_patterns.append(pattern)
            
            logger.info(f"Initialized {len(self.data_patterns)} data patterns")
            
        except Exception as e:
            logger.error(f"Error initializing data patterns: {e}")

    async def _real_time_analytics_loop(self):
        """Real-time analytics loop"""
        while self.is_running:
            try:
                # Perform real-time analytics
                await self._perform_real_time_analytics()
                
                await asyncio.sleep(self.analytics_configs["real_time_processing_interval"])
                
            except Exception as e:
                logger.error(f"Error in real-time analytics loop: {e}")
                await asyncio.sleep(self.analytics_configs["real_time_processing_interval"])

    async def _batch_analytics_loop(self):
        """Batch analytics loop"""
        while self.is_running:
            try:
                # Perform batch analytics
                await self._perform_batch_analytics()
                
                await asyncio.sleep(self.analytics_configs["batch_processing_interval"])
                
            except Exception as e:
                logger.error(f"Error in batch analytics loop: {e}")
                await asyncio.sleep(self.analytics_configs["batch_processing_interval"])

    async def _streaming_analytics_loop(self):
        """Streaming analytics loop"""
        while self.is_running:
            try:
                # Perform streaming analytics
                await self._perform_streaming_analytics()
                
                await asyncio.sleep(5)  # Process every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in streaming analytics loop: {e}")
                await asyncio.sleep(5)

    async def _model_training_loop(self):
        """Model training loop"""
        while self.is_running:
            try:
                # Retrain models
                await self._retrain_models()
                
                await asyncio.sleep(self.analytics_configs["model_retraining_interval"])
                
            except Exception as e:
                logger.error(f"Error in model training loop: {e}")
                await asyncio.sleep(self.analytics_configs["model_retraining_interval"])

    async def _insight_generation_loop(self):
        """Insight generation loop"""
        while self.is_running:
            try:
                # Generate insights
                await self._generate_insights()
                
                await asyncio.sleep(300)  # Generate every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in insight generation loop: {e}")
                await asyncio.sleep(300)

    async def _perform_real_time_analytics(self):
        """Perform real-time analytics"""
        try:
            # Simulate real-time analytics processing
            # In a real implementation, this would process live sensor data
            
            # Generate mock real-time insights
            num_insights = secrets.randbelow(5) + 1
            
            for _ in range(num_insights):
                insight = DataInsight(
                    insight_id=f"insight_{secrets.token_hex(8)}",
                    device_id=f"device_{secrets.token_hex(8)}",
                    sensor_type=secrets.choice(["temperature", "humidity", "pressure", "motion", "sound"]),
                    insight_type=secrets.choice(list(InsightType)),
                    title=f"Real-time {secrets.choice(['anomaly', 'trend', 'pattern'])} detected",
                    description=f"Real-time analysis detected {secrets.choice(['unusual', 'normal', 'expected'])} behavior",
                    confidence=random.uniform(0.7, 0.95),
                    severity=secrets.choice(list(AlertSeverity)),
                    timestamp=datetime.now(),
                    data_points=[random.uniform(0, 100) for _ in range(10)],
                    trend=secrets.choice(["increasing", "decreasing", "stable", "volatile"]),
                    anomaly_score=random.uniform(0, 1) if secrets.choice([True, False]) else None
                )
                
                self.data_insights.append(insight)
            
            # Keep only last 10000 insights
            if len(self.data_insights) > 10000:
                self.data_insights = self.data_insights[-10000:]
            
            logger.info(f"Generated {num_insights} real-time insights")
            
        except Exception as e:
            logger.error(f"Error performing real-time analytics: {e}")

    async def _perform_batch_analytics(self):
        """Perform batch analytics"""
        try:
            # Create batch analytics job
            job = AnalyticsJob(
                job_id=f"batch_job_{secrets.token_hex(8)}",
                job_type=AnalyticsType.BATCH,
                device_ids=[f"device_{secrets.token_hex(8)}" for _ in range(5)],
                sensor_types=["temperature", "humidity", "pressure", "motion", "sound"],
                parameters={"window_size": 1000, "analysis_type": "comprehensive"},
                status="running",
                started_at=datetime.now()
            )
            
            self.analytics_jobs.append(job)
            
            # Simulate batch processing
            for progress in [20, 40, 60, 80, 100]:
                await asyncio.sleep(1)  # Simulate processing time
                job.progress = progress
            
            # Complete job
            job.status = "completed"
            job.completed_at = datetime.now()
            job.results = {
                "total_data_points": secrets.randbelow(10000) + 1000,
                "anomalies_detected": secrets.randbelow(50) + 5,
                "patterns_found": secrets.randbelow(20) + 2,
                "trends_identified": secrets.randbelow(10) + 1,
                "processing_time": random.uniform(10, 60)
            }
            
            # Keep only last 1000 jobs
            if len(self.analytics_jobs) > 1000:
                self.analytics_jobs = self.analytics_jobs[-1000:]
            
            logger.info(f"Completed batch analytics job: {job.job_id}")
            
        except Exception as e:
            logger.error(f"Error performing batch analytics: {e}")

    async def _perform_streaming_analytics(self):
        """Perform streaming analytics"""
        try:
            # Simulate streaming analytics processing
            # In a real implementation, this would process streaming sensor data
            
            # Generate mock streaming insights
            num_streaming_insights = secrets.randbelow(3) + 1
            
            for _ in range(num_streaming_insights):
                insight = DataInsight(
                    insight_id=f"streaming_insight_{secrets.token_hex(8)}",
                    device_id=f"device_{secrets.token_hex(8)}",
                    sensor_type=secrets.choice(["temperature", "humidity", "pressure", "motion", "sound"]),
                    insight_type=InsightType.STREAMING,
                    title=f"Streaming {secrets.choice(['anomaly', 'trend', 'pattern'])} detected",
                    description=f"Streaming analysis detected {secrets.choice(['unusual', 'normal', 'expected'])} behavior",
                    confidence=random.uniform(0.8, 0.98),
                    severity=secrets.choice(list(AlertSeverity)),
                    timestamp=datetime.now(),
                    data_points=[random.uniform(0, 100) for _ in range(5)],
                    trend=secrets.choice(["increasing", "decreasing", "stable", "volatile"])
                )
                
                self.data_insights.append(insight)
            
            logger.info(f"Generated {num_streaming_insights} streaming insights")
            
        except Exception as e:
            logger.error(f"Error performing streaming analytics: {e}")

    async def _retrain_models(self):
        """Retrain predictive models"""
        try:
            # Retrain models that need updating
            models_to_retrain = [m for m in self.predictive_models.values() 
                               if datetime.now() - m.last_trained > timedelta(days=7)]
            
            for model in models_to_retrain:
                # Simulate model retraining
                await asyncio.sleep(1)  # Simulate training time
                
                # Update model
                model.last_trained = datetime.now()
                model.accuracy = min(0.99, model.accuracy + random.uniform(0, 0.05))
                model.training_data_size += secrets.randbelow(1000) + 100
                
                logger.info(f"Retrained model: {model.model_id}")
            
            logger.info(f"Retrained {len(models_to_retrain)} models")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")

    async def _generate_insights(self):
        """Generate insights from analytics results"""
        try:
            # Generate insights based on recent analytics jobs
            recent_jobs = [j for j in self.analytics_jobs 
                          if j.status == "completed" and
                          j.completed_at and
                          j.completed_at > datetime.now() - timedelta(hours=1)]
            
            for job in recent_jobs:
                if not job.results:
                    continue
                
                # Generate insights based on job results
                insights = await self._analyze_job_results(job)
                
                for insight in insights:
                    self.data_insights.append(insight)
            
            logger.info(f"Generated insights from {len(recent_jobs)} analytics jobs")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")

    async def _analyze_job_results(self, job: AnalyticsJob) -> List[DataInsight]:
        """Analyze job results for insights"""
        try:
            insights = []
            results = job.results
            
            if not results:
                return insights
            
            # Generate insights based on results
            if results.get("anomalies_detected", 0) > 10:
                insight = DataInsight(
                    insight_id=f"insight_{secrets.token_hex(8)}",
                    device_id=job.device_ids[0] if job.device_ids else "unknown",
                    sensor_type=job.sensor_types[0] if job.sensor_types else "unknown",
                    insight_type=InsightType.ANOMALY,
                    title="High Anomaly Rate Detected",
                    description=f"Analytics detected {results['anomalies_detected']} anomalies in recent data",
                    confidence=0.9,
                    severity=AlertSeverity.HIGH,
                    timestamp=datetime.now(),
                    data_points=[],
                    trend="anomalous",
                    anomaly_score=0.9,
                    recommendations=["Investigate sensor calibration", "Check for environmental changes"]
                )
                insights.append(insight)
            
            if results.get("patterns_found", 0) > 5:
                insight = DataInsight(
                    insight_id=f"insight_{secrets.token_hex(8)}",
                    device_id=job.device_ids[0] if job.device_ids else "unknown",
                    sensor_type=job.sensor_types[0] if job.sensor_types else "unknown",
                    insight_type=InsightType.PATTERN,
                    title="Multiple Patterns Detected",
                    description=f"Analytics identified {results['patterns_found']} distinct patterns",
                    confidence=0.8,
                    severity=AlertSeverity.MEDIUM,
                    timestamp=datetime.now(),
                    data_points=[],
                    trend="patterned",
                    recommendations=["Analyze pattern significance", "Consider pattern-based predictions"]
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing job results: {e}")
            return []

    # Public API methods
    async def get_analytics_jobs(self, job_type: Optional[AnalyticsType] = None,
                               status: Optional[str] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get analytics jobs"""
        try:
            jobs = self.analytics_jobs
            
            # Filter by job type
            if job_type:
                jobs = [j for j in jobs if j.job_type == job_type]
            
            # Filter by status
            if status:
                jobs = [j for j in jobs if j.status == status]
            
            # Sort by created time (most recent first)
            jobs = sorted(jobs, key=lambda x: x.created_at, reverse=True)
            
            # Limit results
            jobs = jobs[:limit]
            
            return [
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type.value,
                    "device_ids": job.device_ids,
                    "sensor_types": job.sensor_types,
                    "parameters": job.parameters,
                    "status": job.status,
                    "progress": job.progress,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "results": job.results,
                    "error_message": job.error_message,
                    "metadata": job.metadata
                }
                for job in jobs
            ]
            
        except Exception as e:
            logger.error(f"Error getting analytics jobs: {e}")
            return []

    async def get_data_insights(self, device_id: Optional[str] = None,
                              sensor_type: Optional[str] = None,
                              insight_type: Optional[InsightType] = None,
                              severity: Optional[AlertSeverity] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get data insights"""
        try:
            insights = self.data_insights
            
            # Filter by device ID
            if device_id:
                insights = [i for i in insights if i.device_id == device_id]
            
            # Filter by sensor type
            if sensor_type:
                insights = [i for i in insights if i.sensor_type == sensor_type]
            
            # Filter by insight type
            if insight_type:
                insights = [i for i in insights if i.insight_type == insight_type]
            
            # Filter by severity
            if severity:
                insights = [i for i in insights if i.severity == severity]
            
            # Sort by timestamp (most recent first)
            insights = sorted(insights, key=lambda x: x.timestamp, reverse=True)
            
            # Limit results
            insights = insights[:limit]
            
            return [
                {
                    "insight_id": insight.insight_id,
                    "device_id": insight.device_id,
                    "sensor_type": insight.sensor_type,
                    "insight_type": insight.insight_type.value,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "severity": insight.severity.value,
                    "timestamp": insight.timestamp.isoformat(),
                    "data_points": insight.data_points,
                    "trend": insight.trend,
                    "anomaly_score": insight.anomaly_score,
                    "correlation_coefficient": insight.correlation_coefficient,
                    "prediction_accuracy": insight.prediction_accuracy,
                    "recommendations": insight.recommendations,
                    "metadata": insight.metadata
                }
                for insight in insights
            ]
            
        except Exception as e:
            logger.error(f"Error getting data insights: {e}")
            return []

    async def get_predictive_models(self, device_id: Optional[str] = None,
                                  sensor_type: Optional[str] = None,
                                  active_only: bool = False) -> List[Dict[str, Any]]:
        """Get predictive models"""
        try:
            models = list(self.predictive_models.values())
            
            # Filter by device ID
            if device_id:
                models = [m for m in models if m.device_id == device_id]
            
            # Filter by sensor type
            if sensor_type:
                models = [m for m in models if m.sensor_type == sensor_type]
            
            # Filter by active status
            if active_only:
                models = [m for m in models if m.is_active]
            
            return [
                {
                    "model_id": model.model_id,
                    "model_type": model.model_type,
                    "device_id": model.device_id,
                    "sensor_type": model.sensor_type,
                    "algorithm": model.algorithm,
                    "accuracy": model.accuracy,
                    "training_data_size": model.training_data_size,
                    "last_trained": model.last_trained.isoformat(),
                    "is_active": model.is_active,
                    "parameters": model.parameters,
                    "metadata": model.metadata
                }
                for model in models
            ]
            
        except Exception as e:
            logger.error(f"Error getting predictive models: {e}")
            return []

    async def get_data_patterns(self, device_id: Optional[str] = None,
                              sensor_type: Optional[str] = None,
                              pattern_type: Optional[str] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get data patterns"""
        try:
            patterns = self.data_patterns
            
            # Filter by device ID
            if device_id:
                patterns = [p for p in patterns if p.device_id == device_id]
            
            # Filter by sensor type
            if sensor_type:
                patterns = [p for p in patterns if p.sensor_type == sensor_type]
            
            # Filter by pattern type
            if pattern_type:
                patterns = [p for p in patterns if p.pattern_type == pattern_type]
            
            # Sort by last detected (most recent first)
            patterns = sorted(patterns, key=lambda x: x.last_detected, reverse=True)
            
            # Limit results
            patterns = patterns[:limit]
            
            return [
                {
                    "pattern_id": pattern.pattern_id,
                    "device_id": pattern.device_id,
                    "sensor_type": pattern.sensor_type,
                    "pattern_type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "duration": pattern.duration,
                    "confidence": pattern.confidence,
                    "first_detected": pattern.first_detected.isoformat(),
                    "last_detected": pattern.last_detected.isoformat(),
                    "pattern_data": pattern.pattern_data,
                    "metadata": pattern.metadata
                }
                for pattern in patterns
            ]
            
        except Exception as e:
            logger.error(f"Error getting data patterns: {e}")
            return []

    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        try:
            return {
                "engine_id": self.engine_id,
                "is_running": self.is_running,
                "total_jobs": len(self.analytics_jobs),
                "total_insights": len(self.data_insights),
                "total_models": len(self.predictive_models),
                "total_patterns": len(self.data_patterns),
                "completed_jobs": len([j for j in self.analytics_jobs if j.status == "completed"]),
                "failed_jobs": len([j for j in self.analytics_jobs if j.status == "failed"]),
                "active_models": len([m for m in self.predictive_models.values() if m.is_active]),
                "high_severity_insights": len([i for i in self.data_insights if i.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]]),
                "analytics_configs": self.analytics_configs,
                "supported_analytics_types": [at.value for at in AnalyticsType],
                "supported_insight_types": [it.value for it in InsightType],
                "uptime": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting engine metrics: {e}")
            return {}

    async def create_analytics_job(self, job_type: AnalyticsType, device_ids: List[str],
                                 sensor_types: List[str], parameters: Dict[str, Any]) -> str:
        """Create an analytics job"""
        try:
            job = AnalyticsJob(
                job_id=f"job_{secrets.token_hex(8)}",
                job_type=job_type,
                device_ids=device_ids,
                sensor_types=sensor_types,
                parameters=parameters,
                status="pending"
            )
            
            self.analytics_jobs.append(job)
            
            logger.info(f"Created analytics job: {job.job_id}")
            return job.job_id
            
        except Exception as e:
            logger.error(f"Error creating analytics job: {e}")
            raise

    async def update_model_status(self, model_id: str, is_active: bool) -> bool:
        """Update model status"""
        try:
            if model_id not in self.predictive_models:
                return False
            
            model = self.predictive_models[model_id]
            model.is_active = is_active
            
            logger.info(f"Updated model status: {model_id} -> {is_active}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model status {model_id}: {e}")
            return False

# Global instance
iot_analytics_engine = IoTAnalyticsEngine()
