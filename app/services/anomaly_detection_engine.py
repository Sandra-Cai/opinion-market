"""
Anomaly Detection Engine
Advanced real-time anomaly detection and alerting system
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
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Anomaly types"""
    STATISTICAL = "statistical"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    PATTERN = "pattern"
    THRESHOLD = "threshold"
    MACHINE_LEARNING = "machine_learning"
    ENSEMBLE = "ensemble"

class SeverityLevel(Enum):
    """Severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DetectionMethod(Enum):
    """Detection methods"""
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    DBSCAN = "dbscan"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ENSEMBLE = "ensemble"

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_id: str
    asset: str
    anomaly_type: AnomalyType
    detection_method: DetectionMethod
    severity: SeverityLevel
    confidence: float  # 0.0 to 1.0
    score: float  # Anomaly score
    detected_at: datetime
    value: float
    expected_value: float
    deviation: float
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnomalyPattern:
    """Anomaly pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    last_seen: datetime
    severity_distribution: Dict[SeverityLevel, int] = field(default_factory=dict)
    affected_assets: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectionRule:
    """Detection rule"""
    rule_id: str
    name: str
    description: str
    asset: str
    method: DetectionMethod
    parameters: Dict[str, Any]
    threshold: float
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

class AnomalyDetectionEngine:
    """Advanced Anomaly Detection Engine"""
    
    def __init__(self):
        self.engine_id = f"anomaly_detection_{secrets.token_hex(8)}"
        self.is_running = False
        
        # Anomaly data
        self.anomaly_detections: List[AnomalyDetection] = []
        self.anomaly_patterns: Dict[str, AnomalyPattern] = {}
        self.detection_rules: Dict[str, DetectionRule] = {}
        
        # Historical data for detection
        self.historical_data: Dict[str, List[float]] = {}
        self.feature_data: Dict[str, pd.DataFrame] = {}
        
        # Detection models
        self.detection_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Configuration
        self.detection_config = {
            "window_size": 100,  # Number of data points for detection
            "min_anomaly_score": 0.7,  # Minimum score to consider as anomaly
            "max_anomalies_per_hour": 50,  # Rate limiting
            "pattern_min_frequency": 3,  # Minimum frequency for pattern detection
            "severity_thresholds": {
                SeverityLevel.LOW: 0.7,
                SeverityLevel.MEDIUM: 0.8,
                SeverityLevel.HIGH: 0.9,
                SeverityLevel.CRITICAL: 0.95
            }
        }
        
        # Performance tracking
        self.detection_performance: Dict[str, List[float]] = {}
        self.false_positive_rate: Dict[str, float] = {}
        self.detection_latency: List[float] = []
        
        # Processing tasks
        self.detection_task: Optional[asyncio.Task] = None
        self.pattern_analysis_task: Optional[asyncio.Task] = None
        self.model_training_task: Optional[asyncio.Task] = None
        self.performance_monitoring_task: Optional[asyncio.Task] = None
        
        logger.info(f"Anomaly Detection Engine {self.engine_id} initialized")

    async def start_anomaly_detection_engine(self):
        """Start the anomaly detection engine"""
        if self.is_running:
            return
        
        logger.info("Starting Anomaly Detection Engine...")
        
        # Initialize detection models and data
        await self._initialize_detection_models()
        await self._initialize_historical_data()
        await self._initialize_detection_rules()
        
        # Start processing tasks
        self.is_running = True
        
        self.detection_task = asyncio.create_task(self._detection_loop())
        self.pattern_analysis_task = asyncio.create_task(self._pattern_analysis_loop())
        self.model_training_task = asyncio.create_task(self._model_training_loop())
        self.performance_monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        
        logger.info("Anomaly Detection Engine started")

    async def stop_anomaly_detection_engine(self):
        """Stop the anomaly detection engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping Anomaly Detection Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.detection_task,
            self.pattern_analysis_task,
            self.model_training_task,
            self.performance_monitoring_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Anomaly Detection Engine stopped")

    async def _initialize_detection_models(self):
        """Initialize detection models"""
        try:
            # Initialize models for each detection method
            for method in DetectionMethod:
                if method == DetectionMethod.ISOLATION_FOREST:
                    self.detection_models[method.value] = IsolationForest(
                        contamination=0.1, random_state=42
                    )
                elif method == DetectionMethod.DBSCAN:
                    self.detection_models[method.value] = DBSCAN(
                        eps=0.5, min_samples=5
                    )
                else:
                    # For statistical methods, we don't need sklearn models
                    self.detection_models[method.value] = None
            
            # Initialize scalers
            for method in DetectionMethod:
                self.scalers[method.value] = StandardScaler()
            
            logger.info(f"Initialized {len(self.detection_models)} detection models")
            
        except Exception as e:
            logger.error(f"Error initializing detection models: {e}")

    async def _initialize_historical_data(self):
        """Initialize historical data for detection"""
        try:
            # Generate mock historical data for major assets
            assets = ["BTC", "ETH", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]
            
            for asset in assets:
                # Generate 1000 data points
                n_points = 1000
                
                # Generate different types of data
                price_data = self._generate_price_data(n_points)
                volume_data = self._generate_volume_data(n_points)
                volatility_data = self._generate_volatility_data(n_points)
                
                # Store historical data
                self.historical_data[f"{asset}_price"] = price_data
                self.historical_data[f"{asset}_volume"] = volume_data
                self.historical_data[f"{asset}_volatility"] = volatility_data
                
                # Create feature DataFrame
                feature_df = pd.DataFrame({
                    'price': price_data,
                    'volume': volume_data,
                    'volatility': volatility_data,
                    'price_change': np.diff(price_data, prepend=price_data[0]),
                    'volume_change': np.diff(volume_data, prepend=volume_data[0]),
                    'volatility_change': np.diff(volatility_data, prepend=volatility_data[0])
                })
                
                self.feature_data[asset] = feature_df
            
            logger.info(f"Initialized historical data for {len(assets)} assets")
            
        except Exception as e:
            logger.error(f"Error initializing historical data: {e}")

    def _generate_price_data(self, n_points: int) -> List[float]:
        """Generate realistic price data with some anomalies"""
        try:
            # Generate base price series
            base_price = 100.0
            prices = [base_price]
            
            for i in range(1, n_points):
                # Normal price movement
                if i < n_points * 0.9:  # 90% normal data
                    change = np.random.normal(0, 0.02)
                else:  # 10% anomalous data
                    change = np.random.normal(0, 0.1)  # Higher volatility
                
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 0.01))
            
            return prices
            
        except Exception as e:
            logger.error(f"Error generating price data: {e}")
            return [100.0] * n_points

    def _generate_volume_data(self, n_points: int) -> List[float]:
        """Generate realistic volume data with some anomalies"""
        try:
            # Generate base volume series
            base_volume = 1000000.0
            volumes = [base_volume]
            
            for i in range(1, n_points):
                # Normal volume
                if i < n_points * 0.9:  # 90% normal data
                    change = np.random.normal(0, 0.1)
                else:  # 10% anomalous data
                    change = np.random.normal(0, 0.5)  # Higher volatility
                
                new_volume = volumes[-1] * (1 + change)
                volumes.append(max(new_volume, 1000))
            
            return volumes
            
        except Exception as e:
            logger.error(f"Error generating volume data: {e}")
            return [1000000.0] * n_points

    def _generate_volatility_data(self, n_points: int) -> List[float]:
        """Generate realistic volatility data with some anomalies"""
        try:
            # Generate base volatility series
            base_volatility = 0.02
            volatilities = [base_volatility]
            
            for i in range(1, n_points):
                # Normal volatility
                if i < n_points * 0.9:  # 90% normal data
                    change = np.random.normal(0, 0.01)
                else:  # 10% anomalous data
                    change = np.random.normal(0, 0.05)  # Higher volatility
                
                new_volatility = volatilities[-1] + change
                volatilities.append(max(new_volatility, 0.001))
            
            return volatilities
            
        except Exception as e:
            logger.error(f"Error generating volatility data: {e}")
            return [0.02] * n_points

    async def _initialize_detection_rules(self):
        """Initialize detection rules"""
        try:
            # Create default detection rules for each asset
            assets = ["BTC", "ETH", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]
            
            for asset in assets:
                # Z-score rule
                z_score_rule = DetectionRule(
                    rule_id=f"{asset}_z_score",
                    name=f"{asset} Z-Score Anomaly Detection",
                    description=f"Detect anomalies in {asset} using Z-score method",
                    asset=asset,
                    method=DetectionMethod.Z_SCORE,
                    parameters={"threshold": 2.5},
                    threshold=0.8
                )
                self.detection_rules[z_score_rule.rule_id] = z_score_rule
                
                # IQR rule
                iqr_rule = DetectionRule(
                    rule_id=f"{asset}_iqr",
                    name=f"{asset} IQR Anomaly Detection",
                    description=f"Detect anomalies in {asset} using IQR method",
                    asset=asset,
                    method=DetectionMethod.IQR,
                    parameters={"factor": 1.5},
                    threshold=0.7
                )
                self.detection_rules[iqr_rule.rule_id] = iqr_rule
                
                # Isolation Forest rule
                isolation_rule = DetectionRule(
                    rule_id=f"{asset}_isolation_forest",
                    name=f"{asset} Isolation Forest Anomaly Detection",
                    description=f"Detect anomalies in {asset} using Isolation Forest",
                    asset=asset,
                    method=DetectionMethod.ISOLATION_FOREST,
                    parameters={"contamination": 0.1},
                    threshold=0.9
                )
                self.detection_rules[isolation_rule.rule_id] = isolation_rule
            
            logger.info(f"Initialized {len(self.detection_rules)} detection rules")
            
        except Exception as e:
            logger.error(f"Error initializing detection rules: {e}")

    async def _detection_loop(self):
        """Main anomaly detection loop"""
        while self.is_running:
            try:
                # Run detection for all assets
                for asset in ["BTC", "ETH", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]:
                    await self._detect_anomalies_for_asset(asset)
                
                await asyncio.sleep(60)  # Detect every minute
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                await asyncio.sleep(60)

    async def _pattern_analysis_loop(self):
        """Pattern analysis loop"""
        while self.is_running:
            try:
                # Analyze anomaly patterns
                await self._analyze_anomaly_patterns()
                
                await asyncio.sleep(1800)  # Analyze every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in pattern analysis loop: {e}")
                await asyncio.sleep(1800)

    async def _model_training_loop(self):
        """Model training loop"""
        while self.is_running:
            try:
                # Retrain detection models
                await self._retrain_detection_models()
                
                await asyncio.sleep(3600)  # Retrain every hour
                
            except Exception as e:
                logger.error(f"Error in model training loop: {e}")
                await asyncio.sleep(3600)

    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                await asyncio.sleep(600)  # Update every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(600)

    async def _detect_anomalies_for_asset(self, asset: str):
        """Detect anomalies for a specific asset"""
        try:
            if asset not in self.feature_data:
                return
            
            feature_df = self.feature_data[asset]
            
            # Get recent data window
            window_size = self.detection_config["window_size"]
            recent_data = feature_df.tail(window_size)
            
            if len(recent_data) < window_size:
                return
            
            # Run detection using different methods
            for rule_id, rule in self.detection_rules.items():
                if rule.asset == asset and rule.enabled:
                    await self._run_detection_rule(rule, recent_data)
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {asset}: {e}")

    async def _run_detection_rule(self, rule: DetectionRule, data: pd.DataFrame):
        """Run a specific detection rule"""
        try:
            start_time = time.time()
            
            # Select appropriate data column
            if rule.method in [DetectionMethod.Z_SCORE, DetectionMethod.IQR]:
                # Use price data for statistical methods
                values = data['price'].values
            else:
                # Use all features for ML methods
                values = data.values
            
            # Run detection based on method
            if rule.method == DetectionMethod.Z_SCORE:
                anomalies = await self._z_score_detection(values, rule.parameters)
            elif rule.method == DetectionMethod.IQR:
                anomalies = await self._iqr_detection(values, rule.parameters)
            elif rule.method == DetectionMethod.ISOLATION_FOREST:
                anomalies = await self._isolation_forest_detection(values, rule.parameters)
            elif rule.method == DetectionMethod.DBSCAN:
                anomalies = await self._dbscan_detection(values, rule.parameters)
            else:
                return
            
            # Process detected anomalies
            for anomaly_idx, anomaly_score in anomalies:
                if anomaly_score >= rule.threshold:
                    await self._create_anomaly_detection(
                        rule, data, anomaly_idx, anomaly_score
                    )
            
            # Record detection latency
            detection_time = time.time() - start_time
            self.detection_latency.append(detection_time)
            
            # Keep only last 1000 latency measurements
            if len(self.detection_latency) > 1000:
                self.detection_latency = self.detection_latency[-1000:]
            
        except Exception as e:
            logger.error(f"Error running detection rule {rule.rule_id}: {e}")

    async def _z_score_detection(self, values: np.ndarray, parameters: Dict[str, Any]) -> List[Tuple[int, float]]:
        """Z-score based anomaly detection"""
        try:
            threshold = parameters.get("threshold", 2.5)
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                return []
            
            z_scores = np.abs((values - mean) / std)
            anomalies = []
            
            for i, z_score in enumerate(z_scores):
                if z_score > threshold:
                    # Convert z-score to anomaly score (0-1)
                    anomaly_score = min(z_score / (threshold * 2), 1.0)
                    anomalies.append((i, anomaly_score))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in Z-score detection: {e}")
            return []

    async def _iqr_detection(self, values: np.ndarray, parameters: Dict[str, Any]) -> List[Tuple[int, float]]:
        """IQR based anomaly detection"""
        try:
            factor = parameters.get("factor", 1.5)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            
            anomalies = []
            
            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    # Calculate anomaly score based on distance from bounds
                    if value < lower_bound:
                        distance = lower_bound - value
                    else:
                        distance = value - upper_bound
                    
                    anomaly_score = min(distance / (factor * iqr), 1.0)
                    anomalies.append((i, anomaly_score))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in IQR detection: {e}")
            return []

    async def _isolation_forest_detection(self, values: np.ndarray, parameters: Dict[str, Any]) -> List[Tuple[int, float]]:
        """Isolation Forest based anomaly detection"""
        try:
            model = self.detection_models[DetectionMethod.ISOLATION_FOREST.value]
            
            # Reshape data for sklearn
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            
            # Fit model and predict
            model.fit(values)
            anomaly_scores = model.decision_function(values)
            
            # Convert to anomaly scores (0-1)
            min_score = np.min(anomaly_scores)
            max_score = np.max(anomaly_scores)
            
            if max_score - min_score == 0:
                return []
            
            normalized_scores = (anomaly_scores - min_score) / (max_score - min_score)
            
            anomalies = []
            for i, score in enumerate(normalized_scores):
                if score < 0.5:  # Isolation Forest returns negative scores for anomalies
                    anomaly_score = 1.0 - score
                    anomalies.append((i, anomaly_score))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest detection: {e}")
            return []

    async def _dbscan_detection(self, values: np.ndarray, parameters: Dict[str, Any]) -> List[Tuple[int, float]]:
        """DBSCAN based anomaly detection"""
        try:
            model = self.detection_models[DetectionMethod.DBSCAN.value]
            
            # Reshape data for sklearn
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            
            # Fit model and predict
            labels = model.fit_predict(values)
            
            anomalies = []
            for i, label in enumerate(labels):
                if label == -1:  # DBSCAN marks outliers as -1
                    # Calculate anomaly score based on distance to nearest cluster
                    anomaly_score = 0.8  # Default score for DBSCAN outliers
                    anomalies.append((i, anomaly_score))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in DBSCAN detection: {e}")
            return []

    async def _create_anomaly_detection(self, rule: DetectionRule, data: pd.DataFrame, 
                                      anomaly_idx: int, anomaly_score: float):
        """Create anomaly detection record"""
        try:
            # Get the actual value and expected value
            actual_value = data.iloc[anomaly_idx]['price']
            expected_value = data['price'].mean()
            
            # Determine severity level
            severity = self._determine_severity(anomaly_score)
            
            # Determine anomaly type
            anomaly_type = self._determine_anomaly_type(rule.method, data, anomaly_idx)
            
            # Create anomaly detection
            anomaly = AnomalyDetection(
                anomaly_id=f"anomaly_{secrets.token_hex(8)}",
                asset=rule.asset,
                anomaly_type=anomaly_type,
                detection_method=rule.method,
                severity=severity,
                confidence=anomaly_score,
                score=anomaly_score,
                detected_at=datetime.now(),
                value=actual_value,
                expected_value=expected_value,
                deviation=abs(actual_value - expected_value),
                context={
                    "rule_id": rule.rule_id,
                    "data_index": anomaly_idx,
                    "window_size": len(data)
                },
                metadata={
                    "detection_engine": self.engine_id,
                    "method_parameters": rule.parameters
                }
            )
            
            self.anomaly_detections.append(anomaly)
            
            # Keep only last 10000 anomalies
            if len(self.anomaly_detections) > 10000:
                self.anomaly_detections = self.anomaly_detections[-10000:]
            
            logger.info(f"Detected {severity.value} anomaly for {rule.asset}: "
                       f"value={actual_value:.2f}, score={anomaly_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error creating anomaly detection: {e}")

    def _determine_severity(self, anomaly_score: float) -> SeverityLevel:
        """Determine severity level based on anomaly score"""
        if anomaly_score >= self.detection_config["severity_thresholds"][SeverityLevel.CRITICAL]:
            return SeverityLevel.CRITICAL
        elif anomaly_score >= self.detection_config["severity_thresholds"][SeverityLevel.HIGH]:
            return SeverityLevel.HIGH
        elif anomaly_score >= self.detection_config["severity_thresholds"][SeverityLevel.MEDIUM]:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def _determine_anomaly_type(self, method: DetectionMethod, data: pd.DataFrame, 
                               anomaly_idx: int) -> AnomalyType:
        """Determine anomaly type based on detection method and context"""
        if method in [DetectionMethod.Z_SCORE, DetectionMethod.IQR]:
            return AnomalyType.STATISTICAL
        elif method in [DetectionMethod.ISOLATION_FOREST, DetectionMethod.DBSCAN]:
            return AnomalyType.MACHINE_LEARNING
        else:
            return AnomalyType.BEHAVIORAL

    async def _analyze_anomaly_patterns(self):
        """Analyze anomaly patterns"""
        try:
            # Group anomalies by type and asset
            pattern_groups = {}
            
            for anomaly in self.anomaly_detections:
                key = f"{anomaly.anomaly_type.value}_{anomaly.asset}"
                
                if key not in pattern_groups:
                    pattern_groups[key] = []
                
                pattern_groups[key].append(anomaly)
            
            # Analyze each pattern group
            for pattern_key, anomalies in pattern_groups.items():
                if len(anomalies) >= self.detection_config["pattern_min_frequency"]:
                    await self._create_anomaly_pattern(pattern_key, anomalies)
            
        except Exception as e:
            logger.error(f"Error analyzing anomaly patterns: {e}")

    async def _create_anomaly_pattern(self, pattern_key: str, anomalies: List[AnomalyDetection]):
        """Create anomaly pattern"""
        try:
            # Calculate pattern statistics
            severity_distribution = {}
            for severity in SeverityLevel:
                count = sum(1 for a in anomalies if a.severity == severity)
                severity_distribution[severity] = count
            
            affected_assets = list(set(a.asset for a in anomalies))
            
            pattern = AnomalyPattern(
                pattern_id=f"pattern_{secrets.token_hex(8)}",
                pattern_type=pattern_key,
                description=f"Pattern of {len(anomalies)} {anomalies[0].anomaly_type.value} anomalies",
                frequency=len(anomalies),
                last_seen=datetime.now(),
                severity_distribution=severity_distribution,
                affected_assets=affected_assets,
                metadata={
                    "first_detected": min(a.detected_at for a in anomalies).isoformat(),
                    "last_detected": max(a.detected_at for a in anomalies).isoformat(),
                    "average_confidence": np.mean([a.confidence for a in anomalies])
                }
            )
            
            self.anomaly_patterns[pattern_key] = pattern
            
        except Exception as e:
            logger.error(f"Error creating anomaly pattern: {e}")

    async def _retrain_detection_models(self):
        """Retrain detection models"""
        try:
            # Retrain ML-based models with recent data
            for method in [DetectionMethod.ISOLATION_FOREST, DetectionMethod.DBSCAN]:
                model = self.detection_models[method.value]
                
                if model is not None:
                    # Collect training data from all assets
                    training_data = []
                    
                    for asset in self.feature_data:
                        feature_df = self.feature_data[asset]
                        recent_data = feature_df.tail(100)  # Use last 100 points
                        training_data.append(recent_data.values)
                    
                    if training_data:
                        # Combine all training data
                        combined_data = np.vstack(training_data)
                        
                        # Retrain model
                        model.fit(combined_data)
                        
                        logger.info(f"Retrained {method.value} model with {len(combined_data)} samples")
            
        except Exception as e:
            logger.error(f"Error retraining detection models: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate detection performance
            total_detections = len(self.anomaly_detections)
            
            if total_detections > 0:
                # Calculate severity distribution
                severity_counts = {}
                for severity in SeverityLevel:
                    count = sum(1 for a in self.anomaly_detections if a.severity == severity)
                    severity_counts[severity.value] = count
                
                # Calculate average detection latency
                avg_latency = np.mean(self.detection_latency) if self.detection_latency else 0
                
                # Store performance metrics
                self.detection_performance["total_detections"] = total_detections
                self.detection_performance["severity_distribution"] = severity_counts
                self.detection_performance["average_latency"] = avg_latency
                self.detection_performance["patterns_detected"] = len(self.anomaly_patterns)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    # Public API methods
    async def get_anomaly_detections(self, asset: Optional[str] = None, 
                                   severity: Optional[SeverityLevel] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get anomaly detections"""
        try:
            anomalies = self.anomaly_detections
            
            # Filter by asset
            if asset:
                anomalies = [a for a in anomalies if a.asset == asset]
            
            # Filter by severity
            if severity:
                anomalies = [a for a in anomalies if a.severity == severity]
            
            # Sort by detection time (most recent first)
            anomalies = sorted(anomalies, key=lambda x: x.detected_at, reverse=True)
            
            # Limit results
            anomalies = anomalies[:limit]
            
            return [
                {
                    "anomaly_id": anomaly.anomaly_id,
                    "asset": anomaly.asset,
                    "anomaly_type": anomaly.anomaly_type.value,
                    "detection_method": anomaly.detection_method.value,
                    "severity": anomaly.severity.value,
                    "confidence": anomaly.confidence,
                    "score": anomaly.score,
                    "detected_at": anomaly.detected_at.isoformat(),
                    "value": anomaly.value,
                    "expected_value": anomaly.expected_value,
                    "deviation": anomaly.deviation,
                    "context": anomaly.context,
                    "metadata": anomaly.metadata
                }
                for anomaly in anomalies
            ]
            
        except Exception as e:
            logger.error(f"Error getting anomaly detections: {e}")
            return []

    async def get_anomaly_patterns(self) -> List[Dict[str, Any]]:
        """Get anomaly patterns"""
        try:
            return [
                {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "description": pattern.description,
                    "frequency": pattern.frequency,
                    "last_seen": pattern.last_seen.isoformat(),
                    "severity_distribution": {
                        severity.value: count 
                        for severity, count in pattern.severity_distribution.items()
                    },
                    "affected_assets": pattern.affected_assets,
                    "metadata": pattern.metadata
                }
                for pattern in self.anomaly_patterns.values()
            ]
            
        except Exception as e:
            logger.error(f"Error getting anomaly patterns: {e}")
            return []

    async def get_detection_rules(self) -> List[Dict[str, Any]]:
        """Get detection rules"""
        try:
            return [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "asset": rule.asset,
                    "method": rule.method.value,
                    "parameters": rule.parameters,
                    "threshold": rule.threshold,
                    "enabled": rule.enabled,
                    "created_at": rule.created_at.isoformat()
                }
                for rule in self.detection_rules.values()
            ]
            
        except Exception as e:
            logger.error(f"Error getting detection rules: {e}")
            return []

    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        try:
            return {
                "engine_id": self.engine_id,
                "is_running": self.is_running,
                "total_anomalies": len(self.anomaly_detections),
                "total_patterns": len(self.anomaly_patterns),
                "total_rules": len(self.detection_rules),
                "active_rules": len([r for r in self.detection_rules.values() if r.enabled]),
                "detection_performance": self.detection_performance,
                "average_latency": np.mean(self.detection_latency) if self.detection_latency else 0,
                "uptime": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting engine metrics: {e}")
            return {}

    async def add_detection_rule(self, name: str, description: str, asset: str,
                               method: DetectionMethod, parameters: Dict[str, Any],
                               threshold: float) -> str:
        """Add a new detection rule"""
        try:
            rule = DetectionRule(
                rule_id=f"rule_{secrets.token_hex(8)}",
                name=name,
                description=description,
                asset=asset,
                method=method,
                parameters=parameters,
                threshold=threshold
            )
            
            self.detection_rules[rule.rule_id] = rule
            
            logger.info(f"Added detection rule: {rule.rule_id}")
            return rule.rule_id
            
        except Exception as e:
            logger.error(f"Error adding detection rule: {e}")
            raise

    async def update_detection_rule(self, rule_id: str, **kwargs) -> bool:
        """Update a detection rule"""
        try:
            if rule_id not in self.detection_rules:
                return False
            
            rule = self.detection_rules[rule_id]
            
            # Update allowed fields
            allowed_fields = ['name', 'description', 'parameters', 'threshold', 'enabled']
            for field, value in kwargs.items():
                if field in allowed_fields:
                    setattr(rule, field, value)
            
            logger.info(f"Updated detection rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating detection rule {rule_id}: {e}")
            return False

# Global instance
anomaly_detection_engine = AnomalyDetectionEngine()
