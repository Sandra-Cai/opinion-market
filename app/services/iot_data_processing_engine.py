"""
IoT Data Processing Engine
Advanced IoT sensor data collection, processing, and analysis
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
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SensorType(Enum):
    """Sensor types"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    MOTION = "motion"
    SOUND = "sound"
    VIBRATION = "vibration"
    AIR_QUALITY = "air_quality"
    GPS = "gps"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    PROXIMITY = "proximity"
    CAMERA = "camera"
    MICROPHONE = "microphone"

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"

class ProcessingStatus(Enum):
    """Processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class IoTDevice:
    """IoT device"""
    device_id: str
    name: str
    device_type: str
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    sensors: List[str] = field(default_factory=list)
    firmware_version: str = "1.0.0"
    battery_level: Optional[float] = None
    signal_strength: Optional[float] = None
    last_seen: datetime = field(default_factory=datetime.now)
    is_online: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SensorData:
    """Sensor data"""
    data_id: str
    device_id: str
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: datetime
    quality: DataQuality = DataQuality.GOOD
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataProcessingJob:
    """Data processing job"""
    job_id: str
    device_id: str
    sensor_type: SensorType
    processing_type: str
    input_data: List[SensorData]
    output_data: Optional[Dict[str, Any]] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataInsight:
    """Data insight"""
    insight_id: str
    device_id: str
    sensor_type: SensorType
    insight_type: str
    description: str
    confidence: float
    severity: str
    timestamp: datetime
    data_points: List[float]
    trend: str
    anomaly_score: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class IoTDataProcessingEngine:
    """Advanced IoT Data Processing Engine"""
    
    def __init__(self):
        self.engine_id = f"iot_processing_{secrets.token_hex(8)}"
        self.is_running = False
        
        # IoT data
        self.devices: Dict[str, IoTDevice] = {}
        self.sensor_data: List[SensorData] = []
        self.processing_jobs: List[DataProcessingJob] = []
        self.data_insights: List[DataInsight] = []
        
        # Processing configurations
        self.processing_configs = {
            "batch_size": 1000,
            "processing_interval": 60,  # seconds
            "data_retention_days": 30,
            "anomaly_threshold": 0.8,
            "quality_thresholds": {
                DataQuality.EXCELLENT: 0.95,
                DataQuality.GOOD: 0.85,
                DataQuality.FAIR: 0.70,
                DataQuality.POOR: 0.50
            }
        }
        
        # Sensor configurations
        self.sensor_configs = {
            SensorType.TEMPERATURE: {
                "unit": "°C",
                "range": (-50, 100),
                "precision": 2,
                "sampling_rate": 1  # Hz
            },
            SensorType.HUMIDITY: {
                "unit": "%",
                "range": (0, 100),
                "precision": 1,
                "sampling_rate": 1
            },
            SensorType.PRESSURE: {
                "unit": "hPa",
                "range": (800, 1200),
                "precision": 1,
                "sampling_rate": 1
            },
            SensorType.LIGHT: {
                "unit": "lux",
                "range": (0, 100000),
                "precision": 0,
                "sampling_rate": 1
            },
            SensorType.MOTION: {
                "unit": "boolean",
                "range": (0, 1),
                "precision": 0,
                "sampling_rate": 10
            },
            SensorType.SOUND: {
                "unit": "dB",
                "range": (0, 140),
                "precision": 1,
                "sampling_rate": 10
            },
            SensorType.AIR_QUALITY: {
                "unit": "AQI",
                "range": (0, 500),
                "precision": 0,
                "sampling_rate": 1
            },
            SensorType.GPS: {
                "unit": "coordinates",
                "range": None,
                "precision": 6,
                "sampling_rate": 1
            }
        }
        
        # Processing tasks
        self.data_collection_task: Optional[asyncio.Task] = None
        self.data_processing_task: Optional[asyncio.Task] = None
        self.insight_generation_task: Optional[asyncio.Task] = None
        self.device_monitoring_task: Optional[asyncio.Task] = None
        self.data_cleanup_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.processing_stats: Dict[str, List[float]] = {}
        self.data_quality_stats: Dict[str, float] = {}
        self.insight_generation_stats: Dict[str, int] = {}
        
        logger.info(f"IoT Data Processing Engine {self.engine_id} initialized")

    async def start_iot_processing_engine(self):
        """Start the IoT processing engine"""
        if self.is_running:
            return
        
        logger.info("Starting IoT Data Processing Engine...")
        
        # Initialize IoT data
        await self._initialize_devices()
        await self._initialize_sensor_data()
        
        # Start processing tasks
        self.is_running = True
        
        self.data_collection_task = asyncio.create_task(self._data_collection_loop())
        self.data_processing_task = asyncio.create_task(self._data_processing_loop())
        self.insight_generation_task = asyncio.create_task(self._insight_generation_loop())
        self.device_monitoring_task = asyncio.create_task(self._device_monitoring_loop())
        self.data_cleanup_task = asyncio.create_task(self._data_cleanup_loop())
        
        logger.info("IoT Data Processing Engine started")

    async def stop_iot_processing_engine(self):
        """Stop the IoT processing engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping IoT Data Processing Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.data_collection_task,
            self.data_processing_task,
            self.insight_generation_task,
            self.device_monitoring_task,
            self.data_cleanup_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("IoT Data Processing Engine stopped")

    async def _initialize_devices(self):
        """Initialize IoT devices"""
        try:
            # Create mock IoT devices
            device_types = ["sensor_node", "gateway", "camera", "actuator", "beacon"]
            locations = ["office", "warehouse", "factory", "home", "outdoor", "vehicle"]
            sensor_types = list(SensorType)
            
            for i in range(50):  # Generate 50 mock devices
                device = IoTDevice(
                    device_id=f"device_{secrets.token_hex(8)}",
                    name=f"IoT Device {i+1}",
                    device_type=secrets.choice(device_types),
                    location=secrets.choice(locations),
                    latitude=secrets.uniform(-90, 90),
                    longitude=secrets.uniform(-180, 180),
                    altitude=secrets.uniform(0, 1000),
                    sensors=[secrets.choice(sensor_types).value for _ in range(secrets.randbelow(5) + 1)],
                    firmware_version=f"{secrets.randbelow(3) + 1}.{secrets.randbelow(10)}.{secrets.randbelow(10)}",
                    battery_level=secrets.uniform(0, 100),
                    signal_strength=secrets.uniform(-100, -30),
                    last_seen=datetime.now() - timedelta(minutes=secrets.randbelow(60)),
                    is_online=secrets.choice([True, False])
                )
                
                self.devices[device.device_id] = device
            
            logger.info(f"Initialized {len(self.devices)} IoT devices")
            
        except Exception as e:
            logger.error(f"Error initializing devices: {e}")

    async def _initialize_sensor_data(self):
        """Initialize sensor data"""
        try:
            # Generate mock sensor data
            for device in self.devices.values():
                for sensor_name in device.sensors:
                    try:
                        sensor_type = SensorType(sensor_name)
                        config = self.sensor_configs.get(sensor_type, {})
                        
                        # Generate historical data
                        for i in range(100):  # 100 data points per sensor
                            value = self._generate_sensor_value(sensor_type, config)
                            
                            sensor_data = SensorData(
                                data_id=f"data_{secrets.token_hex(8)}",
                                device_id=device.device_id,
                                sensor_type=sensor_type,
                                value=value,
                                unit=config.get("unit", ""),
                                timestamp=datetime.now() - timedelta(minutes=i),
                                quality=secrets.choice(list(DataQuality)),
                                location=device.location,
                                latitude=device.latitude,
                                longitude=device.longitude
                            )
                            
                            self.sensor_data.append(sensor_data)
                    
                    except ValueError:
                        continue  # Skip invalid sensor types
            
            logger.info(f"Initialized {len(self.sensor_data)} sensor data points")
            
        except Exception as e:
            logger.error(f"Error initializing sensor data: {e}")

    def _generate_sensor_value(self, sensor_type: SensorType, config: Dict[str, Any]) -> float:
        """Generate realistic sensor value"""
        try:
            range_config = config.get("range")
            if range_config is None:
                return 0.0
            
            if sensor_type == SensorType.TEMPERATURE:
                # Temperature with daily cycle
                hour = datetime.now().hour
                base_temp = 20 + 10 * np.sin(2 * np.pi * hour / 24)
                return base_temp + secrets.uniform(-5, 5)
            
            elif sensor_type == SensorType.HUMIDITY:
                # Humidity with some variation
                return secrets.uniform(30, 80)
            
            elif sensor_type == SensorType.PRESSURE:
                # Atmospheric pressure
                return secrets.uniform(980, 1020)
            
            elif sensor_type == SensorType.LIGHT:
                # Light with day/night cycle
                hour = datetime.now().hour
                if 6 <= hour <= 18:
                    return secrets.uniform(100, 1000)
                else:
                    return secrets.uniform(0, 10)
            
            elif sensor_type == SensorType.MOTION:
                # Motion detection
                return float(secrets.choice([0, 1]))
            
            elif sensor_type == SensorType.SOUND:
                # Sound level
                return secrets.uniform(30, 80)
            
            elif sensor_type == SensorType.AIR_QUALITY:
                # Air quality index
                return secrets.uniform(0, 150)
            
            else:
                # Default random value within range
                if isinstance(range_config, tuple):
                    return secrets.uniform(range_config[0], range_config[1])
                else:
                    return 0.0
            
        except Exception as e:
            logger.error(f"Error generating sensor value: {e}")
            return 0.0

    async def _data_collection_loop(self):
        """Data collection loop"""
        while self.is_running:
            try:
                # Collect new sensor data
                await self._collect_sensor_data()
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(10)

    async def _data_processing_loop(self):
        """Data processing loop"""
        while self.is_running:
            try:
                # Process sensor data
                await self._process_sensor_data()
                
                await asyncio.sleep(self.processing_configs["processing_interval"])
                
            except Exception as e:
                logger.error(f"Error in data processing loop: {e}")
                await asyncio.sleep(self.processing_configs["processing_interval"])

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

    async def _device_monitoring_loop(self):
        """Device monitoring loop"""
        while self.is_running:
            try:
                # Monitor device health
                await self._monitor_device_health()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in device monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _data_cleanup_loop(self):
        """Data cleanup loop"""
        while self.is_running:
            try:
                # Clean up old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error in data cleanup loop: {e}")
                await asyncio.sleep(3600)

    async def _collect_sensor_data(self):
        """Collect new sensor data"""
        try:
            # Generate new sensor data for online devices
            online_devices = [d for d in self.devices.values() if d.is_online]
            
            for device in online_devices:
                for sensor_name in device.sensors:
                    try:
                        sensor_type = SensorType(sensor_name)
                        config = self.sensor_configs.get(sensor_type, {})
                        
                        value = self._generate_sensor_value(sensor_type, config)
                        
                        sensor_data = SensorData(
                            data_id=f"data_{secrets.token_hex(8)}",
                            device_id=device.device_id,
                            sensor_type=sensor_type,
                            value=value,
                            unit=config.get("unit", ""),
                            timestamp=datetime.now(),
                            quality=secrets.choice(list(DataQuality)),
                            location=device.location,
                            latitude=device.latitude,
                            longitude=device.longitude
                        )
                        
                        self.sensor_data.append(sensor_data)
                    
                    except ValueError:
                        continue
            
            # Keep only last 100000 data points
            if len(self.sensor_data) > 100000:
                self.sensor_data = self.sensor_data[-100000:]
            
            logger.info(f"Collected sensor data from {len(online_devices)} devices")
            
        except Exception as e:
            logger.error(f"Error collecting sensor data: {e}")

    async def _process_sensor_data(self):
        """Process sensor data"""
        try:
            # Get recent data for processing
            recent_data = [d for d in self.sensor_data 
                          if d.timestamp > datetime.now() - timedelta(minutes=5)]
            
            if not recent_data:
                return
            
            # Group data by device and sensor type
            grouped_data = {}
            for data in recent_data:
                key = f"{data.device_id}_{data.sensor_type.value}"
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(data)
            
            # Process each group
            for key, data_group in grouped_data.items():
                await self._process_data_group(data_group)
            
            logger.info(f"Processed {len(recent_data)} sensor data points")
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")

    async def _process_data_group(self, data_group: List[SensorData]):
        """Process a group of sensor data"""
        try:
            if not data_group:
                return
            
            # Create processing job
            job = DataProcessingJob(
                job_id=f"job_{secrets.token_hex(8)}",
                device_id=data_group[0].device_id,
                sensor_type=data_group[0].sensor_type,
                processing_type="statistical_analysis",
                input_data=data_group,
                status=ProcessingStatus.PROCESSING,
                started_at=datetime.now()
            )
            
            self.processing_jobs.append(job)
            
            # Perform statistical analysis
            values = [d.value for d in data_group]
            
            analysis = {
                "count": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": float(np.max(values) - np.min(values)),
                "quality_distribution": {
                    quality.value: len([d for d in data_group if d.quality == quality])
                    for quality in DataQuality
                }
            }
            
            # Update job
            job.output_data = analysis
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Keep only last 10000 jobs
            if len(self.processing_jobs) > 10000:
                self.processing_jobs = self.processing_jobs[-10000:]
            
        except Exception as e:
            logger.error(f"Error processing data group: {e}")

    async def _generate_insights(self):
        """Generate insights from processed data"""
        try:
            # Get recent processing jobs
            recent_jobs = [j for j in self.processing_jobs 
                          if j.status == ProcessingStatus.COMPLETED and
                          j.completed_at and
                          j.completed_at > datetime.now() - timedelta(hours=1)]
            
            for job in recent_jobs:
                if not job.output_data:
                    continue
                
                # Generate insights based on analysis
                insights = await self._analyze_for_insights(job)
                
                for insight in insights:
                    self.data_insights.append(insight)
            
            # Keep only last 5000 insights
            if len(self.data_insights) > 5000:
                self.data_insights = self.data_insights[-5000:]
            
            logger.info(f"Generated {len(recent_jobs)} insights")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")

    async def _analyze_for_insights(self, job: DataProcessingJob) -> List[DataInsight]:
        """Analyze job data for insights"""
        try:
            insights = []
            analysis = job.output_data
            
            if not analysis:
                return insights
            
            # Check for anomalies
            if analysis.get("std", 0) > analysis.get("mean", 0) * 0.5:
                insight = DataInsight(
                    insight_id=f"insight_{secrets.token_hex(8)}",
                    device_id=job.device_id,
                    sensor_type=job.sensor_type,
                    insight_type="anomaly",
                    description=f"High variability detected in {job.sensor_type.value} data",
                    confidence=0.8,
                    severity="medium",
                    timestamp=datetime.now(),
                    data_points=[d.value for d in job.input_data],
                    trend="volatile",
                    anomaly_score=0.8,
                    recommendations=["Check sensor calibration", "Investigate environmental factors"]
                )
                insights.append(insight)
            
            # Check for trends
            if len(job.input_data) >= 10:
                values = [d.value for d in job.input_data[-10:]]
                if all(values[i] <= values[i+1] for i in range(len(values)-1)):
                    insight = DataInsight(
                        insight_id=f"insight_{secrets.token_hex(8)}",
                        device_id=job.device_id,
                        sensor_type=job.sensor_type,
                        insight_type="trend",
                        description=f"Increasing trend detected in {job.sensor_type.value}",
                        confidence=0.9,
                        severity="low",
                        timestamp=datetime.now(),
                        data_points=values,
                        trend="increasing",
                        recommendations=["Monitor for continued increase"]
                    )
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing for insights: {e}")
            return []

    async def _monitor_device_health(self):
        """Monitor device health"""
        try:
            for device in self.devices.values():
                # Check if device is online
                time_since_last_seen = datetime.now() - device.last_seen
                
                if time_since_last_seen > timedelta(minutes=5):
                    device.is_online = False
                else:
                    device.is_online = True
                
                # Update battery level (simulate battery drain)
                if device.battery_level is not None:
                    device.battery_level = max(0, device.battery_level - secrets.uniform(0, 0.1))
                
                # Update signal strength (simulate signal variation)
                if device.signal_strength is not None:
                    device.signal_strength += secrets.uniform(-5, 5)
                    device.signal_strength = max(-100, min(-30, device.signal_strength))
            
            online_count = len([d for d in self.devices.values() if d.is_online])
            logger.info(f"Device health check: {online_count}/{len(self.devices)} devices online")
            
        except Exception as e:
            logger.error(f"Error monitoring device health: {e}")

    async def _cleanup_old_data(self):
        """Clean up old data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.processing_configs["data_retention_days"])
            
            # Clean up old sensor data
            initial_count = len(self.sensor_data)
            self.sensor_data = [d for d in self.sensor_data if d.timestamp > cutoff_time]
            removed_sensor_data = initial_count - len(self.sensor_data)
            
            # Clean up old processing jobs
            initial_jobs = len(self.processing_jobs)
            self.processing_jobs = [j for j in self.processing_jobs if j.created_at > cutoff_time]
            removed_jobs = initial_jobs - len(self.processing_jobs)
            
            # Clean up old insights
            initial_insights = len(self.data_insights)
            self.data_insights = [i for i in self.data_insights if i.timestamp > cutoff_time]
            removed_insights = initial_insights - len(self.data_insights)
            
            logger.info(f"Data cleanup: removed {removed_sensor_data} sensor data, "
                       f"{removed_jobs} jobs, {removed_insights} insights")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    # Public API methods
    async def get_devices(self, device_type: Optional[str] = None,
                         location: Optional[str] = None,
                         online_only: bool = False) -> List[Dict[str, Any]]:
        """Get IoT devices"""
        try:
            devices = list(self.devices.values())
            
            # Filter by device type
            if device_type:
                devices = [d for d in devices if d.device_type == device_type]
            
            # Filter by location
            if location:
                devices = [d for d in devices if d.location == location]
            
            # Filter by online status
            if online_only:
                devices = [d for d in devices if d.is_online]
            
            return [
                {
                    "device_id": device.device_id,
                    "name": device.name,
                    "device_type": device.device_type,
                    "location": device.location,
                    "latitude": device.latitude,
                    "longitude": device.longitude,
                    "altitude": device.altitude,
                    "sensors": device.sensors,
                    "firmware_version": device.firmware_version,
                    "battery_level": device.battery_level,
                    "signal_strength": device.signal_strength,
                    "last_seen": device.last_seen.isoformat(),
                    "is_online": device.is_online,
                    "metadata": device.metadata
                }
                for device in devices
            ]
            
        except Exception as e:
            logger.error(f"Error getting devices: {e}")
            return []

    async def get_sensor_data(self, device_id: Optional[str] = None,
                            sensor_type: Optional[SensorType] = None,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: int = 1000) -> List[Dict[str, Any]]:
        """Get sensor data"""
        try:
            data = self.sensor_data
            
            # Filter by device ID
            if device_id:
                data = [d for d in data if d.device_id == device_id]
            
            # Filter by sensor type
            if sensor_type:
                data = [d for d in data if d.sensor_type == sensor_type]
            
            # Filter by time range
            if start_time:
                data = [d for d in data if d.timestamp >= start_time]
            
            if end_time:
                data = [d for d in data if d.timestamp <= end_time]
            
            # Sort by timestamp (most recent first)
            data = sorted(data, key=lambda x: x.timestamp, reverse=True)
            
            # Limit results
            data = data[:limit]
            
            return [
                {
                    "data_id": d.data_id,
                    "device_id": d.device_id,
                    "sensor_type": d.sensor_type.value,
                    "value": d.value,
                    "unit": d.unit,
                    "timestamp": d.timestamp.isoformat(),
                    "quality": d.quality.value,
                    "location": d.location,
                    "latitude": d.latitude,
                    "longitude": d.longitude,
                    "metadata": d.metadata
                }
                for d in data
            ]
            
        except Exception as e:
            logger.error(f"Error getting sensor data: {e}")
            return []

    async def get_processing_jobs(self, device_id: Optional[str] = None,
                                status: Optional[ProcessingStatus] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Get processing jobs"""
        try:
            jobs = self.processing_jobs
            
            # Filter by device ID
            if device_id:
                jobs = [j for j in jobs if j.device_id == device_id]
            
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
                    "device_id": job.device_id,
                    "sensor_type": job.sensor_type.value,
                    "processing_type": job.processing_type,
                    "input_data_count": len(job.input_data),
                    "output_data": job.output_data,
                    "status": job.status.value,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error_message": job.error_message,
                    "metadata": job.metadata
                }
                for job in jobs
            ]
            
        except Exception as e:
            logger.error(f"Error getting processing jobs: {e}")
            return []

    async def get_data_insights(self, device_id: Optional[str] = None,
                              sensor_type: Optional[SensorType] = None,
                              insight_type: Optional[str] = None,
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
            
            # Sort by timestamp (most recent first)
            insights = sorted(insights, key=lambda x: x.timestamp, reverse=True)
            
            # Limit results
            insights = insights[:limit]
            
            return [
                {
                    "insight_id": insight.insight_id,
                    "device_id": insight.device_id,
                    "sensor_type": insight.sensor_type.value,
                    "insight_type": insight.insight_type,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "severity": insight.severity,
                    "timestamp": insight.timestamp.isoformat(),
                    "data_points": insight.data_points,
                    "trend": insight.trend,
                    "anomaly_score": insight.anomaly_score,
                    "recommendations": insight.recommendations,
                    "metadata": insight.metadata
                }
                for insight in insights
            ]
            
        except Exception as e:
            logger.error(f"Error getting data insights: {e}")
            return []

    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        try:
            return {
                "engine_id": self.engine_id,
                "is_running": self.is_running,
                "total_devices": len(self.devices),
                "online_devices": len([d for d in self.devices.values() if d.is_online]),
                "total_sensor_data": len(self.sensor_data),
                "total_processing_jobs": len(self.processing_jobs),
                "total_insights": len(self.data_insights),
                "completed_jobs": len([j for j in self.processing_jobs if j.status == ProcessingStatus.COMPLETED]),
                "failed_jobs": len([j for j in self.processing_jobs if j.status == ProcessingStatus.FAILED]),
                "supported_sensor_types": [st.value for st in SensorType],
                "processing_configs": self.processing_configs,
                "uptime": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting engine metrics: {e}")
            return {}

    async def add_device(self, name: str, device_type: str, location: str,
                        latitude: Optional[float] = None, longitude: Optional[float] = None,
                        sensors: List[str] = None) -> str:
        """Add a new IoT device"""
        try:
            device = IoTDevice(
                device_id=f"device_{secrets.token_hex(8)}",
                name=name,
                device_type=device_type,
                location=location,
                latitude=latitude,
                longitude=longitude,
                sensors=sensors or [],
                firmware_version="1.0.0",
                battery_level=100.0,
                signal_strength=-50.0,
                is_online=True
            )
            
            self.devices[device.device_id] = device
            
            logger.info(f"Added IoT device: {device.device_id}")
            return device.device_id
            
        except Exception as e:
            logger.error(f"Error adding device: {e}")
            raise

    async def update_device(self, device_id: str, **kwargs) -> bool:
        """Update an IoT device"""
        try:
            if device_id not in self.devices:
                return False
            
            device = self.devices[device_id]
            
            # Update allowed fields
            allowed_fields = ['name', 'device_type', 'location', 'latitude', 'longitude', 
                            'altitude', 'sensors', 'firmware_version', 'battery_level', 
                            'signal_strength', 'is_online']
            for field, value in kwargs.items():
                if field in allowed_fields:
                    setattr(device, field, value)
            
            device.last_seen = datetime.now()
            
            logger.info(f"Updated IoT device: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating device {device_id}: {e}")
            return False

# Global instance
iot_data_processing_engine = IoTDataProcessingEngine()
