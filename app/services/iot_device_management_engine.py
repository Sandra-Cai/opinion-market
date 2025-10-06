"""
IoT Device Management Engine
Advanced IoT device registration, configuration, and monitoring
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
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DeviceStatus(Enum):
    """Device status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    ERROR = "error"

class DeviceType(Enum):
    """Device types"""
    SENSOR_NODE = "sensor_node"
    GATEWAY = "gateway"
    CAMERA = "camera"
    ACTUATOR = "actuator"
    BEACON = "beacon"
    SMART_METER = "smart_meter"
    WEATHER_STATION = "weather_station"
    AIR_QUALITY_MONITOR = "air_quality_monitor"
    SECURITY_CAMERA = "security_camera"
    SMART_LOCK = "smart_lock"

class ConfigurationType(Enum):
    """Configuration types"""
    SENSOR_CONFIG = "sensor_config"
    NETWORK_CONFIG = "network_config"
    SECURITY_CONFIG = "security_config"
    POWER_CONFIG = "power_config"
    DATA_CONFIG = "data_config"

@dataclass
class DeviceConfiguration:
    """Device configuration"""
    config_id: str
    device_id: str
    config_type: ConfigurationType
    parameters: Dict[str, Any]
    version: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceFirmware:
    """Device firmware"""
    firmware_id: str
    device_type: DeviceType
    version: str
    file_path: str
    file_size: int
    checksum: str
    release_notes: str
    is_stable: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceUpdate:
    """Device update"""
    update_id: str
    device_id: str
    update_type: str  # "firmware", "configuration", "security"
    target_version: str
    current_version: str
    status: str  # "pending", "downloading", "installing", "completed", "failed"
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceAlert:
    """Device alert"""
    alert_id: str
    device_id: str
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    timestamp: datetime
    is_acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class IoTDeviceManagementEngine:
    """Advanced IoT Device Management Engine"""
    
    def __init__(self):
        self.engine_id = f"iot_device_mgmt_{secrets.token_hex(8)}"
        self.is_running = False
        
        # Device management data
        self.device_configurations: Dict[str, DeviceConfiguration] = {}
        self.device_firmware: Dict[str, DeviceFirmware] = {}
        self.device_updates: List[DeviceUpdate] = []
        self.device_alerts: List[DeviceAlert] = []
        
        # Device management configurations
        self.management_configs = {
            "update_check_interval": 3600,  # 1 hour
            "health_check_interval": 300,   # 5 minutes
            "alert_retention_days": 30,
            "max_concurrent_updates": 10,
            "update_timeout": 1800,  # 30 minutes
            "firmware_storage_path": "/firmware/",
            "config_backup_retention": 7  # days
        }
        
        # Device type configurations
        self.device_type_configs = {
            DeviceType.SENSOR_NODE: {
                "default_sensors": ["temperature", "humidity", "pressure"],
                "update_frequency": 60,  # seconds
                "battery_life_days": 365,
                "max_range_meters": 100
            },
            DeviceType.GATEWAY: {
                "max_connected_devices": 100,
                "update_frequency": 10,  # seconds
                "battery_life_days": 30,
                "max_range_meters": 1000
            },
            DeviceType.CAMERA: {
                "resolution": "1920x1080",
                "update_frequency": 1,  # seconds
                "battery_life_days": 7,
                "max_range_meters": 50
            },
            DeviceType.ACTUATOR: {
                "response_time_ms": 100,
                "update_frequency": 5,  # seconds
                "battery_life_days": 90,
                "max_range_meters": 200
            },
            DeviceType.BEACON: {
                "transmission_power": -20,  # dBm
                "update_frequency": 1,  # seconds
                "battery_life_days": 730,
                "max_range_meters": 10
            }
        }
        
        # Processing tasks
        self.device_monitoring_task: Optional[asyncio.Task] = None
        self.update_management_task: Optional[asyncio.Task] = None
        self.alert_processing_task: Optional[asyncio.Task] = None
        self.configuration_sync_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.device_health_stats: Dict[str, float] = {}
        self.update_success_stats: Dict[str, int] = {}
        self.alert_stats: Dict[str, int] = {}
        
        logger.info(f"IoT Device Management Engine {self.engine_id} initialized")

    async def start_device_management_engine(self):
        """Start the device management engine"""
        if self.is_running:
            return
        
        logger.info("Starting IoT Device Management Engine...")
        
        # Initialize device management data
        await self._initialize_firmware()
        await self._initialize_configurations()
        
        # Start processing tasks
        self.is_running = True
        
        self.device_monitoring_task = asyncio.create_task(self._device_monitoring_loop())
        self.update_management_task = asyncio.create_task(self._update_management_loop())
        self.alert_processing_task = asyncio.create_task(self._alert_processing_loop())
        self.configuration_sync_task = asyncio.create_task(self._configuration_sync_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("IoT Device Management Engine started")

    async def stop_device_management_engine(self):
        """Stop the device management engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping IoT Device Management Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.device_monitoring_task,
            self.update_management_task,
            self.alert_processing_task,
            self.configuration_sync_task,
            self.health_check_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("IoT Device Management Engine stopped")

    async def _initialize_firmware(self):
        """Initialize device firmware"""
        try:
            # Create firmware versions for each device type
            device_types = list(DeviceType)
            
            for device_type in device_types:
                # Create multiple firmware versions
                for version in ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]:
                    firmware = DeviceFirmware(
                        firmware_id=f"firmware_{secrets.token_hex(8)}",
                        device_type=device_type,
                        version=version,
                        file_path=f"/firmware/{device_type.value}_{version}.bin",
                        file_size=secrets.randbelow(1000000) + 100000,  # 100KB - 1MB
                        checksum=secrets.token_hex(32),
                        release_notes=f"Firmware {version} for {device_type.value}",
                        is_stable=version.endswith(".0")
                    )
                    
                    self.device_firmware[firmware.firmware_id] = firmware
            
            logger.info(f"Initialized {len(self.device_firmware)} firmware versions")
            
        except Exception as e:
            logger.error(f"Error initializing firmware: {e}")

    async def _initialize_configurations(self):
        """Initialize device configurations"""
        try:
            # Create default configurations for each device type
            device_types = list(DeviceType)
            config_types = list(ConfigurationType)
            
            for device_type in device_types:
                for config_type in config_types:
                    config = DeviceConfiguration(
                        config_id=f"config_{secrets.token_hex(8)}",
                        device_id=f"default_{device_type.value}",
                        config_type=config_type,
                        parameters=self._get_default_config_parameters(device_type, config_type),
                        version="1.0.0"
                    )
                    
                    self.device_configurations[config.config_id] = config
            
            logger.info(f"Initialized {len(self.device_configurations)} device configurations")
            
        except Exception as e:
            logger.error(f"Error initializing configurations: {e}")

    def _get_default_config_parameters(self, device_type: DeviceType, config_type: ConfigurationType) -> Dict[str, Any]:
        """Get default configuration parameters"""
        try:
            if config_type == ConfigurationType.SENSOR_CONFIG:
                return {
                    "sampling_rate": 1,
                    "sensitivity": 0.8,
                    "calibration_offset": 0.0,
                    "data_format": "json"
                }
            
            elif config_type == ConfigurationType.NETWORK_CONFIG:
                return {
                    "wifi_ssid": "IoT_Network",
                    "wifi_password": "secure_password",
                    "server_url": "https://api.iot-platform.com",
                    "port": 443,
                    "ssl_enabled": True
                }
            
            elif config_type == ConfigurationType.SECURITY_CONFIG:
                return {
                    "encryption_enabled": True,
                    "certificate_path": "/certs/device.crt",
                    "key_path": "/certs/device.key",
                    "auth_token": secrets.token_hex(32)
                }
            
            elif config_type == ConfigurationType.POWER_CONFIG:
                return {
                    "sleep_mode": True,
                    "sleep_duration": 60,
                    "low_battery_threshold": 20,
                    "power_saving_mode": True
                }
            
            elif config_type == ConfigurationType.DATA_CONFIG:
                return {
                    "data_retention_days": 30,
                    "compression_enabled": True,
                    "batch_size": 100,
                    "upload_frequency": 300
                }
            
            else:
                return {}
            
        except Exception as e:
            logger.error(f"Error getting default config parameters: {e}")
            return {}

    async def _device_monitoring_loop(self):
        """Device monitoring loop"""
        while self.is_running:
            try:
                # Monitor device status
                await self._monitor_device_status()
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in device monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _update_management_loop(self):
        """Update management loop"""
        while self.is_running:
            try:
                # Manage device updates
                await self._manage_device_updates()
                
                await asyncio.sleep(self.management_configs["update_check_interval"])
                
            except Exception as e:
                logger.error(f"Error in update management loop: {e}")
                await asyncio.sleep(self.management_configs["update_check_interval"])

    async def _alert_processing_loop(self):
        """Alert processing loop"""
        while self.is_running:
            try:
                # Process device alerts
                await self._process_device_alerts()
                
                await asyncio.sleep(30)  # Process every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(30)

    async def _configuration_sync_loop(self):
        """Configuration sync loop"""
        while self.is_running:
            try:
                # Sync device configurations
                await self._sync_device_configurations()
                
                await asyncio.sleep(300)  # Sync every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in configuration sync loop: {e}")
                await asyncio.sleep(300)

    async def _health_check_loop(self):
        """Health check loop"""
        while self.is_running:
            try:
                # Perform health checks
                await self._perform_health_checks()
                
                await asyncio.sleep(self.management_configs["health_check_interval"])
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.management_configs["health_check_interval"])

    async def _monitor_device_status(self):
        """Monitor device status"""
        try:
            # Simulate device status monitoring
            # In a real implementation, this would check actual device status
            
            # Generate mock device status updates
            num_updates = secrets.randbelow(10) + 1
            
            for _ in range(num_updates):
                # Simulate device status change
                device_id = f"device_{secrets.token_hex(8)}"
                status = secrets.choice(list(DeviceStatus))
                
                # Create alert if device goes offline or has error
                if status in [DeviceStatus.OFFLINE, DeviceStatus.ERROR]:
                    alert = DeviceAlert(
                        alert_id=f"alert_{secrets.token_hex(8)}",
                        device_id=device_id,
                        alert_type="device_status",
                        severity="high" if status == DeviceStatus.ERROR else "medium",
                        message=f"Device {device_id} status changed to {status.value}",
                        timestamp=datetime.now()
                    )
                    
                    self.device_alerts.append(alert)
            
            # Keep only last 10000 alerts
            if len(self.device_alerts) > 10000:
                self.device_alerts = self.device_alerts[-10000:]
            
            logger.info(f"Monitored device status: {num_updates} updates")
            
        except Exception as e:
            logger.error(f"Error monitoring device status: {e}")

    async def _manage_device_updates(self):
        """Manage device updates"""
        try:
            # Check for pending updates
            pending_updates = [u for u in self.device_updates if u.status == "pending"]
            
            # Process pending updates
            for update in pending_updates[:self.management_configs["max_concurrent_updates"]]:
                await self._process_device_update(update)
            
            # Check for stuck updates
            stuck_updates = [u for u in self.device_updates 
                           if u.status in ["downloading", "installing"] and
                           u.started_at and
                           datetime.now() - u.started_at > timedelta(seconds=self.management_configs["update_timeout"])]
            
            for update in stuck_updates:
                update.status = "failed"
                update.error_message = "Update timeout"
                update.completed_at = datetime.now()
            
            logger.info(f"Managed {len(pending_updates)} pending updates, "
                       f"{len(stuck_updates)} stuck updates")
            
        except Exception as e:
            logger.error(f"Error managing device updates: {e}")

    async def _process_device_update(self, update: DeviceUpdate):
        """Process a device update"""
        try:
            update.status = "downloading"
            update.started_at = datetime.now()
            update.progress = 0.0
            
            # Simulate download progress
            for progress in [25, 50, 75, 100]:
                await asyncio.sleep(1)  # Simulate download time
                update.progress = progress
            
            # Simulate installation
            update.status = "installing"
            await asyncio.sleep(2)  # Simulate installation time
            
            # Complete update
            update.status = "completed"
            update.progress = 100.0
            update.completed_at = datetime.now()
            
            logger.info(f"Completed update {update.update_id} for device {update.device_id}")
            
        except Exception as e:
            update.status = "failed"
            update.error_message = str(e)
            update.completed_at = datetime.now()
            logger.error(f"Error processing update {update.update_id}: {e}")

    async def _process_device_alerts(self):
        """Process device alerts"""
        try:
            # Process unacknowledged alerts
            unacknowledged_alerts = [a for a in self.device_alerts if not a.is_acknowledged]
            
            for alert in unacknowledged_alerts:
                # Simulate alert processing
                if alert.severity == "critical":
                    # Critical alerts need immediate attention
                    logger.warning(f"CRITICAL ALERT: {alert.message}")
                elif alert.severity == "high":
                    # High priority alerts
                    logger.info(f"HIGH PRIORITY ALERT: {alert.message}")
                else:
                    # Normal alerts
                    logger.info(f"ALERT: {alert.message}")
            
            # Auto-resolve old low severity alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            for alert in unacknowledged_alerts:
                if (alert.severity == "low" and 
                    alert.timestamp < cutoff_time and
                    not alert.is_acknowledged):
                    alert.is_acknowledged = True
                    alert.acknowledged_at = datetime.now()
                    alert.acknowledged_by = "system"
                    alert.resolved_at = datetime.now()
            
            logger.info(f"Processed {len(unacknowledged_alerts)} device alerts")
            
        except Exception as e:
            logger.error(f"Error processing device alerts: {e}")

    async def _sync_device_configurations(self):
        """Sync device configurations"""
        try:
            # Simulate configuration sync
            # In a real implementation, this would sync configurations with actual devices
            
            synced_configs = 0
            for config in self.device_configurations.values():
                if not config.applied_at:
                    # Simulate applying configuration
                    config.applied_at = datetime.now()
                    config.updated_at = datetime.now()
                    synced_configs += 1
            
            logger.info(f"Synced {synced_configs} device configurations")
            
        except Exception as e:
            logger.error(f"Error syncing device configurations: {e}")

    async def _perform_health_checks(self):
        """Perform health checks"""
        try:
            # Simulate health checks
            # In a real implementation, this would check actual device health
            
            health_checks = {
                "total_devices": secrets.randbelow(1000) + 500,
                "online_devices": secrets.randbelow(800) + 400,
                "offline_devices": secrets.randbelow(100) + 50,
                "error_devices": secrets.randbelow(20) + 5,
                "battery_low_devices": secrets.randbelow(50) + 10,
                "signal_weak_devices": secrets.randbelow(30) + 5
            }
            
            self.device_health_stats = health_checks
            
            logger.info(f"Health check completed: {health_checks['online_devices']}/{health_checks['total_devices']} devices online")
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")

    # Public API methods
    async def get_device_configurations(self, device_id: Optional[str] = None,
                                      config_type: Optional[ConfigurationType] = None) -> List[Dict[str, Any]]:
        """Get device configurations"""
        try:
            configs = list(self.device_configurations.values())
            
            # Filter by device ID
            if device_id:
                configs = [c for c in configs if c.device_id == device_id]
            
            # Filter by config type
            if config_type:
                configs = [c for c in configs if c.config_type == config_type]
            
            return [
                {
                    "config_id": config.config_id,
                    "device_id": config.device_id,
                    "config_type": config.config_type.value,
                    "parameters": config.parameters,
                    "version": config.version,
                    "is_active": config.is_active,
                    "created_at": config.created_at.isoformat(),
                    "updated_at": config.updated_at.isoformat(),
                    "applied_at": config.applied_at.isoformat() if config.applied_at else None,
                    "metadata": config.metadata
                }
                for config in configs
            ]
            
        except Exception as e:
            logger.error(f"Error getting device configurations: {e}")
            return []

    async def get_device_firmware(self, device_type: Optional[DeviceType] = None,
                                stable_only: bool = False) -> List[Dict[str, Any]]:
        """Get device firmware"""
        try:
            firmware = list(self.device_firmware.values())
            
            # Filter by device type
            if device_type:
                firmware = [f for f in firmware if f.device_type == device_type]
            
            # Filter by stability
            if stable_only:
                firmware = [f for f in firmware if f.is_stable]
            
            return [
                {
                    "firmware_id": fw.firmware_id,
                    "device_type": fw.device_type.value,
                    "version": fw.version,
                    "file_path": fw.file_path,
                    "file_size": fw.file_size,
                    "checksum": fw.checksum,
                    "release_notes": fw.release_notes,
                    "is_stable": fw.is_stable,
                    "created_at": fw.created_at.isoformat(),
                    "metadata": fw.metadata
                }
                for fw in firmware
            ]
            
        except Exception as e:
            logger.error(f"Error getting device firmware: {e}")
            return []

    async def get_device_updates(self, device_id: Optional[str] = None,
                               status: Optional[str] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get device updates"""
        try:
            updates = self.device_updates
            
            # Filter by device ID
            if device_id:
                updates = [u for u in updates if u.device_id == device_id]
            
            # Filter by status
            if status:
                updates = [u for u in updates if u.status == status]
            
            # Sort by created time (most recent first)
            updates = sorted(updates, key=lambda x: x.created_at, reverse=True)
            
            # Limit results
            updates = updates[:limit]
            
            return [
                {
                    "update_id": update.update_id,
                    "device_id": update.device_id,
                    "update_type": update.update_type,
                    "target_version": update.target_version,
                    "current_version": update.current_version,
                    "status": update.status,
                    "progress": update.progress,
                    "created_at": update.created_at.isoformat(),
                    "started_at": update.started_at.isoformat() if update.started_at else None,
                    "completed_at": update.completed_at.isoformat() if update.completed_at else None,
                    "error_message": update.error_message,
                    "metadata": update.metadata
                }
                for update in updates
            ]
            
        except Exception as e:
            logger.error(f"Error getting device updates: {e}")
            return []

    async def get_device_alerts(self, device_id: Optional[str] = None,
                              severity: Optional[str] = None,
                              acknowledged: Optional[bool] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get device alerts"""
        try:
            alerts = self.device_alerts
            
            # Filter by device ID
            if device_id:
                alerts = [a for a in alerts if a.device_id == device_id]
            
            # Filter by severity
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            # Filter by acknowledged status
            if acknowledged is not None:
                alerts = [a for a in alerts if a.is_acknowledged == acknowledged]
            
            # Sort by timestamp (most recent first)
            alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)
            
            # Limit results
            alerts = alerts[:limit]
            
            return [
                {
                    "alert_id": alert.alert_id,
                    "device_id": alert.device_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "is_acknowledged": alert.is_acknowledged,
                    "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    "acknowledged_by": alert.acknowledged_by,
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                    "metadata": alert.metadata
                }
                for alert in alerts
            ]
            
        except Exception as e:
            logger.error(f"Error getting device alerts: {e}")
            return []

    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        try:
            return {
                "engine_id": self.engine_id,
                "is_running": self.is_running,
                "total_configurations": len(self.device_configurations),
                "total_firmware": len(self.device_firmware),
                "total_updates": len(self.device_updates),
                "total_alerts": len(self.device_alerts),
                "pending_updates": len([u for u in self.device_updates if u.status == "pending"]),
                "failed_updates": len([u for u in self.device_updates if u.status == "failed"]),
                "unacknowledged_alerts": len([a for a in self.device_alerts if not a.is_acknowledged]),
                "device_health_stats": self.device_health_stats,
                "supported_device_types": [dt.value for dt in DeviceType],
                "supported_config_types": [ct.value for ct in ConfigurationType],
                "uptime": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting engine metrics: {e}")
            return {}

    async def create_device_update(self, device_id: str, update_type: str,
                                 target_version: str, current_version: str) -> str:
        """Create a device update"""
        try:
            update = DeviceUpdate(
                update_id=f"update_{secrets.token_hex(8)}",
                device_id=device_id,
                update_type=update_type,
                target_version=target_version,
                current_version=current_version,
                status="pending"
            )
            
            self.device_updates.append(update)
            
            logger.info(f"Created device update: {update.update_id}")
            return update.update_id
            
        except Exception as e:
            logger.error(f"Error creating device update: {e}")
            raise

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a device alert"""
        try:
            alert = next((a for a in self.device_alerts if a.alert_id == alert_id), None)
            if not alert:
                return False
            
            alert.is_acknowledged = True
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Acknowledged alert: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False

    async def update_device_configuration(self, config_id: str, parameters: Dict[str, Any]) -> bool:
        """Update device configuration"""
        try:
            if config_id not in self.device_configurations:
                return False
            
            config = self.device_configurations[config_id]
            config.parameters.update(parameters)
            config.updated_at = datetime.now()
            config.applied_at = None  # Reset applied status
            
            logger.info(f"Updated device configuration: {config_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating device configuration {config_id}: {e}")
            return False

# Global instance
iot_device_management_engine = IoTDeviceManagementEngine()
