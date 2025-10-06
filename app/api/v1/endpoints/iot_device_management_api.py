"""
IoT Device Management API Endpoints
REST API for the IoT Device Management Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.iot_device_management_engine import (
    iot_device_management_engine,
    DeviceType,
    ConfigurationType,
    DeviceStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class DeviceConfigurationResponse(BaseModel):
    config_id: str
    device_id: str
    config_type: str
    parameters: Dict[str, Any]
    version: str
    is_active: bool
    created_at: str
    updated_at: str
    applied_at: Optional[str]
    metadata: Dict[str, Any]

class DeviceFirmwareResponse(BaseModel):
    firmware_id: str
    device_type: str
    version: str
    file_path: str
    file_size: int
    checksum: str
    release_notes: str
    is_stable: bool
    created_at: str
    metadata: Dict[str, Any]

class DeviceUpdateResponse(BaseModel):
    update_id: str
    device_id: str
    update_type: str
    target_version: str
    current_version: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    metadata: Dict[str, Any]

class DeviceAlertResponse(BaseModel):
    alert_id: str
    device_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: str
    is_acknowledged: bool
    acknowledged_at: Optional[str]
    acknowledged_by: Optional[str]
    resolved_at: Optional[str]
    metadata: Dict[str, Any]

class CreateUpdateRequest(BaseModel):
    device_id: str = Field(..., description="Device ID")
    update_type: str = Field(..., description="Update type")
    target_version: str = Field(..., description="Target version")
    current_version: str = Field(..., description="Current version")

class AcknowledgeAlertRequest(BaseModel):
    alert_id: str = Field(..., description="Alert ID")
    acknowledged_by: str = Field(..., description="Acknowledged by")

@router.get("/configurations", response_model=List[DeviceConfigurationResponse])
async def get_device_configurations(
    device_id: Optional[str] = None,
    config_type: Optional[str] = None
):
    """Get device configurations"""
    try:
        # Validate config type
        config_type_enum = None
        if config_type:
            try:
                config_type_enum = ConfigurationType(config_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid config type: {config_type}")
        
        configs = await iot_device_management_engine.get_device_configurations(
            device_id=device_id,
            config_type=config_type_enum
        )
        return configs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting device configurations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/firmware", response_model=List[DeviceFirmwareResponse])
async def get_device_firmware(
    device_type: Optional[str] = None,
    stable_only: bool = False
):
    """Get device firmware"""
    try:
        # Validate device type
        device_type_enum = None
        if device_type:
            try:
                device_type_enum = DeviceType(device_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid device type: {device_type}")
        
        firmware = await iot_device_management_engine.get_device_firmware(
            device_type=device_type_enum,
            stable_only=stable_only
        )
        return firmware
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting device firmware: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/updates", response_model=List[DeviceUpdateResponse])
async def get_device_updates(
    device_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Get device updates"""
    try:
        updates = await iot_device_management_engine.get_device_updates(
            device_id=device_id,
            status=status,
            limit=limit
        )
        return updates
        
    except Exception as e:
        logger.error(f"Error getting device updates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", response_model=List[DeviceAlertResponse])
async def get_device_alerts(
    device_id: Optional[str] = None,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = 100
):
    """Get device alerts"""
    try:
        alerts = await iot_device_management_engine.get_device_alerts(
            device_id=device_id,
            severity=severity,
            acknowledged=acknowledged,
            limit=limit
        )
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting device alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/updates", response_model=Dict[str, str])
async def create_device_update(update_request: CreateUpdateRequest):
    """Create a device update"""
    try:
        update_id = await iot_device_management_engine.create_device_update(
            device_id=update_request.device_id,
            update_type=update_request.update_type,
            target_version=update_request.target_version,
            current_version=update_request.current_version
        )
        
        return {
            "update_id": update_id,
            "status": "created",
            "message": f"Update for device {update_request.device_id} created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating device update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/acknowledge", response_model=Dict[str, str])
async def acknowledge_alert(acknowledge_request: AcknowledgeAlertRequest):
    """Acknowledge a device alert"""
    try:
        success = await iot_device_management_engine.acknowledge_alert(
            alert_id=acknowledge_request.alert_id,
            acknowledged_by=acknowledge_request.acknowledged_by
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "alert_id": acknowledge_request.alert_id,
            "status": "acknowledged",
            "message": "Alert acknowledged successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/configurations/{config_id}")
async def update_device_configuration(config_id: str, parameters: Dict[str, Any]):
    """Update device configuration"""
    try:
        success = await iot_device_management_engine.update_device_configuration(
            config_id=config_id,
            parameters=parameters
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        return {
            "config_id": config_id,
            "status": "updated",
            "message": "Configuration updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating device configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/engine-metrics")
async def get_engine_metrics():
    """Get engine metrics"""
    try:
        metrics = await iot_device_management_engine.get_engine_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-device-types")
async def get_available_device_types():
    """Get available device types and configuration types"""
    try:
        return {
            "device_types": [
                {
                    "name": device_type.value,
                    "display_name": device_type.value.replace("_", " ").title(),
                    "description": f"{device_type.value.replace('_', ' ').title()} device type"
                }
                for device_type in DeviceType
            ],
            "configuration_types": [
                {
                    "name": config_type.value,
                    "display_name": config_type.value.replace("_", " ").title(),
                    "description": f"{config_type.value.replace('_', ' ').title()} configuration type"
                }
                for config_type in ConfigurationType
            ],
            "device_statuses": [
                {
                    "name": status.value,
                    "display_name": status.value.replace("_", " ").title(),
                    "description": f"{status.value.replace('_', ' ').title()} device status"
                }
                for status in DeviceStatus
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting available device types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_iot_device_management_health():
    """Get IoT device management engine health status"""
    try:
        return {
            "engine_id": iot_device_management_engine.engine_id,
            "is_running": iot_device_management_engine.is_running,
            "total_configurations": len(iot_device_management_engine.device_configurations),
            "total_firmware": len(iot_device_management_engine.device_firmware),
            "total_updates": len(iot_device_management_engine.device_updates),
            "total_alerts": len(iot_device_management_engine.device_alerts),
            "pending_updates": len([u for u in iot_device_management_engine.device_updates if u.status == "pending"]),
            "unacknowledged_alerts": len([a for a in iot_device_management_engine.device_alerts if not a.is_acknowledged]),
            "supported_device_types": [dt.value for dt in DeviceType],
            "uptime": "active" if iot_device_management_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting IoT device management health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
