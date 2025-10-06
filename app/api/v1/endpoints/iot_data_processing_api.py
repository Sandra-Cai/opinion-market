"""
IoT Data Processing API Endpoints
REST API for the IoT Data Processing Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.iot_data_processing_engine import (
    iot_data_processing_engine,
    SensorType,
    DataQuality,
    ProcessingStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class IoTDeviceResponse(BaseModel):
    device_id: str
    name: str
    device_type: str
    location: str
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    sensors: List[str]
    firmware_version: str
    battery_level: Optional[float]
    signal_strength: Optional[float]
    last_seen: str
    is_online: bool
    metadata: Dict[str, Any]

class SensorDataResponse(BaseModel):
    data_id: str
    device_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: str
    quality: str
    location: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    metadata: Dict[str, Any]

class DataProcessingJobResponse(BaseModel):
    job_id: str
    device_id: str
    sensor_type: str
    processing_type: str
    input_data_count: int
    output_data: Optional[Dict[str, Any]]
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    metadata: Dict[str, Any]

class DataInsightResponse(BaseModel):
    insight_id: str
    device_id: str
    sensor_type: str
    insight_type: str
    description: str
    confidence: float
    severity: str
    timestamp: str
    data_points: List[float]
    trend: str
    anomaly_score: Optional[float]
    recommendations: List[str]
    metadata: Dict[str, Any]

class AddDeviceRequest(BaseModel):
    name: str = Field(..., description="Device name")
    device_type: str = Field(..., description="Device type")
    location: str = Field(..., description="Device location")
    latitude: Optional[float] = Field(None, description="Latitude")
    longitude: Optional[float] = Field(None, description="Longitude")
    sensors: List[str] = Field(..., description="List of sensors")

@router.get("/devices", response_model=List[IoTDeviceResponse])
async def get_devices(
    device_type: Optional[str] = None,
    location: Optional[str] = None,
    online_only: bool = False
):
    """Get IoT devices"""
    try:
        devices = await iot_data_processing_engine.get_devices(
            device_type=device_type,
            location=location,
            online_only=online_only
        )
        return devices
        
    except Exception as e:
        logger.error(f"Error getting devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sensor-data", response_model=List[SensorDataResponse])
async def get_sensor_data(
    device_id: Optional[str] = None,
    sensor_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 1000
):
    """Get sensor data"""
    try:
        # Parse time parameters
        start_datetime = None
        end_datetime = None
        
        if start_time:
            try:
                start_datetime = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_time format")
        
        if end_time:
            try:
                end_datetime = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_time format")
        
        # Validate sensor type
        sensor_type_enum = None
        if sensor_type:
            try:
                sensor_type_enum = SensorType(sensor_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid sensor type: {sensor_type}")
        
        data = await iot_data_processing_engine.get_sensor_data(
            device_id=device_id,
            sensor_type=sensor_type_enum,
            start_time=start_datetime,
            end_time=end_datetime,
            limit=limit
        )
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processing-jobs", response_model=List[DataProcessingJobResponse])
async def get_processing_jobs(
    device_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Get data processing jobs"""
    try:
        # Validate status
        status_enum = None
        if status:
            try:
                status_enum = ProcessingStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        jobs = await iot_data_processing_engine.get_processing_jobs(
            device_id=device_id,
            status=status_enum,
            limit=limit
        )
        return jobs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting processing jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights", response_model=List[DataInsightResponse])
async def get_data_insights(
    device_id: Optional[str] = None,
    sensor_type: Optional[str] = None,
    insight_type: Optional[str] = None,
    limit: int = 100
):
    """Get data insights"""
    try:
        # Validate sensor type
        sensor_type_enum = None
        if sensor_type:
            try:
                sensor_type_enum = SensorType(sensor_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid sensor type: {sensor_type}")
        
        insights = await iot_data_processing_engine.get_data_insights(
            device_id=device_id,
            sensor_type=sensor_type_enum,
            insight_type=insight_type,
            limit=limit
        )
        return insights
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/devices", response_model=Dict[str, str])
async def add_device(device_request: AddDeviceRequest):
    """Add a new IoT device"""
    try:
        device_id = await iot_data_processing_engine.add_device(
            name=device_request.name,
            device_type=device_request.device_type,
            location=device_request.location,
            latitude=device_request.latitude,
            longitude=device_request.longitude,
            sensors=device_request.sensors
        )
        
        return {
            "device_id": device_id,
            "status": "added",
            "message": f"Device '{device_request.name}' added successfully"
        }
        
    except Exception as e:
        logger.error(f"Error adding device: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/devices/{device_id}")
async def update_device(device_id: str, **kwargs):
    """Update an IoT device"""
    try:
        success = await iot_data_processing_engine.update_device(device_id, **kwargs)
        if not success:
            raise HTTPException(status_code=404, detail="Device not found")
        
        return {
            "device_id": device_id,
            "status": "updated",
            "message": "Device updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating device {device_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/engine-metrics")
async def get_engine_metrics():
    """Get engine metrics"""
    try:
        metrics = await iot_data_processing_engine.get_engine_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-sensors")
async def get_supported_sensors():
    """Get supported sensor types"""
    try:
        return {
            "sensor_types": [
                {
                    "name": sensor.value,
                    "display_name": sensor.value.replace("_", " ").title(),
                    "description": f"{sensor.value.replace('_', ' ').title()} sensor"
                }
                for sensor in SensorType
            ],
            "data_quality_levels": [
                {
                    "name": quality.value,
                    "display_name": quality.value.replace("_", " ").title(),
                    "description": f"{quality.value.replace('_', ' ').title()} data quality"
                }
                for quality in DataQuality
            ],
            "processing_statuses": [
                {
                    "name": status.value,
                    "display_name": status.value.replace("_", " ").title(),
                    "description": f"{status.value.replace('_', ' ').title()} processing status"
                }
                for status in ProcessingStatus
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting supported sensors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_iot_data_processing_health():
    """Get IoT data processing engine health status"""
    try:
        return {
            "engine_id": iot_data_processing_engine.engine_id,
            "is_running": iot_data_processing_engine.is_running,
            "total_devices": len(iot_data_processing_engine.devices),
            "online_devices": len([d for d in iot_data_processing_engine.devices.values() if d.is_online]),
            "total_sensor_data": len(iot_data_processing_engine.sensor_data),
            "total_processing_jobs": len(iot_data_processing_engine.processing_jobs),
            "total_insights": len(iot_data_processing_engine.data_insights),
            "supported_sensor_types": [st.value for st in SensorType],
            "uptime": "active" if iot_data_processing_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting IoT data processing health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
