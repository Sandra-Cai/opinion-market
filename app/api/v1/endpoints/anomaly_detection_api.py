"""
Anomaly Detection API Endpoints
REST API for the Anomaly Detection Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.anomaly_detection_engine import (
    anomaly_detection_engine,
    AnomalyType,
    SeverityLevel,
    DetectionMethod
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class AnomalyDetectionResponse(BaseModel):
    anomaly_id: str
    asset: str
    anomaly_type: str
    detection_method: str
    severity: str
    confidence: float
    score: float
    detected_at: str
    value: float
    expected_value: float
    deviation: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]

class AnomalyPatternResponse(BaseModel):
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    last_seen: str
    severity_distribution: Dict[str, int]
    affected_assets: List[str]
    metadata: Dict[str, Any]

class DetectionRuleResponse(BaseModel):
    rule_id: str
    name: str
    description: str
    asset: str
    method: str
    parameters: Dict[str, Any]
    threshold: float
    enabled: bool
    created_at: str

class AddDetectionRuleRequest(BaseModel):
    name: str = Field(..., description="Rule name")
    description: str = Field(..., description="Rule description")
    asset: str = Field(..., description="Asset symbol")
    method: str = Field(..., description="Detection method")
    parameters: Dict[str, Any] = Field(..., description="Method parameters")
    threshold: float = Field(..., description="Detection threshold")

@router.get("/anomalies", response_model=List[AnomalyDetectionResponse])
async def get_anomaly_detections(
    asset: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100
):
    """Get anomaly detections"""
    try:
        # Validate severity if provided
        severity_level = None
        if severity:
            try:
                severity_level = SeverityLevel(severity.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity level: {severity}")
        
        anomalies = await anomaly_detection_engine.get_anomaly_detections(
            asset=asset,
            severity=severity_level,
            limit=limit
        )
        return anomalies
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting anomaly detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patterns", response_model=List[AnomalyPatternResponse])
async def get_anomaly_patterns():
    """Get anomaly patterns"""
    try:
        patterns = await anomaly_detection_engine.get_anomaly_patterns()
        return patterns
        
    except Exception as e:
        logger.error(f"Error getting anomaly patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/detection-rules", response_model=List[DetectionRuleResponse])
async def get_detection_rules():
    """Get detection rules"""
    try:
        rules = await anomaly_detection_engine.get_detection_rules()
        return rules
        
    except Exception as e:
        logger.error(f"Error getting detection rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detection-rules", response_model=Dict[str, str])
async def add_detection_rule(rule_request: AddDetectionRuleRequest):
    """Add a new detection rule"""
    try:
        # Validate detection method
        try:
            method = DetectionMethod(rule_request.method.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid detection method: {rule_request.method}")
        
        rule_id = await anomaly_detection_engine.add_detection_rule(
            name=rule_request.name,
            description=rule_request.description,
            asset=rule_request.asset,
            method=method,
            parameters=rule_request.parameters,
            threshold=rule_request.threshold
        )
        
        return {
            "rule_id": rule_id,
            "status": "created",
            "message": f"Detection rule '{rule_request.name}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding detection rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/detection-rules/{rule_id}")
async def update_detection_rule(rule_id: str, **kwargs):
    """Update a detection rule"""
    try:
        success = await anomaly_detection_engine.update_detection_rule(rule_id, **kwargs)
        if not success:
            raise HTTPException(status_code=404, detail="Detection rule not found")
        
        return {
            "rule_id": rule_id,
            "status": "updated",
            "message": "Detection rule updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating detection rule {rule_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/engine-metrics")
async def get_engine_metrics():
    """Get engine metrics"""
    try:
        metrics = await anomaly_detection_engine.get_engine_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-methods")
async def get_available_detection_methods():
    """Get available detection methods"""
    try:
        return {
            "detection_methods": [
                {
                    "name": method.value,
                    "description": f"{method.value.replace('_', ' ').title()} detection method",
                    "enabled": True
                }
                for method in DetectionMethod
            ],
            "anomaly_types": [
                {
                    "name": anomaly_type.value,
                    "description": f"{anomaly_type.value.replace('_', ' ').title()} anomaly type"
                }
                for anomaly_type in AnomalyType
            ],
            "severity_levels": [
                {
                    "name": severity.value,
                    "description": f"{severity.value.title()} severity level"
                }
                for severity in SeverityLevel
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting available detection methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_anomaly_detection_health():
    """Get anomaly detection engine health status"""
    try:
        return {
            "engine_id": anomaly_detection_engine.engine_id,
            "is_running": anomaly_detection_engine.is_running,
            "total_anomalies": len(anomaly_detection_engine.anomaly_detections),
            "total_patterns": len(anomaly_detection_engine.anomaly_patterns),
            "total_rules": len(anomaly_detection_engine.detection_rules),
            "active_rules": len([r for r in anomaly_detection_engine.detection_rules.values() if r.enabled]),
            "detection_performance": anomaly_detection_engine.detection_performance,
            "uptime": "active" if anomaly_detection_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting anomaly detection health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
