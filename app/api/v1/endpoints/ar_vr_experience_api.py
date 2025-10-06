"""
AR/VR Experience API Endpoints
REST API for the AR/VR Experience Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

from app.services.ar_vr_experience_engine import (
    ar_vr_experience_engine,
    ExperienceType,
    DeviceType,
    InteractionType,
    EnvironmentType
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class ARVREnvironmentResponse(BaseModel):
    environment_id: str
    name: str
    experience_type: str
    environment_type: str
    description: str
    world_size: Tuple[float, float, float]
    max_users: int
    is_active: bool
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]

class ARVRUserResponse(BaseModel):
    user_id: str
    username: str
    device_type: str
    interaction_type: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    is_connected: bool
    connected_at: str
    last_activity: str
    session_duration: float
    metadata: Dict[str, Any]

class ARVRObjectResponse(BaseModel):
    object_id: str
    name: str
    object_type: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    is_interactive: bool
    physics_enabled: bool
    collision_detection: bool
    created_at: str
    metadata: Dict[str, Any]

class ARVREventResponse(BaseModel):
    event_id: str
    event_type: str
    user_id: str
    environment_id: str
    object_id: Optional[str]
    position: Tuple[float, float, float]
    timestamp: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]

class ARVRSessionResponse(BaseModel):
    session_id: str
    user_id: str
    environment_id: str
    device_type: str
    start_time: str
    end_time: Optional[str]
    duration: float
    interactions_count: int
    objects_created: int
    events_count: int
    performance_metrics: Dict[str, float]
    metadata: Dict[str, Any]

class CreateEnvironmentRequest(BaseModel):
    name: str = Field(..., description="Environment name")
    experience_type: str = Field(..., description="Experience type")
    environment_type: str = Field(..., description="Environment type")
    description: str = Field(..., description="Environment description")
    world_size: Tuple[float, float, float] = Field(..., description="World size (width, height, depth)")
    max_users: int = Field(..., description="Maximum number of users")

class AddUserRequest(BaseModel):
    username: str = Field(..., description="Username")
    device_type: str = Field(..., description="Device type")
    interaction_type: str = Field(..., description="Interaction type")

class CreateObjectRequest(BaseModel):
    name: str = Field(..., description="Object name")
    object_type: str = Field(..., description="Object type")
    position: Tuple[float, float, float] = Field(..., description="Object position")
    rotation: Tuple[float, float, float] = Field(..., description="Object rotation")
    scale: Tuple[float, float, float] = Field(..., description="Object scale")

@router.get("/environments", response_model=List[ARVREnvironmentResponse])
async def get_environments(
    experience_type: Optional[str] = None,
    environment_type: Optional[str] = None,
    limit: int = 100
):
    """Get AR/VR environments"""
    try:
        # Validate experience type
        experience_type_enum = None
        if experience_type:
            try:
                experience_type_enum = ExperienceType(experience_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid experience type: {experience_type}")
        
        # Validate environment type
        environment_type_enum = None
        if environment_type:
            try:
                environment_type_enum = EnvironmentType(environment_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid environment type: {environment_type}")
        
        environments = await ar_vr_experience_engine.get_environments(
            experience_type=experience_type_enum,
            environment_type=environment_type_enum,
            limit=limit
        )
        return environments
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting environments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users", response_model=List[ARVRUserResponse])
async def get_users(
    device_type: Optional[str] = None,
    interaction_type: Optional[str] = None,
    connected_only: bool = False,
    limit: int = 100
):
    """Get AR/VR users"""
    try:
        # Validate device type
        device_type_enum = None
        if device_type:
            try:
                device_type_enum = DeviceType(device_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid device type: {device_type}")
        
        # Validate interaction type
        interaction_type_enum = None
        if interaction_type:
            try:
                interaction_type_enum = InteractionType(interaction_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid interaction type: {interaction_type}")
        
        users = await ar_vr_experience_engine.get_users(
            device_type=device_type_enum,
            interaction_type=interaction_type_enum,
            connected_only=connected_only,
            limit=limit
        )
        return users
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/objects", response_model=List[ARVRObjectResponse])
async def get_objects(
    object_type: Optional[str] = None,
    interactive_only: bool = False,
    physics_enabled: bool = False,
    limit: int = 100
):
    """Get AR/VR objects"""
    try:
        objects = await ar_vr_experience_engine.get_objects(
            object_type=object_type,
            interactive_only=interactive_only,
            physics_enabled=physics_enabled,
            limit=limit
        )
        return objects
        
    except Exception as e:
        logger.error(f"Error getting objects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events", response_model=List[ARVREventResponse])
async def get_events(
    user_id: Optional[str] = None,
    environment_id: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100
):
    """Get AR/VR events"""
    try:
        events = await ar_vr_experience_engine.get_events(
            user_id=user_id,
            environment_id=environment_id,
            event_type=event_type,
            limit=limit
        )
        return events
        
    except Exception as e:
        logger.error(f"Error getting events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions", response_model=List[ARVRSessionResponse])
async def get_sessions(
    user_id: Optional[str] = None,
    environment_id: Optional[str] = None,
    active_only: bool = False,
    limit: int = 100
):
    """Get AR/VR sessions"""
    try:
        sessions = await ar_vr_experience_engine.get_sessions(
            user_id=user_id,
            environment_id=environment_id,
            active_only=active_only,
            limit=limit
        )
        return sessions
        
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/environments", response_model=Dict[str, str])
async def create_environment(environment_request: CreateEnvironmentRequest):
    """Create a new AR/VR environment"""
    try:
        # Validate experience type
        try:
            experience_type_enum = ExperienceType(environment_request.experience_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid experience type: {environment_request.experience_type}")
        
        # Validate environment type
        try:
            environment_type_enum = EnvironmentType(environment_request.environment_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid environment type: {environment_request.environment_type}")
        
        environment_id = await ar_vr_experience_engine.create_environment(
            name=environment_request.name,
            experience_type=experience_type_enum,
            environment_type=environment_type_enum,
            description=environment_request.description,
            world_size=environment_request.world_size,
            max_users=environment_request.max_users
        )
        
        return {
            "environment_id": environment_id,
            "status": "created",
            "message": f"Environment '{environment_request.name}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating environment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/users", response_model=Dict[str, str])
async def add_user(user_request: AddUserRequest):
    """Add a new AR/VR user"""
    try:
        # Validate device type
        try:
            device_type_enum = DeviceType(user_request.device_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid device type: {user_request.device_type}")
        
        # Validate interaction type
        try:
            interaction_type_enum = InteractionType(user_request.interaction_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid interaction type: {user_request.interaction_type}")
        
        user_id = await ar_vr_experience_engine.add_user(
            username=user_request.username,
            device_type=device_type_enum,
            interaction_type=interaction_type_enum
        )
        
        return {
            "user_id": user_id,
            "status": "added",
            "message": f"User '{user_request.username}' added successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/objects", response_model=Dict[str, str])
async def create_object(object_request: CreateObjectRequest):
    """Create a new AR/VR object"""
    try:
        object_id = await ar_vr_experience_engine.create_object(
            name=object_request.name,
            object_type=object_request.object_type,
            position=object_request.position,
            rotation=object_request.rotation,
            scale=object_request.scale
        )
        
        return {
            "object_id": object_id,
            "status": "created",
            "message": f"Object '{object_request.name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating object: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/engine-metrics")
async def get_engine_metrics():
    """Get engine metrics"""
    try:
        metrics = await ar_vr_experience_engine.get_engine_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-types")
async def get_available_types():
    """Get available AR/VR types"""
    try:
        return {
            "experience_types": [
                {
                    "name": exp_type.value,
                    "display_name": exp_type.value.replace("_", " ").title(),
                    "description": f"{exp_type.value.replace('_', ' ').title()} experience"
                }
                for exp_type in ExperienceType
            ],
            "device_types": [
                {
                    "name": device_type.value,
                    "display_name": device_type.value.replace("_", " ").title(),
                    "description": f"{device_type.value.replace('_', ' ').title()} device"
                }
                for device_type in DeviceType
            ],
            "interaction_types": [
                {
                    "name": interaction_type.value,
                    "display_name": interaction_type.value.replace("_", " ").title(),
                    "description": f"{interaction_type.value.replace('_', ' ').title()} interaction"
                }
                for interaction_type in InteractionType
            ],
            "environment_types": [
                {
                    "name": env_type.value,
                    "display_name": env_type.value.replace("_", " ").title(),
                    "description": f"{env_type.value.replace('_', ' ').title()} environment"
                }
                for env_type in EnvironmentType
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting available types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_ar_vr_experience_health():
    """Get AR/VR experience engine health status"""
    try:
        return {
            "engine_id": ar_vr_experience_engine.engine_id,
            "is_running": ar_vr_experience_engine.is_running,
            "total_environments": len(ar_vr_experience_engine.environments),
            "total_users": len(ar_vr_experience_engine.users),
            "total_objects": len(ar_vr_experience_engine.objects),
            "total_events": len(ar_vr_experience_engine.events),
            "total_sessions": len(ar_vr_experience_engine.sessions),
            "active_users": len([u for u in ar_vr_experience_engine.users.values() if u.is_connected]),
            "active_sessions": len([s for s in ar_vr_experience_engine.sessions if s.end_time is None]),
            "supported_experience_types": [et.value for et in ExperienceType],
            "uptime": "active" if ar_vr_experience_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting AR/VR experience health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
