"""
Immersive Content API Endpoints
REST API for the Immersive Content Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

from app.services.immersive_content_engine import (
    immersive_content_engine,
    ContentType,
    MediaFormat,
    AudioType,
    QualityLevel
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class ImmersiveContentResponse(BaseModel):
    content_id: str
    name: str
    content_type: str
    media_format: str
    file_path: str
    file_size: int
    duration: Optional[float]
    resolution: Optional[Tuple[int, int]]
    quality_level: str
    is_optimized: bool
    compression_ratio: float
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]

class SpatialAudioResponse(BaseModel):
    audio_id: str
    name: str
    audio_type: str
    file_path: str
    position: Tuple[float, float, float]
    volume: float
    pitch: float
    loop: bool
    fade_in: float
    fade_out: float
    spatial_blend: float
    min_distance: float
    max_distance: float
    rolloff_mode: str
    created_at: str
    metadata: Dict[str, Any]

class ThreeDModelResponse(BaseModel):
    model_id: str
    name: str
    file_path: str
    vertices_count: int
    triangles_count: int
    materials_count: int
    textures_count: int
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    lod_levels: List[int]
    is_animated: bool
    animation_count: int
    created_at: str
    metadata: Dict[str, Any]

class ImmersiveSceneResponse(BaseModel):
    scene_id: str
    name: str
    description: str
    content_objects: List[str]
    spatial_audio: List[str]
    lighting_setup: Dict[str, Any]
    environment_settings: Dict[str, Any]
    performance_target: int
    is_optimized: bool
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]

class ContentOptimizationResponse(BaseModel):
    optimization_id: str
    content_id: str
    optimization_type: str
    original_size: int
    optimized_size: int
    compression_ratio: float
    quality_loss: float
    processing_time: float
    created_at: str
    metadata: Dict[str, Any]

class CreateContentRequest(BaseModel):
    name: str = Field(..., description="Content name")
    content_type: str = Field(..., description="Content type")
    media_format: str = Field(..., description="Media format")
    file_path: str = Field(..., description="File path")
    file_size: int = Field(..., description="File size in bytes")

class CreateSpatialAudioRequest(BaseModel):
    name: str = Field(..., description="Audio name")
    audio_type: str = Field(..., description="Audio type")
    file_path: str = Field(..., description="File path")
    position: Tuple[float, float, float] = Field(..., description="Audio position")

class CreateSceneRequest(BaseModel):
    name: str = Field(..., description="Scene name")
    description: str = Field(..., description="Scene description")
    content_objects: List[str] = Field(..., description="List of content object IDs")
    spatial_audio: List[str] = Field(..., description="List of spatial audio IDs")

@router.get("/content", response_model=List[ImmersiveContentResponse])
async def get_immersive_content(
    content_type: Optional[str] = None,
    media_format: Optional[str] = None,
    quality_level: Optional[str] = None,
    optimized_only: bool = False,
    limit: int = 100
):
    """Get immersive content"""
    try:
        # Validate content type
        content_type_enum = None
        if content_type:
            try:
                content_type_enum = ContentType(content_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid content type: {content_type}")
        
        # Validate media format
        media_format_enum = None
        if media_format:
            try:
                media_format_enum = MediaFormat(media_format.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid media format: {media_format}")
        
        # Validate quality level
        quality_level_enum = None
        if quality_level:
            try:
                quality_level_enum = QualityLevel(quality_level.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid quality level: {quality_level}")
        
        content = await immersive_content_engine.get_immersive_content(
            content_type=content_type_enum,
            media_format=media_format_enum,
            quality_level=quality_level_enum,
            optimized_only=optimized_only,
            limit=limit
        )
        return content
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting immersive content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/spatial-audio", response_model=List[SpatialAudioResponse])
async def get_spatial_audio(
    audio_type: Optional[str] = None,
    limit: int = 100
):
    """Get spatial audio"""
    try:
        # Validate audio type
        audio_type_enum = None
        if audio_type:
            try:
                audio_type_enum = AudioType(audio_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid audio type: {audio_type}")
        
        audio = await immersive_content_engine.get_spatial_audio(
            audio_type=audio_type_enum,
            limit=limit
        )
        return audio
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting spatial audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/3d-models", response_model=List[ThreeDModelResponse])
async def get_three_d_models(
    animated_only: bool = False,
    min_vertices: Optional[int] = None,
    limit: int = 100
):
    """Get 3D models"""
    try:
        models = await immersive_content_engine.get_three_d_models(
            animated_only=animated_only,
            min_vertices=min_vertices,
            limit=limit
        )
        return models
        
    except Exception as e:
        logger.error(f"Error getting 3D models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/scenes", response_model=List[ImmersiveSceneResponse])
async def get_immersive_scenes(
    performance_target: Optional[int] = None,
    optimized_only: bool = False,
    limit: int = 100
):
    """Get immersive scenes"""
    try:
        scenes = await immersive_content_engine.get_immersive_scenes(
            performance_target=performance_target,
            optimized_only=optimized_only,
            limit=limit
        )
        return scenes
        
    except Exception as e:
        logger.error(f"Error getting immersive scenes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimizations", response_model=List[ContentOptimizationResponse])
async def get_content_optimizations(
    content_id: Optional[str] = None,
    limit: int = 100
):
    """Get content optimizations"""
    try:
        optimizations = await immersive_content_engine.get_content_optimizations(
            content_id=content_id,
            limit=limit
        )
        return optimizations
        
    except Exception as e:
        logger.error(f"Error getting content optimizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/content", response_model=Dict[str, str])
async def create_immersive_content(content_request: CreateContentRequest):
    """Create immersive content"""
    try:
        # Validate content type
        try:
            content_type_enum = ContentType(content_request.content_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid content type: {content_request.content_type}")
        
        # Validate media format
        try:
            media_format_enum = MediaFormat(content_request.media_format.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid media format: {content_request.media_format}")
        
        content_id = await immersive_content_engine.create_immersive_content(
            name=content_request.name,
            content_type=content_type_enum,
            media_format=media_format_enum,
            file_path=content_request.file_path,
            file_size=content_request.file_size
        )
        
        return {
            "content_id": content_id,
            "status": "created",
            "message": f"Content '{content_request.name}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating immersive content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/spatial-audio", response_model=Dict[str, str])
async def create_spatial_audio(audio_request: CreateSpatialAudioRequest):
    """Create spatial audio"""
    try:
        # Validate audio type
        try:
            audio_type_enum = AudioType(audio_request.audio_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid audio type: {audio_request.audio_type}")
        
        audio_id = await immersive_content_engine.create_spatial_audio(
            name=audio_request.name,
            audio_type=audio_type_enum,
            file_path=audio_request.file_path,
            position=audio_request.position
        )
        
        return {
            "audio_id": audio_id,
            "status": "created",
            "message": f"Spatial audio '{audio_request.name}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating spatial audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scenes", response_model=Dict[str, str])
async def create_immersive_scene(scene_request: CreateSceneRequest):
    """Create immersive scene"""
    try:
        scene_id = await immersive_content_engine.create_immersive_scene(
            name=scene_request.name,
            description=scene_request.description,
            content_objects=scene_request.content_objects,
            spatial_audio=scene_request.spatial_audio
        )
        
        return {
            "scene_id": scene_id,
            "status": "created",
            "message": f"Scene '{scene_request.name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating immersive scene: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/engine-metrics")
async def get_engine_metrics():
    """Get engine metrics"""
    try:
        metrics = await immersive_content_engine.get_engine_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting engine metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-formats")
async def get_available_formats():
    """Get available content formats and types"""
    try:
        return {
            "content_types": [
                {
                    "name": content_type.value,
                    "display_name": content_type.value.replace("_", " ").title(),
                    "description": f"{content_type.value.replace('_', ' ').title()} content"
                }
                for content_type in ContentType
            ],
            "media_formats": [
                {
                    "name": media_format.value,
                    "display_name": media_format.value.upper(),
                    "description": f"{media_format.value.upper()} format"
                }
                for media_format in MediaFormat
            ],
            "audio_types": [
                {
                    "name": audio_type.value,
                    "display_name": audio_type.value.replace("_", " ").title(),
                    "description": f"{audio_type.value.replace('_', ' ').title()} audio"
                }
                for audio_type in AudioType
            ],
            "quality_levels": [
                {
                    "name": quality_level.value,
                    "display_name": quality_level.value.title(),
                    "description": f"{quality_level.value.title()} quality"
                }
                for quality_level in QualityLevel
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting available formats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_immersive_content_health():
    """Get immersive content engine health status"""
    try:
        return {
            "engine_id": immersive_content_engine.engine_id,
            "is_running": immersive_content_engine.is_running,
            "total_content": len(immersive_content_engine.immersive_content),
            "total_audio": len(immersive_content_engine.spatial_audio),
            "total_models": len(immersive_content_engine.three_d_models),
            "total_scenes": len(immersive_content_engine.immersive_scenes),
            "total_optimizations": len(immersive_content_engine.content_optimizations),
            "optimized_content": len([c for c in immersive_content_engine.immersive_content.values() if c.is_optimized]),
            "supported_content_types": [ct.value for ct in ContentType],
            "uptime": "active" if immersive_content_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting immersive content health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
