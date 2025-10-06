"""
Immersive Content Engine
Advanced 3D content creation, spatial audio, and immersive media management
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
from decimal import Decimal
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Content types"""
    THREE_D_MODEL = "3d_model"
    TEXTURE = "texture"
    AUDIO = "audio"
    VIDEO = "video"
    ANIMATION = "animation"
    PARTICLE_SYSTEM = "particle_system"
    LIGHTING = "lighting"
    MATERIAL = "material"
    SHADER = "shader"
    SCENE = "scene"

class MediaFormat(Enum):
    """Media formats"""
    OBJ = "obj"
    FBX = "fbx"
    GLTF = "gltf"
    USDZ = "usdz"
    PNG = "png"
    JPG = "jpg"
    MP4 = "mp4"
    WEBM = "webm"
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"

class AudioType(Enum):
    """Audio types"""
    SPATIAL_AUDIO = "spatial_audio"
    AMBIENT_SOUND = "ambient_sound"
    VOICE_OVER = "voice_over"
    MUSIC = "music"
    SOUND_EFFECT = "sound_effect"
    BINAURAL_AUDIO = "binaural_audio"
    SURROUND_SOUND = "surround_sound"

class QualityLevel(Enum):
    """Quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    CINEMATIC = "cinematic"

@dataclass
class ImmersiveContent:
    """Immersive content"""
    content_id: str
    name: str
    content_type: ContentType
    media_format: MediaFormat
    file_path: str
    file_size: int
    duration: Optional[float] = None  # seconds
    resolution: Optional[Tuple[int, int]] = None
    quality_level: QualityLevel = QualityLevel.HIGH
    is_optimized: bool = False
    compression_ratio: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SpatialAudio:
    """Spatial audio"""
    audio_id: str
    name: str
    audio_type: AudioType
    file_path: str
    position: Tuple[float, float, float]
    volume: float
    pitch: float
    loop: bool = False
    fade_in: float = 0.0
    fade_out: float = 0.0
    spatial_blend: float = 1.0  # 0 = 2D, 1 = 3D
    min_distance: float = 1.0
    max_distance: float = 500.0
    rolloff_mode: str = "logarithmic"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThreeDModel:
    """3D model"""
    model_id: str
    name: str
    file_path: str
    vertices_count: int
    triangles_count: int
    materials_count: int
    textures_count: int
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    lod_levels: List[int] = field(default_factory=list)
    is_animated: bool = False
    animation_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ImmersiveScene:
    """Immersive scene"""
    scene_id: str
    name: str
    description: str
    content_objects: List[str]  # content IDs
    spatial_audio: List[str]  # audio IDs
    lighting_setup: Dict[str, Any]
    environment_settings: Dict[str, Any]
    performance_target: int  # FPS
    is_optimized: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentOptimization:
    """Content optimization"""
    optimization_id: str
    content_id: str
    optimization_type: str
    original_size: int
    optimized_size: int
    compression_ratio: float
    quality_loss: float
    processing_time: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ImmersiveContentEngine:
    """Advanced Immersive Content Engine"""
    
    def __init__(self):
        self.engine_id = f"immersive_content_{secrets.token_hex(8)}"
        self.is_running = False
        
        # Content data
        self.immersive_content: Dict[str, ImmersiveContent] = {}
        self.spatial_audio: Dict[str, SpatialAudio] = {}
        self.three_d_models: Dict[str, ThreeDModel] = {}
        self.immersive_scenes: Dict[str, ImmersiveScene] = {}
        self.content_optimizations: List[ContentOptimization] = []
        
        # Engine configurations
        self.engine_configs = {
            "max_content_size": 100 * 1024 * 1024,  # 100MB
            "supported_formats": [fmt.value for fmt in MediaFormat],
            "optimization_enabled": True,
            "auto_compression": True,
            "quality_presets": {
                QualityLevel.LOW: {"compression": 0.1, "resolution": 0.5},
                QualityLevel.MEDIUM: {"compression": 0.3, "resolution": 0.75},
                QualityLevel.HIGH: {"compression": 0.5, "resolution": 1.0},
                QualityLevel.ULTRA: {"compression": 0.7, "resolution": 1.0},
                QualityLevel.CINEMATIC: {"compression": 0.9, "resolution": 1.0}
            },
            "spatial_audio_config": {
                "max_audio_sources": 32,
                "audio_buffer_size": 4096,
                "sample_rate": 48000,
                "bit_depth": 24
            },
            "rendering_config": {
                "max_polygons": 1000000,
                "max_textures": 100,
                "max_lights": 8,
                "shadow_quality": "high"
            }
        }
        
        # Processing tasks
        self.content_processing_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.audio_processing_task: Optional[asyncio.Task] = None
        self.scene_rendering_task: Optional[asyncio.Task] = None
        self.performance_monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.processing_stats: Dict[str, List[float]] = {}
        self.optimization_stats: Dict[str, float] = {}
        self.rendering_stats: Dict[str, float] = {}
        
        logger.info(f"Immersive Content Engine {self.engine_id} initialized")

    async def start_content_engine(self):
        """Start the immersive content engine"""
        if self.is_running:
            return
        
        logger.info("Starting Immersive Content Engine...")
        
        # Initialize content data
        await self._initialize_content()
        await self._initialize_spatial_audio()
        await self._initialize_three_d_models()
        await self._initialize_scenes()
        
        # Start processing tasks
        self.is_running = True
        
        self.content_processing_task = asyncio.create_task(self._content_processing_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        self.audio_processing_task = asyncio.create_task(self._audio_processing_loop())
        self.scene_rendering_task = asyncio.create_task(self._scene_rendering_loop())
        self.performance_monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        
        logger.info("Immersive Content Engine started")

    async def stop_content_engine(self):
        """Stop the immersive content engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping Immersive Content Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.content_processing_task,
            self.optimization_task,
            self.audio_processing_task,
            self.scene_rendering_task,
            self.performance_monitoring_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Immersive Content Engine stopped")

    async def _initialize_content(self):
        """Initialize immersive content"""
        try:
            # Create mock immersive content
            content_types = list(ContentType)
            media_formats = list(MediaFormat)
            quality_levels = list(QualityLevel)
            
            for i in range(50):  # Generate 50 mock content items
                content = ImmersiveContent(
                    content_id=f"content_{secrets.token_hex(8)}",
                    name=f"Content {i+1}",
                    content_type=secrets.choice(content_types),
                    media_format=secrets.choice(media_formats),
                    file_path=f"/content/{secrets.choice(media_formats).value}/content_{i+1}.{secrets.choice(media_formats).value}",
                    file_size=random.randint(1024, 50 * 1024 * 1024),  # 1KB to 50MB
                    duration=random.uniform(1, 300) if secrets.choice([True, False]) else None,
                    resolution=(random.randint(256, 4096), random.randint(256, 4096)) if secrets.choice([True, False]) else None,
                    quality_level=secrets.choice(quality_levels),
                    is_optimized=secrets.choice([True, False]),
                    compression_ratio=random.uniform(0.1, 0.9)
                )
                
                self.immersive_content[content.content_id] = content
            
            logger.info(f"Initialized {len(self.immersive_content)} immersive content items")
            
        except Exception as e:
            logger.error(f"Error initializing content: {e}")

    async def _initialize_spatial_audio(self):
        """Initialize spatial audio"""
        try:
            # Create mock spatial audio
            audio_types = list(AudioType)
            
            for i in range(30):  # Generate 30 mock audio items
                audio = SpatialAudio(
                    audio_id=f"audio_{secrets.token_hex(8)}",
                    name=f"Spatial Audio {i+1}",
                    audio_type=secrets.choice(audio_types),
                    file_path=f"/audio/spatial_audio_{i+1}.wav",
                    position=(
                        random.uniform(-50, 50),  # x
                        random.uniform(0, 10),    # y
                        random.uniform(-50, 50)   # z
                    ),
                    volume=random.uniform(0.1, 1.0),
                    pitch=random.uniform(0.5, 2.0),
                    loop=secrets.choice([True, False]),
                    fade_in=random.uniform(0, 5),
                    fade_out=random.uniform(0, 5),
                    spatial_blend=random.uniform(0, 1),
                    min_distance=random.uniform(1, 10),
                    max_distance=random.uniform(100, 1000)
                )
                
                self.spatial_audio[audio.audio_id] = audio
            
            logger.info(f"Initialized {len(self.spatial_audio)} spatial audio items")
            
        except Exception as e:
            logger.error(f"Error initializing spatial audio: {e}")

    async def _initialize_three_d_models(self):
        """Initialize 3D models"""
        try:
            # Create mock 3D models
            model_names = [
                "Character Model", "Building Model", "Vehicle Model", "Furniture Model",
                "Nature Model", "Architecture Model", "Abstract Model", "Technical Model"
            ]
            
            for i, name in enumerate(model_names):
                for j in range(5):  # 5 models per category
                    model = ThreeDModel(
                        model_id=f"model_{secrets.token_hex(8)}",
                        name=f"{name} {j+1}",
                        file_path=f"/models/{name.lower().replace(' ', '_')}_{j+1}.gltf",
                        vertices_count=random.randint(1000, 100000),
                        triangles_count=random.randint(500, 50000),
                        materials_count=random.randint(1, 10),
                        textures_count=random.randint(0, 20),
                        bounding_box=(
                            (random.uniform(-10, -1), random.uniform(-10, -1), random.uniform(-10, -1)),
                            (random.uniform(1, 10), random.uniform(1, 10), random.uniform(1, 10))
                        ),
                        lod_levels=[100, 50, 25, 10],
                        is_animated=secrets.choice([True, False]),
                        animation_count=random.randint(0, 10) if secrets.choice([True, False]) else 0
                    )
                    
                    self.three_d_models[model.model_id] = model
            
            logger.info(f"Initialized {len(self.three_d_models)} 3D models")
            
        except Exception as e:
            logger.error(f"Error initializing 3D models: {e}")

    async def _initialize_scenes(self):
        """Initialize immersive scenes"""
        try:
            # Create mock immersive scenes
            scene_names = [
                "Virtual Office", "Fantasy Forest", "Space Station", "Underwater World",
                "Medieval Castle", "Modern City", "Desert Oasis", "Arctic Tundra"
            ]
            
            for i, name in enumerate(scene_names):
                scene = ImmersiveScene(
                    scene_id=f"scene_{secrets.token_hex(8)}",
                    name=name,
                    description=f"Immersive {name.lower()} environment",
                    content_objects=[secrets.choice(list(self.immersive_content.keys())) for _ in range(random.randint(5, 20))],
                    spatial_audio=[secrets.choice(list(self.spatial_audio.keys())) for _ in range(random.randint(2, 8))],
                    lighting_setup={
                        "ambient_light": {"color": [0.2, 0.2, 0.2], "intensity": 0.5},
                        "directional_lights": random.randint(1, 4),
                        "point_lights": random.randint(2, 8),
                        "spot_lights": random.randint(0, 3)
                    },
                    environment_settings={
                        "skybox": f"skybox_{i+1}",
                        "fog_enabled": secrets.choice([True, False]),
                        "weather_effects": secrets.choice(["none", "rain", "snow", "fog"]),
                        "time_of_day": random.uniform(0, 24)
                    },
                    performance_target=random.choice([30, 60, 90, 120])
                )
                
                self.immersive_scenes[scene.scene_id] = scene
            
            logger.info(f"Initialized {len(self.immersive_scenes)} immersive scenes")
            
        except Exception as e:
            logger.error(f"Error initializing scenes: {e}")

    async def _content_processing_loop(self):
        """Content processing loop"""
        while self.is_running:
            try:
                # Process content
                await self._process_content()
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error in content processing loop: {e}")
                await asyncio.sleep(1)

    async def _optimization_loop(self):
        """Optimization loop"""
        while self.is_running:
            try:
                # Optimize content
                await self._optimize_content()
                
                await asyncio.sleep(30)  # Optimize every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(30)

    async def _audio_processing_loop(self):
        """Audio processing loop"""
        while self.is_running:
            try:
                # Process spatial audio
                await self._process_spatial_audio()
                
                await asyncio.sleep(0.1)  # Process every 100ms
                
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                await asyncio.sleep(0.1)

    async def _scene_rendering_loop(self):
        """Scene rendering loop"""
        while self.is_running:
            try:
                # Render scenes
                await self._render_scenes()
                
                await asyncio.sleep(1/60)  # 60 FPS
                
            except Exception as e:
                logger.error(f"Error in scene rendering loop: {e}")
                await asyncio.sleep(1/60)

    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                # Monitor performance
                await self._monitor_performance()
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _process_content(self):
        """Process content"""
        try:
            # Simulate content processing
            unprocessed_content = [c for c in self.immersive_content.values() if not c.is_optimized]
            
            for content in unprocessed_content[:5]:  # Process up to 5 items
                # Simulate processing
                await asyncio.sleep(0.1)
                
                # Mark as processed
                content.is_optimized = True
                content.updated_at = datetime.now()
            
            if unprocessed_content:
                logger.info(f"Processed {min(5, len(unprocessed_content))} content items")
            
        except Exception as e:
            logger.error(f"Error processing content: {e}")

    async def _optimize_content(self):
        """Optimize content"""
        try:
            # Find content that needs optimization
            content_to_optimize = [c for c in self.immersive_content.values() 
                                 if c.file_size > self.engine_configs["max_content_size"] * 0.5]
            
            for content in content_to_optimize[:3]:  # Optimize up to 3 items
                # Simulate optimization
                original_size = content.file_size
                optimization_ratio = random.uniform(0.3, 0.7)
                optimized_size = int(original_size * optimization_ratio)
                
                # Create optimization record
                optimization = ContentOptimization(
                    optimization_id=f"opt_{secrets.token_hex(8)}",
                    content_id=content.content_id,
                    optimization_type="compression",
                    original_size=original_size,
                    optimized_size=optimized_size,
                    compression_ratio=optimization_ratio,
                    quality_loss=random.uniform(0.05, 0.2),
                    processing_time=random.uniform(1, 10)
                )
                
                self.content_optimizations.append(optimization)
                
                # Update content
                content.file_size = optimized_size
                content.compression_ratio = optimization_ratio
                content.is_optimized = True
                content.updated_at = datetime.now()
            
            if content_to_optimize:
                logger.info(f"Optimized {min(3, len(content_to_optimize))} content items")
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")

    async def _process_spatial_audio(self):
        """Process spatial audio"""
        try:
            # Simulate spatial audio processing
            active_audio = [a for a in self.spatial_audio.values() if a.volume > 0]
            
            for audio in active_audio:
                # Simulate audio processing (volume, pitch, spatial effects)
                audio.volume = max(0, min(1, audio.volume + random.uniform(-0.01, 0.01)))
                audio.pitch = max(0.1, min(3, audio.pitch + random.uniform(-0.01, 0.01)))
            
            logger.info(f"Processed {len(active_audio)} spatial audio sources")
            
        except Exception as e:
            logger.error(f"Error processing spatial audio: {e}")

    async def _render_scenes(self):
        """Render scenes"""
        try:
            # Simulate scene rendering
            active_scenes = [s for s in self.immersive_scenes.values() if s.is_optimized]
            
            for scene in active_scenes:
                # Simulate rendering metrics
                render_time = random.uniform(1, 16)  # 1-16ms
                fps = 1000 / render_time if render_time > 0 else 60
                
                # Update performance metrics
                if "render_time" not in self.rendering_stats:
                    self.rendering_stats["render_time"] = []
                self.rendering_stats["render_time"].append(render_time)
                
                if "fps" not in self.rendering_stats:
                    self.rendering_stats["fps"] = []
                self.rendering_stats["fps"].append(fps)
            
            logger.info(f"Rendered {len(active_scenes)} scenes")
            
        except Exception as e:
            logger.error(f"Error rendering scenes: {e}")

    async def _monitor_performance(self):
        """Monitor performance"""
        try:
            # Collect performance metrics
            metrics = {
                "total_content": len(self.immersive_content),
                "total_audio": len(self.spatial_audio),
                "total_models": len(self.three_d_models),
                "total_scenes": len(self.immersive_scenes),
                "optimized_content": len([c for c in self.immersive_content.values() if c.is_optimized]),
                "total_optimizations": len(self.content_optimizations),
                "average_compression_ratio": np.mean([c.compression_ratio for c in self.immersive_content.values()]) if self.immersive_content else 0,
                "average_render_time": np.mean(self.rendering_stats.get("render_time", [0])) if self.rendering_stats.get("render_time") else 0,
                "average_fps": np.mean(self.rendering_stats.get("fps", [60])) if self.rendering_stats.get("fps") else 60
            }
            
            # Store metrics
            for key, value in metrics.items():
                if key not in self.processing_stats:
                    self.processing_stats[key] = []
                self.processing_stats[key].append(value)
                
                # Keep only last 1000 measurements
                if len(self.processing_stats[key]) > 1000:
                    self.processing_stats[key] = self.processing_stats[key][-1000:]
            
            logger.info(f"Performance monitoring: {metrics['total_content']} content, {metrics['average_fps']:.1f} FPS")
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")

    # Public API methods
    async def get_immersive_content(self, content_type: Optional[ContentType] = None,
                                  media_format: Optional[MediaFormat] = None,
                                  quality_level: Optional[QualityLevel] = None,
                                  optimized_only: bool = False,
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get immersive content"""
        try:
            content = list(self.immersive_content.values())
            
            # Filter by content type
            if content_type:
                content = [c for c in content if c.content_type == content_type]
            
            # Filter by media format
            if media_format:
                content = [c for c in content if c.media_format == media_format]
            
            # Filter by quality level
            if quality_level:
                content = [c for c in content if c.quality_level == quality_level]
            
            # Filter by optimization status
            if optimized_only:
                content = [c for c in content if c.is_optimized]
            
            # Limit results
            content = content[:limit]
            
            return [
                {
                    "content_id": c.content_id,
                    "name": c.name,
                    "content_type": c.content_type.value,
                    "media_format": c.media_format.value,
                    "file_path": c.file_path,
                    "file_size": c.file_size,
                    "duration": c.duration,
                    "resolution": c.resolution,
                    "quality_level": c.quality_level.value,
                    "is_optimized": c.is_optimized,
                    "compression_ratio": c.compression_ratio,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                    "metadata": c.metadata
                }
                for c in content
            ]
            
        except Exception as e:
            logger.error(f"Error getting immersive content: {e}")
            return []

    async def get_spatial_audio(self, audio_type: Optional[AudioType] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get spatial audio"""
        try:
            audio = list(self.spatial_audio.values())
            
            # Filter by audio type
            if audio_type:
                audio = [a for a in audio if a.audio_type == audio_type]
            
            # Limit results
            audio = audio[:limit]
            
            return [
                {
                    "audio_id": a.audio_id,
                    "name": a.name,
                    "audio_type": a.audio_type.value,
                    "file_path": a.file_path,
                    "position": a.position,
                    "volume": a.volume,
                    "pitch": a.pitch,
                    "loop": a.loop,
                    "fade_in": a.fade_in,
                    "fade_out": a.fade_out,
                    "spatial_blend": a.spatial_blend,
                    "min_distance": a.min_distance,
                    "max_distance": a.max_distance,
                    "rolloff_mode": a.rolloff_mode,
                    "created_at": a.created_at.isoformat(),
                    "metadata": a.metadata
                }
                for a in audio
            ]
            
        except Exception as e:
            logger.error(f"Error getting spatial audio: {e}")
            return []

    async def get_three_d_models(self, animated_only: bool = False,
                               min_vertices: Optional[int] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Get 3D models"""
        try:
            models = list(self.three_d_models.values())
            
            # Filter by animation status
            if animated_only:
                models = [m for m in models if m.is_animated]
            
            # Filter by minimum vertices
            if min_vertices:
                models = [m for m in models if m.vertices_count >= min_vertices]
            
            # Limit results
            models = models[:limit]
            
            return [
                {
                    "model_id": m.model_id,
                    "name": m.name,
                    "file_path": m.file_path,
                    "vertices_count": m.vertices_count,
                    "triangles_count": m.triangles_count,
                    "materials_count": m.materials_count,
                    "textures_count": m.textures_count,
                    "bounding_box": m.bounding_box,
                    "lod_levels": m.lod_levels,
                    "is_animated": m.is_animated,
                    "animation_count": m.animation_count,
                    "created_at": m.created_at.isoformat(),
                    "metadata": m.metadata
                }
                for m in models
            ]
            
        except Exception as e:
            logger.error(f"Error getting 3D models: {e}")
            return []

    async def get_immersive_scenes(self, performance_target: Optional[int] = None,
                                 optimized_only: bool = False,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """Get immersive scenes"""
        try:
            scenes = list(self.immersive_scenes.values())
            
            # Filter by performance target
            if performance_target:
                scenes = [s for s in scenes if s.performance_target == performance_target]
            
            # Filter by optimization status
            if optimized_only:
                scenes = [s for s in scenes if s.is_optimized]
            
            # Limit results
            scenes = scenes[:limit]
            
            return [
                {
                    "scene_id": s.scene_id,
                    "name": s.name,
                    "description": s.description,
                    "content_objects": s.content_objects,
                    "spatial_audio": s.spatial_audio,
                    "lighting_setup": s.lighting_setup,
                    "environment_settings": s.environment_settings,
                    "performance_target": s.performance_target,
                    "is_optimized": s.is_optimized,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                    "metadata": s.metadata
                }
                for s in scenes
            ]
            
        except Exception as e:
            logger.error(f"Error getting immersive scenes: {e}")
            return []

    async def get_content_optimizations(self, content_id: Optional[str] = None,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """Get content optimizations"""
        try:
            optimizations = self.content_optimizations
            
            # Filter by content ID
            if content_id:
                optimizations = [o for o in optimizations if o.content_id == content_id]
            
            # Sort by created time (most recent first)
            optimizations = sorted(optimizations, key=lambda x: x.created_at, reverse=True)
            
            # Limit results
            optimizations = optimizations[:limit]
            
            return [
                {
                    "optimization_id": o.optimization_id,
                    "content_id": o.content_id,
                    "optimization_type": o.optimization_type,
                    "original_size": o.original_size,
                    "optimized_size": o.optimized_size,
                    "compression_ratio": o.compression_ratio,
                    "quality_loss": o.quality_loss,
                    "processing_time": o.processing_time,
                    "created_at": o.created_at.isoformat(),
                    "metadata": o.metadata
                }
                for o in optimizations
            ]
            
        except Exception as e:
            logger.error(f"Error getting content optimizations: {e}")
            return []

    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        try:
            return {
                "engine_id": self.engine_id,
                "is_running": self.is_running,
                "total_content": len(self.immersive_content),
                "total_audio": len(self.spatial_audio),
                "total_models": len(self.three_d_models),
                "total_scenes": len(self.immersive_scenes),
                "total_optimizations": len(self.content_optimizations),
                "optimized_content": len([c for c in self.immersive_content.values() if c.is_optimized]),
                "supported_content_types": [ct.value for ct in ContentType],
                "supported_media_formats": [mf.value for mf in MediaFormat],
                "supported_audio_types": [at.value for at in AudioType],
                "engine_configs": self.engine_configs,
                "uptime": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting engine metrics: {e}")
            return {}

    async def create_immersive_content(self, name: str, content_type: ContentType,
                                     media_format: MediaFormat, file_path: str,
                                     file_size: int) -> str:
        """Create immersive content"""
        try:
            content = ImmersiveContent(
                content_id=f"content_{secrets.token_hex(8)}",
                name=name,
                content_type=content_type,
                media_format=media_format,
                file_path=file_path,
                file_size=file_size
            )
            
            self.immersive_content[content.content_id] = content
            
            logger.info(f"Created immersive content: {content.content_id}")
            return content.content_id
            
        except Exception as e:
            logger.error(f"Error creating immersive content: {e}")
            raise

    async def create_spatial_audio(self, name: str, audio_type: AudioType,
                                 file_path: str, position: Tuple[float, float, float],
                                 volume: float = 1.0, pitch: float = 1.0) -> str:
        """Create spatial audio"""
        try:
            audio = SpatialAudio(
                audio_id=f"audio_{secrets.token_hex(8)}",
                name=name,
                audio_type=audio_type,
                file_path=file_path,
                position=position,
                volume=volume,
                pitch=pitch
            )
            
            self.spatial_audio[audio.audio_id] = audio
            
            logger.info(f"Created spatial audio: {audio.audio_id}")
            return audio.audio_id
            
        except Exception as e:
            logger.error(f"Error creating spatial audio: {e}")
            raise

    async def create_immersive_scene(self, name: str, description: str,
                                   content_objects: List[str], spatial_audio: List[str]) -> str:
        """Create immersive scene"""
        try:
            scene = ImmersiveScene(
                scene_id=f"scene_{secrets.token_hex(8)}",
                name=name,
                description=description,
                content_objects=content_objects,
                spatial_audio=spatial_audio,
                lighting_setup={"ambient_light": {"color": [0.2, 0.2, 0.2], "intensity": 0.5}},
                environment_settings={"skybox": "default", "fog_enabled": False},
                performance_target=60
            )
            
            self.immersive_scenes[scene.scene_id] = scene
            
            logger.info(f"Created immersive scene: {scene.scene_id}")
            return scene.scene_id
            
        except Exception as e:
            logger.error(f"Error creating immersive scene: {e}")
            raise

# Global instance
immersive_content_engine = ImmersiveContentEngine()
