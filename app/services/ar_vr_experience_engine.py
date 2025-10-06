"""
AR/VR Experience Engine
Advanced Augmented Reality and Virtual Reality experience management
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

class ExperienceType(Enum):
    """Experience types"""
    VIRTUAL_REALITY = "virtual_reality"
    AUGMENTED_REALITY = "augmented_reality"
    MIXED_REALITY = "mixed_reality"
    IMMERSIVE_360 = "immersive_360"
    HOLOGRAPHIC = "holographic"
    SPATIAL_COMPUTING = "spatial_computing"

class DeviceType(Enum):
    """AR/VR device types"""
    VR_HEADSET = "vr_headset"
    AR_GLASSES = "ar_glasses"
    MIXED_REALITY_HEADSET = "mixed_reality_headset"
    MOBILE_AR = "mobile_ar"
    HOLOLENS = "hololens"
    OCULUS = "oculus"
    HTC_VIVE = "htc_vive"
    PLAYSTATION_VR = "playstation_vr"
    CARDBOARD = "cardboard"

class InteractionType(Enum):
    """Interaction types"""
    HAND_TRACKING = "hand_tracking"
    EYE_TRACKING = "eye_tracking"
    VOICE_COMMAND = "voice_command"
    GESTURE_CONTROL = "gesture_control"
    CONTROLLER = "controller"
    TOUCH = "touch"
    GAZE = "gaze"
    BRAIN_COMPUTER_INTERFACE = "brain_computer_interface"

class EnvironmentType(Enum):
    """Environment types"""
    VIRTUAL_WORLD = "virtual_world"
    REAL_WORLD_OVERLAY = "real_world_overlay"
    HYBRID_ENVIRONMENT = "hybrid_environment"
    SIMULATION = "simulation"
    GAMIFIED_EXPERIENCE = "gamified_experience"
    EDUCATIONAL_ENVIRONMENT = "educational_environment"
    TRAINING_SIMULATION = "training_simulation"

@dataclass
class ARVREnvironment:
    """AR/VR environment"""
    environment_id: str
    name: str
    experience_type: ExperienceType
    environment_type: EnvironmentType
    description: str
    world_size: Tuple[float, float, float]  # width, height, depth
    max_users: int
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ARVRUser:
    """AR/VR user"""
    user_id: str
    username: str
    device_type: DeviceType
    interaction_type: InteractionType
    position: Tuple[float, float, float]  # x, y, z
    rotation: Tuple[float, float, float]  # pitch, yaw, roll
    is_connected: bool = True
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    session_duration: float = 0.0  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ARVRObject:
    """AR/VR object"""
    object_id: str
    name: str
    object_type: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]
    is_interactive: bool = True
    physics_enabled: bool = False
    collision_detection: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ARVREvent:
    """AR/VR event"""
    event_id: str
    event_type: str
    user_id: str
    environment_id: str
    object_id: Optional[str]
    position: Tuple[float, float, float]
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ARVRSession:
    """AR/VR session"""
    session_id: str
    user_id: str
    environment_id: str
    device_type: DeviceType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: float = 0.0  # seconds
    interactions_count: int = 0
    objects_created: int = 0
    events_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ARVRExperienceEngine:
    """Advanced AR/VR Experience Engine"""
    
    def __init__(self):
        self.engine_id = f"ar_vr_experience_{secrets.token_hex(8)}"
        self.is_running = False
        
        # AR/VR data
        self.environments: Dict[str, ARVREnvironment] = {}
        self.users: Dict[str, ARVRUser] = {}
        self.objects: Dict[str, ARVRObject] = {}
        self.events: List[ARVREvent] = []
        self.sessions: List[ARVRSession] = []
        
        # Engine configurations
        self.engine_configs = {
            "max_environments": 100,
            "max_users_per_environment": 50,
            "max_objects_per_environment": 1000,
            "event_retention_hours": 24,
            "session_retention_days": 30,
            "update_frequency": 60,  # Hz
            "physics_timestep": 1/60,  # seconds
            "collision_detection_enabled": True,
            "spatial_audio_enabled": True,
            "hand_tracking_enabled": True,
            "eye_tracking_enabled": True
        }
        
        # Device configurations
        self.device_configs = {
            DeviceType.VR_HEADSET: {
                "field_of_view": 110,  # degrees
                "resolution": (2160, 1200),
                "refresh_rate": 90,  # Hz
                "tracking_accuracy": 0.1,  # mm
                "latency": 20  # ms
            },
            DeviceType.AR_GLASSES: {
                "field_of_view": 52,  # degrees
                "resolution": (1280, 720),
                "refresh_rate": 60,  # Hz
                "tracking_accuracy": 1.0,  # mm
                "latency": 30  # ms
            },
            DeviceType.MIXED_REALITY_HEADSET: {
                "field_of_view": 95,  # degrees
                "resolution": (2880, 1440),
                "refresh_rate": 90,  # Hz
                "tracking_accuracy": 0.5,  # mm
                "latency": 25  # ms
            },
            DeviceType.MOBILE_AR: {
                "field_of_view": 60,  # degrees
                "resolution": (1920, 1080),
                "refresh_rate": 60,  # Hz
                "tracking_accuracy": 2.0,  # mm
                "latency": 50  # ms
            }
        }
        
        # Processing tasks
        self.user_tracking_task: Optional[asyncio.Task] = None
        self.object_physics_task: Optional[asyncio.Task] = None
        self.event_processing_task: Optional[asyncio.Task] = None
        self.session_management_task: Optional[asyncio.Task] = None
        self.performance_monitoring_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {}
        self.user_activity_stats: Dict[str, int] = {}
        self.environment_usage_stats: Dict[str, int] = {}
        
        logger.info(f"AR/VR Experience Engine {self.engine_id} initialized")

    async def start_ar_vr_engine(self):
        """Start the AR/VR experience engine"""
        if self.is_running:
            return
        
        logger.info("Starting AR/VR Experience Engine...")
        
        # Initialize AR/VR data
        await self._initialize_environments()
        await self._initialize_users()
        await self._initialize_objects()
        
        # Start processing tasks
        self.is_running = True
        
        self.user_tracking_task = asyncio.create_task(self._user_tracking_loop())
        self.object_physics_task = asyncio.create_task(self._object_physics_loop())
        self.event_processing_task = asyncio.create_task(self._event_processing_loop())
        self.session_management_task = asyncio.create_task(self._session_management_loop())
        self.performance_monitoring_task = asyncio.create_task(self._performance_monitoring_loop())
        
        logger.info("AR/VR Experience Engine started")

    async def stop_ar_vr_engine(self):
        """Stop the AR/VR experience engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping AR/VR Experience Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.user_tracking_task,
            self.object_physics_task,
            self.event_processing_task,
            self.session_management_task,
            self.performance_monitoring_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("AR/VR Experience Engine stopped")

    async def _initialize_environments(self):
        """Initialize AR/VR environments"""
        try:
            # Create mock AR/VR environments
            environment_names = [
                "Virtual Office Space", "Augmented City Tour", "Mixed Reality Training",
                "VR Gaming Arena", "AR Shopping Experience", "Immersive Learning Lab",
                "Virtual Conference Room", "AR Navigation System", "VR Therapy Session",
                "Mixed Reality Workspace"
            ]
            
            for i, name in enumerate(environment_names):
                environment = ARVREnvironment(
                    environment_id=f"env_{secrets.token_hex(8)}",
                    name=name,
                    experience_type=secrets.choice(list(ExperienceType)),
                    environment_type=secrets.choice(list(EnvironmentType)),
                    description=f"Immersive {name.lower()} environment",
                    world_size=(
                        random.uniform(10, 100),  # width
                        random.uniform(5, 20),    # height
                        random.uniform(10, 100)   # depth
                    ),
                    max_users=random.randint(5, 50)
                )
                
                self.environments[environment.environment_id] = environment
            
            logger.info(f"Initialized {len(self.environments)} AR/VR environments")
            
        except Exception as e:
            logger.error(f"Error initializing environments: {e}")

    async def _initialize_users(self):
        """Initialize AR/VR users"""
        try:
            # Create mock AR/VR users
            usernames = [
                "VRExplorer", "ARDesigner", "MixedRealityUser", "ImmersiveGamer",
                "VirtualTourist", "AugmentedLearner", "SpatialComputingFan", "HolographicUser",
                "VRDeveloper", "ARArtist", "MixedRealityCreator", "ImmersiveEducator"
            ]
            
            for i, username in enumerate(usernames):
                user = ARVRUser(
                    user_id=f"user_{secrets.token_hex(8)}",
                    username=username,
                    device_type=secrets.choice(list(DeviceType)),
                    interaction_type=secrets.choice(list(InteractionType)),
                    position=(
                        random.uniform(-10, 10),  # x
                        random.uniform(0, 5),     # y
                        random.uniform(-10, 10)   # z
                    ),
                    rotation=(
                        random.uniform(-180, 180),  # pitch
                        random.uniform(-180, 180),  # yaw
                        random.uniform(-180, 180)   # roll
                    ),
                    connected_at=datetime.now() - timedelta(minutes=random.randint(1, 60)),
                    last_activity=datetime.now() - timedelta(seconds=random.randint(1, 300))
                )
                
                self.users[user.user_id] = user
            
            logger.info(f"Initialized {len(self.users)} AR/VR users")
            
        except Exception as e:
            logger.error(f"Error initializing users: {e}")

    async def _initialize_objects(self):
        """Initialize AR/VR objects"""
        try:
            # Create mock AR/VR objects
            object_types = [
                "cube", "sphere", "cylinder", "plane", "mesh", "light", "camera",
                "audio_source", "interactive_button", "3d_model", "particle_system",
                "texture", "material", "animation", "collider"
            ]
            
            for i in range(100):  # Generate 100 mock objects
                object_id = f"obj_{secrets.token_hex(8)}"
                obj = ARVRObject(
                    object_id=object_id,
                    name=f"Object {i+1}",
                    object_type=secrets.choice(object_types),
                    position=(
                        random.uniform(-20, 20),  # x
                        random.uniform(0, 10),    # y
                        random.uniform(-20, 20)   # z
                    ),
                    rotation=(
                        random.uniform(0, 360),   # pitch
                        random.uniform(0, 360),   # yaw
                        random.uniform(0, 360)    # roll
                    ),
                    scale=(
                        random.uniform(0.1, 5),   # x
                        random.uniform(0.1, 5),   # y
                        random.uniform(0.1, 5)    # z
                    ),
                    is_interactive=secrets.choice([True, False]),
                    physics_enabled=secrets.choice([True, False]),
                    collision_detection=secrets.choice([True, False])
                )
                
                self.objects[object_id] = obj
            
            logger.info(f"Initialized {len(self.objects)} AR/VR objects")
            
        except Exception as e:
            logger.error(f"Error initializing objects: {e}")

    async def _user_tracking_loop(self):
        """User tracking loop"""
        while self.is_running:
            try:
                # Update user positions and activities
                await self._update_user_positions()
                
                await asyncio.sleep(1/self.engine_configs["update_frequency"])
                
            except Exception as e:
                logger.error(f"Error in user tracking loop: {e}")
                await asyncio.sleep(1/self.engine_configs["update_frequency"])

    async def _object_physics_loop(self):
        """Object physics loop"""
        while self.is_running:
            try:
                # Update object physics
                await self._update_object_physics()
                
                await asyncio.sleep(self.engine_configs["physics_timestep"])
                
            except Exception as e:
                logger.error(f"Error in object physics loop: {e}")
                await asyncio.sleep(self.engine_configs["physics_timestep"])

    async def _event_processing_loop(self):
        """Event processing loop"""
        while self.is_running:
            try:
                # Process AR/VR events
                await self._process_ar_vr_events()
                
                await asyncio.sleep(0.1)  # Process every 100ms
                
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(0.1)

    async def _session_management_loop(self):
        """Session management loop"""
        while self.is_running:
            try:
                # Manage AR/VR sessions
                await self._manage_ar_vr_sessions()
                
                await asyncio.sleep(10)  # Manage every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in session management loop: {e}")
                await asyncio.sleep(10)

    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                # Monitor performance metrics
                await self._monitor_performance()
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _update_user_positions(self):
        """Update user positions and activities"""
        try:
            for user in self.users.values():
                if not user.is_connected:
                    continue
                
                # Simulate user movement
                movement_speed = random.uniform(0.1, 2.0)  # units per second
                direction = random.uniform(0, 2 * np.pi)
                
                # Update position
                user.position = (
                    user.position[0] + movement_speed * np.cos(direction) * self.engine_configs["physics_timestep"],
                    user.position[1],
                    user.position[2] + movement_speed * np.sin(direction) * self.engine_configs["physics_timestep"]
                )
                
                # Update rotation
                user.rotation = (
                    user.rotation[0] + random.uniform(-5, 5),
                    user.rotation[1] + random.uniform(-10, 10),
                    user.rotation[2] + random.uniform(-2, 2)
                )
                
                # Update last activity
                user.last_activity = datetime.now()
                
                # Update session duration
                user.session_duration += self.engine_configs["physics_timestep"]
            
            logger.info(f"Updated positions for {len([u for u in self.users.values() if u.is_connected])} active users")
            
        except Exception as e:
            logger.error(f"Error updating user positions: {e}")

    async def _update_object_physics(self):
        """Update object physics"""
        try:
            physics_objects = [obj for obj in self.objects.values() if obj.physics_enabled]
            
            for obj in physics_objects:
                # Simulate physics (gravity, movement, etc.)
                if obj.object_type in ["cube", "sphere", "cylinder"]:
                    # Apply gravity
                    obj.position = (
                        obj.position[0],
                        max(0, obj.position[1] - 9.8 * self.engine_configs["physics_timestep"]),
                        obj.position[2]
                    )
                    
                    # Add some random movement
                    obj.position = (
                        obj.position[0] + random.uniform(-0.1, 0.1),
                        obj.position[1],
                        obj.position[2] + random.uniform(-0.1, 0.1)
                    )
            
            logger.info(f"Updated physics for {len(physics_objects)} objects")
            
        except Exception as e:
            logger.error(f"Error updating object physics: {e}")

    async def _process_ar_vr_events(self):
        """Process AR/VR events"""
        try:
            # Generate mock events
            if self.users and self.environments:
                num_events = random.randint(1, 5)
                
                for _ in range(num_events):
                    user_id = secrets.choice(list(self.users.keys()))
                    environment_id = secrets.choice(list(self.environments.keys()))
                    object_id = secrets.choice(list(self.objects.keys())) if self.objects else None
                    
                    event = ARVREvent(
                        event_id=f"event_{secrets.token_hex(8)}",
                        event_type=secrets.choice(["interaction", "movement", "gaze", "gesture", "voice"]),
                        user_id=user_id,
                        environment_id=environment_id,
                        object_id=object_id,
                        position=self.users[user_id].position,
                        timestamp=datetime.now()
                    )
                    
                    self.events.append(event)
            
            # Keep only recent events
            cutoff_time = datetime.now() - timedelta(hours=self.engine_configs["event_retention_hours"])
            self.events = [e for e in self.events if e.timestamp > cutoff_time]
            
            logger.info(f"Processed {len(self.events)} AR/VR events")
            
        except Exception as e:
            logger.error(f"Error processing AR/VR events: {e}")

    async def _manage_ar_vr_sessions(self):
        """Manage AR/VR sessions"""
        try:
            # Create new sessions for connected users
            for user in self.users.values():
                if user.is_connected:
                    # Check if user has an active session
                    active_session = next((s for s in self.sessions 
                                         if s.user_id == user.user_id and s.end_time is None), None)
                    
                    if not active_session:
                        # Create new session
                        session = ARVRSession(
                            session_id=f"session_{secrets.token_hex(8)}",
                            user_id=user.user_id,
                            environment_id=secrets.choice(list(self.environments.keys())),
                            device_type=user.device_type,
                            start_time=datetime.now()
                        )
                        self.sessions.append(session)
            
            # Update session metrics
            for session in self.sessions:
                if session.end_time is None:
                    session.duration = (datetime.now() - session.start_time).total_seconds()
                    session.events_count = len([e for e in self.events if e.user_id == session.user_id])
            
            # Clean up old sessions
            cutoff_time = datetime.now() - timedelta(days=self.engine_configs["session_retention_days"])
            self.sessions = [s for s in self.sessions if s.start_time > cutoff_time]
            
            logger.info(f"Managed {len(self.sessions)} AR/VR sessions")
            
        except Exception as e:
            logger.error(f"Error managing AR/VR sessions: {e}")

    async def _monitor_performance(self):
        """Monitor performance metrics"""
        try:
            # Collect performance metrics
            metrics = {
                "active_users": len([u for u in self.users.values() if u.is_connected]),
                "total_environments": len(self.environments),
                "total_objects": len(self.objects),
                "total_events": len(self.events),
                "active_sessions": len([s for s in self.sessions if s.end_time is None]),
                "average_session_duration": np.mean([s.duration for s in self.sessions if s.duration > 0]) if self.sessions else 0,
                "events_per_second": len(self.events) / max(1, (datetime.now() - min([e.timestamp for e in self.events])).total_seconds()) if self.events else 0
            }
            
            # Store metrics
            for key, value in metrics.items():
                if key not in self.performance_metrics:
                    self.performance_metrics[key] = []
                self.performance_metrics[key].append(value)
                
                # Keep only last 1000 measurements
                if len(self.performance_metrics[key]) > 1000:
                    self.performance_metrics[key] = self.performance_metrics[key][-1000:]
            
            logger.info(f"Performance monitoring: {metrics['active_users']} users, {metrics['active_sessions']} sessions")
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {e}")

    # Public API methods
    async def get_environments(self, experience_type: Optional[ExperienceType] = None,
                             environment_type: Optional[EnvironmentType] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get AR/VR environments"""
        try:
            environments = list(self.environments.values())
            
            # Filter by experience type
            if experience_type:
                environments = [e for e in environments if e.experience_type == experience_type]
            
            # Filter by environment type
            if environment_type:
                environments = [e for e in environments if e.environment_type == environment_type]
            
            # Limit results
            environments = environments[:limit]
            
            return [
                {
                    "environment_id": env.environment_id,
                    "name": env.name,
                    "experience_type": env.experience_type.value,
                    "environment_type": env.environment_type.value,
                    "description": env.description,
                    "world_size": env.world_size,
                    "max_users": env.max_users,
                    "is_active": env.is_active,
                    "created_at": env.created_at.isoformat(),
                    "updated_at": env.updated_at.isoformat(),
                    "metadata": env.metadata
                }
                for env in environments
            ]
            
        except Exception as e:
            logger.error(f"Error getting environments: {e}")
            return []

    async def get_users(self, device_type: Optional[DeviceType] = None,
                       interaction_type: Optional[InteractionType] = None,
                       connected_only: bool = False,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Get AR/VR users"""
        try:
            users = list(self.users.values())
            
            # Filter by device type
            if device_type:
                users = [u for u in users if u.device_type == device_type]
            
            # Filter by interaction type
            if interaction_type:
                users = [u for u in users if u.interaction_type == interaction_type]
            
            # Filter by connection status
            if connected_only:
                users = [u for u in users if u.is_connected]
            
            # Limit results
            users = users[:limit]
            
            return [
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "device_type": user.device_type.value,
                    "interaction_type": user.interaction_type.value,
                    "position": user.position,
                    "rotation": user.rotation,
                    "is_connected": user.is_connected,
                    "connected_at": user.connected_at.isoformat(),
                    "last_activity": user.last_activity.isoformat(),
                    "session_duration": user.session_duration,
                    "metadata": user.metadata
                }
                for user in users
            ]
            
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []

    async def get_objects(self, object_type: Optional[str] = None,
                         interactive_only: bool = False,
                         physics_enabled: bool = False,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get AR/VR objects"""
        try:
            objects = list(self.objects.values())
            
            # Filter by object type
            if object_type:
                objects = [o for o in objects if o.object_type == object_type]
            
            # Filter by interactive status
            if interactive_only:
                objects = [o for o in objects if o.is_interactive]
            
            # Filter by physics status
            if physics_enabled:
                objects = [o for o in objects if o.physics_enabled]
            
            # Limit results
            objects = objects[:limit]
            
            return [
                {
                    "object_id": obj.object_id,
                    "name": obj.name,
                    "object_type": obj.object_type,
                    "position": obj.position,
                    "rotation": obj.rotation,
                    "scale": obj.scale,
                    "is_interactive": obj.is_interactive,
                    "physics_enabled": obj.physics_enabled,
                    "collision_detection": obj.collision_detection,
                    "created_at": obj.created_at.isoformat(),
                    "metadata": obj.metadata
                }
                for obj in objects
            ]
            
        except Exception as e:
            logger.error(f"Error getting objects: {e}")
            return []

    async def get_events(self, user_id: Optional[str] = None,
                        environment_id: Optional[str] = None,
                        event_type: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Get AR/VR events"""
        try:
            events = self.events
            
            # Filter by user ID
            if user_id:
                events = [e for e in events if e.user_id == user_id]
            
            # Filter by environment ID
            if environment_id:
                events = [e for e in events if e.environment_id == environment_id]
            
            # Filter by event type
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            # Sort by timestamp (most recent first)
            events = sorted(events, key=lambda x: x.timestamp, reverse=True)
            
            # Limit results
            events = events[:limit]
            
            return [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "user_id": event.user_id,
                    "environment_id": event.environment_id,
                    "object_id": event.object_id,
                    "position": event.position,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.data,
                    "metadata": event.metadata
                }
                for event in events
            ]
            
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            return []

    async def get_sessions(self, user_id: Optional[str] = None,
                          environment_id: Optional[str] = None,
                          active_only: bool = False,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get AR/VR sessions"""
        try:
            sessions = self.sessions
            
            # Filter by user ID
            if user_id:
                sessions = [s for s in sessions if s.user_id == user_id]
            
            # Filter by environment ID
            if environment_id:
                sessions = [s for s in sessions if s.environment_id == environment_id]
            
            # Filter by active status
            if active_only:
                sessions = [s for s in sessions if s.end_time is None]
            
            # Sort by start time (most recent first)
            sessions = sorted(sessions, key=lambda x: x.start_time, reverse=True)
            
            # Limit results
            sessions = sessions[:limit]
            
            return [
                {
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "environment_id": session.environment_id,
                    "device_type": session.device_type.value,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat() if session.end_time else None,
                    "duration": session.duration,
                    "interactions_count": session.interactions_count,
                    "objects_created": session.objects_created,
                    "events_count": session.events_count,
                    "performance_metrics": session.performance_metrics,
                    "metadata": session.metadata
                }
                for session in sessions
            ]
            
        except Exception as e:
            logger.error(f"Error getting sessions: {e}")
            return []

    async def get_engine_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        try:
            return {
                "engine_id": self.engine_id,
                "is_running": self.is_running,
                "total_environments": len(self.environments),
                "total_users": len(self.users),
                "total_objects": len(self.objects),
                "total_events": len(self.events),
                "total_sessions": len(self.sessions),
                "active_users": len([u for u in self.users.values() if u.is_connected]),
                "active_sessions": len([s for s in self.sessions if s.end_time is None]),
                "supported_experience_types": [et.value for et in ExperienceType],
                "supported_device_types": [dt.value for dt in DeviceType],
                "supported_interaction_types": [it.value for it in InteractionType],
                "engine_configs": self.engine_configs,
                "uptime": "active" if self.is_running else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error getting engine metrics: {e}")
            return {}

    async def create_environment(self, name: str, experience_type: ExperienceType,
                               environment_type: EnvironmentType, description: str,
                               world_size: Tuple[float, float, float], max_users: int) -> str:
        """Create a new AR/VR environment"""
        try:
            environment = ARVREnvironment(
                environment_id=f"env_{secrets.token_hex(8)}",
                name=name,
                experience_type=experience_type,
                environment_type=environment_type,
                description=description,
                world_size=world_size,
                max_users=max_users
            )
            
            self.environments[environment.environment_id] = environment
            
            logger.info(f"Created AR/VR environment: {environment.environment_id}")
            return environment.environment_id
            
        except Exception as e:
            logger.error(f"Error creating environment: {e}")
            raise

    async def add_user(self, username: str, device_type: DeviceType,
                      interaction_type: InteractionType) -> str:
        """Add a new AR/VR user"""
        try:
            user = ARVRUser(
                user_id=f"user_{secrets.token_hex(8)}",
                username=username,
                device_type=device_type,
                interaction_type=interaction_type,
                position=(0, 0, 0),
                rotation=(0, 0, 0)
            )
            
            self.users[user.user_id] = user
            
            logger.info(f"Added AR/VR user: {user.user_id}")
            return user.user_id
            
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            raise

    async def create_object(self, name: str, object_type: str,
                           position: Tuple[float, float, float],
                           rotation: Tuple[float, float, float],
                           scale: Tuple[float, float, float]) -> str:
        """Create a new AR/VR object"""
        try:
            object_id = f"obj_{secrets.token_hex(8)}"
            obj = ARVRObject(
                object_id=object_id,
                name=name,
                object_type=object_type,
                position=position,
                rotation=rotation,
                scale=scale
            )
            
            self.objects[object_id] = obj
            
            logger.info(f"Created AR/VR object: {object_id}")
            return object_id
            
        except Exception as e:
            logger.error(f"Error creating object: {e}")
            raise

# Global instance
ar_vr_experience_engine = ARVRExperienceEngine()
