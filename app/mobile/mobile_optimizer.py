"""
Mobile Optimizer
Optimizes content and performance for mobile devices
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re
import hashlib

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Device type enumeration"""
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"
    UNKNOWN = "unknown"


class ConnectionType(Enum):
    """Connection type enumeration"""
    WIFI = "wifi"
    CELLULAR_4G = "4g"
    CELLULAR_3G = "3g"
    CELLULAR_2G = "2g"
    UNKNOWN = "unknown"


@dataclass
class DeviceInfo:
    """Device information"""
    device_type: DeviceType
    screen_width: int
    screen_height: int
    pixel_ratio: float
    connection_type: ConnectionType
    user_agent: str
    is_touch_device: bool
    supports_webp: bool
    supports_avif: bool
    memory_limit_mb: int
    cpu_cores: int


@dataclass
class OptimizationConfig:
    """Mobile optimization configuration"""
    enable_image_compression: bool = True
    enable_lazy_loading: bool = True
    enable_code_splitting: bool = True
    enable_service_worker: bool = True
    max_image_size_kb: int = 500
    max_bundle_size_kb: int = 1000
    enable_offline_support: bool = True
    cache_strategy: str = "aggressive"
    enable_push_notifications: bool = True


class MobileOptimizer:
    """Mobile optimization engine"""
    
    def __init__(self):
        self.device_detection_cache = {}
        self.optimization_cache = {}
        self.performance_metrics = {}
        
        # Default optimization config
        self.default_config = OptimizationConfig()
        
        # Device detection patterns
        self.device_patterns = {
            DeviceType.MOBILE: [
                r'Mobile', r'Android', r'iPhone', r'iPod', r'BlackBerry',
                r'Windows Phone', r'Opera Mini', r'IEMobile'
            ],
            DeviceType.TABLET: [
                r'iPad', r'Android.*Tablet', r'Kindle', r'Silk', r'PlayBook'
            ]
        }
        
        # Connection type patterns
        self.connection_patterns = {
            ConnectionType.WIFI: [r'WiFi', r'WLAN'],
            ConnectionType.CELLULAR_4G: [r'4G', r'LTE'],
            ConnectionType.CELLULAR_3G: [r'3G', r'UMTS', r'CDMA'],
            ConnectionType.CELLULAR_2G: [r'2G', r'EDGE', r'GPRS']
        }
    
    async def detect_device(self, user_agent: str, 
                          additional_info: Optional[Dict[str, Any]] = None) -> DeviceInfo:
        """Detect device information from user agent and additional data"""
        try:
            # Check cache first
            cache_key = f"device_info_{hashlib.md5(user_agent.encode()).hexdigest()}"
            cached_info = await enhanced_cache.get(cache_key)
            if cached_info:
                return DeviceInfo(**cached_info)
            
            # Detect device type
            device_type = self._detect_device_type(user_agent)
            
            # Detect screen dimensions
            screen_width, screen_height = self._detect_screen_dimensions(
                user_agent, additional_info
            )
            
            # Detect pixel ratio
            pixel_ratio = self._detect_pixel_ratio(user_agent, additional_info)
            
            # Detect connection type
            connection_type = self._detect_connection_type(user_agent, additional_info)
            
            # Detect touch capability
            is_touch_device = self._detect_touch_capability(user_agent)
            
            # Detect image format support
            supports_webp = self._detect_webp_support(user_agent)
            supports_avif = self._detect_avif_support(user_agent)
            
            # Estimate device capabilities
            memory_limit_mb = self._estimate_memory_limit(device_type, screen_width, screen_height)
            cpu_cores = self._estimate_cpu_cores(device_type)
            
            device_info = DeviceInfo(
                device_type=device_type,
                screen_width=screen_width,
                screen_height=screen_height,
                pixel_ratio=pixel_ratio,
                connection_type=connection_type,
                user_agent=user_agent,
                is_touch_device=is_touch_device,
                supports_webp=supports_webp,
                supports_avif=supports_avif,
                memory_limit_mb=memory_limit_mb,
                cpu_cores=cpu_cores
            )
            
            # Cache device info
            await enhanced_cache.set(
                cache_key,
                device_info.__dict__,
                ttl=86400,  # 24 hours
                tags=["device_detection", device_type.value]
            )
            
            return device_info
            
        except Exception as e:
            logger.error(f"Failed to detect device: {e}")
            # Return default device info
            return DeviceInfo(
                device_type=DeviceType.UNKNOWN,
                screen_width=320,
                screen_height=568,
                pixel_ratio=1.0,
                connection_type=ConnectionType.UNKNOWN,
                user_agent=user_agent,
                is_touch_device=True,
                supports_webp=False,
                supports_avif=False,
                memory_limit_mb=512,
                cpu_cores=2
            )
    
    def _detect_device_type(self, user_agent: str) -> DeviceType:
        """Detect device type from user agent"""
        user_agent_lower = user_agent.lower()
        
        # Check for tablet patterns first
        for pattern in self.device_patterns[DeviceType.TABLET]:
            if re.search(pattern, user_agent_lower, re.IGNORECASE):
                return DeviceType.TABLET
        
        # Check for mobile patterns
        for pattern in self.device_patterns[DeviceType.MOBILE]:
            if re.search(pattern, user_agent_lower, re.IGNORECASE):
                return DeviceType.MOBILE
        
        # Default to desktop if no mobile patterns found
        return DeviceType.DESKTOP
    
    def _detect_screen_dimensions(self, user_agent: str, 
                                additional_info: Optional[Dict[str, Any]] = None) -> tuple:
        """Detect screen dimensions"""
        if additional_info and "screen_width" in additional_info and "screen_height" in additional_info:
            return additional_info["screen_width"], additional_info["screen_height"]
        
        # Extract from user agent if possible
        width_match = re.search(r'(\d{3,4})x(\d{3,4})', user_agent)
        if width_match:
            return int(width_match.group(1)), int(width_match.group(2))
        
        # Default dimensions based on device type
        device_type = self._detect_device_type(user_agent)
        if device_type == DeviceType.MOBILE:
            return 375, 667  # iPhone 6/7/8 dimensions
        elif device_type == DeviceType.TABLET:
            return 768, 1024  # iPad dimensions
        else:
            return 1920, 1080  # Desktop dimensions
    
    def _detect_pixel_ratio(self, user_agent: str, 
                          additional_info: Optional[Dict[str, Any]] = None) -> float:
        """Detect pixel ratio"""
        if additional_info and "pixel_ratio" in additional_info:
            return additional_info["pixel_ratio"]
        
        # Extract from user agent
        ratio_match = re.search(r'pixel[_-]?ratio[:\s]*(\d+\.?\d*)', user_agent, re.IGNORECASE)
        if ratio_match:
            return float(ratio_match.group(1))
        
        # Default based on device type
        device_type = self._detect_device_type(user_agent)
        if device_type == DeviceType.MOBILE:
            return 2.0  # High DPI mobile
        elif device_type == DeviceType.TABLET:
            return 1.5  # Medium DPI tablet
        else:
            return 1.0  # Standard desktop
    
    def _detect_connection_type(self, user_agent: str, 
                              additional_info: Optional[Dict[str, Any]] = None) -> ConnectionType:
        """Detect connection type"""
        if additional_info and "connection_type" in additional_info:
            return ConnectionType(additional_info["connection_type"])
        
        # Extract from user agent
        user_agent_lower = user_agent.lower()
        for connection_type, patterns in self.connection_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_agent_lower, re.IGNORECASE):
                    return connection_type
        
        return ConnectionType.UNKNOWN
    
    def _detect_touch_capability(self, user_agent: str) -> bool:
        """Detect if device supports touch"""
        touch_indicators = ['touch', 'mobile', 'android', 'iphone', 'ipad', 'tablet']
        user_agent_lower = user_agent.lower()
        
        for indicator in touch_indicators:
            if indicator in user_agent_lower:
                return True
        
        return False
    
    def _detect_webp_support(self, user_agent: str) -> bool:
        """Detect WebP image format support"""
        # Modern browsers support WebP
        modern_browsers = ['chrome', 'firefox', 'safari', 'edge']
        user_agent_lower = user_agent.lower()
        
        for browser in modern_browsers:
            if browser in user_agent_lower:
                return True
        
        return False
    
    def _detect_avif_support(self, user_agent: str) -> bool:
        """Detect AVIF image format support"""
        # Only very modern browsers support AVIF
        avif_browsers = ['chrome/9', 'firefox/93', 'safari/16']
        user_agent_lower = user_agent.lower()
        
        for browser in avif_browsers:
            if browser in user_agent_lower:
                return True
        
        return False
    
    def _estimate_memory_limit(self, device_type: DeviceType, 
                             screen_width: int, screen_height: int) -> int:
        """Estimate device memory limit in MB"""
        if device_type == DeviceType.MOBILE:
            # Mobile devices typically have 2-8GB RAM
            if screen_width >= 414:  # Large phones
                return 4096
            else:
                return 2048
        elif device_type == DeviceType.TABLET:
            # Tablets typically have 4-8GB RAM
            return 4096
        else:
            # Desktop devices
            return 8192
    
    def _estimate_cpu_cores(self, device_type: DeviceType) -> int:
        """Estimate CPU core count"""
        if device_type == DeviceType.MOBILE:
            return 4  # Modern mobile devices have 4-8 cores
        elif device_type == DeviceType.TABLET:
            return 6  # Tablets typically have 6-8 cores
        else:
            return 8  # Desktop devices
    
    async def optimize_content(self, content: Dict[str, Any], 
                             device_info: DeviceInfo) -> Dict[str, Any]:
        """Optimize content for specific device"""
        try:
            # Create optimization config based on device
            config = self._create_optimization_config(device_info)
            
            optimized_content = content.copy()
            
            # Optimize images
            if "images" in optimized_content:
                optimized_content["images"] = await self._optimize_images(
                    optimized_content["images"], device_info, config
                )
            
            # Optimize data payload
            if "data" in optimized_content:
                optimized_content["data"] = await self._optimize_data_payload(
                    optimized_content["data"], device_info, config
                )
            
            # Add mobile-specific optimizations
            optimized_content["mobile_optimizations"] = {
                "lazy_loading": config.enable_lazy_loading,
                "code_splitting": config.enable_code_splitting,
                "service_worker": config.enable_service_worker,
                "offline_support": config.enable_offline_support,
                "push_notifications": config.enable_push_notifications
            }
            
            # Add device-specific metadata
            optimized_content["device_info"] = {
                "type": device_info.device_type.value,
                "screen_width": device_info.screen_width,
                "screen_height": device_info.screen_height,
                "connection_type": device_info.connection_type.value,
                "touch_device": device_info.is_touch_device
            }
            
            return optimized_content
            
        except Exception as e:
            logger.error(f"Failed to optimize content: {e}")
            return content
    
    def _create_optimization_config(self, device_info: DeviceInfo) -> OptimizationConfig:
        """Create optimization config based on device capabilities"""
        config = OptimizationConfig()
        
        # Adjust based on device type
        if device_info.device_type == DeviceType.MOBILE:
            config.max_image_size_kb = 300
            config.max_bundle_size_kb = 500
            config.cache_strategy = "aggressive"
        elif device_info.device_type == DeviceType.TABLET:
            config.max_image_size_kb = 400
            config.max_bundle_size_kb = 750
            config.cache_strategy = "moderate"
        else:
            config.max_image_size_kb = 500
            config.max_bundle_size_kb = 1000
            config.cache_strategy = "conservative"
        
        # Adjust based on connection type
        if device_info.connection_type in [ConnectionType.CELLULAR_2G, ConnectionType.CELLULAR_3G]:
            config.max_image_size_kb = min(config.max_image_size_kb, 200)
            config.max_bundle_size_kb = min(config.max_bundle_size_kb, 300)
            config.enable_image_compression = True
        elif device_info.connection_type == ConnectionType.WIFI:
            # Can be more generous with WiFi
            config.max_image_size_kb = min(config.max_image_size_kb * 1.5, 800)
            config.max_bundle_size_kb = min(config.max_bundle_size_kb * 1.5, 1500)
        
        # Adjust based on memory
        if device_info.memory_limit_mb < 2048:
            config.cache_strategy = "minimal"
            config.enable_offline_support = False
        
        return config
    
    async def _optimize_images(self, images: List[Dict[str, Any]], 
                             device_info: DeviceInfo, 
                             config: OptimizationConfig) -> List[Dict[str, Any]]:
        """Optimize images for device"""
        optimized_images = []
        
        for image in images:
            optimized_image = image.copy()
            
            # Choose best image format
            if device_info.supports_avif and config.enable_image_compression:
                optimized_image["format"] = "avif"
                optimized_image["quality"] = 80
            elif device_info.supports_webp and config.enable_image_compression:
                optimized_image["format"] = "webp"
                optimized_image["quality"] = 85
            else:
                optimized_image["format"] = "jpeg"
                optimized_image["quality"] = 90
            
            # Resize based on screen dimensions
            if "width" in image and "height" in image:
                max_width = min(device_info.screen_width, config.max_image_size_kb * 2)
                max_height = min(device_info.screen_height, config.max_image_size_kb * 2)
                
                if image["width"] > max_width or image["height"] > max_height:
                    # Calculate new dimensions maintaining aspect ratio
                    aspect_ratio = image["width"] / image["height"]
                    if image["width"] > max_width:
                        optimized_image["width"] = max_width
                        optimized_image["height"] = int(max_width / aspect_ratio)
                    else:
                        optimized_image["height"] = max_height
                        optimized_image["width"] = int(max_height * aspect_ratio)
            
            # Add lazy loading
            if config.enable_lazy_loading:
                optimized_image["lazy_loading"] = True
            
            optimized_images.append(optimized_image)
        
        return optimized_images
    
    async def _optimize_data_payload(self, data: Any, 
                                   device_info: DeviceInfo, 
                                   config: OptimizationConfig) -> Any:
        """Optimize data payload for device"""
        # For mobile devices, limit data size
        if device_info.device_type == DeviceType.MOBILE:
            # Convert to JSON to check size
            json_data = json.dumps(data)
            max_size = config.max_bundle_size_kb * 1024  # Convert to bytes
            
            if len(json_data) > max_size:
                # Truncate or paginate data
                if isinstance(data, list):
                    # Paginate lists
                    page_size = max(1, len(data) // 2)
                    return data[:page_size]
                elif isinstance(data, dict):
                    # Keep only essential fields
                    essential_fields = ["id", "title", "description", "status"]
                    return {k: v for k, v in data.items() if k in essential_fields}
        
        return data
    
    async def get_performance_metrics(self, device_info: DeviceInfo) -> Dict[str, Any]:
        """Get performance metrics for device"""
        try:
            cache_key = f"perf_metrics_{device_info.device_type.value}"
            cached_metrics = await enhanced_cache.get(cache_key)
            
            if cached_metrics:
                return cached_metrics
            
            # Calculate performance metrics
            metrics = {
                "device_type": device_info.device_type.value,
                "estimated_load_time": self._estimate_load_time(device_info),
                "recommended_cache_ttl": self._get_recommended_cache_ttl(device_info),
                "max_concurrent_requests": self._get_max_concurrent_requests(device_info),
                "recommended_batch_size": self._get_recommended_batch_size(device_info),
                "memory_usage_estimate": self._estimate_memory_usage(device_info),
                "battery_impact": self._estimate_battery_impact(device_info)
            }
            
            # Cache metrics
            await enhanced_cache.set(
                cache_key,
                metrics,
                ttl=3600,
                tags=["performance", "mobile", device_info.device_type.value]
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    def _estimate_load_time(self, device_info: DeviceInfo) -> float:
        """Estimate page load time in seconds"""
        base_time = 2.0  # Base load time
        
        # Adjust based on device type
        if device_info.device_type == DeviceType.MOBILE:
            base_time *= 1.5
        elif device_info.device_type == DeviceType.TABLET:
            base_time *= 1.2
        
        # Adjust based on connection
        if device_info.connection_type == ConnectionType.CELLULAR_2G:
            base_time *= 3.0
        elif device_info.connection_type == ConnectionType.CELLULAR_3G:
            base_time *= 2.0
        elif device_info.connection_type == ConnectionType.CELLULAR_4G:
            base_time *= 1.2
        elif device_info.connection_type == ConnectionType.WIFI:
            base_time *= 0.8
        
        return base_time
    
    def _get_recommended_cache_ttl(self, device_info: DeviceInfo) -> int:
        """Get recommended cache TTL in seconds"""
        if device_info.device_type == DeviceType.MOBILE:
            return 1800  # 30 minutes
        elif device_info.device_type == DeviceType.TABLET:
            return 3600  # 1 hour
        else:
            return 7200  # 2 hours
    
    def _get_max_concurrent_requests(self, device_info: DeviceInfo) -> int:
        """Get maximum concurrent requests for device"""
        if device_info.device_type == DeviceType.MOBILE:
            return 3
        elif device_info.device_type == DeviceType.TABLET:
            return 5
        else:
            return 10
    
    def _get_recommended_batch_size(self, device_info: DeviceInfo) -> int:
        """Get recommended batch size for data requests"""
        if device_info.device_type == DeviceType.MOBILE:
            return 10
        elif device_info.device_type == DeviceType.TABLET:
            return 20
        else:
            return 50
    
    def _estimate_memory_usage(self, device_info: DeviceInfo) -> Dict[str, Any]:
        """Estimate memory usage for different operations"""
        return {
            "page_load_mb": min(50, device_info.memory_limit_mb * 0.1),
            "cache_mb": min(100, device_info.memory_limit_mb * 0.2),
            "max_images_mb": min(200, device_info.memory_limit_mb * 0.3)
        }
    
    def _estimate_battery_impact(self, device_info: DeviceInfo) -> str:
        """Estimate battery impact level"""
        if device_info.device_type == DeviceType.MOBILE:
            if device_info.connection_type in [ConnectionType.CELLULAR_2G, ConnectionType.CELLULAR_3G]:
                return "high"
            else:
                return "medium"
        elif device_info.device_type == DeviceType.TABLET:
            return "low"
        else:
            return "minimal"


# Global mobile optimizer instance
mobile_optimizer = MobileOptimizer()
