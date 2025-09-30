"""
Mobile Optimization Engine
Advanced mobile optimization and Progressive Web App features
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import hashlib
import gzip
import base64

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
    OFFLINE = "offline"
    UNKNOWN = "unknown"


@dataclass
class DeviceInfo:
    """Device information data structure"""
    device_id: str
    device_type: DeviceType
    user_agent: str
    screen_width: int
    screen_height: int
    pixel_ratio: float
    connection_type: ConnectionType
    is_touch_device: bool
    browser: str
    os: str
    capabilities: Dict[str, bool] = field(default_factory=dict)
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class MobileOptimization:
    """Mobile optimization data structure"""
    optimization_id: str
    device_id: str
    optimization_type: str
    original_size: int
    optimized_size: int
    compression_ratio: float
    optimization_time: float
    applied_at: datetime


@dataclass
class PWAManifest:
    """PWA manifest data structure"""
    name: str
    short_name: str
    description: str
    start_url: str
    display: str
    orientation: str
    theme_color: str
    background_color: str
    icons: List[Dict[str, Any]]
    categories: List[str]
    lang: str
    scope: str


class MobileOptimizationEngine:
    """Mobile Optimization Engine for enhanced mobile experience"""
    
    def __init__(self):
        self.devices: Dict[str, DeviceInfo] = {}
        self.optimizations: List[MobileOptimization] = []
        self.pwa_manifests: Dict[str, PWAManifest] = {}
        
        # Configuration
        self.config = {
            "image_compression_quality": 80,
            "max_image_width": 1920,
            "max_image_height": 1080,
            "enable_lazy_loading": True,
            "enable_service_worker": True,
            "cache_strategy": "cache_first",
            "offline_fallback": True,
            "push_notifications": True,
            "background_sync": True,
            "max_cache_size": 50 * 1024 * 1024,  # 50MB
            "cache_expiry": 86400 * 7,  # 7 days
            "compression_threshold": 1024  # 1KB
        }
        
        # Optimization strategies
        self.optimization_strategies = {
            "image": {
                "webp_conversion": True,
                "responsive_images": True,
                "lazy_loading": True,
                "compression": True
            },
            "css": {
                "minification": True,
                "critical_css": True,
                "unused_css_removal": True,
                "css_purging": True
            },
            "javascript": {
                "minification": True,
                "tree_shaking": True,
                "code_splitting": True,
                "lazy_loading": True
            },
            "html": {
                "minification": True,
                "critical_path": True,
                "preloading": True,
                "resource_hints": True
            }
        }
        
        # PWA features
        self.pwa_features = {
            "installable": True,
            "offline_support": True,
            "push_notifications": True,
            "background_sync": True,
            "app_shortcuts": True,
            "share_target": True,
            "file_handling": True
        }
        
        # Monitoring
        self.mobile_active = False
        self.mobile_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.mobile_stats = {
            "devices_optimized": 0,
            "optimizations_applied": 0,
            "bandwidth_saved": 0,
            "load_time_improvements": 0,
            "pwa_installs": 0,
            "offline_usage": 0
        }
        
        # Initialize default PWA manifest
        self._initialize_default_manifest()
        
    def _initialize_default_manifest(self):
        """Initialize default PWA manifest"""
        default_manifest = PWAManifest(
            name="Opinion Market Platform",
            short_name="OpinionMarket",
            description="Advanced opinion trading and market analysis platform",
            start_url="/",
            display="standalone",
            orientation="portrait-primary",
            theme_color="#1a365d",
            background_color="#ffffff",
            icons=[
                {
                    "src": "/static/icons/icon-192x192.png",
                    "sizes": "192x192",
                    "type": "image/png",
                    "purpose": "any maskable"
                },
                {
                    "src": "/static/icons/icon-512x512.png",
                    "sizes": "512x512",
                    "type": "image/png",
                    "purpose": "any maskable"
                }
            ],
            categories=["finance", "business", "productivity"],
            lang="en",
            scope="/"
        )
        
        self.pwa_manifests["default"] = default_manifest
        
    async def start_mobile_optimization(self):
        """Start the mobile optimization engine"""
        if self.mobile_active:
            logger.warning("Mobile optimization already active")
            return
            
        self.mobile_active = True
        self.mobile_task = asyncio.create_task(self._mobile_optimization_loop())
        logger.info("Mobile Optimization Engine started")
        
    async def stop_mobile_optimization(self):
        """Stop the mobile optimization engine"""
        self.mobile_active = False
        if self.mobile_task:
            self.mobile_task.cancel()
            try:
                await self.mobile_task
            except asyncio.CancelledError:
                pass
        logger.info("Mobile Optimization Engine stopped")
        
    async def _mobile_optimization_loop(self):
        """Main mobile optimization loop"""
        while self.mobile_active:
            try:
                # Optimize images
                await self._optimize_images()
                
                # Optimize CSS
                await self._optimize_css()
                
                # Optimize JavaScript
                await self._optimize_javascript()
                
                # Update service worker
                await self._update_service_worker()
                
                # Clean up old optimizations
                await self._cleanup_old_optimizations()
                
                # Wait before next cycle
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in mobile optimization loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def register_device(self, device_info: DeviceInfo):
        """Register a new device"""
        try:
            self.devices[device_info.device_id] = device_info
            
            # Store in cache
            await enhanced_cache.set(
                f"device_{device_info.device_id}",
                device_info,
                ttl=86400 * 30  # 30 days
            )
            
            logger.info(f"Device registered: {device_info.device_id}")
            
        except Exception as e:
            logger.error(f"Error registering device: {e}")
            
    async def optimize_content(self, content: str, content_type: str, device_id: str) -> Dict[str, Any]:
        """Optimize content for a specific device"""
        try:
            start_time = time.time()
            
            # Get device info
            device_info = self.devices.get(device_id)
            if not device_info:
                return {"optimized_content": content, "optimization_applied": False}
                
            # Apply optimizations based on content type
            optimized_content = content
            optimizations_applied = []
            
            if content_type == "image":
                optimized_content, optimizations = await self._optimize_image_content(content, device_info)
                optimizations_applied.extend(optimizations)
                
            elif content_type == "css":
                optimized_content, optimizations = await self._optimize_css_content(content, device_info)
                optimizations_applied.extend(optimizations)
                
            elif content_type == "javascript":
                optimized_content, optimizations = await self._optimize_js_content(content, device_info)
                optimizations_applied.extend(optimizations)
                
            elif content_type == "html":
                optimized_content, optimizations = await self._optimize_html_content(content, device_info)
                optimizations_applied.extend(optimizations)
                
            # Calculate optimization metrics
            optimization_time = time.time() - start_time
            original_size = len(content.encode('utf-8'))
            optimized_size = len(optimized_content.encode('utf-8'))
            compression_ratio = (original_size - optimized_size) / original_size if original_size > 0 else 0
            
            # Record optimization
            optimization = MobileOptimization(
                optimization_id=f"opt_{int(time.time())}_{device_id}",
                device_id=device_id,
                optimization_type=content_type,
                original_size=original_size,
                optimized_size=optimized_size,
                compression_ratio=compression_ratio,
                optimization_time=optimization_time,
                applied_at=datetime.now()
            )
            
            self.optimizations.append(optimization)
            self.mobile_stats["optimizations_applied"] += 1
            self.mobile_stats["bandwidth_saved"] += (original_size - optimized_size)
            
            return {
                "optimized_content": optimized_content,
                "optimization_applied": True,
                "compression_ratio": compression_ratio,
                "optimization_time": optimization_time,
                "optimizations_applied": optimizations_applied
            }
            
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            return {"optimized_content": content, "optimization_applied": False, "error": str(e)}
            
    async def _optimize_image_content(self, content: str, device_info: DeviceInfo) -> Tuple[str, List[str]]:
        """Optimize image content"""
        try:
            optimizations = []
            
            # For demo purposes, we'll simulate image optimization
            # In a real implementation, you would use libraries like Pillow, OpenCV, etc.
            
            # Simulate WebP conversion
            if self.optimization_strategies["image"]["webp_conversion"]:
                optimizations.append("webp_conversion")
                
            # Simulate responsive image sizing
            if self.optimization_strategies["image"]["responsive_images"]:
                optimizations.append("responsive_images")
                
            # Simulate compression
            if self.optimization_strategies["image"]["compression"]:
                optimizations.append("compression")
                
            return content, optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing image content: {e}")
            return content, []
            
    async def _optimize_css_content(self, content: str, device_info: DeviceInfo) -> Tuple[str, List[str]]:
        """Optimize CSS content"""
        try:
            optimizations = []
            optimized_content = content
            
            # Minification
            if self.optimization_strategies["css"]["minification"]:
                optimized_content = await self._minify_css(optimized_content)
                optimizations.append("minification")
                
            # Critical CSS extraction
            if self.optimization_strategies["css"]["critical_css"]:
                optimized_content = await self._extract_critical_css(optimized_content, device_info)
                optimizations.append("critical_css")
                
            # Unused CSS removal
            if self.optimization_strategies["css"]["unused_css_removal"]:
                optimized_content = await self._remove_unused_css(optimized_content)
                optimizations.append("unused_css_removal")
                
            return optimized_content, optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing CSS content: {e}")
            return content, []
            
    async def _optimize_js_content(self, content: str, device_info: DeviceInfo) -> Tuple[str, List[str]]:
        """Optimize JavaScript content"""
        try:
            optimizations = []
            optimized_content = content
            
            # Minification
            if self.optimization_strategies["javascript"]["minification"]:
                optimized_content = await self._minify_js(optimized_content)
                optimizations.append("minification")
                
            # Tree shaking
            if self.optimization_strategies["javascript"]["tree_shaking"]:
                optimized_content = await self._tree_shake_js(optimized_content)
                optimizations.append("tree_shaking")
                
            # Code splitting
            if self.optimization_strategies["javascript"]["code_splitting"]:
                optimized_content = await self._split_js_code(optimized_content)
                optimizations.append("code_splitting")
                
            return optimized_content, optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing JavaScript content: {e}")
            return content, []
            
    async def _optimize_html_content(self, content: str, device_info: DeviceInfo) -> Tuple[str, List[str]]:
        """Optimize HTML content"""
        try:
            optimizations = []
            optimized_content = content
            
            # Minification
            if self.optimization_strategies["html"]["minification"]:
                optimized_content = await self._minify_html(optimized_content)
                optimizations.append("minification")
                
            # Critical path optimization
            if self.optimization_strategies["html"]["critical_path"]:
                optimized_content = await self._optimize_critical_path(optimized_content)
                optimizations.append("critical_path")
                
            # Resource hints
            if self.optimization_strategies["html"]["resource_hints"]:
                optimized_content = await self._add_resource_hints(optimized_content)
                optimizations.append("resource_hints")
                
            return optimized_content, optimizations
            
        except Exception as e:
            logger.error(f"Error optimizing HTML content: {e}")
            return content, []
            
    async def _minify_css(self, css_content: str) -> str:
        """Minify CSS content"""
        try:
            # Simple CSS minification (remove comments, extra whitespace)
            import re
            
            # Remove comments
            css_content = re.sub(r'/\*.*?\*/', '', css_content, flags=re.DOTALL)
            
            # Remove extra whitespace
            css_content = re.sub(r'\s+', ' ', css_content)
            css_content = re.sub(r';\s*}', '}', css_content)
            css_content = re.sub(r'{\s*', '{', css_content)
            css_content = re.sub(r';\s*', ';', css_content)
            
            return css_content.strip()
            
        except Exception as e:
            logger.error(f"Error minifying CSS: {e}")
            return css_content
            
    async def _minify_js(self, js_content: str) -> str:
        """Minify JavaScript content"""
        try:
            # Simple JavaScript minification
            import re
            
            # Remove single-line comments
            js_content = re.sub(r'//.*$', '', js_content, flags=re.MULTILINE)
            
            # Remove multi-line comments
            js_content = re.sub(r'/\*.*?\*/', '', js_content, flags=re.DOTALL)
            
            # Remove extra whitespace
            js_content = re.sub(r'\s+', ' ', js_content)
            
            return js_content.strip()
            
        except Exception as e:
            logger.error(f"Error minifying JavaScript: {e}")
            return js_content
            
    async def _minify_html(self, html_content: str) -> str:
        """Minify HTML content"""
        try:
            # Simple HTML minification
            import re
            
            # Remove HTML comments
            html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
            
            # Remove extra whitespace
            html_content = re.sub(r'\s+', ' ', html_content)
            
            return html_content.strip()
            
        except Exception as e:
            logger.error(f"Error minifying HTML: {e}")
            return html_content
            
    async def _extract_critical_css(self, css_content: str, device_info: DeviceInfo) -> str:
        """Extract critical CSS for above-the-fold content"""
        try:
            # This is a simplified implementation
            # In a real implementation, you would use tools like critical, penthouse, etc.
            
            # For demo purposes, return the first 1000 characters as "critical"
            return css_content[:1000]
            
        except Exception as e:
            logger.error(f"Error extracting critical CSS: {e}")
            return css_content
            
    async def _remove_unused_css(self, css_content: str) -> str:
        """Remove unused CSS rules"""
        try:
            # This is a simplified implementation
            # In a real implementation, you would use tools like PurgeCSS, UnCSS, etc.
            
            # For demo purposes, return the content as-is
            return css_content
            
        except Exception as e:
            logger.error(f"Error removing unused CSS: {e}")
            return css_content
            
    async def _tree_shake_js(self, js_content: str) -> str:
        """Remove unused JavaScript code"""
        try:
            # This is a simplified implementation
            # In a real implementation, you would use tools like Webpack, Rollup, etc.
            
            # For demo purposes, return the content as-is
            return js_content
            
        except Exception as e:
            logger.error(f"Error tree shaking JavaScript: {e}")
            return js_content
            
    async def _split_js_code(self, js_content: str) -> str:
        """Split JavaScript code into chunks"""
        try:
            # This is a simplified implementation
            # In a real implementation, you would use tools like Webpack, Rollup, etc.
            
            # For demo purposes, return the content as-is
            return js_content
            
        except Exception as e:
            logger.error(f"Error splitting JavaScript code: {e}")
            return js_content
            
    async def _optimize_critical_path(self, html_content: str) -> str:
        """Optimize critical rendering path"""
        try:
            # This is a simplified implementation
            # In a real implementation, you would inline critical CSS, defer non-critical resources, etc.
            
            # For demo purposes, return the content as-is
            return html_content
            
        except Exception as e:
            logger.error(f"Error optimizing critical path: {e}")
            return html_content
            
    async def _add_resource_hints(self, html_content: str) -> str:
        """Add resource hints for better performance"""
        try:
            # Add preload hints for critical resources
            preload_hints = [
                '<link rel="preload" href="/static/css/critical.css" as="style">',
                '<link rel="preload" href="/static/js/critical.js" as="script">'
            ]
            
            # Insert hints in the head section
            if '<head>' in html_content:
                html_content = html_content.replace(
                    '<head>',
                    '<head>\n' + '\n'.join(preload_hints)
                )
                
            return html_content
            
        except Exception as e:
            logger.error(f"Error adding resource hints: {e}")
            return html_content
            
    async def _optimize_images(self):
        """Optimize images in the system"""
        try:
            # This would implement image optimization for all images in the system
            # For demo purposes, we'll just log the action
            logger.debug("Optimizing images...")
            
        except Exception as e:
            logger.error(f"Error optimizing images: {e}")
            
    async def _optimize_css(self):
        """Optimize CSS files in the system"""
        try:
            # This would implement CSS optimization for all CSS files in the system
            # For demo purposes, we'll just log the action
            logger.debug("Optimizing CSS...")
            
        except Exception as e:
            logger.error(f"Error optimizing CSS: {e}")
            
    async def _optimize_javascript(self):
        """Optimize JavaScript files in the system"""
        try:
            # This would implement JavaScript optimization for all JS files in the system
            # For demo purposes, we'll just log the action
            logger.debug("Optimizing JavaScript...")
            
        except Exception as e:
            logger.error(f"Error optimizing JavaScript: {e}")
            
    async def _update_service_worker(self):
        """Update service worker for PWA functionality"""
        try:
            # This would implement service worker updates
            # For demo purposes, we'll just log the action
            logger.debug("Updating service worker...")
            
        except Exception as e:
            logger.error(f"Error updating service worker: {e}")
            
    async def _cleanup_old_optimizations(self):
        """Clean up old optimization data"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=30)
            
            # Clean up old optimizations
            self.optimizations = [
                opt for opt in self.optimizations
                if opt.applied_at > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning up old optimizations: {e}")
            
    def get_pwa_manifest(self, manifest_id: str = "default") -> Optional[PWAManifest]:
        """Get PWA manifest"""
        return self.pwa_manifests.get(manifest_id)
        
    def update_pwa_manifest(self, manifest_id: str, manifest: PWAManifest):
        """Update PWA manifest"""
        self.pwa_manifests[manifest_id] = manifest
        logger.info(f"PWA manifest updated: {manifest_id}")
        
    def get_mobile_summary(self) -> Dict[str, Any]:
        """Get comprehensive mobile optimization summary"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "mobile_active": self.mobile_active,
                "registered_devices": len(self.devices),
                "total_optimizations": len(self.optimizations),
                "pwa_manifests": len(self.pwa_manifests),
                "stats": self.mobile_stats,
                "config": self.config,
                "optimization_strategies": self.optimization_strategies,
                "pwa_features": self.pwa_features
            }
            
        except Exception as e:
            logger.error(f"Error getting mobile summary: {e}")
            return {"error": str(e)}


# Global instance
mobile_optimization_engine = MobileOptimizationEngine()
