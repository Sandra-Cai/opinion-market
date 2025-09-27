"""
Mobile Optimization API Endpoints
Provides mobile-specific features and optimizations
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import time

from app.core.auth import get_current_user
from app.models.user import User
from app.mobile.mobile_optimizer import mobile_optimizer, DeviceType, ConnectionType
from app.websocket.websocket_manager import websocket_manager, MessageType
from app.notifications.notification_manager import notification_manager

router = APIRouter()


@router.post("/device/detect")
async def detect_device(
    request: Request,
    device_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Detect device information and optimize content"""
    try:
        user_agent = request.headers.get("User-Agent", "")
        additional_info = device_data.get("additional_info", {})
        
        # Detect device
        device_info = await mobile_optimizer.detect_device(user_agent, additional_info)
        
        # Get performance metrics
        performance_metrics = await mobile_optimizer.get_performance_metrics(device_info)
        
        return {
            "success": True,
            "data": {
                "device_info": {
                    "device_type": device_info.device_type.value,
                    "screen_width": device_info.screen_width,
                    "screen_height": device_info.screen_height,
                    "pixel_ratio": device_info.pixel_ratio,
                    "connection_type": device_info.connection_type.value,
                    "is_touch_device": device_info.is_touch_device,
                    "supports_webp": device_info.supports_webp,
                    "supports_avif": device_info.supports_avif,
                    "memory_limit_mb": device_info.memory_limit_mb,
                    "cpu_cores": device_info.cpu_cores
                },
                "performance_metrics": performance_metrics,
                "optimization_recommendations": {
                    "max_image_size_kb": 300 if device_info.device_type == DeviceType.MOBILE else 500,
                    "enable_lazy_loading": True,
                    "enable_compression": device_info.connection_type != ConnectionType.WIFI,
                    "cache_strategy": "aggressive" if device_info.device_type == DeviceType.MOBILE else "moderate"
                }
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect device: {str(e)}")


@router.post("/content/optimize")
async def optimize_content(
    content_data: Dict[str, Any],
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Optimize content for mobile devices"""
    try:
        user_agent = request.headers.get("User-Agent", "")
        device_info = await mobile_optimizer.detect_device(user_agent)
        
        # Optimize content
        optimized_content = await mobile_optimizer.optimize_content(content_data, device_info)
        
        return {
            "success": True,
            "data": optimized_content,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize content: {str(e)}")


@router.get("/performance/metrics")
async def get_mobile_performance_metrics(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get mobile performance metrics"""
    try:
        user_agent = request.headers.get("User-Agent", "")
        device_info = await mobile_optimizer.detect_device(user_agent)
        
        performance_metrics = await mobile_optimizer.get_performance_metrics(device_info)
        
        return {
            "success": True,
            "data": {
                "device_info": {
                    "device_type": device_info.device_type.value,
                    "connection_type": device_info.connection_type.value,
                    "memory_limit_mb": device_info.memory_limit_mb
                },
                "performance_metrics": performance_metrics
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.post("/websocket/connect")
async def connect_websocket(
    websocket_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Get WebSocket connection information"""
    try:
        # In a real implementation, this would return WebSocket connection details
        # For now, return connection information
        connection_info = {
            "websocket_url": f"ws://localhost:8000/ws/{current_user.id}",
            "connection_id": f"conn_{int(time.time())}",
            "auth_token": f"ws_token_{current_user.id}_{int(time.time())}",
            "heartbeat_interval": 30,
            "max_reconnect_attempts": 5,
            "reconnect_delay": 1000
        }
        
        return {
            "success": True,
            "data": connection_info,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get WebSocket info: {str(e)}")


@router.post("/notifications/register")
async def register_for_notifications(
    notification_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Register device for push notifications"""
    try:
        device_token = notification_data.get("device_token")
        device_type = notification_data.get("device_type", "mobile")
        notification_preferences = notification_data.get("preferences", {})
        
        if not device_token:
            raise HTTPException(status_code=400, detail="Device token is required")
        
        # Store device token and preferences
        # In a real implementation, this would be stored in the database
        device_info = {
            "user_id": current_user.id,
            "device_token": device_token,
            "device_type": device_type,
            "preferences": notification_preferences,
            "registered_at": time.time()
        }
        
        return {
            "success": True,
            "message": "Device registered for notifications",
            "data": {
                "device_id": f"device_{current_user.id}_{int(time.time())}",
                "notification_types": ["market_updates", "trade_notifications", "price_alerts"]
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register for notifications: {str(e)}")


@router.post("/notifications/send")
async def send_notification(
    notification_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Send a test notification"""
    try:
        notification_type = notification_data.get("type", "test")
        message = notification_data.get("message", "Test notification")
        
        # Create a simple notification
        notification_info = {
            "type": notification_type,
            "message": message,
            "user_id": current_user.id,
            "timestamp": time.time()
        }
        
        # Send via WebSocket if connected
        await websocket_manager.broadcast_to_user(
            str(current_user.id),
            MessageType.NOTIFICATION,
            notification_info
        )
        
        return {
            "success": True,
            "message": "Notification sent successfully",
            "data": notification_info,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send notification: {str(e)}")


@router.get("/notifications/history")
async def get_notification_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user)
):
    """Get notification history for user"""
    try:
        # Get notifications from notification manager
        notifications = await notification_manager.get_user_notifications(
            str(current_user.id), 
            limit
        )
        
        return {
            "success": True,
            "data": {
                "notifications": notifications,
                "total_count": len(notifications)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get notification history: {str(e)}")


@router.post("/offline/sync")
async def sync_offline_data(
    sync_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Sync offline data when device comes back online"""
    try:
        offline_actions = sync_data.get("actions", [])
        last_sync_time = sync_data.get("last_sync_time", 0)
        
        # Process offline actions
        synced_actions = []
        for action in offline_actions:
            # In a real implementation, this would process each offline action
            # For now, just acknowledge them
            synced_actions.append({
                "action_id": action.get("id"),
                "status": "synced",
                "timestamp": time.time()
            })
        
        # Get updates since last sync
        updates = {
            "markets": [],  # New/updated markets
            "trades": [],   # New trades
            "notifications": []  # New notifications
        }
        
        return {
            "success": True,
            "data": {
                "synced_actions": synced_actions,
                "updates": updates,
                "sync_timestamp": time.time()
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync offline data: {str(e)}")


@router.get("/cache/status")
async def get_cache_status(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get mobile cache status and recommendations"""
    try:
        user_agent = request.headers.get("User-Agent", "")
        device_info = await mobile_optimizer.detect_device(user_agent)
        
        # Get cache recommendations based on device
        cache_recommendations = {
            "max_cache_size_mb": min(100, device_info.memory_limit_mb * 0.1),
            "cache_ttl_seconds": 1800 if device_info.device_type == DeviceType.MOBILE else 3600,
            "enable_compression": device_info.connection_type != ConnectionType.WIFI,
            "cache_strategy": "aggressive" if device_info.device_type == DeviceType.MOBILE else "moderate",
            "preload_critical_data": True,
            "lazy_load_images": True
        }
        
        return {
            "success": True,
            "data": {
                "device_info": {
                    "device_type": device_info.device_type.value,
                    "memory_limit_mb": device_info.memory_limit_mb,
                    "connection_type": device_info.connection_type.value
                },
                "cache_recommendations": cache_recommendations
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")


@router.post("/analytics/track")
async def track_mobile_analytics(
    analytics_data: Dict[str, Any],
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Track mobile-specific analytics"""
    try:
        user_agent = request.headers.get("User-Agent", "")
        device_info = await mobile_optimizer.detect_device(user_agent)
        
        # Track analytics event
        event_data = {
            "user_id": current_user.id,
            "event_type": analytics_data.get("event_type", "page_view"),
            "event_data": analytics_data.get("event_data", {}),
            "device_info": {
                "device_type": device_info.device_type.value,
                "screen_width": device_info.screen_width,
                "screen_height": device_info.screen_height,
                "connection_type": device_info.connection_type.value
            },
            "timestamp": time.time()
        }
        
        # Store analytics (in real implementation, send to analytics service)
        analytics_id = f"analytics_{current_user.id}_{int(time.time())}"
        
        return {
            "success": True,
            "message": "Analytics tracked successfully",
            "data": {
                "analytics_id": analytics_id,
                "event_data": event_data
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to track analytics: {str(e)}")


@router.get("/health")
async def mobile_health_check(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Get mobile-specific health status"""
    try:
        user_agent = request.headers.get("User-Agent", "")
        device_info = await mobile_optimizer.detect_device(user_agent)
        
        # Get WebSocket connection stats
        websocket_stats = websocket_manager.get_connection_stats()
        
        # Get notification stats
        notification_stats = notification_manager.get_notification_stats()
        
        health_status = {
            "mobile_optimization": "healthy",
            "websocket_connections": websocket_stats["active_connections"],
            "notification_system": "healthy",
            "device_support": {
                "device_type": device_info.device_type.value,
                "optimization_level": "high" if device_info.device_type == DeviceType.MOBILE else "medium"
            }
        }
        
        return {
            "success": True,
            "data": health_status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")