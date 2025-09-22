"""
Notification Management API Endpoints
Provides management and monitoring of notification system
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from app.core.advanced_notifications import (
    notification_manager,
    NotificationChannel,
    NotificationPriority,
    NotificationType,
    send_notification,
    send_custom_notification,
    set_user_preferences,
    configure_channel
)
from app.core.advanced_auth import get_current_user, User

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/stats")
async def get_notification_stats():
    """Get notification system statistics"""
    try:
        stats = notification_manager.get_stats()
        return {"success": True, "data": stats}
    except Exception as e:
        logger.error(f"Error getting notification stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send")
async def send_notification_endpoint(
    user_id: str,
    template_id: str,
    variables: Optional[Dict[str, Any]] = None,
    channel: Optional[str] = None,
    priority: str = "normal",
    scheduled_at: Optional[datetime] = None
):
    """Send notification using template"""
    try:
        # Validate channel
        notification_channel = None
        if channel:
            try:
                notification_channel = NotificationChannel(channel)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid channel: {channel}")
        
        # Validate priority
        try:
            notification_priority = NotificationPriority(priority)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        notification_id = await send_notification(
            user_id=user_id,
            template_id=template_id,
            variables=variables,
            channel=notification_channel,
            priority=notification_priority,
            scheduled_at=scheduled_at
        )
        
        if not notification_id:
            raise HTTPException(status_code=400, detail="Notification blocked by user preferences or rate limits")
        
        return {
            "success": True,
            "message": "Notification sent successfully",
            "data": {"notification_id": notification_id}
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_notification_templates():
    """Get all notification templates"""
    try:
        templates = []
        for template in notification_manager.templates.values():
            templates.append({
                "id": template.id,
                "name": template.name,
                "type": template.type.value,
                "channel": template.channel.value,
                "subject": template.subject,
                "content": template.content,
                "is_active": template.is_active
            })
        
        return {"success": True, "data": {"templates": templates, "total": len(templates)}}
    except Exception as e:
        logger.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/users/{user_id}/notifications")
async def get_user_notifications(
    user_id: str,
    limit: int = Query(100, description="Maximum number of notifications to return")
):
    """Get user notifications"""
    try:
        notifications = notification_manager.get_user_notifications(user_id, limit)
        
        notification_data = []
        for notification in notifications:
            notification_data.append({
                "id": notification.id,
                "type": notification.type.value,
                "channel": notification.channel.value,
                "priority": notification.priority.value,
                "subject": notification.subject,
                "content": notification.content,
                "status": notification.status.value,
                "created_at": notification.created_at.isoformat()
            })
        
        return {
            "success": True,
            "data": {
                "notifications": notification_data,
                "total": len(notification_data),
                "user_id": user_id
            }
        }
    except Exception as e:
        logger.error(f"Error getting user notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/my-notifications")
async def get_my_notifications(
    current_user: User = Depends(get_current_user),
    limit: int = Query(100, description="Maximum number of notifications to return")
):
    """Get current user's notifications"""
    try:
        notifications = notification_manager.get_user_notifications(current_user.id, limit)
        
        notification_data = []
        for notification in notifications:
            notification_data.append({
                "id": notification.id,
                "type": notification.type.value,
                "channel": notification.channel.value,
                "priority": notification.priority.value,
                "subject": notification.subject,
                "content": notification.content,
                "status": notification.status.value,
                "created_at": notification.created_at.isoformat()
            })
        
        return {
            "success": True,
            "data": {
                "notifications": notification_data,
                "total": len(notification_data),
                "user_id": current_user.id
            }
        }
    except Exception as e:
        logger.error(f"Error getting user notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/channels")
async def get_available_channels():
    """Get available notification channels"""
    try:
        channels = []
        for channel in NotificationChannel:
            channels.append({
                "name": channel.value,
                "description": f"{channel.value.title()} notifications"
            })
        
        return {"success": True, "data": {"channels": channels, "total": len(channels)}}
    except Exception as e:
        logger.error(f"Error getting channels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def get_notification_types():
    """Get available notification types"""
    try:
        types = []
        for notification_type in NotificationType:
            types.append({
                "name": notification_type.value,
                "description": f"{notification_type.value.title()} notifications"
            })
        
        return {"success": True, "data": {"types": types, "total": len(types)}}
    except Exception as e:
        logger.error(f"Error getting notification types: {e}")
        raise HTTPException(status_code=500, detail=str(e))