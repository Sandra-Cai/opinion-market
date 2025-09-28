"""
WebSocket API Endpoints
Provides real-time communication endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List, Optional
import asyncio
import time
import json

from app.core.auth import get_current_user
from app.models.user import User
from app.websocket.websocket_manager import websocket_manager, MessageType
from app.notifications.notification_manager import notification_manager

router = APIRouter()


@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication"""
    connection_id = None
    
    try:
        # Accept connection
        connection_id = await websocket_manager.connect(websocket)
        
        # Authenticate connection
        # In a real implementation, you would validate the JWT token
        # For now, we'll use a simple authentication
        auth_success = await websocket_manager.authenticate_connection(
            connection_id, 
            user_id, 
            f"token_{user_id}_{int(time.time())}"
        )
        
        if not auth_success:
            await websocket.close()
            return
        
        # Main message loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Handle message
                await websocket_manager.handle_message(connection_id, message_data)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager.send_message(
                    connection_id, 
                    MessageType.ERROR, 
                    {"error": "Invalid JSON format"}
                )
            except Exception as e:
                await websocket_manager.send_message(
                    connection_id, 
                    MessageType.ERROR, 
                    {"error": f"Message processing error: {str(e)}"}
                )
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id, "client_disconnect")


@router.get("/connections/stats")
async def get_websocket_stats(current_user: User = Depends(get_current_user)):
    """Get WebSocket connection statistics"""
    try:
        stats = websocket_manager.get_connection_stats()
        
        return {
            "success": True,
            "data": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get WebSocket stats: {str(e)}")


@router.get("/connections/{user_id}")
async def get_user_connections(
    user_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get WebSocket connections for a user"""
    try:
        connections = websocket_manager.get_user_connections(user_id)
        
        return {
            "success": True,
            "data": {
                "user_id": user_id,
                "connections": connections,
                "connection_count": len(connections)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user connections: {str(e)}")


@router.get("/rooms/{room_id}")
async def get_room_info(
    room_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get information about a WebSocket room"""
    try:
        room_info = websocket_manager.get_room_info(room_id)
        
        if not room_info:
            raise HTTPException(status_code=404, detail=f"Room {room_id} not found")
        
        return {
            "success": True,
            "data": room_info,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get room info: {str(e)}")


@router.post("/rooms/{room_id}/join")
async def join_room(
    room_id: str,
    connection_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Join a WebSocket room"""
    try:
        connection_id = connection_data.get("connection_id")
        
        if not connection_id:
            raise HTTPException(status_code=400, detail="Connection ID is required")
        
        success = await websocket_manager.join_room(connection_id, room_id)
        
        if success:
            return {
                "success": True,
                "message": f"Successfully joined room {room_id}",
                "data": {
                    "room_id": room_id,
                    "connection_id": connection_id
                },
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to join room")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to join room: {str(e)}")


@router.post("/rooms/{room_id}/leave")
async def leave_room(
    room_id: str,
    connection_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Leave a WebSocket room"""
    try:
        connection_id = connection_data.get("connection_id")
        
        if not connection_id:
            raise HTTPException(status_code=400, detail="Connection ID is required")
        
        success = await websocket_manager.leave_room(connection_id, room_id)
        
        if success:
            return {
                "success": True,
                "message": f"Successfully left room {room_id}",
                "data": {
                    "room_id": room_id,
                    "connection_id": connection_id
                },
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to leave room")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to leave room: {str(e)}")


@router.post("/broadcast/room/{room_id}")
async def broadcast_to_room(
    room_id: str,
    broadcast_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Broadcast a message to all connections in a room"""
    try:
        message_type_str = broadcast_data.get("message_type", "notification")
        message_data = broadcast_data.get("data", {})
        exclude_connection = broadcast_data.get("exclude_connection")
        
        try:
            message_type = MessageType(message_type_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid message type: {message_type_str}")
        
        sent_count = await websocket_manager.broadcast_to_room(
            room_id, 
            message_type, 
            message_data, 
            exclude_connection
        )
        
        return {
            "success": True,
            "message": f"Message broadcasted to {sent_count} connections",
            "data": {
                "room_id": room_id,
                "message_type": message_type_str,
                "sent_count": sent_count
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast to room: {str(e)}")


@router.post("/broadcast/user/{user_id}")
async def broadcast_to_user(
    user_id: str,
    broadcast_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Broadcast a message to all connections of a user"""
    try:
        message_type_str = broadcast_data.get("message_type", "notification")
        message_data = broadcast_data.get("data", {})
        
        try:
            message_type = MessageType(message_type_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid message type: {message_type_str}")
        
        sent_count = await websocket_manager.broadcast_to_user(
            user_id, 
            message_type, 
            message_data
        )
        
        return {
            "success": True,
            "message": f"Message broadcasted to {sent_count} connections",
            "data": {
                "user_id": user_id,
                "message_type": message_type_str,
                "sent_count": sent_count
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast to user: {str(e)}")


@router.post("/send/{connection_id}")
async def send_to_connection(
    connection_id: str,
    message_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Send a message to a specific connection"""
    try:
        message_type_str = message_data.get("message_type", "notification")
        data = message_data.get("data", {})
        
        try:
            message_type = MessageType(message_type_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid message type: {message_type_str}")
        
        success = await websocket_manager.send_message(
            connection_id, 
            message_type, 
            data
        )
        
        if success:
            return {
                "success": True,
                "message": "Message sent successfully",
                "data": {
                    "connection_id": connection_id,
                    "message_type": message_type_str
                },
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=404, detail="Connection not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@router.post("/disconnect/{connection_id}")
async def disconnect_connection(
    connection_id: str,
    current_user: User = Depends(get_current_user)
):
    """Disconnect a specific WebSocket connection"""
    try:
        await websocket_manager.disconnect(connection_id, "admin_disconnect")
        
        return {
            "success": True,
            "message": f"Connection {connection_id} disconnected",
            "data": {
                "connection_id": connection_id
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disconnect connection: {str(e)}")


@router.get("/health")
async def websocket_health_check(current_user: User = Depends(get_current_user)):
    """Get WebSocket system health status"""
    try:
        stats = websocket_manager.get_connection_stats()
        
        # Determine health status
        if stats["active_connections"] > 1000:
            health_status = "degraded"
        elif stats["authentication_failures"] > 100:
            health_status = "degraded"
        else:
            health_status = "healthy"
        
        return {
            "success": True,
            "data": {
                "status": health_status,
                "stats": stats,
                "system_info": {
                    "max_connections_per_user": websocket_manager.max_connections_per_user,
                    "max_rooms_per_connection": websocket_manager.max_rooms_per_connection,
                    "heartbeat_interval": websocket_manager.heartbeat_interval,
                    "connection_timeout": websocket_manager.connection_timeout
                }
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")