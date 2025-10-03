"""
Event Sourcing API
API endpoints for event sourcing and CQRS management
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.services.event_sourcing_engine import event_sourcing_engine, EventType, EventStatus, AggregateType

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class EventRequest(BaseModel):
    """Event request model"""
    aggregate_id: str
    aggregate_type: str
    event_type: str
    event_data: Dict[str, Any]
    event_metadata: Optional[Dict[str, Any]] = None
    version: int = 1
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None


class CommandRequest(BaseModel):
    """Command request model"""
    aggregate_id: str
    aggregate_type: str
    command_type: str
    command_data: Dict[str, Any]
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None


class ProjectionRequest(BaseModel):
    """Projection request model"""
    name: str
    description: str
    aggregate_type: str
    event_types: List[str]


# API Endpoints
@router.get("/status")
async def get_event_sourcing_status():
    """Get event sourcing system status"""
    try:
        summary = event_sourcing_engine.get_event_sourcing_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting event sourcing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events")
async def create_event(event_request: EventRequest):
    """Create a new event"""
    try:
        event_data = {
            "aggregate_id": event_request.aggregate_id,
            "aggregate_type": event_request.aggregate_type,
            "event_type": event_request.event_type,
            "event_data": event_request.event_data,
            "event_metadata": event_request.event_metadata or {},
            "version": event_request.version,
            "correlation_id": event_request.correlation_id,
            "causation_id": event_request.causation_id
        }
        
        event = await event_sourcing_engine.create_event(event_data)
        
        return JSONResponse(content={
            "message": "Event created successfully",
            "event_id": event.event_id,
            "aggregate_id": event.aggregate_id,
            "event_type": event.event_type.value,
            "status": event.status.value
        })
        
    except Exception as e:
        logger.error(f"Error creating event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def get_events():
    """Get all events"""
    try:
        events = []
        for event_id, event in event_sourcing_engine.events.items():
            events.append({
                "event_id": event.event_id,
                "aggregate_id": event.aggregate_id,
                "aggregate_type": event.aggregate_type.value,
                "event_type": event.event_type.value,
                "event_data": event.event_data,
                "event_metadata": event.event_metadata,
                "version": event.version,
                "timestamp": event.timestamp.isoformat(),
                "correlation_id": event.correlation_id,
                "causation_id": event.causation_id,
                "status": event.status.value,
                "processed_at": event.processed_at.isoformat() if event.processed_at else None,
                "error_message": event.error_message
            })
            
        return JSONResponse(content={"events": events})
        
    except Exception as e:
        logger.error(f"Error getting events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/{event_id}")
async def get_event(event_id: str):
    """Get a specific event"""
    try:
        event = event_sourcing_engine.events.get(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
            
        return JSONResponse(content={
            "event_id": event.event_id,
            "aggregate_id": event.aggregate_id,
            "aggregate_type": event.aggregate_type.value,
            "event_type": event.event_type.value,
            "event_data": event.event_data,
            "event_metadata": event.event_metadata,
            "version": event.version,
            "timestamp": event.timestamp.isoformat(),
            "correlation_id": event.correlation_id,
            "causation_id": event.causation_id,
            "status": event.status.value,
            "processed_at": event.processed_at.isoformat() if event.processed_at else None,
            "error_message": event.error_message
        })
        
    except Exception as e:
        logger.error(f"Error getting event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/commands")
async def process_command(command_request: CommandRequest):
    """Process a command"""
    try:
        command_data = {
            "aggregate_id": command_request.aggregate_id,
            "aggregate_type": command_request.aggregate_type,
            "command_type": command_request.command_type,
            "command_data": command_request.command_data,
            "correlation_id": command_request.correlation_id,
            "causation_id": command_request.causation_id
        }
        
        command = await event_sourcing_engine.process_command(command_data)
        
        return JSONResponse(content={
            "message": "Command processed successfully",
            "command_id": command.command_id,
            "aggregate_id": command.aggregate_id,
            "command_type": command.command_type,
            "status": command.status,
            "result": command.result,
            "error_message": command.error_message
        })
        
    except Exception as e:
        logger.error(f"Error processing command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/commands")
async def get_commands():
    """Get all commands"""
    try:
        commands = []
        for command_id, command in event_sourcing_engine.commands.items():
            commands.append({
                "command_id": command.command_id,
                "aggregate_id": command.aggregate_id,
                "aggregate_type": command.aggregate_type.value,
                "command_type": command.command_type,
                "command_data": command.command_data,
                "correlation_id": command.correlation_id,
                "causation_id": command.causation_id,
                "timestamp": command.timestamp.isoformat(),
                "status": command.status,
                "result": command.result,
                "error_message": command.error_message
            })
            
        return JSONResponse(content={"commands": commands})
        
    except Exception as e:
        logger.error(f"Error getting commands: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aggregates")
async def get_aggregates():
    """Get all aggregates"""
    try:
        aggregates = []
        for aggregate_id, aggregate in event_sourcing_engine.aggregates.items():
            aggregates.append({
                "aggregate_id": aggregate.aggregate_id,
                "aggregate_type": aggregate.aggregate_type.value,
                "version": aggregate.version,
                "state": aggregate.state,
                "events_count": len(aggregate.events),
                "created_at": aggregate.created_at.isoformat(),
                "updated_at": aggregate.updated_at.isoformat()
            })
            
        return JSONResponse(content={"aggregates": aggregates})
        
    except Exception as e:
        logger.error(f"Error getting aggregates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aggregates/{aggregate_id}")
async def get_aggregate(aggregate_id: str):
    """Get a specific aggregate"""
    try:
        aggregate = event_sourcing_engine.aggregates.get(aggregate_id)
        if not aggregate:
            raise HTTPException(status_code=404, detail="Aggregate not found")
            
        return JSONResponse(content={
            "aggregate_id": aggregate.aggregate_id,
            "aggregate_type": aggregate.aggregate_type.value,
            "version": aggregate.version,
            "state": aggregate.state,
            "events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "status": event.status.value
                }
                for event in aggregate.events
            ],
            "created_at": aggregate.created_at.isoformat(),
            "updated_at": aggregate.updated_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting aggregate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projections")
async def get_projections():
    """Get all projections"""
    try:
        projections = []
        for projection_id, projection in event_sourcing_engine.projections.items():
            projections.append({
                "projection_id": projection.projection_id,
                "name": projection.name,
                "description": projection.description,
                "aggregate_type": projection.aggregate_type.value,
                "event_types": [event_type.value for event_type in projection.event_types],
                "state": projection.state,
                "last_processed_event": projection.last_processed_event,
                "created_at": projection.created_at.isoformat(),
                "updated_at": projection.updated_at.isoformat()
            })
            
        return JSONResponse(content={"projections": projections})
        
    except Exception as e:
        logger.error(f"Error getting projections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projections/{projection_id}")
async def get_projection(projection_id: str):
    """Get a specific projection"""
    try:
        projection = event_sourcing_engine.projections.get(projection_id)
        if not projection:
            raise HTTPException(status_code=404, detail="Projection not found")
            
        return JSONResponse(content={
            "projection_id": projection.projection_id,
            "name": projection.name,
            "description": projection.description,
            "aggregate_type": projection.aggregate_type.value,
            "event_types": [event_type.value for event_type in projection.event_types],
            "state": projection.state,
            "last_processed_event": projection.last_processed_event,
            "created_at": projection.created_at.isoformat(),
            "updated_at": projection.updated_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting projection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_event_sourcing_dashboard():
    """Get event sourcing dashboard data"""
    try:
        summary = event_sourcing_engine.get_event_sourcing_summary()
        
        # Get recent events
        recent_events = list(event_sourcing_engine.events.values())[-10:]
        
        # Get recent commands
        recent_commands = list(event_sourcing_engine.commands.values())[-10:]
        
        dashboard_data = {
            "summary": summary,
            "recent_events": [
                {
                    "event_id": event.event_id,
                    "aggregate_id": event.aggregate_id,
                    "event_type": event.event_type.value,
                    "status": event.status.value,
                    "timestamp": event.timestamp.isoformat()
                }
                for event in recent_events
            ],
            "recent_commands": [
                {
                    "command_id": command.command_id,
                    "aggregate_id": command.aggregate_id,
                    "command_type": command.command_type,
                    "status": command.status,
                    "timestamp": command.timestamp.isoformat()
                }
                for command in recent_commands
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting event sourcing dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_event_sourcing():
    """Start event sourcing engine"""
    try:
        await event_sourcing_engine.start_event_sourcing_engine()
        return JSONResponse(content={"message": "Event sourcing engine started"})
        
    except Exception as e:
        logger.error(f"Error starting event sourcing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_event_sourcing():
    """Stop event sourcing engine"""
    try:
        await event_sourcing_engine.stop_event_sourcing_engine()
        return JSONResponse(content={"message": "Event sourcing engine stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping event sourcing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
