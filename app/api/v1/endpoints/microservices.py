"""
Microservices Management API Endpoints
Provides management and monitoring for microservices architecture
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import asyncio
import time

from app.core.auth import get_current_user
from app.models.user import User
from app.services.service_registry import service_registry
from app.services.inter_service_communication import inter_service_comm
from app.api_gateway.gateway import api_gateway
from app.events.event_bus import event_bus
from app.events.event_store import event_store

router = APIRouter()


@router.get("/services")
async def get_services(current_user: User = Depends(get_current_user)):
    """Get all registered services"""
    try:
        registry_status = service_registry.get_registry_status()
        return {
            "success": True,
            "data": registry_status,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get services: {str(e)}")


@router.get("/services/{service_name}")
async def get_service_details(service_name: str, current_user: User = Depends(get_current_user)):
    """Get details for a specific service"""
    try:
        instances = await service_registry.discover_services(service_name)
        
        if not instances:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        return {
            "success": True,
            "data": {
                "service_name": service_name,
                "instances": [
                    {
                        "instance_id": inst.instance_id,
                        "host": inst.host,
                        "port": inst.port,
                        "version": inst.version,
                        "status": inst.status,
                        "last_heartbeat": inst.last_heartbeat,
                        "metadata": inst.metadata
                    }
                    for inst in instances
                ]
            },
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service details: {str(e)}")


@router.post("/services/{service_name}/call")
async def call_service(
    service_name: str,
    request_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Call a specific microservice"""
    try:
        endpoint = request_data.get("endpoint", "/")
        method = request_data.get("method", "GET")
        data = request_data.get("data")
        headers = request_data.get("headers", {})
        
        result = await inter_service_comm.call_service(
            service_name=service_name,
            endpoint=endpoint,
            method=method,
            data=data,
            headers=headers
        )
        
        return {
            "success": True,
            "data": result,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to call service: {str(e)}")


@router.get("/gateway/stats")
async def get_gateway_stats(current_user: User = Depends(get_current_user)):
    """Get API Gateway statistics"""
    try:
        stats = api_gateway.get_gateway_stats()
        return {
            "success": True,
            "data": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get gateway stats: {str(e)}")


@router.get("/communication/stats")
async def get_communication_stats(current_user: User = Depends(get_current_user)):
    """Get inter-service communication statistics"""
    try:
        stats = inter_service_comm.get_communication_stats()
        return {
            "success": True,
            "data": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get communication stats: {str(e)}")


@router.post("/events/publish")
async def publish_event(
    event_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Publish an event to the event bus"""
    try:
        from app.events.event_store import Event
        
        event = Event(
            event_id=event_data.get("event_id", f"event_{int(time.time())}"),
            event_type=event_data["event_type"],
            aggregate_id=event_data["aggregate_id"],
            aggregate_type=event_data["aggregate_type"],
            event_data=event_data.get("event_data", {}),
            metadata=event_data.get("metadata", {}),
            timestamp=time.time(),
            version=event_data.get("version", 1),
            causation_id=event_data.get("causation_id"),
            correlation_id=event_data.get("correlation_id")
        )
        
        success = await event_bus.publish_event(event)
        
        return {
            "success": success,
            "event_id": event.event_id,
            "message": "Event published successfully" if success else "Failed to publish event",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish event: {str(e)}")


@router.get("/events/{aggregate_type}/{aggregate_id}")
async def get_events(
    aggregate_type: str,
    aggregate_id: str,
    from_version: int = 0,
    current_user: User = Depends(get_current_user)
):
    """Get events for a specific aggregate"""
    try:
        events = await event_store.get_events(aggregate_id, aggregate_type, from_version)
        
        return {
            "success": True,
            "data": {
                "aggregate_type": aggregate_type,
                "aggregate_id": aggregate_id,
                "events": [
                    {
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "event_data": event.event_data,
                        "metadata": event.metadata,
                        "timestamp": event.timestamp,
                        "version": event.version
                    }
                    for event in events
                ],
                "count": len(events)
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get events: {str(e)}")


@router.get("/events/statistics")
async def get_event_statistics(current_user: User = Depends(get_current_user)):
    """Get event store statistics"""
    try:
        stats = await event_store.get_event_statistics()
        bus_stats = await event_bus.get_event_bus_stats()
        
        return {
            "success": True,
            "data": {
                "event_store": stats,
                "event_bus": bus_stats
            },
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get event statistics: {str(e)}")


@router.post("/services/{service_name}/health-check")
async def health_check_service(
    service_name: str,
    current_user: User = Depends(get_current_user)
):
    """Perform health check on a specific service"""
    try:
        instances = await service_registry.discover_services(service_name)
        
        if not instances:
            return {
                "success": False,
                "message": f"No instances found for service {service_name}",
                "timestamp": time.time()
            }
        
        # Test communication with the service
        test_result = await inter_service_comm.call_service(
            service_name=service_name,
            endpoint="/health",
            method="GET"
        )
        
        return {
            "success": True,
            "data": {
                "service_name": service_name,
                "instances_count": len(instances),
                "health_check_result": test_result,
                "instances": [
                    {
                        "instance_id": inst.instance_id,
                        "status": inst.status,
                        "last_heartbeat": inst.last_heartbeat
                    }
                    for inst in instances
                ]
            },
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Health check failed: {str(e)}",
            "timestamp": time.time()
        }


@router.post("/services/register")
async def register_service(
    service_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Register a new service instance"""
    try:
        success = await service_registry.register_service(
            service_name=service_data["service_name"],
            instance_id=service_data["instance_id"],
            host=service_data["host"],
            port=service_data["port"],
            version=service_data.get("version", "1.0.0"),
            metadata=service_data.get("metadata", {})
        )
        
        return {
            "success": success,
            "message": "Service registered successfully" if success else "Failed to register service",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register service: {str(e)}")


@router.delete("/services/{service_name}/{instance_id}")
async def unregister_service(
    service_name: str,
    instance_id: str,
    current_user: User = Depends(get_current_user)
):
    """Unregister a service instance"""
    try:
        success = await service_registry.unregister_service(service_name, instance_id)
        
        return {
            "success": success,
            "message": "Service unregistered successfully" if success else "Failed to unregister service",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unregister service: {str(e)}")


@router.get("/architecture/overview")
async def get_architecture_overview(current_user: User = Depends(get_current_user)):
    """Get microservices architecture overview"""
    try:
        # Get service registry status
        registry_status = service_registry.get_registry_status()
        
        # Get gateway stats
        gateway_stats = api_gateway.get_gateway_stats()
        
        # Get communication stats
        communication_stats = inter_service_comm.get_communication_stats()
        
        # Get event statistics
        event_stats = await event_store.get_event_statistics()
        bus_stats = await event_bus.get_event_bus_stats()
        
        overview = {
            "services": {
                "total_services": registry_status["total_services"],
                "total_instances": registry_status["total_instances"],
                "healthy_instances": registry_status["healthy_instances"],
                "service_list": list(registry_status["services"].keys())
            },
            "gateway": {
                "routing_rules": len(gateway_stats["routing_rules"]),
                "middleware_count": gateway_stats["middleware_count"],
                "service_routers": len(gateway_stats["service_routers"])
            },
            "communication": {
                "circuit_breakers": len(communication_stats["circuit_breakers"]),
                "cache_size": communication_stats["cache_size"]
            },
            "events": {
                "total_events": event_stats.get("total_events", 0),
                "total_aggregates": event_stats.get("total_aggregates", 0),
                "published_events": bus_stats["published_events"],
                "total_subscriptions": bus_stats["total_subscriptions"]
            }
        }
        
        return {
            "success": True,
            "data": overview,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get architecture overview: {str(e)}")
