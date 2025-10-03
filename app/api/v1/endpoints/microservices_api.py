"""
Microservices API
API endpoints for microservices and service mesh management
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.services.microservices_engine import microservices_engine, ServiceType, ServiceStatus, LoadBalancingStrategy, CircuitBreakerState

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class ServiceMeshRequest(BaseModel):
    """Service mesh request model"""
    name: str


class ServiceRegistrationRequest(BaseModel):
    """Service registration request model"""
    service_type: str
    host: str = "localhost"
    port: Optional[int] = None


class ServiceCallRequest(BaseModel):
    """Service call request model"""
    service_name: str
    method: str
    endpoint: str
    data: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None


class LoadBalancingConfigRequest(BaseModel):
    """Load balancing configuration request model"""
    strategy: str
    weights: Optional[Dict[str, int]] = None


# API Endpoints
@router.get("/status")
async def get_microservices_status():
    """Get microservices system status"""
    try:
        summary = microservices_engine.get_microservices_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting microservices status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mesh")
async def create_service_mesh(mesh_request: ServiceMeshRequest):
    """Create a new service mesh"""
    try:
        mesh = await microservices_engine.create_service_mesh(mesh_request.name)
        
        return JSONResponse(content={
            "message": "Service mesh created successfully",
            "mesh_id": mesh.mesh_id,
            "name": mesh.name
        })
        
    except Exception as e:
        logger.error(f"Error creating service mesh: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mesh")
async def get_service_mesh():
    """Get current service mesh information"""
    try:
        if not microservices_engine.service_mesh:
            raise HTTPException(status_code=404, detail="No service mesh found")
            
        mesh = microservices_engine.service_mesh
        
        return JSONResponse(content={
            "mesh_id": mesh.mesh_id,
            "name": mesh.name,
            "services": {
                service_name: [
                    {
                        "instance_id": instance.instance_id,
                        "host": instance.host,
                        "port": instance.port,
                        "version": instance.version,
                        "status": instance.status.value,
                        "created_at": instance.created_at.isoformat(),
                        "last_heartbeat": instance.last_heartbeat.isoformat(),
                        "load_balancer_weight": instance.load_balancer_weight,
                        "circuit_breaker_state": instance.circuit_breaker_state.value,
                        "failure_count": instance.failure_count,
                        "success_count": instance.success_count
                    }
                    for instance in instances
                ]
                for service_name, instances in mesh.services.items()
            },
            "load_balancing_strategy": mesh.load_balancing_strategy.value,
            "circuit_breaker_config": mesh.circuit_breaker_config,
            "retry_config": mesh.retry_config,
            "timeout_config": mesh.timeout_config,
            "created_at": mesh.created_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting service mesh: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services")
async def register_service(service_request: ServiceRegistrationRequest):
    """Register a new service instance"""
    try:
        service_type = ServiceType(service_request.service_type)
        
        instance = await microservices_engine.register_service(
            service_type,
            service_request.host,
            service_request.port
        )
        
        return JSONResponse(content={
            "message": "Service registered successfully",
            "instance_id": instance.instance_id,
            "service_name": instance.service_name,
            "service_type": instance.service_type.value,
            "host": instance.host,
            "port": instance.port,
            "status": instance.status.value
        })
        
    except Exception as e:
        logger.error(f"Error registering service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services")
async def get_services():
    """Get all registered services"""
    try:
        services = []
        for instance_id, instance in microservices_engine.service_instances.items():
            services.append({
                "instance_id": instance.instance_id,
                "service_name": instance.service_name,
                "service_type": instance.service_type.value,
                "host": instance.host,
                "port": instance.port,
                "version": instance.version,
                "status": instance.status.value,
                "health_check_url": instance.health_check_url,
                "metadata": instance.metadata,
                "created_at": instance.created_at.isoformat(),
                "last_heartbeat": instance.last_heartbeat.isoformat(),
                "load_balancer_weight": instance.load_balancer_weight,
                "circuit_breaker_state": instance.circuit_breaker_state.value,
                "failure_count": instance.failure_count,
                "success_count": instance.success_count
            })
            
        return JSONResponse(content={"services": services})
        
    except Exception as e:
        logger.error(f"Error getting services: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/call")
async def call_service(call_request: ServiceCallRequest):
    """Call a service through the service mesh"""
    try:
        response = await microservices_engine.call_service(
            call_request.service_name,
            call_request.method,
            call_request.endpoint,
            call_request.data,
            call_request.headers
        )
        
        return JSONResponse(content={
            "message": "Service call successful",
            "response": response
        })
        
    except Exception as e:
        logger.error(f"Error calling service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/{service_name}/instances")
async def get_service_instances(service_name: str):
    """Get instances of a specific service"""
    try:
        if not microservices_engine.service_mesh or service_name not in microservices_engine.service_mesh.services:
            raise HTTPException(status_code=404, detail=f"Service not found: {service_name}")
            
        instances = microservices_engine.service_mesh.services[service_name]
        
        instance_data = []
        for instance in instances:
            instance_data.append({
                "instance_id": instance.instance_id,
                "host": instance.host,
                "port": instance.port,
                "version": instance.version,
                "status": instance.status.value,
                "health_check_url": instance.health_check_url,
                "created_at": instance.created_at.isoformat(),
                "last_heartbeat": instance.last_heartbeat.isoformat(),
                "load_balancer_weight": instance.load_balancer_weight,
                "circuit_breaker_state": instance.circuit_breaker_state.value,
                "failure_count": instance.failure_count,
                "success_count": instance.success_count
            })
            
        return JSONResponse(content={"instances": instance_data})
        
    except Exception as e:
        logger.error(f"Error getting service instances: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/{service_name}/health")
async def get_service_health(service_name: str):
    """Get health status of a specific service"""
    try:
        if not microservices_engine.service_mesh or service_name not in microservices_engine.service_mesh.services:
            raise HTTPException(status_code=404, detail=f"Service not found: {service_name}")
            
        instances = microservices_engine.service_mesh.services[service_name]
        
        health_data = {
            "service_name": service_name,
            "total_instances": len(instances),
            "healthy_instances": len([i for i in instances if i.status == ServiceStatus.RUNNING]),
            "unhealthy_instances": len([i for i in instances if i.status != ServiceStatus.RUNNING]),
            "circuit_breaker_open": len([i for i in instances if i.circuit_breaker_state == CircuitBreakerState.OPEN]),
            "instances": []
        }
        
        for instance in instances:
            health_data["instances"].append({
                "instance_id": instance.instance_id,
                "status": instance.status.value,
                "circuit_breaker_state": instance.circuit_breaker_state.value,
                "failure_count": instance.failure_count,
                "success_count": instance.success_count,
                "last_heartbeat": instance.last_heartbeat.isoformat()
            })
            
        return JSONResponse(content=health_data)
        
    except Exception as e:
        logger.error(f"Error getting service health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mesh/load-balancing")
async def configure_load_balancing(config_request: LoadBalancingConfigRequest):
    """Configure load balancing strategy"""
    try:
        if not microservices_engine.service_mesh:
            raise HTTPException(status_code=404, detail="No service mesh found")
            
        strategy = LoadBalancingStrategy(config_request.strategy)
        microservices_engine.service_mesh.load_balancing_strategy = strategy
        
        # Update weights if provided
        if config_request.weights:
            for service_name, weight in config_request.weights.items():
                if service_name in microservices_engine.service_mesh.services:
                    for instance in microservices_engine.service_mesh.services[service_name]:
                        instance.load_balancer_weight = weight
                        
        return JSONResponse(content={
            "message": "Load balancing configured successfully",
            "strategy": strategy.value,
            "weights": config_request.weights
        })
        
    except Exception as e:
        logger.error(f"Error configuring load balancing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calls")
async def get_service_calls():
    """Get recent service calls"""
    try:
        calls = []
        for call in microservices_engine.service_calls[-50:]:  # Last 50 calls
            calls.append({
                "call_id": call.call_id,
                "from_service": call.from_service,
                "to_service": call.to_service,
                "method": call.method,
                "endpoint": call.endpoint,
                "status_code": call.status_code,
                "duration_ms": call.duration_ms,
                "timestamp": call.timestamp.isoformat(),
                "success": call.success,
                "error_message": call.error_message
            })
            
        return JSONResponse(content={"calls": calls})
        
    except Exception as e:
        logger.error(f"Error getting service calls: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_microservices_dashboard():
    """Get microservices dashboard data"""
    try:
        summary = microservices_engine.get_microservices_summary()
        
        # Get service statistics
        services_by_status = {}
        services_by_type = {}
        
        for instance in microservices_engine.service_instances.values():
            status = instance.status.value
            service_type = instance.service_type.value
            
            services_by_status[status] = services_by_status.get(status, 0) + 1
            services_by_type[service_type] = services_by_type.get(service_type, 0) + 1
            
        # Get recent service calls
        recent_calls = microservices_engine.service_calls[-10:]
        
        dashboard_data = {
            "summary": summary,
            "services_by_status": services_by_status,
            "services_by_type": services_by_type,
            "recent_calls": [
                {
                    "from_service": call.from_service,
                    "to_service": call.to_service,
                    "method": call.method,
                    "endpoint": call.endpoint,
                    "success": call.success,
                    "duration_ms": call.duration_ms,
                    "timestamp": call.timestamp.isoformat()
                }
                for call in recent_calls
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting microservices dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_microservices():
    """Start microservices engine"""
    try:
        await microservices_engine.start_microservices_engine()
        return JSONResponse(content={"message": "Microservices engine started"})
        
    except Exception as e:
        logger.error(f"Error starting microservices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_microservices():
    """Stop microservices engine"""
    try:
        await microservices_engine.stop_microservices_engine()
        return JSONResponse(content={"message": "Microservices engine stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping microservices: {e}")
        raise HTTPException(status_code=500, detail=str(e))
