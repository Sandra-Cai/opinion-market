"""
Advanced API Gateway API
API endpoints for advanced API gateway management
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

from app.services.advanced_api_gateway import advanced_api_gateway, GatewayRoute, AuthenticationMethod, RateLimitType

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class APIRouteRequest(BaseModel):
    """API route request model"""
    name: str
    path: str
    methods: List[str]
    target_service: str
    target_url: str
    authentication_required: bool = True
    authentication_method: str = "jwt"
    rate_limit: Optional[Dict[str, Any]] = None
    middleware: Optional[List[str]] = None


class APIRequestRequest(BaseModel):
    """API request request model"""
    client_ip: str = "127.0.0.1"
    user_id: Optional[str] = None
    method: str = "GET"
    path: str = "/"
    headers: Optional[Dict[str, str]] = None
    query_params: Optional[Dict[str, Any]] = None
    body: Optional[bytes] = None


# API Endpoints
@router.get("/status")
async def get_api_gateway_status():
    """Get API gateway system status"""
    try:
        summary = advanced_api_gateway.get_gateway_summary()
        return JSONResponse(content=summary)
        
    except Exception as e:
        logger.error(f"Error getting API gateway status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/routes")
async def add_api_route(route_request: APIRouteRequest):
    """Add a new API route"""
    try:
        route_data = {
            "name": route_request.name,
            "path": route_request.path,
            "methods": route_request.methods,
            "target_service": route_request.target_service,
            "target_url": route_request.target_url,
            "authentication_required": route_request.authentication_required,
            "authentication_method": route_request.authentication_method,
            "rate_limit": route_request.rate_limit,
            "middleware": route_request.middleware
        }
        
        route = await advanced_api_gateway.add_route(route_data)
        
        return JSONResponse(content={
            "message": "API route added successfully",
            "route_id": route.route_id,
            "name": route.name,
            "path": route.path
        })
        
    except Exception as e:
        logger.error(f"Error adding API route: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/routes")
async def get_api_routes():
    """Get all API routes"""
    try:
        routes = []
        for route_id, route in advanced_api_gateway.routes.items():
            routes.append({
                "route_id": route.route_id,
                "name": route.name,
                "path": route.path,
                "methods": route.methods,
                "target_service": route.target_service,
                "target_url": route.target_url,
                "authentication_required": route.authentication_required,
                "authentication_method": route.authentication_method.value,
                "rate_limit": route.rate_limit,
                "middleware": route.middleware,
                "created_at": route.created_at.isoformat(),
                "updated_at": route.updated_at.isoformat(),
                "metadata": route.metadata
            })
            
        return JSONResponse(content={"routes": routes})
        
    except Exception as e:
        logger.error(f"Error getting API routes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/requests")
async def process_api_request(request_request: APIRequestRequest):
    """Process an API request through the gateway"""
    try:
        request_data = {
            "client_ip": request_request.client_ip,
            "user_id": request_request.user_id,
            "method": request_request.method,
            "path": request_request.path,
            "headers": request_request.headers or {},
            "query_params": request_request.query_params or {},
            "body": request_request.body
        }
        
        response = await advanced_api_gateway.process_request(request_data)
        
        return JSONResponse(content={
            "message": "API request processed",
            "response": response
        })
        
    except Exception as e:
        logger.error(f"Error processing API request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/requests")
async def get_request_logs():
    """Get recent request logs"""
    try:
        logs = []
        for request in advanced_api_gateway.request_logs[-100:]:  # Last 100 requests
            logs.append({
                "request_id": request.request_id,
                "client_ip": request.client_ip,
                "user_id": request.user_id,
                "method": request.method,
                "path": request.path,
                "timestamp": request.timestamp.isoformat(),
                "response_time": request.response_time,
                "status_code": request.status_code,
                "error_message": request.error_message
            })
            
        return JSONResponse(content={"logs": logs})
        
    except Exception as e:
        logger.error(f"Error getting request logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rate-limits")
async def get_rate_limit_status():
    """Get rate limit status"""
    try:
        rate_limit_status = {}
        for key, counter in advanced_api_gateway.rate_limit_counters.items():
            rate_limit_status[key] = {
                "current_requests": len(counter),
                "window_start": counter[0] if counter else None,
                "window_end": counter[-1] if counter else None
            }
            
        return JSONResponse(content={"rate_limits": rate_limit_status})
        
    except Exception as e:
        logger.error(f"Error getting rate limit status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tokens")
async def get_authentication_tokens():
    """Get authentication tokens"""
    try:
        tokens = []
        for token_id, token in advanced_api_gateway.authentication_tokens.items():
            tokens.append({
                "token_id": token.token_id,
                "user_id": token.user_id,
                "token_type": token.token_type,
                "expires_at": token.expires_at.isoformat(),
                "scopes": token.scopes,
                "created_at": token.created_at.isoformat(),
                "metadata": token.metadata
            })
            
        return JSONResponse(content={"tokens": tokens})
        
    except Exception as e:
        logger.error(f"Error getting authentication tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/middleware")
async def get_middleware_status():
    """Get middleware status"""
    try:
        middleware_status = {
            "available_middleware": list(advanced_api_gateway.middleware.keys()),
            "config": {
                "rate_limiting_enabled": advanced_api_gateway.config["rate_limiting_enabled"],
                "authentication_enabled": advanced_api_gateway.config["authentication_enabled"],
                "request_logging_enabled": advanced_api_gateway.config["request_logging_enabled"],
                "caching_enabled": advanced_api_gateway.config["caching_enabled"],
                "load_balancing_enabled": advanced_api_gateway.config["load_balancing_enabled"],
                "circuit_breaker_enabled": advanced_api_gateway.config["circuit_breaker_enabled"]
            }
        }
        
        return JSONResponse(content=middleware_status)
        
    except Exception as e:
        logger.error(f"Error getting middleware status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_api_gateway_dashboard():
    """Get API gateway dashboard data"""
    try:
        summary = advanced_api_gateway.get_gateway_summary()
        
        # Get recent requests
        recent_requests = advanced_api_gateway.request_logs[-10:]
        
        # Get route statistics
        route_stats = {}
        for route in advanced_api_gateway.routes.values():
            route_stats[route.name] = {
                "path": route.path,
                "methods": route.methods,
                "authentication_required": route.authentication_required,
                "middleware": route.middleware
            }
        
        dashboard_data = {
            "summary": summary,
            "recent_requests": [
                {
                    "request_id": request.request_id,
                    "method": request.method,
                    "path": request.path,
                    "client_ip": request.client_ip,
                    "response_time": request.response_time,
                    "status_code": request.status_code,
                    "timestamp": request.timestamp.isoformat()
                }
                for request in recent_requests
            ],
            "route_stats": route_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error getting API gateway dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_api_gateway():
    """Start API gateway"""
    try:
        await advanced_api_gateway.start_gateway()
        return JSONResponse(content={"message": "API gateway started"})
        
    except Exception as e:
        logger.error(f"Error starting API gateway: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_api_gateway():
    """Stop API gateway"""
    try:
        await advanced_api_gateway.stop_gateway()
        return JSONResponse(content={"message": "API gateway stopped"})
        
    except Exception as e:
        logger.error(f"Error stopping API gateway: {e}")
        raise HTTPException(status_code=500, detail=str(e))
