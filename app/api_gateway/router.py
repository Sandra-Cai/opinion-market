"""
Service Router for API Gateway
Routes requests to appropriate microservices
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import JSONResponse

from app.services.service_registry import service_registry
from app.services.inter_service_communication import inter_service_comm

logger = logging.getLogger(__name__)


class ServiceRouter:
    """Router for specific microservice"""
    
    def __init__(self, service_name: str, path_prefix: str):
        self.service_name = service_name
        self.path_prefix = path_prefix
        self.router = APIRouter()
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup routing for the service"""
        
        @self.router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def route_to_service(request: Request, path: str):
            """Route request to microservice"""
            return await self._handle_request(request, path)
    
    async def _handle_request(self, request: Request, path: str) -> JSONResponse:
        """Handle request routing to microservice"""
        start_time = time.time()
        
        try:
            # Get service instance
            instance = await service_registry.get_service_instance(self.service_name)
            if not instance:
                raise HTTPException(
                    status_code=503,
                    detail=f"Service {self.service_name} is not available"
                )
            
            # Prepare request data
            request_data = await self._prepare_request_data(request, path)
            
            # Call microservice
            response = await inter_service_comm.call_service(
                service_name=self.service_name,
                endpoint=f"/{path}",
                method=request.method,
                data=request_data.get("body"),
                headers=request_data.get("headers")
            )
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(response_time, False)
            
            # Return response
            return JSONResponse(
                status_code=response.get("status_code", 200),
                content=response.get("data", {})
            )
            
        except HTTPException:
            raise
        except Exception as e:
            # Update error metrics
            response_time = time.time() - start_time
            self._update_metrics(response_time, True)
            
            logger.error(f"Error routing to {self.service_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error communicating with {self.service_name}"
            )
    
    async def _prepare_request_data(self, request: Request, path: str) -> Dict[str, Any]:
        """Prepare request data for microservice"""
        try:
            # Get request body
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                body_bytes = await request.body()
                if body_bytes:
                    try:
                        body = json.loads(body_bytes.decode("utf-8"))
                    except json.JSONDecodeError:
                        body = body_bytes.decode("utf-8")
            
            # Get headers (filter out some headers)
            headers = {}
            excluded_headers = {"host", "content-length", "connection"}
            for name, value in request.headers.items():
                if name.lower() not in excluded_headers:
                    headers[name] = value
            
            # Add user information if available
            if hasattr(request.state, 'user'):
                headers["X-User-ID"] = request.state.user.get("user_id", "")
                headers["X-Username"] = request.state.user.get("username", "")
            
            return {
                "body": body,
                "headers": headers,
                "query_params": dict(request.query_params),
                "path": path,
                "method": request.method
            }
            
        except Exception as e:
            logger.error(f"Error preparing request data: {e}")
            return {}
    
    def _update_metrics(self, response_time: float, is_error: bool):
        """Update router metrics"""
        self.request_count += 1
        self.response_times.append(response_time)
        
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        if is_error:
            self.error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            "service_name": self.service_name,
            "path_prefix": self.path_prefix,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "average_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "response_times_count": len(self.response_times)
        }
