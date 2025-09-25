"""
API Gateway Implementation
Routes requests to appropriate microservices with middleware support
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api_gateway.middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    LoggingMiddleware,
    SecurityMiddleware,
    MetricsMiddleware
)
from app.api_gateway.router import ServiceRouter
from app.services.service_registry import service_registry
from app.services.inter_service_communication import inter_service_comm

logger = logging.getLogger(__name__)


class APIGateway:
    """API Gateway for microservices routing"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Opinion Market API Gateway",
            version="1.0.0",
            description="API Gateway for Opinion Market Microservices"
        )
        
        self.middleware_stack: List[Callable] = []
        self.service_routers: Dict[str, ServiceRouter] = {}
        self.routing_rules: Dict[str, str] = {}
        
        # Initialize middleware
        self._setup_middleware()
        
        # Setup routing
        self._setup_routing()
        
        # Setup error handlers
        self._setup_error_handlers()
    
    def _setup_middleware(self):
        """Setup middleware stack"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Custom middleware
        self.middleware_stack = [
            SecurityMiddleware(),
            RateLimitMiddleware(),
            AuthenticationMiddleware(),
            LoggingMiddleware(),
            MetricsMiddleware()
        ]
        
        # Add middleware to FastAPI app
        for middleware in self.middleware_stack:
            self.app.middleware("http")(middleware)
    
    def _setup_routing(self):
        """Setup service routing rules"""
        self.routing_rules = {
            "/api/v1/markets": "market-service",
            "/api/v1/trades": "trading-service", 
            "/api/v1/users": "user-service",
            "/api/v1/auth": "auth-service",
            "/api/v1/notifications": "notification-service",
            "/api/v1/analytics": "analytics-service",
            "/api/v1/admin": "admin-service"
        }
        
        # Create service routers
        for path_prefix, service_name in self.routing_rules.items():
            router = ServiceRouter(service_name, path_prefix)
            self.service_routers[service_name] = router
            self.app.include_router(router.router, prefix=path_prefix)
    
    def _setup_error_handlers(self):
        """Setup error handlers"""
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "success": False,
                    "error": exc.detail,
                    "timestamp": time.time(),
                    "path": str(request.url)
                }
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal server error",
                    "timestamp": time.time(),
                    "path": str(request.url)
                }
            )
    
    async def start(self):
        """Start the API Gateway"""
        logger.info("Starting API Gateway")
        
        # Initialize service registry
        await service_registry.start_health_monitoring()
        
        # Initialize inter-service communication
        await inter_service_comm.initialize()
        
        # Register gateway service
        await service_registry.register_service(
            service_name="api-gateway",
            instance_id="gateway-1",
            host="localhost",
            port=8000,
            version="1.0.0",
            metadata={"type": "gateway"}
        )
        
        logger.info("API Gateway started successfully")
    
    async def stop(self):
        """Stop the API Gateway"""
        logger.info("Stopping API Gateway")
        
        # Unregister gateway service
        await service_registry.unregister_service("api-gateway", "gateway-1")
        
        # Cleanup inter-service communication
        await inter_service_comm.cleanup()
        
        logger.info("API Gateway stopped")
    
    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        stats = {
            "routing_rules": self.routing_rules,
            "service_routers": {
                name: router.get_stats() 
                for name, router in self.service_routers.items()
            },
            "middleware_count": len(self.middleware_stack),
            "service_registry": service_registry.get_registry_status(),
            "communication_stats": inter_service_comm.get_communication_stats()
        }
        
        return stats


# Global API Gateway instance
api_gateway = APIGateway()
