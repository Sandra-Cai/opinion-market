"""
Base Service Class for Microservices Architecture
Provides common functionality for all microservices
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core.enhanced_cache import enhanced_cache
from app.core.security_manager import security_manager

logger = logging.getLogger(__name__)


@dataclass
class ServiceHealth:
    """Service health status"""
    service_name: str
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: float
    version: str
    uptime: float
    metrics: Dict[str, Any]
    dependencies: List[str]


@dataclass
class ServiceMetrics:
    """Service performance metrics"""
    request_count: int
    error_count: int
    average_response_time: float
    memory_usage: float
    cpu_usage: float
    active_connections: int


class BaseService(ABC):
    """Base class for all microservices"""
    
    def __init__(self, service_name: str, version: str = "1.0.0"):
        self.service_name = service_name
        self.version = version
        self.start_time = time.time()
        self.metrics = ServiceMetrics(
            request_count=0,
            error_count=0,
            average_response_time=0.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            active_connections=0
        )
        self.dependencies = []
        self.is_healthy = True
        self.health_checks = []
        
        # Service configuration
        self.config = {
            "max_retries": 3,
            "timeout": 30,
            "circuit_breaker_threshold": 5,
            "health_check_interval": 30
        }
    
    async def start(self):
        """Start the service"""
        logger.info(f"Starting service: {self.service_name} v{self.version}")
        
        # Initialize service-specific components
        await self.initialize()
        
        # Start health monitoring
        asyncio.create_task(self._health_monitoring_loop())
        
        # Start metrics collection
        asyncio.create_task(self._metrics_collection_loop())
        
        logger.info(f"Service {self.service_name} started successfully")
    
    async def stop(self):
        """Stop the service"""
        logger.info(f"Stopping service: {self.service_name}")
        
        # Cleanup service-specific resources
        await self.cleanup()
        
        logger.info(f"Service {self.service_name} stopped")
    
    @abstractmethod
    async def initialize(self):
        """Initialize service-specific components"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup service-specific resources"""
        pass
    
    async def health_check(self) -> ServiceHealth:
        """Perform comprehensive health check"""
        try:
            # Check service-specific health
            service_health = await self.check_service_health()
            
            # Check dependencies
            dependency_health = await self.check_dependencies()
            
            # Calculate overall health status
            if not service_health or not dependency_health:
                status = "unhealthy"
            elif any(dep.get("status") != "healthy" for dep in dependency_health):
                status = "degraded"
            else:
                status = "healthy"
            
            health = ServiceHealth(
                service_name=self.service_name,
                status=status,
                timestamp=time.time(),
                version=self.version,
                uptime=time.time() - self.start_time,
                metrics=self.metrics.__dict__,
                dependencies=[dep["name"] for dep in dependency_health]
            )
            
            # Cache health status
            await enhanced_cache.set(
                f"health_{self.service_name}",
                health,
                ttl=60,
                tags=["health", "service"]
            )
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed for {self.service_name}: {e}")
            return ServiceHealth(
                service_name=self.service_name,
                status="unhealthy",
                timestamp=time.time(),
                version=self.version,
                uptime=time.time() - self.start_time,
                metrics=self.metrics.__dict__,
                dependencies=[]
            )
    
    async def check_service_health(self) -> bool:
        """Check service-specific health (to be implemented by subclasses)"""
        return True
    
    async def check_dependencies(self) -> List[Dict[str, Any]]:
        """Check health of service dependencies"""
        dependency_health = []
        
        for dependency in self.dependencies:
            try:
                # Check if dependency is healthy
                health_data = await enhanced_cache.get(f"health_{dependency}")
                if health_data:
                    dependency_health.append({
                        "name": dependency,
                        "status": health_data.status,
                        "last_check": health_data.timestamp
                    })
                else:
                    dependency_health.append({
                        "name": dependency,
                        "status": "unknown",
                        "last_check": time.time()
                    })
            except Exception as e:
                logger.error(f"Failed to check dependency {dependency}: {e}")
                dependency_health.append({
                    "name": dependency,
                    "status": "unhealthy",
                    "last_check": time.time()
                })
        
        return dependency_health
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                health = await self.health_check()
                self.is_healthy = health.status == "healthy"
                
                # Log health status changes
                if health.status != "healthy":
                    logger.warning(f"Service {self.service_name} health status: {health.status}")
                
                await asyncio.sleep(self.config["health_check_interval"])
                
            except Exception as e:
                logger.error(f"Error in health monitoring for {self.service_name}: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while True:
            try:
                await self._collect_metrics()
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
            except Exception as e:
                logger.error(f"Error collecting metrics for {self.service_name}: {e}")
                await asyncio.sleep(30)
    
    async def _collect_metrics(self):
        """Collect service metrics"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Update metrics
            self.metrics.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            self.metrics.cpu_usage = process.cpu_percent()
            self.metrics.active_connections = len(process.connections())
            
            # Cache metrics
            await enhanced_cache.set(
                f"metrics_{self.service_name}",
                self.metrics,
                ttl=60,
                tags=["metrics", "service"]
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming request with metrics tracking"""
        start_time = time.time()
        
        try:
            # Check rate limits
            client_ip = request_data.get("client_ip", "unknown")
            rate_limit_ok = await security_manager.check_rate_limit(client_ip, "api")
            
            if not rate_limit_ok:
                self.metrics.error_count += 1
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "service": self.service_name
                }
            
            # Process request
            result = await self.process_request(request_data)
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics.request_count += 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.request_count - 1) + response_time) 
                / self.metrics.request_count
            )
            
            return {
                "success": True,
                "data": result,
                "service": self.service_name,
                "response_time": response_time
            }
            
        except Exception as e:
            # Update error metrics
            self.metrics.error_count += 1
            logger.error(f"Error handling request in {self.service_name}: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "service": self.service_name
            }
    
    @abstractmethod
    async def process_request(self, request_data: Dict[str, Any]) -> Any:
        """Process service-specific request (to be implemented by subclasses)"""
        pass
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "name": self.service_name,
            "version": self.version,
            "uptime": time.time() - self.start_time,
            "is_healthy": self.is_healthy,
            "dependencies": self.dependencies,
            "metrics": self.metrics.__dict__
        }
    
    async def register_service(self):
        """Register service with service registry"""
        try:
            service_info = self.get_service_info()
            await enhanced_cache.set(
                f"service_{self.service_name}",
                service_info,
                ttl=300,  # 5 minutes
                tags=["service", "registry"]
            )
            logger.info(f"Service {self.service_name} registered")
        except Exception as e:
            logger.error(f"Failed to register service {self.service_name}: {e}")
    
    async def unregister_service(self):
        """Unregister service from service registry"""
        try:
            await enhanced_cache.delete(f"service_{self.service_name}")
            logger.info(f"Service {self.service_name} unregistered")
        except Exception as e:
            logger.error(f"Failed to unregister service {self.service_name}: {e}")
