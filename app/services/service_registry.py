"""
Service Registry for Microservices Architecture
Manages service registration, discovery, and health monitoring
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


@dataclass
class ServiceInstance:
    """Service instance information"""
    service_name: str
    instance_id: str
    host: str
    port: int
    version: str
    status: str  # "healthy", "degraded", "unhealthy"
    last_heartbeat: float
    metadata: Dict[str, Any]


class ServiceRegistry:
    """Service registry for microservices discovery"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.heartbeat_interval = 30  # seconds
        self.health_check_interval = 60  # seconds
        self.service_timeout = 120  # seconds
        
    async def register_service(self, service_name: str, instance_id: str, 
                             host: str, port: int, version: str = "1.0.0",
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a service instance"""
        try:
            instance = ServiceInstance(
                service_name=service_name,
                instance_id=instance_id,
                host=host,
                port=port,
                version=version,
                status="healthy",
                last_heartbeat=time.time(),
                metadata=metadata or {}
            )
            
            # Add to local registry
            if service_name not in self.services:
                self.services[service_name] = []
            
            # Remove existing instance with same ID
            self.services[service_name] = [
                inst for inst in self.services[service_name] 
                if inst.instance_id != instance_id
            ]
            
            # Add new instance
            self.services[service_name].append(instance)
            
            # Cache in distributed cache
            await self._cache_service_instance(instance)
            
            logger.info(f"Registered service instance: {service_name}:{instance_id} at {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service_name}:{instance_id}: {e}")
            return False
    
    async def unregister_service(self, service_name: str, instance_id: str) -> bool:
        """Unregister a service instance"""
        try:
            # Remove from local registry
            if service_name in self.services:
                self.services[service_name] = [
                    inst for inst in self.services[service_name] 
                    if inst.instance_id != instance_id
                ]
                
                if not self.services[service_name]:
                    del self.services[service_name]
            
            # Remove from distributed cache
            await enhanced_cache.delete(f"service_instance_{service_name}_{instance_id}")
            
            logger.info(f"Unregistered service instance: {service_name}:{instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister service {service_name}:{instance_id}: {e}")
            return False
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover healthy instances of a service"""
        try:
            # Get from local registry first
            local_instances = self.services.get(service_name, [])
            
            # Filter healthy instances
            healthy_instances = [
                inst for inst in local_instances
                if inst.status == "healthy" and 
                time.time() - inst.last_heartbeat < self.service_timeout
            ]
            
            # If no healthy instances locally, try distributed cache
            if not healthy_instances:
                healthy_instances = await self._discover_from_cache(service_name)
            
            return healthy_instances
            
        except Exception as e:
            logger.error(f"Failed to discover services for {service_name}: {e}")
            return []
    
    async def get_service_instance(self, service_name: str, 
                                 load_balancing: str = "round_robin") -> Optional[ServiceInstance]:
        """Get a service instance using load balancing"""
        try:
            instances = await self.discover_services(service_name)
            
            if not instances:
                return None
            
            if load_balancing == "round_robin":
                # Simple round-robin (could be improved with proper state)
                return instances[0]
            elif load_balancing == "random":
                import random
                return random.choice(instances)
            elif load_balancing == "least_connections":
                # Return instance with least connections (if metadata available)
                return min(instances, key=lambda x: x.metadata.get("connections", 0))
            else:
                return instances[0]
                
        except Exception as e:
            logger.error(f"Failed to get service instance for {service_name}: {e}")
            return None
    
    async def update_heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Update service heartbeat"""
        try:
            # Update local registry
            if service_name in self.services:
                for instance in self.services[service_name]:
                    if instance.instance_id == instance_id:
                        instance.last_heartbeat = time.time()
                        break
            
            # Update distributed cache
            cache_key = f"service_instance_{service_name}_{instance_id}"
            instance_data = await enhanced_cache.get(cache_key)
            if instance_data:
                instance_data.last_heartbeat = time.time()
                await enhanced_cache.set(cache_key, instance_data, ttl=300)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for {service_name}:{instance_id}: {e}")
            return False
    
    async def start_health_monitoring(self):
        """Start background health monitoring"""
        logger.info("Starting service registry health monitoring")
        asyncio.create_task(self._health_monitoring_loop())
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                await self._check_service_health()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _check_service_health(self):
        """Check health of all registered services"""
        try:
            current_time = time.time()
            
            for service_name, instances in self.services.items():
                for instance in instances:
                    # Check if instance is still alive
                    if current_time - instance.last_heartbeat > self.service_timeout:
                        instance.status = "unhealthy"
                        logger.warning(f"Service instance {service_name}:{instance.instance_id} is unhealthy")
                    else:
                        instance.status = "healthy"
                    
                    # Update cache
                    await self._cache_service_instance(instance)
            
        except Exception as e:
            logger.error(f"Error checking service health: {e}")
    
    async def _cache_service_instance(self, instance: ServiceInstance):
        """Cache service instance in distributed cache"""
        try:
            cache_key = f"service_instance_{instance.service_name}_{instance.instance_id}"
            await enhanced_cache.set(
                cache_key,
                instance,
                ttl=300,  # 5 minutes
                tags=["service", "registry", instance.service_name]
            )
        except Exception as e:
            logger.error(f"Failed to cache service instance: {e}")
    
    async def _discover_from_cache(self, service_name: str) -> List[ServiceInstance]:
        """Discover services from distributed cache"""
        try:
            # This would typically use a more sophisticated cache query
            # For now, we'll simulate by checking for cached instances
            instances = []
            
            # In a real implementation, you'd query the cache for all instances
            # of the service and filter by health status
            
            return instances
            
        except Exception as e:
            logger.error(f"Failed to discover services from cache: {e}")
            return []
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status and statistics"""
        total_services = len(self.services)
        total_instances = sum(len(instances) for instances in self.services.values())
        healthy_instances = sum(
            len([inst for inst in instances if inst.status == "healthy"])
            for instances in self.services.values()
        )
        
        return {
            "total_services": total_services,
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "unhealthy_instances": total_instances - healthy_instances,
            "services": {
                name: {
                    "total_instances": len(instances),
                    "healthy_instances": len([inst for inst in instances if inst.status == "healthy"]),
                    "instances": [
                        {
                            "id": inst.instance_id,
                            "host": inst.host,
                            "port": inst.port,
                            "status": inst.status,
                            "last_heartbeat": inst.last_heartbeat
                        }
                        for inst in instances
                    ]
                }
                for name, instances in self.services.items()
            }
        }


# Global service registry instance
service_registry = ServiceRegistry()
