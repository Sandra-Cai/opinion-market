"""
Service Discovery Mock
Simple mock implementation for service discovery functionality
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ServiceInstance:
    """Service instance data structure"""
    service_name: str
    instance_id: str
    host: str
    port: int
    version: str
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = None
    last_heartbeat: datetime = None


class ServiceRegistry:
    """Simple service registry implementation"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.health_check_interval = 30
        self.health_check_task: Optional[asyncio.Task] = None
        
    async def register_service(self, service_name: str, instance_id: str, 
                             host: str, port: int, version: str = "1.0.0",
                             health_check_url: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """Register a service instance"""
        instance = ServiceInstance(
            service_name=service_name,
            instance_id=instance_id,
            host=host,
            port=port,
            version=version,
            health_check_url=health_check_url,
            metadata=metadata or {},
            last_heartbeat=datetime.now()
        )
        
        if service_name not in self.services:
            self.services[service_name] = []
            
        # Remove existing instance with same ID
        self.services[service_name] = [
            s for s in self.services[service_name] 
            if s.instance_id != instance_id
        ]
        
        self.services[service_name].append(instance)
        
    async def unregister_service(self, service_name: str, instance_id: str):
        """Unregister a service instance"""
        if service_name in self.services:
            self.services[service_name] = [
                s for s in self.services[service_name] 
                if s.instance_id != instance_id
            ]
            
    async def get_service_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Get a service instance"""
        if service_name in self.services and self.services[service_name]:
            return self.services[service_name][0]  # Return first available instance
        return None
        
    async def get_all_service_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get all service instances for a service"""
        return self.services.get(service_name, [])
        
    async def discover_services(self) -> Dict[str, List[ServiceInstance]]:
        """Discover all registered services"""
        return self.services.copy()
        
    async def health_check(self, instance: ServiceInstance) -> bool:
        """Perform health check on a service instance"""
        # Simple mock health check - always return True
        instance.last_heartbeat = datetime.now()
        return True
        
    async def start_health_checking(self):
        """Start health checking task"""
        if self.health_check_task is None:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
    async def stop_health_checking(self):
        """Stop health checking task"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            
    async def _health_check_loop(self):
        """Health check loop"""
        while True:
            try:
                for service_name, instances in self.services.items():
                    for instance in instances:
                        await self.health_check(instance)
                        
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)


class ServiceDiscovery:
    """Service discovery client"""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        
    async def register_service(self, service_name: str, instance_id: str, 
                             host: str, port: int, version: str = "1.0.0",
                             health_check_url: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """Register a service instance"""
        await self.registry.register_service(
            service_name, instance_id, host, port, version,
            health_check_url, metadata
        )
        
    async def unregister_service(self, service_name: str, instance_id: str):
        """Unregister a service instance"""
        await self.registry.unregister_service(service_name, instance_id)
        
    async def discover_service(self, service_name: str) -> Optional[ServiceInstance]:
        """Discover a service instance"""
        return await self.registry.get_service_instance(service_name)
        
    async def discover_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """Discover all services"""
        return await self.registry.discover_services()


# Global instances
service_registry = ServiceRegistry()
service_discovery = ServiceDiscovery()
