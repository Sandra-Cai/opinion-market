"""
Inter-Service Communication for Microservices Architecture
Handles communication between microservices with retry, circuit breaker, and load balancing
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
import hashlib

from app.services.service_registry import service_registry, ServiceInstance

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.next_attempt_time = 0
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if time.time() < self.next_attempt_time:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")
        else:
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class InterServiceCommunication:
    """Inter-service communication manager"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_config = RetryConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize communication components"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
        )
        
        # Start cache cleanup task
        asyncio.create_task(self._cache_cleanup_loop())
    
    async def cleanup(self):
        """Cleanup communication components"""
        if self.session:
            await self.session.close()
    
    async def call_service(self, service_name: str, endpoint: str, 
                          method: str = "GET", data: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None,
                          use_cache: bool = False,
                          cache_ttl: int = 300) -> Dict[str, Any]:
        """Call another microservice"""
        try:
            # Check cache first
            if use_cache and method == "GET":
                cache_key = self._generate_cache_key(service_name, endpoint, data)
                cached_result = self.request_cache.get(cache_key)
                if cached_result and time.time() - cached_result["timestamp"] < cache_ttl:
                    return cached_result["data"]
            
            # Get service instance
            instance = await service_registry.get_service_instance(service_name)
            if not instance:
                raise Exception(f"No healthy instances found for service: {service_name}")
            
            # Build URL
            url = f"http://{instance.host}:{instance.port}{endpoint}"
            
            # Get or create circuit breaker
            circuit_breaker = self._get_circuit_breaker(service_name)
            
            # Make request with circuit breaker protection
            result = await circuit_breaker.call(
                self._make_http_request,
                url, method, data, headers
            )
            
            # Cache result if requested
            if use_cache and method == "GET":
                self.request_cache[cache_key] = {
                    "data": result,
                    "timestamp": time.time()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to call service {service_name}: {e}")
            raise e
    
    async def _make_http_request(self, url: str, method: str, 
                               data: Optional[Dict[str, Any]] = None,
                               headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Prepare request
                request_headers = {"Content-Type": "application/json"}
                if headers:
                    request_headers.update(headers)
                
                # Make request
                if method.upper() == "GET":
                    async with self.session.get(url, headers=request_headers) as response:
                        return await self._handle_response(response)
                elif method.upper() == "POST":
                    async with self.session.post(url, json=data, headers=request_headers) as response:
                        return await self._handle_response(response)
                elif method.upper() == "PUT":
                    async with self.session.put(url, json=data, headers=request_headers) as response:
                        return await self._handle_response(response)
                elif method.upper() == "DELETE":
                    async with self.session.delete(url, headers=request_headers) as response:
                        return await self._handle_response(response)
                else:
                    raise Exception(f"Unsupported HTTP method: {method}")
                    
            except Exception as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    delay = min(
                        self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
                        self.retry_config.max_delay
                    )
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Request failed after {self.retry_config.max_retries + 1} attempts: {e}")
        
        raise last_exception
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle HTTP response"""
        try:
            response_data = await response.json()
            
            if response.status >= 400:
                raise Exception(f"HTTP {response.status}: {response_data.get('error', 'Unknown error')}")
            
            return {
                "status_code": response.status,
                "data": response_data,
                "headers": dict(response.headers)
            }
            
        except aiohttp.ContentTypeError:
            # Handle non-JSON responses
            text_data = await response.text()
            return {
                "status_code": response.status,
                "data": text_data,
                "headers": dict(response.headers)
            }
    
    def _get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self.circuit_breakers:
            config = CircuitBreakerConfig()
            self.circuit_breakers[service_name] = CircuitBreaker(config)
        
        return self.circuit_breakers[service_name]
    
    def _generate_cache_key(self, service_name: str, endpoint: str, 
                          data: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for request"""
        key_data = f"{service_name}:{endpoint}:{json.dumps(data or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _cache_cleanup_loop(self):
        """Background cache cleanup loop"""
        while True:
            try:
                current_time = time.time()
                expired_keys = []
                
                for key, value in self.request_cache.items():
                    if current_time - value["timestamp"] > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.request_cache[key]
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(60)
    
    async def broadcast_event(self, event_type: str, event_data: Dict[str, Any], 
                            target_services: Optional[List[str]] = None):
        """Broadcast event to multiple services"""
        try:
            if target_services is None:
                # Broadcast to all registered services
                target_services = list(service_registry.services.keys())
            
            tasks = []
            for service_name in target_services:
                task = asyncio.create_task(
                    self._send_event_to_service(service_name, event_type, event_data)
                )
                tasks.append(task)
            
            # Wait for all broadcasts to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to broadcast to {target_services[i]}: {result}")
                else:
                    logger.info(f"Successfully broadcasted to {target_services[i]}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")
    
    async def _send_event_to_service(self, service_name: str, event_type: str, 
                                   event_data: Dict[str, Any]):
        """Send event to specific service"""
        try:
            await self.call_service(
                service_name,
                "/events",
                method="POST",
                data={
                    "event_type": event_type,
                    "event_data": event_data,
                    "timestamp": time.time()
                }
            )
        except Exception as e:
            logger.error(f"Failed to send event to {service_name}: {e}")
            raise e
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        circuit_breaker_stats = {}
        for service_name, cb in self.circuit_breakers.items():
            circuit_breaker_stats[service_name] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count
            }
        
        return {
            "circuit_breakers": circuit_breaker_stats,
            "cache_size": len(self.request_cache),
            "retry_config": self.retry_config.__dict__
        }


# Global inter-service communication instance
inter_service_comm = InterServiceCommunication()
