"""
Base service class for Opinion Market services
Provides common functionality for all services
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from app.core.database import get_db_session
from app.core.cache import cache, cached
from app.core.logging import LoggerMixin, PerformanceLogger, log_system_metric
from app.core.config import settings

T = TypeVar('T')


class BaseService(ABC, LoggerMixin):
    """Base service class with common functionality"""
    
    def __init__(self):
        self.logger = self.logger
        self._initialized = False
        self._startup_time = None
        self._background_tasks = []
    
    async def initialize(self) -> None:
        """Initialize the service"""
        if self._initialized:
            return
        
        self._startup_time = datetime.utcnow()
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
        try:
            await self._initialize_internal()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            raise
    
    @abstractmethod
    async def _initialize_internal(self) -> None:
        """Internal initialization logic - to be implemented by subclasses"""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup service resources"""
        self.logger.info(f"Cleaning up {self.__class__.__name__}")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        try:
            await self._cleanup_internal()
        except Exception as e:
            self.logger.error(f"Error during cleanup of {self.__class__.__name__}: {e}")
        
        self._initialized = False
        self.logger.info(f"{self.__class__.__name__} cleanup completed")
    
    async def _cleanup_internal(self) -> None:
        """Internal cleanup logic - to be implemented by subclasses"""
        pass
    
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._initialized
    
    def get_uptime(self) -> Optional[float]:
        """Get service uptime in seconds"""
        if self._startup_time:
            return (datetime.utcnow() - self._startup_time).total_seconds()
        return None
    
    def add_background_task(self, coro) -> None:
        """Add a background task"""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "service": self.__class__.__name__,
            "initialized": self._initialized,
            "uptime": self.get_uptime(),
            "background_tasks": len(self._background_tasks)
        }


class CRUDService(BaseService, Generic[T]):
    """Base CRUD service with database operations"""
    
    def __init__(self, model_class: type):
        super().__init__()
        self.model_class = model_class
    
    async def _initialize_internal(self) -> None:
        """Initialize CRUD service"""
        pass
    
    async def _cleanup_internal(self) -> None:
        """Cleanup CRUD service"""
        pass
    
    def get_by_id(self, db: Session, id: int) -> Optional[T]:
        """Get entity by ID"""
        with PerformanceLogger(f"get_{self.model_class.__name__}_by_id"):
            return db.query(self.model_class).filter(self.model_class.id == id).first()
    
    def get_by_ids(self, db: Session, ids: List[int]) -> List[T]:
        """Get entities by IDs"""
        with PerformanceLogger(f"get_{self.model_class.__name__}_by_ids"):
            return db.query(self.model_class).filter(self.model_class.id.in_(ids)).all()
    
    def get_all(self, db: Session, skip: int = 0, limit: int = 100) -> List[T]:
        """Get all entities with pagination"""
        with PerformanceLogger(f"get_all_{self.model_class.__name__}"):
            return db.query(self.model_class).offset(skip).limit(limit).all()
    
    def count(self, db: Session) -> int:
        """Count total entities"""
        with PerformanceLogger(f"count_{self.model_class.__name__}"):
            return db.query(self.model_class).count()
    
    def create(self, db: Session, obj_in: Dict[str, Any]) -> T:
        """Create new entity"""
        with PerformanceLogger(f"create_{self.model_class.__name__}"):
            db_obj = self.model_class(**obj_in)
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj
    
    def update(self, db: Session, db_obj: T, obj_in: Dict[str, Any]) -> T:
        """Update entity"""
        with PerformanceLogger(f"update_{self.model_class.__name__}"):
            for field, value in obj_in.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            db.commit()
            db.refresh(db_obj)
            return db_obj
    
    def delete(self, db: Session, id: int) -> bool:
        """Delete entity by ID"""
        with PerformanceLogger(f"delete_{self.model_class.__name__}"):
            obj = db.query(self.model_class).filter(self.model_class.id == id).first()
            if obj:
                db.delete(obj)
                db.commit()
                return True
            return False
    
    def search(self, db: Session, query: str, fields: List[str], skip: int = 0, limit: int = 100) -> List[T]:
        """Search entities by text in specified fields"""
        with PerformanceLogger(f"search_{self.model_class.__name__}"):
            search_conditions = []
            search_term = f"%{query}%"
            
            for field in fields:
                if hasattr(self.model_class, field):
                    search_conditions.append(getattr(self.model_class, field).ilike(search_term))
            
            if not search_conditions:
                return []
            
            return (
                db.query(self.model_class)
                .filter(or_(*search_conditions))
                .offset(skip)
                .limit(limit)
                .all()
            )


class CacheableService(BaseService):
    """Service with caching capabilities"""
    
    def __init__(self, cache_ttl: int = 300):
        super().__init__()
        self.cache_ttl = cache_ttl
        self.cache_prefix = self.__class__.__name__.lower()
    
    def get_cache_key(self, *parts: str) -> str:
        """Generate cache key"""
        return f"{self.cache_prefix}:{':'.join(str(part) for part in parts)}"
    
    def get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not cache:
            return None
        return cache.get(key)
    
    def set_cached(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not cache:
            return False
        return cache.set(key, value, ttl or self.cache_ttl)
    
    def delete_cached(self, key: str) -> bool:
        """Delete value from cache"""
        if not cache:
            return False
        return cache.delete(key)
    
    def clear_cache_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern"""
        if not cache:
            return 0
        return cache.delete_pattern(pattern)


class MetricsService(BaseService):
    """Service with metrics collection"""
    
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.metrics_start_time = datetime.utcnow()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "timestamp": datetime.utcnow(),
            "tags": tags or {}
        })
        
        # Log to system metrics
        log_system_metric(f"{self.__class__.__name__}_{name}", value, tags or {})
    
    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metric summary"""
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        values = [m["value"] for m in self.metrics[name]]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return {
            name: self.get_metric_summary(name)
            for name in self.metrics.keys()
        }


class BackgroundTaskService(BaseService):
    """Service with background task management"""
    
    def __init__(self):
        super().__init__()
        self.tasks = {}
        self.task_intervals = {}
    
    async def _initialize_internal(self) -> None:
        """Initialize background tasks"""
        pass
    
    async def _cleanup_internal(self) -> None:
        """Stop all background tasks"""
        for task_name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    def start_background_task(self, name: str, coro, interval: Optional[int] = None):
        """Start a background task"""
        if name in self.tasks and not self.tasks[name].done():
            self.logger.warning(f"Task {name} is already running")
            return
        
        async def task_wrapper():
            try:
                if interval:
                    while True:
                        await coro()
                        await asyncio.sleep(interval)
                else:
                    await coro()
            except asyncio.CancelledError:
                self.logger.info(f"Background task {name} cancelled")
            except Exception as e:
                self.logger.error(f"Background task {name} failed: {e}")
        
        task = asyncio.create_task(task_wrapper())
        self.tasks[name] = task
        self.task_intervals[name] = interval
        
        self.logger.info(f"Started background task: {name}")
    
    def stop_background_task(self, name: str):
        """Stop a background task"""
        if name in self.tasks:
            self.tasks[name].cancel()
            del self.tasks[name]
            if name in self.task_intervals:
                del self.task_intervals[name]
            self.logger.info(f"Stopped background task: {name}")
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all background tasks"""
        return {
            name: {
                "running": not task.done(),
                "interval": self.task_intervals.get(name),
                "exception": task.exception() if task.done() and task.exception() else None
            }
            for name, task in self.tasks.items()
        }


class ServiceRegistry:
    """Registry for managing services"""
    
    def __init__(self):
        self.services = {}
        self._initialized = False
    
    def register(self, name: str, service: BaseService):
        """Register a service"""
        self.services[name] = service
        self.logger.info(f"Registered service: {name}")
    
    def get(self, name: str) -> Optional[BaseService]:
        """Get a service by name"""
        return self.services.get(name)
    
    async def initialize_all(self):
        """Initialize all registered services"""
        if self._initialized:
            return
        
        for name, service in self.services.items():
            try:
                await service.initialize()
            except Exception as e:
                self.logger.error(f"Failed to initialize service {name}: {e}")
        
        self._initialized = True
    
    async def cleanup_all(self):
        """Cleanup all registered services"""
        for name, service in self.services.items():
            try:
                await service.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup service {name}: {e}")
        
        self._initialized = False
    
    def get_all_health_status(self) -> Dict[str, Any]:
        """Get health status of all services"""
        return {
            name: service.get_health_status()
            for name, service in self.services.items()
        }


# Global service registry
service_registry = ServiceRegistry()