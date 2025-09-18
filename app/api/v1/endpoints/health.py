"""
Health Check Endpoints
Provides comprehensive health monitoring for the Opinion Market API
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import psutil
import asyncio
from typing import Dict, Any, Optional

from app.core.database import get_db
from app.core.enhanced_config import enhanced_config_manager
from app.core.enhanced_error_handler import enhanced_error_handler
from app.core.advanced_performance_optimizer import advanced_performance_optimizer

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    Returns the current status of the API
    """
    return {
        "status": "healthy",
        "service": "opinion-market-api",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check endpoint
    Returns comprehensive system health information
    """
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get application metrics
        config_status = "healthy" if enhanced_config_manager else "unhealthy"
        error_handler_status = "healthy" if enhanced_error_handler else "unhealthy"
        performance_optimizer_status = "healthy" if advanced_performance_optimizer else "unhealthy"
        
        # Calculate overall health
        overall_status = "healthy"
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            overall_status = "degraded"
        
        if config_status == "unhealthy" or error_handler_status == "unhealthy":
            overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "service": "opinion-market-api",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "components": {
                "configuration": {
                    "status": config_status,
                    "environment": enhanced_config_manager.get("environment", "unknown")
                },
                "error_handler": {
                    "status": error_handler_status,
                    "error_count": len(enhanced_error_handler.error_history)
                },
                "performance_optimizer": {
                    "status": performance_optimizer_status,
                    "monitoring_active": advanced_performance_optimizer.monitoring_active
                }
            },
            "system": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "uptime": "running"  # This would be calculated from actual start time
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "opinion-market-api",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/health/readiness")
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes/container orchestration
    Returns whether the service is ready to accept traffic
    """
    try:
        # Check if all critical components are ready
        config_ready = enhanced_config_manager is not None
        error_handler_ready = enhanced_error_handler is not None
        
        # Check system resources
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resources_ready = memory.percent < 95 and disk.percent < 95
        
        ready = config_ready and error_handler_ready and resources_ready
        
        return {
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "configuration": config_ready,
                "error_handler": error_handler_ready,
                "system_resources": resources_ready
            }
        }
        
    except Exception as e:
        return {
            "ready": False,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/health/liveness")
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes/container orchestration
    Returns whether the service is alive and should not be restarted
    """
    try:
        # Simple check - if we can respond, we're alive
        return {
            "alive": True,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "opinion-market-api"
        }
        
    except Exception as e:
        return {
            "alive": False,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/health/database")
async def database_health_check(db: Session = Depends(get_db)):
    """
    Database health check endpoint
    Tests database connectivity and basic operations
    """
    try:
        # Test database connection with a simple query
        result = db.execute("SELECT 1 as test").fetchone()
        
        if result and result[0] == 1:
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "unhealthy",
                "database": "connection_failed",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/health/metrics")
async def health_metrics():
    """
    Health metrics endpoint
    Returns key performance and health metrics
    """
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get application metrics
        error_count = len(enhanced_error_handler.error_history)
        performance_snapshots = len(advanced_performance_optimizer.performance_history)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "application_metrics": {
                "error_count": error_count,
                "performance_snapshots": performance_snapshots,
                "configuration_loaded": enhanced_config_manager is not None
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
