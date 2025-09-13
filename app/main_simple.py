from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.core.performance_optimizer import performance_monitor
from app.core.health_monitor import health_monitor
from app.core.metrics import metrics_collector, MetricsMiddleware
from app.core.rate_limiter import rate_limiter
from app.core.caching import memory_cache
from app.core.security import SecurityManager, SecurityHeaders
from app.core.database_pool import db_pool_manager
from app.core.logging_config import app_logger, LoggingMiddleware
from app.core.config_manager import config_manager
from app.api.v1.endpoints.analytics_enhanced import router as analytics_router
from app.api.v1.endpoints.security import router as security_router

app = FastAPI(
    title="Opinion Market API",
    description="A comprehensive prediction market platform with advanced features",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
app.add_middleware(MetricsMiddleware, metrics_collector=metrics_collector)

# Add logging middleware
app.add_middleware(LoggingMiddleware, logger=app_logger)

# Include enhanced analytics router
app.include_router(analytics_router, prefix="/api/v1", tags=["Enhanced Analytics"])

# Include security router
app.include_router(security_router, prefix="/api/v1", tags=["Security & Authentication"])


@app.get("/")
@performance_monitor
async def root():
    """Root endpoint with enhanced information"""
    # Get configuration
    config = config_manager.get_config()
    
    # Get database health
    db_health = await db_pool_manager.health_check()
    
    return {
        "message": "Welcome to Opinion Market API",
        "version": config.api.version,
        "environment": config.environment.value,
        "description": "A comprehensive prediction market platform with advanced features",
        "status": "operational",
        "database_status": db_health["status"],
        "docs": config.api.docs_url,
        "redoc": config.api.redoc_url,
        "openapi": config.api.openapi_url,
        "features": [
            "Prediction Markets",
            "Real-time Trading",
            "AI Analytics",
            "Social Features",
            "Blockchain Integration",
            "Advanced Orders",
            "Enterprise Security",
            "Performance Optimization",
            "Advanced Caching",
            "Structured Logging",
            "Rate Limiting",
            "Health Monitoring"
        ],
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "analytics": "/api/v1/analytics"
        }
    }


@app.get("/health")
@performance_monitor
async def health_check():
    """Enhanced health check endpoint"""
    health_status = await health_monitor.get_comprehensive_health()
    return health_status


@app.get("/ready")
async def readiness_check():
    return {"status": "ready", "service": "opinion-market-api"}


@app.get("/metrics")
@performance_monitor
async def metrics():
    """Get comprehensive application metrics"""
    # Collect system metrics
    import psutil
    import time
    
    # Get metrics from collector
    metrics_data = await metrics_collector.get_metrics()
    
    # Add system metrics
    metrics_data['system'] = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'uptime': time.time()
    }
    
    # Add cache statistics
    cache_stats = await memory_cache.get_stats()
    metrics_data['cache'] = cache_stats
    
    return metrics_data


@app.get("/api/v1/")
async def api_root():
    return {
        "message": "Opinion Market API v1",
        "endpoints": ["/health", "/ready", "/metrics", "/docs"],
    }


@app.get("/api/v1/health")
async def api_health():
    return {"status": "healthy", "api_version": "v1"}


@app.get("/api/v1/markets")
async def get_markets():
    return {
        "markets": [
            {
                "id": 1,
                "title": "Will Bitcoin reach $100k by end of year?",
                "description": "Prediction market for Bitcoin price",
                "status": "active",
                "total_volume": 1000000,
                "participant_count": 150,
            },
            {
                "id": 2,
                "title": "Will Tesla deliver 2M vehicles in 2024?",
                "description": "Tesla delivery prediction",
                "status": "active",
                "total_volume": 750000,
                "participant_count": 89,
            },
        ]
    }


if __name__ == "__main__":
    uvicorn.run("app.main_simple:app", host="0.0.0.0", port=8000, reload=True)
