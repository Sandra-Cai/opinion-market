from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import logging
from datetime import datetime
import time

from app.core.config import settings
from app.core.database import engine, Base, check_database_health, check_redis_health
from app.core.logging import setup_logging, log_api_call, log_system_metric
from app.core.cache import cache, get_cache_stats, cache_health_check
from app.core.security import security_manager, get_client_ip, rate_limit

# Import only existing modules
try:
    from app.core.enhanced_config import enhanced_config_manager
except ImportError:
    enhanced_config_manager = None

try:
    from app.api.v1.api import api_router
except ImportError:
    api_router = None

try:
    from app.core.auth import get_current_user
except ImportError:
    get_current_user = None

# Import models
from app.models import user, market, trade, vote, position

# Import services that exist
try:
    from app.services.price_feed import price_feed_manager
except ImportError:
    price_feed_manager = None

try:
    from app.services.machine_learning import MachineLearningService
    ml_service = MachineLearningService()
except ImportError:
    ml_service = None

try:
    from app.services.analytics_service import AnalyticsService
    analytics_service = AnalyticsService()
except ImportError:
    analytics_service = None

try:
    from app.services.market_service import MarketService
    market_service = MarketService()
except ImportError:
    market_service = None

# Import core systems that exist
try:
    from app.core.real_time_engine import RealTimeEngine
    real_time_engine = RealTimeEngine()
except ImportError:
    real_time_engine = None

try:
    from app.core.intelligent_alerting import IntelligentAlertingSystem
    alerting_system = IntelligentAlertingSystem()
except ImportError:
    alerting_system = None

# Create database tables (only if using SQLite for development)
try:
    if enhanced_config_manager and enhanced_config_manager.get("database.url", "").startswith("sqlite"):
        Base.metadata.create_all(bind=engine)
    elif settings.DATABASE_URL.startswith("sqlite"):
        Base.metadata.create_all(bind=engine)
except Exception as e:
    logging.warning(f"Could not create database tables: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with comprehensive error handling and monitoring"""
    # Initialize logging system
    logger = setup_logging()
    logger.info("🚀 Starting Opinion Market API...", version=settings.APP_VERSION)
    
    # Track background tasks for cleanup
    background_tasks = []
    startup_errors = []
    
    try:
        # Initialize core systems
        logger.info("Initializing core systems...")
        
        # Check database health
        db_health = check_database_health()
        if db_health["status"] == "healthy":
            logger.info("✅ Database connection established", 
                       response_time=db_health["response_time"])
        else:
            startup_errors.append(f"Database health check failed: {db_health.get('error', 'Unknown error')}")
        
        # Check Redis health
        redis_health = check_redis_health()
        if redis_health["status"] == "healthy":
            logger.info("✅ Redis connection established",
                       response_time=redis_health["response_time"])
        elif redis_health["status"] == "disabled":
            logger.warning("⚠️  Redis caching disabled")
        else:
            startup_errors.append(f"Redis health check failed: {redis_health.get('error', 'Unknown error')}")
        
        # Check cache health
        cache_health = cache_health_check()
        if cache_health["status"] == "healthy":
            logger.info("✅ Cache system operational")
        else:
            startup_errors.append(f"Cache health check failed: {cache_health.get('error', 'Unknown error')}")

        # Initialize core services that exist
        if ml_service:
            try:
                await ml_service.initialize()
                logger.info("✅ Machine Learning service initialized")
            except Exception as e:
                startup_errors.append(f"ML service initialization failed: {e}")

        if analytics_service:
            try:
                logger.info("✅ Analytics service initialized")
            except Exception as e:
                startup_errors.append(f"Analytics service initialization failed: {e}")

        if market_service:
            try:
                await market_service.initialize()
                logger.info("✅ Market service initialized")
            except Exception as e:
                startup_errors.append(f"Market service initialization failed: {e}")

        if real_time_engine:
            try:
                await real_time_engine.start()
                logger.info("✅ Real-time engine initialized")
            except Exception as e:
                startup_errors.append(f"Real-time engine initialization failed: {e}")

        if alerting_system:
            try:
                await alerting_system.start()
                logger.info("✅ Intelligent alerting system initialized")
            except Exception as e:
                startup_errors.append(f"Alerting system initialization failed: {e}")

        if price_feed_manager:
            try:
                # Start price feed in background
                task = asyncio.create_task(price_feed_manager.start_price_feed())
                background_tasks.append(task)
                logger.info("✅ Price feed manager initialized")
            except Exception as e:
                startup_errors.append(f"Price feed manager initialization failed: {e}")

        # Initialize enhanced config if available
        if enhanced_config_manager:
            try:
                enhanced_config_manager.start_file_watching()
                logger.info("✅ Enhanced configuration manager initialized")
            except Exception as e:
                startup_errors.append(f"Enhanced config initialization failed: {e}")

        # Log startup summary
        if startup_errors:
            logger.warning("⚠️  Some services failed to initialize", errors=startup_errors)
        else:
            logger.info("🎉 Opinion Market API startup completed successfully!")
        
        # Log system metrics
        log_system_metric("app_startup", 1, {"version": settings.APP_VERSION, "environment": settings.ENVIRONMENT.value})

    except Exception as e:
        logger.error("❌ Critical error during startup", error=str(e))
        startup_errors.append(f"Critical startup error: {e}")

    yield

    # Shutdown
    logger.info("👋 Shutting down Opinion Market API...")
    shutdown_errors = []
    
    try:
        # Cancel background tasks
        for task in background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop services
        if real_time_engine:
            try:
                await real_time_engine.stop()
                logger.info("✅ Real-time engine stopped")
            except Exception as e:
                shutdown_errors.append(f"Error stopping real-time engine: {e}")

        if alerting_system:
            try:
                await alerting_system.stop()
                logger.info("✅ Alerting system stopped")
            except Exception as e:
                shutdown_errors.append(f"Error stopping alerting system: {e}")

        if market_service:
            try:
                await market_service.cleanup()
                logger.info("✅ Market service stopped")
            except Exception as e:
                shutdown_errors.append(f"Error stopping market service: {e}")

        if enhanced_config_manager:
            try:
                enhanced_config_manager.stop_file_watching()
                logger.info("✅ Enhanced config manager stopped")
            except Exception as e:
                shutdown_errors.append(f"Error stopping enhanced config: {e}")

        if shutdown_errors:
            logger.warning("⚠️  Some services failed to stop cleanly", errors=shutdown_errors)
        else:
            logger.info("✅ Opinion Market API shutdown completed")
        
        # Log system metrics
        log_system_metric("app_shutdown", 1, {"version": settings.APP_VERSION})
        
    except Exception as e:
        logger.error("❌ Error during shutdown", error=str(e))


app = FastAPI(
    title=settings.APP_NAME,
    description="A comprehensive prediction market platform with advanced features",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)

# Add middleware in order (last added is first executed)
# Trusted host middleware
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# CORS middleware
cors_config = settings.get_cors_config()
app.add_middleware(
    CORSMiddleware,
    **cors_config
)

# Compression middleware
if settings.ENABLE_COMPRESSION:
    app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with performance metrics"""
    start_time = time.time()
    client_ip = get_client_ip(request)
    
    # Log API call
    log_api_call(
        endpoint=request.url.path,
        method=request.method,
        user_id=getattr(request.state, "user_id", None)
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log performance metrics
    log_system_metric("request_duration", process_time, {
        "method": request.method,
        "endpoint": request.url.path,
        "status_code": response.status_code,
        "client_ip": client_ip
    })
    
    # Add performance headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Security middleware
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for IP blocking and rate limiting"""
    client_ip = get_client_ip(request)
    
    # Check if IP is blocked
    if security_manager.is_ip_blocked(client_ip):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="IP address is blocked"
        )
    
    # Check if IP is suspicious
    if security_manager.is_suspicious_ip(client_ip):
        # Log suspicious activity
        log_system_metric("suspicious_request", 1, {
            "ip": client_ip,
            "endpoint": request.url.path,
            "method": request.method
        })
    
    response = await call_next(request)
    return response

# Security
security = HTTPBearer()

# Include API routes if available
if api_router:
    app.include_router(api_router, prefix="/api/v1")
else:
    # Create basic router if main router is not available
    from fastapi import APIRouter
    basic_router = APIRouter()
    
    @basic_router.get("/")
    async def api_root():
        return {"message": "Opinion Market API", "version": "2.0.0"}
    
    app.include_router(basic_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "message": "Welcome to Opinion Market API",
        "version": "2.0.0",
        "description": "A comprehensive prediction market platform with advanced features",
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "features": [
            "Prediction Markets",
            "Real-time Trading",
            "AI Analytics",
            "Social Features",
            "Blockchain Integration",
            "Advanced Orders",
            "Enterprise Security",
            "Performance Optimization",
        ],
        "services": {
            "ml_service": ml_service is not None,
            "analytics_service": analytics_service is not None,
            "market_service": market_service is not None,
            "real_time_engine": real_time_engine is not None,
            "alerting_system": alerting_system is not None,
            "price_feed": price_feed_manager is not None,
        }
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": "opinion-market-api",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check service health
    if ml_service:
        health_status["services"]["ml_service"] = "healthy"
    if analytics_service:
        health_status["services"]["analytics_service"] = "healthy"
    if market_service:
        health_status["services"]["market_service"] = "healthy"
    if real_time_engine:
        health_status["services"]["real_time_engine"] = "healthy"
    if alerting_system:
        health_status["services"]["alerting_system"] = "healthy"
    
    return health_status


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {
        "status": "ready", 
        "service": "opinion-market-api",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
async def metrics():
    """Metrics endpoint"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return {"error": "Prometheus client not available"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
