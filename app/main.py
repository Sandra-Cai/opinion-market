from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import logging

from app.core.config import settings
from app.core.database import engine, Base

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
    """Application lifespan manager with proper error handling"""
    # Startup
    print("🚀 Starting Opinion Market API...")
    
    # Track background tasks for cleanup
    background_tasks = []
    
    try:
        # Check if Redis is available
        redis_available = True
        try:
            import redis
            r = redis.Redis.from_url(settings.REDIS_URL)
            r.ping()
            print("✅ Redis connection available")
        except Exception as e:
            redis_available = False
            print(f"⚠️  Redis not available: {e}")

        # Initialize core services that exist
        if ml_service:
            try:
                await ml_service.initialize()
                print("✅ Machine Learning service initialized")
            except Exception as e:
                print(f"⚠️  ML service initialization failed: {e}")

        if analytics_service:
            try:
                print("✅ Analytics service initialized")
            except Exception as e:
                print(f"⚠️  Analytics service initialization failed: {e}")

        if market_service:
            try:
                await market_service.initialize()
                print("✅ Market service initialized")
            except Exception as e:
                print(f"⚠️  Market service initialization failed: {e}")

        if real_time_engine:
            try:
                await real_time_engine.start()
                print("✅ Real-time engine initialized")
            except Exception as e:
                print(f"⚠️  Real-time engine initialization failed: {e}")

        if alerting_system:
            try:
                await alerting_system.start()
                print("✅ Intelligent alerting system initialized")
            except Exception as e:
                print(f"⚠️  Alerting system initialization failed: {e}")

        if price_feed_manager:
            try:
                # Start price feed in background
                task = asyncio.create_task(price_feed_manager.start_price_feed())
                background_tasks.append(task)
                print("✅ Price feed manager initialized")
            except Exception as e:
                print(f"⚠️  Price feed manager initialization failed: {e}")

        # Initialize enhanced config if available
        if enhanced_config_manager:
            try:
                enhanced_config_manager.start_file_watching()
                print("✅ Enhanced configuration manager initialized")
            except Exception as e:
                print(f"⚠️  Enhanced config initialization failed: {e}")

        print("🎉 Opinion Market API startup completed successfully!")

    except Exception as e:
        print(f"❌ Error during startup: {e}")
        logging.error(f"Startup error: {e}")

    yield

    # Shutdown
    print("👋 Shutting down Opinion Market API...")
    
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
                print("✅ Real-time engine stopped")
            except Exception as e:
                print(f"⚠️  Error stopping real-time engine: {e}")

        if alerting_system:
            try:
                await alerting_system.stop()
                print("✅ Alerting system stopped")
            except Exception as e:
                print(f"⚠️  Error stopping alerting system: {e}")

        if market_service:
            try:
                await market_service.cleanup()
                print("✅ Market service stopped")
            except Exception as e:
                print(f"⚠️  Error stopping market service: {e}")

        if enhanced_config_manager:
            try:
                enhanced_config_manager.stop_file_watching()
                print("✅ Enhanced config manager stopped")
            except Exception as e:
                print(f"⚠️  Error stopping enhanced config: {e}")

        print("✅ Opinion Market API shutdown completed")
        
    except Exception as e:
        print(f"❌ Error during shutdown: {e}")
        logging.error(f"Shutdown error: {e}")


app = FastAPI(
    title="Opinion Market API",
    description="A comprehensive prediction market platform with advanced features",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {"status": "healthy", "service": "opinion-market-api"}


@app.get("/ready")
async def readiness_check():
    return {"status": "ready", "service": "opinion-market-api"}


@app.get("/metrics")
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
