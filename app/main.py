from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
import asyncio

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1.api import api_router
from app.core.auth import get_current_user
from app.models import user, market, trade, vote, position
from app.services.price_feed import price_feed_manager
from app.services.performance_optimization import get_performance_optimizer
from app.services.enterprise_security import get_enterprise_security
from app.services.market_data_feed import get_market_data_feed
from app.services.machine_learning import get_ml_service
from app.services.blockchain_integration import get_blockchain_integration
from app.services.social_features import get_social_features
from app.services.advanced_orders import get_advanced_order_manager
from app.services.system_monitor import get_system_monitor
from app.api.docs import custom_openapi

# Create database tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Opinion Market API...")
    
    # Initialize all services
    try:
        # Initialize performance optimizer
        await get_performance_optimizer().initialize(
            settings.REDIS_URL, 
            settings.DATABASE_URL
        )
        print("✅ Performance optimizer initialized")
        
        # Initialize enterprise security
        await get_enterprise_security().initialize(settings.REDIS_URL)
        print("✅ Enterprise security initialized")
        
        # Initialize market data feed
        await get_market_data_feed().initialize(settings.REDIS_URL)
        print("✅ Market data feed initialized")
        
        # Initialize machine learning service
        await get_ml_service().initialize(settings.REDIS_URL)
        print("✅ Machine learning service initialized")
        
        # Initialize blockchain integration
        await get_blockchain_integration().initialize(settings.REDIS_URL)
        print("✅ Blockchain integration initialized")
        
        # Initialize social features
        await get_social_features().initialize(settings.REDIS_URL)
        print("✅ Social features initialized")
        
        # Initialize advanced orders
        await get_advanced_order_manager().initialize(settings.REDIS_URL)
        print("✅ Advanced orders initialized")
        
        # Initialize system monitor
        await get_system_monitor().initialize(settings.REDIS_URL)
        print("✅ System monitor initialized")
        
        # Start price feed service in background
        price_feed_task = asyncio.create_task(price_feed_manager.start_price_feed())
        
    except Exception as e:
        print(f"❌ Error initializing services: {e}")
        raise
    
    yield
    
    # Shutdown
    print("👋 Shutting down Opinion Market API...")
    price_feed_task.cancel()

app = FastAPI(
    title="Opinion Market API",
    description="A comprehensive prediction market platform with advanced features",
    version="2.0.0",
    lifespan=lifespan
)

# Set custom OpenAPI schema
app.openapi = lambda: custom_openapi(app)

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

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Welcome to Opinion Market API",
        "version": "2.0.0",
        "description": "A comprehensive prediction market platform with advanced features",
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
            "Performance Optimization"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "opinion-market-api"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
