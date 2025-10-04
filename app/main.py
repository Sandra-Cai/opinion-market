from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
import asyncio

from app.core.config import settings
from app.core.database import engine, Base
# Use enhanced configuration for database
from app.core.enhanced_config import enhanced_config_manager
from app.api.v1.api import api_router
from app.core.auth import get_current_user
from app.models import user, market, trade, vote, position
from app.services.price_feed import price_feed_manager
from app.services.performance_optimization import get_performance_optimizer
from app.services.enterprise_security import get_enterprise_security
from app.services.market_data_feed import get_market_data_feed
from app.services.machine_learning import get_ml_service
from app.services.blockchain_integration import get_blockchain_integration_service
from app.services.social_features import get_social_features
from app.services.advanced_orders import get_advanced_order_manager
from app.services.monitoring import get_system_monitor
from app.api.docs import custom_openapi

# Import enhanced systems
from app.core.enhanced_error_handler import enhanced_error_handler
from app.core.advanced_performance_optimizer import advanced_performance_optimizer
from app.core.advanced_security import advanced_security_manager
from app.core.enhanced_testing import enhanced_test_manager
from app.core.enhanced_config import enhanced_config_manager
from app.api.enhanced_docs import create_enhanced_openapi_schema

# Import new performance monitoring systems
from app.core.performance_monitor import performance_monitor
from app.core.enhanced_cache import enhanced_cache
from app.core.advanced_performance_optimizer import advanced_performance_optimizer
from app.services.advanced_analytics_engine import advanced_analytics_engine
from app.services.auto_scaling_manager import auto_scaling_manager
from app.core.performance_optimizer_v2 import performance_optimizer_v2
from app.core.intelligent_alerting import intelligent_alerting_system
from app.core.advanced_security_v2 import advanced_security_v2
from app.services.business_intelligence_engine import business_intelligence_engine
from app.services.mobile_optimization_engine import mobile_optimization_engine
from app.services.blockchain_integration_engine import blockchain_integration_engine
from app.services.advanced_ml_engine import advanced_ml_engine
from app.services.distributed_caching_engine import distributed_caching_engine
from app.services.advanced_monitoring_engine import advanced_monitoring_engine
from app.services.data_governance_engine import data_governance_engine
from app.services.microservices_engine import microservices_engine
from app.services.chaos_engineering_engine import chaos_engineering_engine
from app.services.mlops_pipeline_engine import mlops_pipeline_engine
from app.services.advanced_api_gateway import advanced_api_gateway
from app.services.event_sourcing_engine import event_sourcing_engine
from app.services.advanced_caching_engine import advanced_caching_engine
from app.services.ai_insights_engine import ai_insights_engine
from app.services.real_time_analytics_engine import real_time_analytics_engine
from app.services.edge_computing_engine import edge_computing_engine
from app.services.quantum_security_engine import quantum_security_engine
from app.services.metaverse_web3_engine import metaverse_web3_engine
from app.services.autonomous_systems_engine import autonomous_systems_engine

# Create database tables (only if using SQLite for development)
if enhanced_config_manager.get("database.url", "").startswith("sqlite"):
    Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Opinion Market API...")

    # Initialize all services
    try:
        # Check if Redis is available
        redis_available = True
        try:
            import redis
            r = redis.Redis.from_url(settings.REDIS_URL)
            r.ping()
            print("✅ Redis connection available")
        except:
            redis_available = False
            print("⚠️  Redis not available - running in development mode without Redis")

        # Initialize performance optimizer (with Redis if available)
        if redis_available:
            await get_performance_optimizer().initialize(
                settings.REDIS_URL, settings.DATABASE_URL
            )
            print("✅ Performance optimizer initialized")
        else:
            print("⚠️  Performance optimizer skipped (Redis not available)")

        # Initialize enterprise security (with Redis if available)
        if redis_available:
            await get_enterprise_security().initialize(settings.REDIS_URL)
            print("✅ Enterprise security initialized")
        else:
            print("⚠️  Enterprise security skipped (Redis not available)")

        # Initialize market data feed (with Redis if available)
        if redis_available:
            await get_market_data_feed().initialize(settings.REDIS_URL)
            print("✅ Market data feed initialized")
        else:
            print("⚠️  Market data feed skipped (Redis not available)")

        # Initialize machine learning service (with Redis if available)
        if redis_available:
            await get_ml_service().initialize(settings.REDIS_URL)
            print("✅ Machine learning service initialized")
        else:
            print("⚠️  Machine learning service skipped (Redis not available)")

        # Initialize blockchain integration (with Redis if available)
        if redis_available:
            await get_blockchain_integration_service().initialize(settings.REDIS_URL)
            print("✅ Blockchain integration initialized")
        else:
            print("⚠️  Blockchain integration skipped (Redis not available)")

        # Initialize social features (with Redis if available)
        if redis_available:
            await get_social_features().initialize(settings.REDIS_URL)
            print("✅ Social features initialized")
        else:
            print("⚠️  Social features skipped (Redis not available)")

        # Initialize advanced orders (with Redis if available)
        if redis_available:
            await get_advanced_order_manager().initialize(settings.REDIS_URL)
            print("✅ Advanced orders initialized")
        else:
            print("⚠️  Advanced orders skipped (Redis not available)")

        # Initialize system monitor (with Redis if available)
        if redis_available:
            await get_system_monitor().initialize(settings.REDIS_URL)
            print("✅ System monitor initialized")
        else:
            print("⚠️  System monitor skipped (Redis not available)")

        # Initialize enhanced systems
        await advanced_performance_optimizer.start_monitoring()
        print("✅ Advanced performance optimizer initialized")
        
        enhanced_config_manager.start_file_watching()
        print("✅ Enhanced configuration manager initialized")

        # Initialize new performance monitoring systems
        await performance_monitor.start_monitoring(interval=30)
        print("✅ Performance monitor initialized")
        
        await enhanced_cache.start_cleanup()
        print("✅ Enhanced cache system initialized")
        
        # Initialize advanced performance optimizer
        await advanced_performance_optimizer.start_monitoring()
        await advanced_performance_optimizer.start_optimization()
        print("✅ Advanced performance optimizer initialized")
        
        # Initialize advanced analytics engine
        await advanced_analytics_engine.start_analytics()
        print("✅ Advanced analytics engine initialized")
        
        # Initialize auto-scaling manager
        await auto_scaling_manager.start_scaling()
        print("✅ Auto-scaling manager initialized")
        
        # Initialize performance optimizer V2
        await performance_optimizer_v2.start_optimization()
        print("✅ Performance Optimizer V2 initialized")
        
        # Initialize intelligent alerting system
        await intelligent_alerting_system.start_alerting()
        print("✅ Intelligent alerting system initialized")

        # Initialize advanced security V2
        await advanced_security_v2.start_security_monitoring()
        print("✅ Advanced Security V2 initialized")

        # Initialize business intelligence engine
        await business_intelligence_engine.start_bi_engine()
        print("✅ Business Intelligence Engine initialized")

        # Initialize mobile optimization engine
        await mobile_optimization_engine.start_mobile_optimization()
        print("✅ Mobile Optimization Engine initialized")

        # Initialize blockchain integration engine
        await blockchain_integration_engine.start_blockchain_engine()
        print("✅ Blockchain Integration Engine initialized")

        # Initialize advanced ML engine
        await advanced_ml_engine.start_ml_engine()
        print("✅ Advanced ML Engine initialized")

        # Initialize distributed caching engine
        await distributed_caching_engine.start_caching_engine()
        print("✅ Distributed Caching Engine initialized")

        # Initialize advanced monitoring engine
        await advanced_monitoring_engine.start_monitoring_engine()
        print("✅ Advanced Monitoring Engine initialized")

        # Initialize data governance engine
        await data_governance_engine.start_governance_engine()
        print("✅ Data Governance Engine initialized")

        # Initialize microservices engine
        await microservices_engine.start_microservices_engine()
        print("✅ Microservices Engine initialized")

        # Initialize chaos engineering engine
        await chaos_engineering_engine.start_chaos_engine()
        print("✅ Chaos Engineering Engine initialized")

        # Initialize MLOps pipeline engine
        await mlops_pipeline_engine.start_mlops_engine()
        print("✅ MLOps Pipeline Engine initialized")

        # Initialize advanced API gateway
        await advanced_api_gateway.start_gateway()
        print("✅ Advanced API Gateway initialized")

        # Initialize event sourcing engine
        await event_sourcing_engine.start_event_sourcing_engine()
        print("✅ Event Sourcing Engine initialized")

        # Initialize advanced caching engine
        await advanced_caching_engine.start_caching_engine()
        print("✅ Advanced Caching Engine initialized")

        # Initialize AI insights engine
        await ai_insights_engine.start_ai_insights_engine()
        print("✅ AI Insights Engine initialized")

    # Initialize real-time analytics engine
    await real_time_analytics_engine.start_analytics_engine()
    print("✅ Real-time Analytics Engine initialized")

    # Initialize edge computing engine
    await edge_computing_engine.start_edge_computing_engine()
    print("✅ Edge Computing Engine initialized")

    # Initialize quantum security engine
    await quantum_security_engine.start_quantum_security_engine()
    print("✅ Quantum Security Engine initialized")

    # Initialize metaverse Web3 engine
    await metaverse_web3_engine.start_metaverse_web3_engine()
    print("✅ Metaverse Web3 Engine initialized")

    # Initialize autonomous systems engine
    await autonomous_systems_engine.start_autonomous_systems_engine()
    print("✅ Autonomous Systems Engine initialized")

        # Start price feed service in background
        price_feed_task = asyncio.create_task(price_feed_manager.start_price_feed())

    except Exception as e:
        print(f"❌ Error initializing services: {e}")
        raise

    yield

    # Shutdown
    print("👋 Shutting down Opinion Market API...")
    price_feed_task.cancel()
    
    # Stop enhanced systems
    await advanced_performance_optimizer.stop_monitoring()
    enhanced_config_manager.stop_file_watching()
    
    # Stop new performance monitoring systems
    await performance_monitor.stop_monitoring()
    await enhanced_cache.stop_cleanup()
    
    # Stop advanced performance optimizer
    await advanced_performance_optimizer.stop_monitoring()
    await advanced_performance_optimizer.stop_optimization()
    
    # Stop advanced analytics engine
    await advanced_analytics_engine.stop_analytics()
    
    # Stop auto-scaling manager
    await auto_scaling_manager.stop_scaling()
    
    # Stop performance optimizer V2
    await performance_optimizer_v2.stop_optimization()
    
        # Stop intelligent alerting system
        await intelligent_alerting_system.stop_alerting()

        # Stop advanced security V2
        await advanced_security_v2.stop_security_monitoring()

        # Stop business intelligence engine
        await business_intelligence_engine.stop_bi_engine()

        # Stop mobile optimization engine
        await mobile_optimization_engine.stop_mobile_optimization()

        # Stop blockchain integration engine
        await blockchain_integration_engine.stop_blockchain_engine()

        # Stop advanced ML engine
        await advanced_ml_engine.stop_ml_engine()

        # Stop distributed caching engine
        await distributed_caching_engine.stop_caching_engine()

        # Stop advanced monitoring engine
        await advanced_monitoring_engine.stop_monitoring_engine()

        # Stop data governance engine
        await data_governance_engine.stop_governance_engine()

        # Stop microservices engine
        await microservices_engine.stop_microservices_engine()

        # Stop chaos engineering engine
        await chaos_engineering_engine.stop_chaos_engine()

        # Stop MLOps pipeline engine
        await mlops_pipeline_engine.stop_mlops_engine()

        # Stop advanced API gateway
        await advanced_api_gateway.stop_gateway()

        # Stop event sourcing engine
        await event_sourcing_engine.stop_event_sourcing_engine()

        # Stop advanced caching engine
        await advanced_caching_engine.stop_caching_engine()

        # Stop AI insights engine
        await ai_insights_engine.stop_ai_insights_engine()

    # Stop real-time analytics engine
    await real_time_analytics_engine.stop_analytics_engine()

    # Stop edge computing engine
    await edge_computing_engine.stop_edge_computing_engine()

    # Stop quantum security engine
    await quantum_security_engine.stop_quantum_security_engine()

    # Stop metaverse Web3 engine
    await metaverse_web3_engine.stop_metaverse_web3_engine()

    # Stop autonomous systems engine
    await autonomous_systems_engine.stop_autonomous_systems_engine()
    
    print("✅ Enhanced systems stopped")


app = FastAPI(
    title="Opinion Market API",
    description="A comprehensive prediction market platform with advanced features",
    version="2.0.0",
    lifespan=lifespan,
)

# Set enhanced OpenAPI schema
app.openapi = lambda: create_enhanced_openapi_schema(app)

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
            "Performance Optimization",
        ],
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
