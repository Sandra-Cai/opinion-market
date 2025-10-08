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

    # Initialize advanced AI orchestration engine
    await advanced_ai_orchestration_engine.start_ai_orchestration_engine()
    print("✅ Advanced AI Orchestration Engine initialized")

    # Initialize intelligent decision engine
    await intelligent_decision_engine.start_intelligent_decision_engine()
    print("✅ Intelligent Decision Engine initialized")

    # Initialize advanced pattern recognition engine
    await advanced_pattern_recognition_engine.start_advanced_pattern_recognition_engine()
    print("✅ Advanced Pattern Recognition Engine initialized")

    # Initialize AI-powered risk assessment engine
    await ai_powered_risk_assessment_engine.start_ai_powered_risk_assessment_engine()
    print("✅ AI-Powered Risk Assessment Engine initialized")

    # Initialize advanced trading engine
    await advanced_trading_engine.start_trading_engine()
    print("✅ Advanced Trading Engine initialized")

    # Initialize portfolio optimization engine
    await portfolio_optimization_engine.start_portfolio_optimization_engine()
    print("✅ Portfolio Optimization Engine initialized")

    # Initialize market sentiment engine
    await market_sentiment_engine.start_market_sentiment_engine()
    print("✅ Market Sentiment Engine initialized")

    # Initialize advanced predictive analytics engine
    await advanced_predictive_analytics_engine.start_predictive_analytics_engine()
    print("✅ Advanced Predictive Analytics Engine initialized")

    # Initialize time series forecasting engine
    await time_series_forecasting_engine.start_time_series_forecasting_engine()
    print("✅ Time Series Forecasting Engine initialized")

    # Initialize anomaly detection engine
    await anomaly_detection_engine.start_anomaly_detection_engine()
    print("✅ Anomaly Detection Engine initialized")

    # Initialize advanced blockchain engine
    await advanced_blockchain_engine.start_blockchain_engine()
    print("✅ Advanced Blockchain Engine initialized")

    # Initialize DeFi protocol manager
    await defi_protocol_manager.start_defi_manager()
    print("✅ DeFi Protocol Manager initialized")

    # Initialize smart contract engine
    await smart_contract_engine.start_smart_contract_engine()
    print("✅ Smart Contract Engine initialized")

    # Initialize IoT data processing engine
    await iot_data_processing_engine.start_iot_processing_engine()
    print("✅ IoT Data Processing Engine initialized")

    # Initialize IoT device management engine
    await iot_device_management_engine.start_device_management_engine()
    print("✅ IoT Device Management Engine initialized")

    # Initialize IoT analytics engine
    await iot_analytics_engine.start_analytics_engine()
    print("✅ IoT Analytics Engine initialized")

    # Initialize AR/VR experience engine
    await ar_vr_experience_engine.start_ar_vr_engine()
    print("✅ AR/VR Experience Engine initialized")

    # Initialize immersive content engine
    await immersive_content_engine.start_content_engine()
    print("✅ Immersive Content Engine initialized")

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

    # Stop advanced AI orchestration engine
    await advanced_ai_orchestration_engine.stop_ai_orchestration_engine()

    # Stop intelligent decision engine
    await intelligent_decision_engine.stop_intelligent_decision_engine()

    # Stop advanced pattern recognition engine
    await advanced_pattern_recognition_engine.stop_advanced_pattern_recognition_engine()

    # Stop AI-powered risk assessment engine
    await ai_powered_risk_assessment_engine.stop_ai_powered_risk_assessment_engine()

    # Stop advanced trading engine
    await advanced_trading_engine.stop_trading_engine()

    # Stop portfolio optimization engine
    await portfolio_optimization_engine.stop_portfolio_optimization_engine()

    # Stop market sentiment engine
    await market_sentiment_engine.stop_market_sentiment_engine()

    # Stop advanced predictive analytics engine
    await advanced_predictive_analytics_engine.stop_predictive_analytics_engine()

    # Stop time series forecasting engine
    await time_series_forecasting_engine.stop_time_series_forecasting_engine()

    # Stop anomaly detection engine
    await anomaly_detection_engine.stop_anomaly_detection_engine()

    # Stop advanced blockchain engine
    await advanced_blockchain_engine.stop_blockchain_engine()

    # Stop DeFi protocol manager
    await defi_protocol_manager.stop_defi_manager()

    # Stop smart contract engine
    await smart_contract_engine.stop_smart_contract_engine()

    # Stop IoT data processing engine
    await iot_data_processing_engine.stop_iot_processing_engine()

    # Stop IoT device management engine
    await iot_device_management_engine.stop_device_management_engine()

    # Stop IoT analytics engine
    await iot_analytics_engine.stop_analytics_engine()

    # Stop AR/VR experience engine
    await ar_vr_experience_engine.stop_ar_vr_engine()

    # Stop immersive content engine
    await immersive_content_engine.stop_content_engine()
    
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
