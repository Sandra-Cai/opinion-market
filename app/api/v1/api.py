from fastapi import APIRouter
from app.api.v1.endpoints import (
    auth,
    users,
    markets,
    trades,
    votes,
    positions,
    websocket,
    leaderboard,
    disputes,
    notifications,
    analytics,
    verification,
    orders,
    governance,
    advanced_markets,
    ai_analytics,
    rewards,
    mobile,
    advanced_orders,
    market_data,
    ml_analytics,
    social,
    forex_trading,
    order_management,
    derivatives,
    monitoring_dashboard,
    health,
    api_docs,
    system_monitor,
    metrics_dashboard,
    performance_profiler,
    rate_limiting,
    api_versioning,
    cache_management,
    performance_monitor,
    security_audit,
    enhanced_cache,
    performance_dashboard,
    business_intelligence,
    ai_optimization,
    security,
    database_optimization,
    microservices,
    data_pipeline,
    mobile,
    websocket,
    monitoring,
    search,
    advanced_performance,
    enhanced_api_docs,
    advanced_dashboard,
    advanced_analytics_api,
    auto_scaling_api,
    intelligent_alerting_api,
    advanced_security_api,
    admin_dashboard_api,
    advanced_monitoring_api,
    data_governance_api,
    microservices_api,
    chaos_engineering_api,
    mlops_pipeline_api,
    advanced_api_gateway_api,
    event_sourcing_api,
    advanced_caching_api,
    ai_insights_api,
    real_time_analytics_api,
    edge_computing_api,
    quantum_security_api,
    metaverse_web3_api,
    autonomous_systems_api,
)

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(markets.router, prefix="/markets", tags=["markets"])
api_router.include_router(trades.router, prefix="/trades", tags=["trades"])
api_router.include_router(votes.router, prefix="/votes", tags=["votes"])
api_router.include_router(positions.router, prefix="/positions", tags=["positions"])
api_router.include_router(websocket.router, tags=["websocket"])
api_router.include_router(
    leaderboard.router, prefix="/leaderboard", tags=["leaderboard"]
)
api_router.include_router(disputes.router, prefix="/disputes", tags=["disputes"])
api_router.include_router(
    notifications.router, prefix="/notifications", tags=["notifications"]
)
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(
    verification.router, prefix="/verification", tags=["verification"]
)
api_router.include_router(orders.router, prefix="/orders", tags=["orders"])
api_router.include_router(governance.router, prefix="/governance", tags=["governance"])
api_router.include_router(
    advanced_markets.router, prefix="/advanced-markets", tags=["advanced-markets"]
)
api_router.include_router(
    ai_analytics.router, prefix="/ai-analytics", tags=["ai-analytics"]
)
api_router.include_router(rewards.router, prefix="/rewards", tags=["rewards"])
api_router.include_router(mobile.router, prefix="/mobile", tags=["mobile"])
api_router.include_router(
    advanced_orders.router, prefix="/advanced-orders", tags=["advanced-orders"]
)
api_router.include_router(
    market_data.router, prefix="/market-data", tags=["market-data"]
)
api_router.include_router(
    ml_analytics.router, prefix="/ml-analytics", tags=["ml-analytics"]
)
api_router.include_router(social.router, prefix="/social", tags=["social"])
api_router.include_router(forex_trading.router, prefix="/forex", tags=["forex-trading"])
api_router.include_router(
    order_management.router, prefix="/order-management", tags=["order-management"]
)
api_router.include_router(
    derivatives.router, prefix="/derivatives", tags=["derivatives"]
)
api_router.include_router(
    monitoring_dashboard.router, prefix="/monitoring", tags=["monitoring"]
)
api_router.include_router(
    health.router, tags=["health"]
)
api_router.include_router(
    api_docs.router, tags=["documentation"]
)
api_router.include_router(
    system_monitor.router, prefix="/system-monitor", tags=["system-monitoring"]
)
api_router.include_router(
    metrics_dashboard.router, prefix="/metrics-dashboard", tags=["metrics-dashboard"]
)
api_router.include_router(
    performance_profiler.router, prefix="/performance-profiler", tags=["performance-profiler"]
)
api_router.include_router(
    rate_limiting.router, prefix="/rate-limiting", tags=["rate-limiting"]
)
api_router.include_router(
    api_versioning.router, prefix="/api-versioning", tags=["api-versioning"]
)
api_router.include_router(
    cache_management.router, prefix="/cache-management", tags=["cache-management"]
)
api_router.include_router(
    performance_monitor.router, prefix="/performance", tags=["performance-monitoring"]
)
api_router.include_router(
    security_audit.router, prefix="/security", tags=["security-audit"]
)
api_router.include_router(
    enhanced_cache.router, prefix="/enhanced-cache", tags=["enhanced-cache"]
)
api_router.include_router(
    performance_dashboard.router, prefix="/performance-dashboard", tags=["performance-dashboard"]
)
api_router.include_router(
    business_intelligence.router, prefix="/business-intelligence", tags=["business-intelligence"]
)
api_router.include_router(
    ai_optimization.router, prefix="/ai-optimization", tags=["ai-optimization"]
)
api_router.include_router(
    security.router, prefix="/security", tags=["security"]
)
api_router.include_router(
    database_optimization.router, prefix="/database-optimization", tags=["database-optimization"]
)
api_router.include_router(
    microservices.router, prefix="/microservices", tags=["microservices"]
)
api_router.include_router(
    data_pipeline.router, prefix="/data-pipeline", tags=["data-pipeline"]
)
api_router.include_router(
    mobile.router, prefix="/mobile", tags=["mobile"]
)
api_router.include_router(
    websocket.router, prefix="/websocket", tags=["websocket"]
)
api_router.include_router(
    monitoring.router, prefix="/monitoring", tags=["monitoring"]
)
api_router.include_router(
    search.router, prefix="/search", tags=["search"]
)
api_router.include_router(
    notifications.router, prefix="/notifications", tags=["notifications"]
)
api_router.include_router(
    advanced_performance.router, prefix="/advanced-performance", tags=["advanced-performance"]
)
api_router.include_router(
    enhanced_api_docs.router, prefix="/docs", tags=["enhanced-documentation"]
)
api_router.include_router(
    advanced_dashboard.router, prefix="/advanced-dashboard", tags=["advanced-dashboard"]
)
api_router.include_router(
    advanced_analytics_api.router, prefix="/advanced-analytics", tags=["advanced-analytics"]
)
api_router.include_router(
    auto_scaling_api.router, prefix="/auto-scaling", tags=["auto-scaling"]
)
api_router.include_router(
    intelligent_alerting_api.router, prefix="/intelligent-alerting", tags=["intelligent-alerting"]
)
api_router.include_router(
    advanced_security_api.router, prefix="/advanced-security", tags=["advanced-security"]
)
api_router.include_router(
    admin_dashboard_api.router, prefix="/admin-dashboard", tags=["admin-dashboard"]
)
api_router.include_router(
    advanced_monitoring_api.router, prefix="/monitoring", tags=["monitoring"]
)
api_router.include_router(
    data_governance_api.router, prefix="/governance", tags=["governance"]
)
api_router.include_router(
    microservices_api.router, prefix="/microservices", tags=["microservices"]
)
api_router.include_router(
    chaos_engineering_api.router, prefix="/chaos-engineering", tags=["chaos-engineering"]
)
api_router.include_router(
    mlops_pipeline_api.router, prefix="/mlops", tags=["mlops"]
)
api_router.include_router(
    advanced_api_gateway_api.router, prefix="/api-gateway", tags=["api-gateway"]
)
api_router.include_router(
    event_sourcing_api.router, prefix="/event-sourcing", tags=["event-sourcing"]
)
api_router.include_router(
    advanced_caching_api.router, prefix="/advanced-caching", tags=["advanced-caching"]
)
api_router.include_router(
    ai_insights_api.router, prefix="/ai-insights", tags=["ai-insights"]
)
api_router.include_router(
    real_time_analytics_api.router, prefix="/real-time-analytics", tags=["real-time-analytics"]
)
api_router.include_router(
    edge_computing_api.router, prefix="/edge-computing", tags=["edge-computing"]
)
api_router.include_router(
    quantum_security_api.router, prefix="/quantum-security", tags=["quantum-security"]
)
api_router.include_router(
    metaverse_web3_api.router, prefix="/metaverse-web3", tags=["metaverse-web3"]
)
api_router.include_router(
    autonomous_systems_api.router, prefix="/autonomous-systems", tags=["autonomous-systems"]
)
