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
    notifications,
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
    notifications.router, prefix="/notifications", tags=["notifications"]
)
