"""
Main API Router for Opinion Market
Combines all endpoint routers into a single API
"""

import logging
from fastapi import APIRouter
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Import core endpoint routers
try:
    from app.api.v1.endpoints import auth, markets, trades, users, orders, positions
    from app.api.v1.endpoints import analytics, notifications, votes, disputes
    from app.api.v1.endpoints import leaderboard, verification, websocket, security_monitoring, performance_monitoring
except ImportError as e:
    logger.warning(f"Some API endpoints not available: {e}", exc_info=True)

# Create main API router
api_router = APIRouter()

# Include core endpoint routers
try:
    api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    api_router.include_router(users.router, prefix="/users", tags=["Users"])
    api_router.include_router(markets.router, prefix="/markets", tags=["Markets"])
    api_router.include_router(trades.router, prefix="/trades", tags=["Trades"])
    api_router.include_router(orders.router, prefix="/orders", tags=["Orders"])
    api_router.include_router(positions.router, prefix="/positions", tags=["Positions"])
    api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
    api_router.include_router(notifications.router, prefix="/notifications", tags=["Notifications"])
    api_router.include_router(votes.router, prefix="/votes", tags=["Votes"])
    api_router.include_router(disputes.router, prefix="/disputes", tags=["Disputes"])
    api_router.include_router(leaderboard.router, prefix="/leaderboard", tags=["Leaderboard"])
    api_router.include_router(verification.router, prefix="/verification", tags=["Verification"])
    api_router.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
    api_router.include_router(security_monitoring.router, prefix="/security", tags=["Security Monitoring"])
    api_router.include_router(performance_monitoring.router, prefix="/performance", tags=["Performance Monitoring"])
    logger.info("API routers registered successfully")
except Exception as e:
    logger.error(f"Error including API routers: {e}", exc_info=True)

# Add basic endpoints if routers are not available
@api_router.get("/", response_model=Dict[str, Any])
async def api_root() -> Dict[str, Any]:
    """
    API root endpoint
    
    Returns:
        Dict containing API information and available endpoints
    """
    return {
        "message": "Opinion Market API v1",
        "version": "1.0.0",
        "endpoints": [
            "/auth",
            "/users", 
            "/markets",
            "/trades",
            "/orders",
            "/positions",
            "/analytics",
            "/notifications",
            "/votes",
            "/disputes",
            "/leaderboard",
            "/verification",
            "/ws",
            "/security",
            "/performance"
        ]
    }

@api_router.get("/status", response_model=Dict[str, Any])
async def api_status() -> Dict[str, Any]:
    """
    API status endpoint
    
    Returns:
        Dict containing API status information
    """
    from datetime import datetime
    return {
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }