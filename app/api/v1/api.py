from fastapi import APIRouter
from app.api.v1.endpoints import auth, users, markets, trades, votes, positions, websocket, leaderboard, disputes, notifications, analytics, verification

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(markets.router, prefix="/markets", tags=["markets"])
api_router.include_router(trades.router, prefix="/trades", tags=["trades"])
api_router.include_router(votes.router, prefix="/votes", tags=["votes"])
api_router.include_router(positions.router, prefix="/positions", tags=["positions"])
api_router.include_router(websocket.router, tags=["websocket"])
api_router.include_router(leaderboard.router, prefix="/leaderboard", tags=["leaderboard"])
api_router.include_router(disputes.router, prefix="/disputes", tags=["disputes"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(verification.router, prefix="/verification", tags=["verification"])
