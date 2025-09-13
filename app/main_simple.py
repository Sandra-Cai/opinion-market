from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.core.performance_optimizer import performance_monitor
from app.core.health_monitor import health_monitor

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
@performance_monitor
async def health_check():
    """Enhanced health check endpoint"""
    health_status = await health_monitor.get_comprehensive_health()
    return health_status


@app.get("/ready")
async def readiness_check():
    return {"status": "ready", "service": "opinion-market-api"}


@app.get("/metrics")
async def metrics():
    return {
        "metrics": {
            "requests_total": 1000,
            "active_users": 150,
            "markets_created": 25,
            "trades_executed": 500,
        }
    }


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
