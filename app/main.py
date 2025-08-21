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

# Create database tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Opinion Market API...")
    
    # Start price feed service in background
    price_feed_task = asyncio.create_task(price_feed_manager.start_price_feed())
    
    yield
    
    # Shutdown
    print("👋 Shutting down Opinion Market API...")
    price_feed_task.cancel()

app = FastAPI(
    title="Opinion Market API",
    description="A platform where people can trade and vote on their opinions",
    version="1.0.0",
    lifespan=lifespan
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

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Welcome to Opinion Market API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
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
