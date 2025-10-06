"""
Portfolio Optimization API Endpoints
REST API for the Portfolio Optimization Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.portfolio_optimization_engine import (
    portfolio_optimization_engine,
    OptimizationObjective,
    RebalancingFrequency
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class CreatePortfolioRequest(BaseModel):
    name: str = Field(..., description="Portfolio name")
    description: str = Field(..., description="Portfolio description")
    asset_symbols: List[str] = Field(..., description="List of asset symbols")
    initial_weights: Optional[List[float]] = Field(None, description="Initial weights (optional)")

class PortfolioResponse(BaseModel):
    portfolio_id: str
    name: str
    description: str
    assets: List[Dict[str, Any]]
    weights: List[float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    beta: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    created_at: str
    updated_at: str

class RebalancingSignalResponse(BaseModel):
    signal_id: str
    portfolio_id: str
    current_weights: List[float]
    target_weights: List[float]
    rebalancing_amounts: List[float]
    expected_improvement: float
    transaction_costs: float
    urgency: float
    reason: str
    created_at: str

@router.post("/create-portfolio", response_model=Dict[str, str])
async def create_portfolio(portfolio_request: CreatePortfolioRequest):
    """Create a new portfolio"""
    try:
        portfolio_id = await portfolio_optimization_engine.create_portfolio(
            name=portfolio_request.name,
            description=portfolio_request.description,
            asset_symbols=portfolio_request.asset_symbols,
            initial_weights=portfolio_request.initial_weights
        )
        
        return {
            "portfolio_id": portfolio_id,
            "status": "created",
            "message": f"Portfolio '{portfolio_request.name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolios", response_model=List[PortfolioResponse])
async def get_all_portfolios():
    """Get all portfolios"""
    try:
        portfolios = await portfolio_optimization_engine.get_all_portfolios()
        return portfolios
        
    except Exception as e:
        logger.error(f"Error getting portfolios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(portfolio_id: str):
    """Get a specific portfolio"""
    try:
        portfolio = await portfolio_optimization_engine.get_portfolio(portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return portfolio
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assets", response_model=List[Dict[str, Any]])
async def get_available_assets():
    """Get available assets for portfolio construction"""
    try:
        assets = await portfolio_optimization_engine.get_available_assets()
        return assets
        
    except Exception as e:
        logger.error(f"Error getting available assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rebalancing-signals", response_model=List[RebalancingSignalResponse])
async def get_rebalancing_signals():
    """Get rebalancing signals"""
    try:
        signals = await portfolio_optimization_engine.get_rebalancing_signals()
        return signals
        
    except Exception as e:
        logger.error(f"Error getting rebalancing signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization-history")
async def get_optimization_history():
    """Get optimization history"""
    try:
        history = await portfolio_optimization_engine.get_optimization_history()
        return {
            "optimization_history": history,
            "total_optimizations": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance/{portfolio_id}")
async def get_portfolio_performance(portfolio_id: str):
    """Get performance metrics for a portfolio"""
    try:
        performance = await portfolio_optimization_engine.get_performance_metrics(portfolio_id)
        if not performance:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        return performance
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting portfolio performance {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies")
async def get_optimization_strategies():
    """Get available optimization strategies"""
    try:
        strategies = [
            {
                "name": objective.value,
                "description": f"{objective.value.replace('_', ' ').title()} optimization strategy",
                "enabled": True
            }
            for objective in OptimizationObjective
        ]
        
        return {
            "strategies": strategies,
            "current_objective": portfolio_optimization_engine.optimization_objective.value,
            "rebalancing_frequency": portfolio_optimization_engine.rebalancing_frequency.value
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize/{portfolio_id}")
async def optimize_portfolio(portfolio_id: str, objective: str = "maximize_sharpe"):
    """Manually trigger portfolio optimization"""
    try:
        # Validate objective
        try:
            opt_objective = OptimizationObjective(objective)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid optimization objective: {objective}")
        
        # Set optimization objective
        portfolio_optimization_engine.optimization_objective = opt_objective
        
        # Trigger optimization
        await portfolio_optimization_engine._optimize_portfolio(portfolio_id)
        
        return {
            "status": "success",
            "message": f"Portfolio {portfolio_id} optimized using {objective}",
            "objective": objective
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_portfolio_engine_health():
    """Get portfolio optimization engine health status"""
    try:
        return {
            "engine_id": portfolio_optimization_engine.engine_id,
            "is_running": portfolio_optimization_engine.is_running,
            "total_portfolios": len(portfolio_optimization_engine.portfolios),
            "total_assets": len(portfolio_optimization_engine.assets),
            "rebalancing_signals": len(portfolio_optimization_engine.rebalancing_signals),
            "optimization_history": len(portfolio_optimization_engine.optimization_history),
            "current_objective": portfolio_optimization_engine.optimization_objective.value,
            "rebalancing_frequency": portfolio_optimization_engine.rebalancing_frequency.value,
            "uptime": "active" if portfolio_optimization_engine.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio engine health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
