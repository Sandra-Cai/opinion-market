"""
DeFi Protocol API Endpoints
REST API for the DeFi Protocol Manager
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.defi_protocol_manager import (
    defi_protocol_manager,
    DeFiStrategyType,
    RiskLevel,
    PositionStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class DeFiStrategyResponse(BaseModel):
    strategy_id: str
    name: str
    strategy_type: str
    protocol: str
    blockchain: str
    risk_level: str
    expected_apy: float
    min_investment: float
    max_investment: float
    auto_compound: bool
    rebalance_frequency: int
    stop_loss_percentage: Optional[float]
    take_profit_percentage: Optional[float]
    enabled: bool
    created_at: str
    metadata: Dict[str, Any]

class DeFiPositionResponse(BaseModel):
    position_id: str
    user_address: str
    strategy_id: str
    protocol: str
    blockchain: str
    position_type: str
    token_address: str
    token_symbol: str
    amount: float
    value_usd: float
    apy: float
    status: str
    opened_at: str
    closed_at: Optional[str]
    pnl: float
    fees_paid: float
    metadata: Dict[str, Any]

class YieldOpportunityResponse(BaseModel):
    opportunity_id: str
    protocol: str
    blockchain: str
    strategy_type: str
    token_pair: str
    apy: float
    tvl: float
    risk_score: float
    liquidity_score: float
    gas_cost: float
    min_investment: float
    max_investment: float
    discovered_at: str
    expires_at: Optional[str]
    metadata: Dict[str, Any]

class AddStrategyRequest(BaseModel):
    name: str = Field(..., description="Strategy name")
    strategy_type: str = Field(..., description="Strategy type")
    protocol: str = Field(..., description="Protocol name")
    blockchain: str = Field(..., description="Blockchain name")
    risk_level: str = Field(..., description="Risk level")
    expected_apy: float = Field(..., description="Expected APY")
    min_investment: float = Field(..., description="Minimum investment")
    max_investment: float = Field(..., description="Maximum investment")

@router.get("/strategies", response_model=List[DeFiStrategyResponse])
async def get_strategies():
    """Get DeFi strategies"""
    try:
        strategies = await defi_protocol_manager.get_strategies()
        return strategies
        
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions", response_model=List[DeFiPositionResponse])
async def get_positions(
    user_address: Optional[str] = None,
    strategy_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Get DeFi positions"""
    try:
        # Validate status if provided
        position_status = None
        if status:
            try:
                position_status = PositionStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        positions = await defi_protocol_manager.get_positions(
            user_address=user_address,
            strategy_id=strategy_id,
            status=position_status,
            limit=limit
        )
        return positions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/yield-opportunities", response_model=List[YieldOpportunityResponse])
async def get_yield_opportunities(
    protocol: Optional[str] = None,
    min_apy: Optional[float] = None,
    max_risk: Optional[float] = None,
    limit: int = 50
):
    """Get yield opportunities"""
    try:
        opportunities = await defi_protocol_manager.get_yield_opportunities(
            protocol=protocol,
            min_apy=min_apy,
            max_risk=max_risk,
            limit=limit
        )
        return opportunities
        
    except Exception as e:
        logger.error(f"Error getting yield opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/strategies", response_model=Dict[str, str])
async def add_strategy(strategy_request: AddStrategyRequest):
    """Add a new DeFi strategy"""
    try:
        # Validate strategy type
        try:
            strategy_type = DeFiStrategyType(strategy_request.strategy_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid strategy type: {strategy_request.strategy_type}")
        
        # Validate risk level
        try:
            risk_level = RiskLevel(strategy_request.risk_level.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid risk level: {strategy_request.risk_level}")
        
        from decimal import Decimal
        
        strategy_id = await defi_protocol_manager.add_strategy(
            name=strategy_request.name,
            strategy_type=strategy_type,
            protocol=strategy_request.protocol,
            blockchain=strategy_request.blockchain,
            risk_level=risk_level,
            expected_apy=Decimal(str(strategy_request.expected_apy)),
            min_investment=Decimal(str(strategy_request.min_investment)),
            max_investment=Decimal(str(strategy_request.max_investment))
        )
        
        return {
            "strategy_id": strategy_id,
            "status": "created",
            "message": f"Strategy '{strategy_request.name}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/strategies/{strategy_id}")
async def update_strategy(strategy_id: str, **kwargs):
    """Update a DeFi strategy"""
    try:
        success = await defi_protocol_manager.update_strategy(strategy_id, **kwargs)
        if not success:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {
            "strategy_id": strategy_id,
            "status": "updated",
            "message": "Strategy updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy {strategy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-metrics")
async def get_risk_metrics():
    """Get risk metrics"""
    try:
        metrics = await defi_protocol_manager.get_risk_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-metrics")
async def get_performance_metrics():
    """Get performance metrics"""
    try:
        metrics = await defi_protocol_manager.get_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/manager-metrics")
async def get_manager_metrics():
    """Get manager metrics"""
    try:
        metrics = await defi_protocol_manager.get_manager_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting manager metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-strategies")
async def get_available_strategies():
    """Get available strategy types and protocols"""
    try:
        return {
            "strategy_types": [
                {
                    "name": strategy.value,
                    "display_name": strategy.value.replace("_", " ").title(),
                    "description": f"{strategy.value.replace('_', ' ').title()} strategy"
                }
                for strategy in DeFiStrategyType
            ],
            "risk_levels": [
                {
                    "name": risk.value,
                    "display_name": risk.value.replace("_", " ").title(),
                    "description": f"{risk.value.replace('_', ' ').title()} risk level"
                }
                for risk in RiskLevel
            ],
            "supported_protocols": list(defi_protocol_manager.protocol_configs.keys()),
            "supported_blockchains": ["ethereum", "polygon", "bsc", "avalanche", "arbitrum", "optimism"]
        }
        
    except Exception as e:
        logger.error(f"Error getting available strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_defi_protocol_health():
    """Get DeFi protocol manager health status"""
    try:
        return {
            "manager_id": defi_protocol_manager.manager_id,
            "is_running": defi_protocol_manager.is_running,
            "total_strategies": len(defi_protocol_manager.strategies),
            "total_positions": len(defi_protocol_manager.positions),
            "total_opportunities": len(defi_protocol_manager.yield_opportunities),
            "total_transactions": len(defi_protocol_manager.transactions),
            "supported_protocols": list(defi_protocol_manager.protocol_configs.keys()),
            "uptime": "active" if defi_protocol_manager.is_running else "stopped"
        }
        
    except Exception as e:
        logger.error(f"Error getting DeFi protocol health: {e}")
        raise HTTPException(status_code=500, detail=str(e))
