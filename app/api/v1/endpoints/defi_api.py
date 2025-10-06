"""
DeFi API Endpoints
REST API for advanced DeFi features including liquidity mining, yield farming, and staking
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from app.services.defi_engine import (
    defi_engine, DeFiProtocol, StakingType, YieldStrategy, 
    LiquidityPool, DeFiPosition, StakingPosition, YieldFarm
)

router = APIRouter()

# Pydantic models for API

class DeFiPositionCreate(BaseModel):
    user_id: str = Field(..., description="User ID")
    protocol: str = Field(..., description="DeFi protocol")
    pool_type: str = Field(..., description="Liquidity pool type")
    token_pair: Tuple[str, str] = Field(..., description="Token pair")
    amount: float = Field(..., gt=0, description="Amount to deposit")
    value_usd: float = Field(..., gt=0, description="Value in USD")

class StakingPositionCreate(BaseModel):
    user_id: str = Field(..., description="User ID")
    staking_type: str = Field(..., description="Staking type")
    token: str = Field(..., description="Token to stake")
    amount: float = Field(..., gt=0, description="Amount to stake")
    lock_period: int = Field(..., ge=1, le=365, description="Lock period in days")
    auto_compound: bool = Field(False, description="Enable auto-compounding")

class RewardClaim(BaseModel):
    user_id: str = Field(..., description="User ID")
    position_id: str = Field(..., description="Position ID")

class DeFiMetricsResponse(BaseModel):
    metrics: Dict[str, Any]
    total_positions: int
    total_staking: int
    total_yield_farms: int
    total_rewards: int
    protocols_supported: int
    engine_status: str

class YieldFarmResponse(BaseModel):
    farm_id: str
    name: str
    strategy: str
    protocol: str
    token_pairs: List[Tuple[str, str]]
    apy: float
    tvl: float
    risk_level: str
    min_deposit: float
    max_deposit: float
    fees: Dict[str, float]

class UserPositionsResponse(BaseModel):
    defi_positions: List[Dict[str, Any]]
    staking_positions: List[Dict[str, Any]]
    total_value: float
    total_rewards: float

# API Endpoints

@router.post("/positions", response_model=Dict[str, str])
async def create_defi_position(position_data: DeFiPositionCreate):
    """Create a new DeFi position"""
    try:
        # Convert string enums to enum objects
        protocol = DeFiProtocol(position_data.protocol.lower())
        pool_type = LiquidityPool(position_data.pool_type.lower())
        
        position_id = await defi_engine.create_defi_position(
            user_id=position_data.user_id,
            protocol=protocol,
            pool_type=pool_type,
            token_pair=position_data.token_pair,
            amount=position_data.amount,
            value_usd=position_data.value_usd
        )
        
        return {
            "position_id": position_id,
            "status": "created",
            "message": "DeFi position created successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create DeFi position: {str(e)}")

@router.post("/staking", response_model=Dict[str, str])
async def create_staking_position(staking_data: StakingPositionCreate):
    """Create a new staking position"""
    try:
        # Convert string enum to enum object
        staking_type = StakingType(staking_data.staking_type.lower())
        
        staking_id = await defi_engine.create_staking_position(
            user_id=staking_data.user_id,
            staking_type=staking_type,
            token=staking_data.token,
            amount=staking_data.amount,
            lock_period=staking_data.lock_period,
            auto_compound=staking_data.auto_compound
        )
        
        return {
            "staking_id": staking_id,
            "status": "created",
            "message": "Staking position created successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create staking position: {str(e)}")

@router.get("/positions/{user_id}", response_model=UserPositionsResponse)
async def get_user_positions(user_id: str):
    """Get all DeFi and staking positions for a user"""
    try:
        positions = await defi_engine.get_user_positions(user_id)
        
        # Convert dataclass objects to dictionaries
        defi_positions = []
        for pos in positions["defi_positions"]:
            defi_positions.append({
                "position_id": pos.position_id,
                "protocol": pos.protocol.value,
                "pool_type": pos.pool_type.value,
                "token_pair": pos.token_pair,
                "amount": pos.amount,
                "value_usd": pos.value_usd,
                "apy": pos.apy,
                "fees_earned": pos.fees_earned,
                "rewards_earned": pos.rewards_earned,
                "created_at": pos.created_at.isoformat(),
                "last_updated": pos.last_updated.isoformat(),
                "status": pos.status
            })
        
        staking_positions = []
        for staking in positions["staking_positions"]:
            staking_positions.append({
                "staking_id": staking.staking_id,
                "staking_type": staking.staking_type.value,
                "token": staking.token,
                "amount": staking.amount,
                "lock_period": staking.lock_period,
                "apy": staking.apy,
                "rewards_earned": staking.rewards_earned,
                "start_date": staking.start_date.isoformat(),
                "end_date": staking.end_date.isoformat(),
                "status": staking.status,
                "auto_compound": staking.auto_compound
            })
        
        return UserPositionsResponse(
            defi_positions=defi_positions,
            staking_positions=staking_positions,
            total_value=positions["total_value"],
            total_rewards=positions["total_rewards"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user positions: {str(e)}")

@router.get("/yield-farms", response_model=List[YieldFarmResponse])
async def get_yield_farms():
    """Get all available yield farms"""
    try:
        farms_data = await defi_engine.get_yield_farms()
        
        farms = []
        for farm_data in farms_data:
            farms.append(YieldFarmResponse(**farm_data))
        
        return farms
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get yield farms: {str(e)}")

@router.post("/rewards/claim", response_model=Dict[str, Any])
async def claim_rewards(claim_data: RewardClaim):
    """Claim rewards for a position"""
    try:
        result = await defi_engine.claim_rewards(
            user_id=claim_data.user_id,
            position_id=claim_data.position_id
        )
        
        return {
            "status": "success",
            "message": "Rewards claimed successfully",
            "data": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to claim rewards: {str(e)}")

@router.get("/metrics", response_model=DeFiMetricsResponse)
async def get_defi_metrics():
    """Get DeFi engine metrics and statistics"""
    try:
        metrics_data = await defi_engine.get_defi_metrics()
        
        return DeFiMetricsResponse(**metrics_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get DeFi metrics: {str(e)}")

@router.get("/protocols")
async def get_supported_protocols():
    """Get list of supported DeFi protocols"""
    try:
        protocols = []
        for protocol in DeFiProtocol:
            protocols.append({
                "name": protocol.value,
                "display_name": protocol.value.title(),
                "supported": True
            })
        
        return {
            "protocols": protocols,
            "total_count": len(protocols)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get protocols: {str(e)}")

@router.get("/staking-types")
async def get_staking_types():
    """Get list of available staking types"""
    try:
        staking_types = []
        for staking_type in StakingType:
            staking_types.append({
                "name": staking_type.value,
                "display_name": staking_type.value.replace("_", " ").title(),
                "description": f"{staking_type.value.replace('_', ' ').title()} staking"
            })
        
        return {
            "staking_types": staking_types,
            "total_count": len(staking_types)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get staking types: {str(e)}")

@router.get("/yield-strategies")
async def get_yield_strategies():
    """Get list of available yield farming strategies"""
    try:
        strategies = []
        for strategy in YieldStrategy:
            strategies.append({
                "name": strategy.value,
                "display_name": strategy.value.replace("_", " ").title(),
                "description": f"{strategy.value.replace('_', ' ').title()} strategy"
            })
        
        return {
            "strategies": strategies,
            "total_count": len(strategies)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get yield strategies: {str(e)}")

@router.get("/liquidity-pools")
async def get_liquidity_pools():
    """Get list of available liquidity pool types"""
    try:
        pools = []
        for pool in LiquidityPool:
            pools.append({
                "name": pool.value,
                "display_name": pool.value.replace("_", " ").title(),
                "description": f"{pool.value.replace('_', ' ').title()} liquidity pool"
            })
        
        return {
            "liquidity_pools": pools,
            "total_count": len(pools)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get liquidity pools: {str(e)}")

@router.get("/status")
async def get_defi_engine_status():
    """Get DeFi engine status and health"""
    try:
        return {
            "status": "active" if defi_engine.engine_active else "inactive",
            "engine_active": defi_engine.engine_active,
            "processing_task_running": defi_engine.processing_task is not None and not defi_engine.processing_task.done(),
            "uptime": "N/A",  # Could be calculated from start time
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get engine status: {str(e)}")

@router.post("/simulate-apy")
async def simulate_apy(
    protocol: str,
    amount: float,
    duration_days: int = 30
):
    """Simulate APY calculation for a given protocol and amount"""
    try:
        protocol_enum = DeFiProtocol(protocol.lower())
        base_apy = defi_engine._get_protocol_apy(protocol_enum)
        
        # Simulate compound interest
        daily_rate = base_apy / 365 / 100
        final_amount = amount * ((1 + daily_rate) ** duration_days)
        total_earnings = final_amount - amount
        
        return {
            "protocol": protocol,
            "initial_amount": amount,
            "duration_days": duration_days,
            "base_apy": base_apy,
            "final_amount": round(final_amount, 2),
            "total_earnings": round(total_earnings, 2),
            "daily_earnings": round(total_earnings / duration_days, 2)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to simulate APY: {str(e)}")

@router.get("/leaderboard")
async def get_defi_leaderboard(limit: int = 10):
    """Get DeFi leaderboard by total value locked"""
    try:
        # This would typically query a database
        # For now, return mock data
        leaderboard = []
        
        # Mock leaderboard data
        for i in range(min(limit, 10)):
            leaderboard.append({
                "rank": i + 1,
                "user_id": f"user_{i+1}",
                "total_value": 10000 - (i * 500),
                "total_rewards": 1000 - (i * 50),
                "positions_count": 5 - (i // 2),
                "apy": 15.0 - (i * 0.5)
            })
        
        return {
            "leaderboard": leaderboard,
            "total_users": len(leaderboard),
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get leaderboard: {str(e)}")
