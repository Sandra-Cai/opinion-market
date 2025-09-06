"""
Market Microstructure API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from app.services.market_microstructure import (
    MarketMicrostructureService,
    MarketDepth,
    LiquidityMetrics,
    OrderFlow,
    MarketRegimeAnalysis,
    MarketMakingStrategy,
    MarketMakingQuote,
    MarketImpact,
)
from app.services.liquidity_management import (
    LiquidityManagementService,
    LiquidityProfile,
    LiquidityPool,
    LiquidityAllocation,
    LiquidityOptimization,
    LiquidityAlert,
)
from app.schemas.market_microstructure import (
    MarketDepthResponse,
    LiquidityMetricsResponse,
    OrderFlowResponse,
    MarketRegimeResponse,
    MarketMakingStrategyCreate,
    MarketMakingStrategyResponse,
    MarketMakingQuoteResponse,
    MarketImpactResponse,
    LiquidityProfileResponse,
    LiquidityPoolCreate,
    LiquidityPoolResponse,
    LiquidityAllocationCreate,
    LiquidityAllocationResponse,
    LiquidityOptimizationResponse,
    LiquidityAlertResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/market-depth/{symbol}", response_model=MarketDepthResponse)
async def get_market_depth(
    symbol: str,
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Get market depth for a symbol"""
    try:
        market_depth = await mm_service.get_market_depth(symbol)

        if not market_depth:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Market depth not found"
            )

        return MarketDepthResponse(
            symbol=market_depth.symbol,
            timestamp=market_depth.timestamp,
            bid_levels=market_depth.bid_levels,
            ask_levels=market_depth.ask_levels,
            total_bid_volume=market_depth.total_bid_volume,
            total_ask_volume=market_depth.total_ask_volume,
            bid_ask_spread=market_depth.bid_ask_spread,
            mid_price=market_depth.mid_price,
            weighted_mid_price=market_depth.weighted_mid_price,
            imbalance_ratio=market_depth.imbalance_ratio,
            depth_score=market_depth.depth_score,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market depth: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market depth: {str(e)}",
        )


@router.get("/liquidity-metrics/{symbol}", response_model=LiquidityMetricsResponse)
async def get_liquidity_metrics(
    symbol: str,
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Get liquidity metrics for a symbol"""
    try:
        liquidity_metrics = await mm_service.get_liquidity_metrics(symbol)

        if not liquidity_metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Liquidity metrics not found",
            )

        return LiquidityMetricsResponse(
            symbol=liquidity_metrics.symbol,
            timestamp=liquidity_metrics.timestamp,
            bid_ask_spread=liquidity_metrics.bid_ask_spread,
            effective_spread=liquidity_metrics.effective_spread,
            realized_spread=liquidity_metrics.realized_spread,
            price_impact=liquidity_metrics.price_impact,
            market_impact=liquidity_metrics.market_impact,
            liquidity_score=liquidity_metrics.liquidity_score,
            depth_score=liquidity_metrics.depth_score,
            resilience_score=liquidity_metrics.resilience_score,
            turnover_ratio=liquidity_metrics.turnover_ratio,
            volume_weighted_price=liquidity_metrics.volume_weighted_price,
            time_weighted_price=liquidity_metrics.time_weighted_price,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting liquidity metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get liquidity metrics: {str(e)}",
        )


@router.get("/order-flow/{symbol}", response_model=OrderFlowResponse)
async def get_order_flow(
    symbol: str,
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Get order flow analysis for a symbol"""
    try:
        order_flow = await mm_service.get_order_flow(symbol)

        if not order_flow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Order flow not found"
            )

        return OrderFlowResponse(
            symbol=order_flow.symbol,
            timestamp=order_flow.timestamp,
            aggressive_buy_volume=order_flow.aggressive_buy_volume,
            aggressive_sell_volume=order_flow.aggressive_sell_volume,
            passive_buy_volume=order_flow.passive_buy_volume,
            passive_sell_volume=order_flow.passive_sell_volume,
            net_order_flow=order_flow.net_order_flow,
            order_flow_imbalance=order_flow.order_flow_imbalance,
            order_flow_pressure=order_flow.order_flow_pressure,
            flow_type=order_flow.flow_type.value,
            flow_strength=order_flow.flow_strength,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting order flow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get order flow: {str(e)}",
        )


@router.get("/market-regime/{symbol}", response_model=MarketRegimeResponse)
async def get_market_regime(
    symbol: str,
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Get market regime analysis for a symbol"""
    try:
        market_regime = await mm_service.get_market_regime(symbol)

        if not market_regime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Market regime not found"
            )

        return MarketRegimeResponse(
            symbol=market_regime.symbol,
            timestamp=market_regime.timestamp,
            regime=market_regime.regime.value,
            regime_confidence=market_regime.regime_confidence,
            volatility=market_regime.volatility,
            trend_strength=market_regime.trend_strength,
            mean_reversion_strength=market_regime.mean_reversion_strength,
            persistence=market_regime.persistence,
            jump_probability=market_regime.jump_probability,
            regime_duration=market_regime.regime_duration,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market regime: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market regime: {str(e)}",
        )


@router.get("/market-impact/{symbol}", response_model=List[MarketImpactResponse])
async def get_market_impact(
    symbol: str,
    limit: int = 100,
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Get market impact analysis for a symbol"""
    try:
        market_impacts = await mm_service.get_market_impact(symbol, limit)

        return [
            MarketImpactResponse(
                symbol=impact.symbol,
                timestamp=impact.timestamp,
                trade_size=impact.trade_size,
                price_impact=impact.price_impact,
                temporary_impact=impact.temporary_impact,
                permanent_impact=impact.permanent_impact,
                market_impact_cost=impact.market_impact_cost,
                implementation_shortfall=impact.implementation_shortfall,
                volume_impact=impact.volume_impact,
                time_impact=impact.time_impact,
            )
            for impact in market_impacts
        ]

    except Exception as e:
        logger.error(f"Error getting market impact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market impact: {str(e)}",
        )


@router.post(
    "/market-making/strategies",
    response_model=MarketMakingStrategyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_market_making_strategy(
    strategy_data: MarketMakingStrategyCreate,
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Create a market making strategy"""
    try:
        strategy = await mm_service.create_market_making_strategy(
            symbol=strategy_data.symbol,
            user_id=strategy_data.user_id,
            strategy_type=strategy_data.strategy_type,
            parameters=strategy_data.parameters,
        )

        return MarketMakingStrategyResponse(
            strategy_id=strategy.strategy_id,
            symbol=strategy.symbol,
            user_id=strategy.user_id,
            strategy_type=strategy.strategy_type,
            parameters=strategy.parameters,
            is_active=strategy.is_active,
            performance_metrics=strategy.performance_metrics,
            risk_limits=strategy.risk_limits,
            created_at=strategy.created_at,
            last_updated=strategy.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating market making strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create market making strategy: {str(e)}",
        )


@router.get(
    "/market-making/strategies/{strategy_id}/quotes",
    response_model=List[MarketMakingQuoteResponse],
)
async def get_market_making_quotes(
    strategy_id: str,
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Get market making quotes for a strategy"""
    try:
        quotes = await mm_service.get_market_making_quotes(strategy_id)

        return [
            MarketMakingQuoteResponse(
                quote_id=quote.quote_id,
                strategy_id=quote.strategy_id,
                symbol=quote.symbol,
                bid_price=quote.bid_price,
                ask_price=quote.ask_price,
                bid_size=quote.bid_size,
                ask_size=quote.ask_size,
                spread=quote.spread,
                mid_price=quote.mid_price,
                skew=quote.skew,
                timestamp=quote.timestamp,
                is_active=quote.is_active,
            )
            for quote in quotes
        ]

    except Exception as e:
        logger.error(f"Error getting market making quotes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market making quotes: {str(e)}",
        )


@router.put(
    "/market-making/strategies/{strategy_id}/parameters", response_model=Dict[str, str]
)
async def update_market_making_parameters(
    strategy_id: str,
    parameters: Dict[str, Any],
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Update market making strategy parameters"""
    try:
        success = await mm_service.update_market_making_parameters(
            strategy_id, parameters
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found"
            )

        return {
            "message": "Parameters updated successfully",
            "strategy_id": strategy_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating market making parameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update parameters: {str(e)}",
        )


@router.delete("/market-making/strategies/{strategy_id}", response_model=Dict[str, str])
async def stop_market_making_strategy(
    strategy_id: str,
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Stop market making strategy"""
    try:
        success = await mm_service.stop_market_making(strategy_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found"
            )

        return {"message": "Strategy stopped successfully", "strategy_id": strategy_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping market making strategy: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop strategy: {str(e)}",
        )


@router.get("/optimal-spread/{symbol}", response_model=Dict[str, float])
async def calculate_optimal_spread(
    symbol: str,
    trade_size: float,
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Calculate optimal spread for a given trade size"""
    try:
        result = await mm_service.calculate_optimal_spread(symbol, trade_size)
        return result

    except Exception as e:
        logger.error(f"Error calculating optimal spread: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate optimal spread: {str(e)}",
        )


@router.get("/market-impact-estimate/{symbol}", response_model=Dict[str, float])
async def estimate_market_impact(
    symbol: str,
    trade_size: float,
    execution_time: float = 60,
    mm_service: MarketMicrostructureService = Depends(
        get_market_microstructure_service
    ),
):
    """Estimate market impact for a trade"""
    try:
        result = await mm_service.estimate_market_impact(
            symbol, trade_size, execution_time
        )
        return result

    except Exception as e:
        logger.error(f"Error estimating market impact: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to estimate market impact: {str(e)}",
        )


# Liquidity Management Endpoints
@router.get("/liquidity-profile/{symbol}", response_model=LiquidityProfileResponse)
async def get_liquidity_profile(
    symbol: str,
    lm_service: LiquidityManagementService = Depends(get_liquidity_management_service),
):
    """Get liquidity profile for a symbol"""
    try:
        profile = await lm_service.get_liquidity_profile(symbol)

        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Liquidity profile not found",
            )

        return LiquidityProfileResponse(
            symbol=profile.symbol,
            timestamp=profile.timestamp,
            liquidity_score=profile.liquidity_score,
            depth_score=profile.depth_score,
            resilience_score=profile.resilience_score,
            turnover_ratio=profile.turnover_ratio,
            bid_ask_spread=profile.bid_ask_spread,
            effective_spread=profile.effective_spread,
            market_impact=profile.market_impact,
            liquidity_providers={
                k.value: v for k, v in profile.liquidity_providers.items()
            },
            liquidity_events=[e.value for e in profile.liquidity_events],
            volatility=profile.volatility,
            volume_profile=profile.volume_profile,
            price_levels=profile.price_levels,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting liquidity profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get liquidity profile: {str(e)}",
        )


@router.get("/liquidity-pools", response_model=List[LiquidityPoolResponse])
async def get_liquidity_pools(
    symbol: Optional[str] = None,
    lm_service: LiquidityManagementService = Depends(get_liquidity_management_service),
):
    """Get liquidity pools"""
    try:
        pools = await lm_service.get_liquidity_pools(symbol)

        return [
            LiquidityPoolResponse(
                pool_id=pool.pool_id,
                symbol=pool.symbol,
                pool_type=pool.pool_type,
                total_liquidity=pool.total_liquidity,
                available_liquidity=pool.available_liquidity,
                utilized_liquidity=pool.utilized_liquidity,
                utilization_rate=pool.utilization_rate,
                providers=pool.providers,
                fees=pool.fees,
                is_active=pool.is_active,
                created_at=pool.created_at,
                last_updated=pool.last_updated,
            )
            for pool in pools
        ]

    except Exception as e:
        logger.error(f"Error getting liquidity pools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get liquidity pools: {str(e)}",
        )


@router.post(
    "/liquidity-pools",
    response_model=LiquidityPoolResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_liquidity_pool(
    pool_data: LiquidityPoolCreate,
    lm_service: LiquidityManagementService = Depends(get_liquidity_management_service),
):
    """Create a new liquidity pool"""
    try:
        pool = await lm_service.create_liquidity_pool(
            symbol=pool_data.symbol,
            pool_type=pool_data.pool_type,
            total_liquidity=pool_data.total_liquidity,
            providers=pool_data.providers,
            fees=pool_data.fees,
        )

        return LiquidityPoolResponse(
            pool_id=pool.pool_id,
            symbol=pool.symbol,
            pool_type=pool.pool_type,
            total_liquidity=pool.total_liquidity,
            available_liquidity=pool.available_liquidity,
            utilized_liquidity=pool.utilized_liquidity,
            utilization_rate=pool.utilization_rate,
            providers=pool.providers,
            fees=pool.fees,
            is_active=pool.is_active,
            created_at=pool.created_at,
            last_updated=pool.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating liquidity pool: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create liquidity pool: {str(e)}",
        )


@router.post(
    "/liquidity-allocations",
    response_model=LiquidityAllocationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def allocate_liquidity(
    allocation_data: LiquidityAllocationCreate,
    lm_service: LiquidityManagementService = Depends(get_liquidity_management_service),
):
    """Allocate liquidity to a user"""
    try:
        allocation = await lm_service.allocate_liquidity(
            symbol=allocation_data.symbol,
            user_id=allocation_data.user_id,
            allocation_type=allocation_data.allocation_type,
            allocated_amount=allocation_data.allocated_amount,
        )

        return LiquidityAllocationResponse(
            allocation_id=allocation.allocation_id,
            symbol=allocation.symbol,
            user_id=allocation.user_id,
            allocation_type=allocation.allocation_type,
            allocated_amount=allocation.allocated_amount,
            utilized_amount=allocation.utilized_amount,
            utilization_rate=allocation.utilization_rate,
            performance_metrics=allocation.performance_metrics,
            risk_metrics=allocation.risk_metrics,
            created_at=allocation.created_at,
            last_updated=allocation.last_updated,
        )

    except Exception as e:
        logger.error(f"Error allocating liquidity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to allocate liquidity: {str(e)}",
        )


@router.get(
    "/liquidity-optimization/{symbol}", response_model=LiquidityOptimizationResponse
)
async def get_liquidity_optimization(
    symbol: str,
    lm_service: LiquidityManagementService = Depends(get_liquidity_management_service),
):
    """Get liquidity optimization for a symbol"""
    try:
        optimization = await lm_service.get_liquidity_optimization(symbol)

        if not optimization:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Liquidity optimization not found",
            )

        return LiquidityOptimizationResponse(
            symbol=optimization.symbol,
            timestamp=optimization.timestamp,
            optimal_allocation=optimization.optimal_allocation,
            expected_return=optimization.expected_return,
            risk_score=optimization.risk_score,
            liquidity_score=optimization.liquidity_score,
            diversification_ratio=optimization.diversification_ratio,
            efficiency_ratio=optimization.efficiency_ratio,
            recommendations=optimization.recommendations,
            confidence_score=optimization.confidence_score,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting liquidity optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get liquidity optimization: {str(e)}",
        )


@router.get("/liquidity-alerts", response_model=List[LiquidityAlertResponse])
async def get_liquidity_alerts(
    symbol: Optional[str] = None,
    severity: Optional[str] = None,
    lm_service: LiquidityManagementService = Depends(get_liquidity_management_service),
):
    """Get liquidity alerts"""
    try:
        alerts = await lm_service.get_liquidity_alerts(symbol, severity)

        return [
            LiquidityAlertResponse(
                alert_id=alert.alert_id,
                symbol=alert.symbol,
                alert_type=alert.alert_type.value,
                severity=alert.severity,
                message=alert.message,
                threshold=alert.threshold,
                current_value=alert.current_value,
                triggered_at=alert.triggered_at,
                is_acknowledged=alert.is_acknowledged,
                acknowledged_by=alert.acknowledged_by,
                acknowledged_at=alert.acknowledged_at,
            )
            for alert in alerts
        ]

    except Exception as e:
        logger.error(f"Error getting liquidity alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get liquidity alerts: {str(e)}",
        )


@router.put("/liquidity-alerts/{alert_id}/acknowledge", response_model=Dict[str, str])
async def acknowledge_liquidity_alert(
    alert_id: str,
    user_id: str,
    lm_service: LiquidityManagementService = Depends(get_liquidity_management_service),
):
    """Acknowledge a liquidity alert"""
    try:
        success = await lm_service.acknowledge_liquidity_alert(alert_id, user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found"
            )

        return {"message": "Alert acknowledged successfully", "alert_id": alert_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging liquidity alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to acknowledge alert: {str(e)}",
        )


@router.get("/liquidity-score/{symbol}", response_model=Dict[str, float])
async def calculate_liquidity_score(
    symbol: str,
    lm_service: LiquidityManagementService = Depends(get_liquidity_management_service),
):
    """Calculate comprehensive liquidity score"""
    try:
        result = await lm_service.calculate_liquidity_score(symbol)
        return result

    except Exception as e:
        logger.error(f"Error calculating liquidity score: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate liquidity score: {str(e)}",
        )


@router.get("/liquidity-optimization/{symbol}/allocate", response_model=Dict[str, Any])
async def optimize_liquidity_allocation(
    symbol: str,
    total_amount: float,
    lm_service: LiquidityManagementService = Depends(get_liquidity_management_service),
):
    """Optimize liquidity allocation across pools"""
    try:
        result = await lm_service.optimize_liquidity_allocation(symbol, total_amount)
        return result

    except Exception as e:
        logger.error(f"Error optimizing liquidity allocation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize liquidity allocation: {str(e)}",
        )


# Dependency injection functions
async def get_market_microstructure_service() -> MarketMicrostructureService:
    """Get Market Microstructure Service instance"""
    # This would be injected from the main app
    # For now, return a mock instance
    pass


async def get_liquidity_management_service() -> LiquidityManagementService:
    """Get Liquidity Management Service instance"""
    # This would be injected from the main app
    # For now, return a mock instance
    pass
