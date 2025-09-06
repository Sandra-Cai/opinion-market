"""
Quantitative Trading API Endpoints
Provides algorithmic trading, backtesting, signal generation, and portfolio optimization
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.core.redis_client import get_redis_client
from app.services.quantitative_trading import (
    get_quantitative_trading_service,
    TradingSignal,
    TradingStrategy,
    BacktestResult,
    PortfolioOptimization,
)
from app.schemas.quantitative_trading import (
    TradingStrategyRequest,
    TradingStrategyResponse,
    SignalResponse,
    BacktestRequest,
    BacktestResponse,
    PortfolioOptimizationRequest,
    PortfolioOptimizationResponse,
    TechnicalIndicatorsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connections for real-time quantitative trading updates
websocket_connections: Dict[str, WebSocket] = {}


@router.post("/strategies/create", response_model=TradingStrategyResponse)
async def create_trading_strategy(
    request: TradingStrategyRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Create a new trading strategy
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        strategy = await quant_service.create_trading_strategy(
            strategy_name=request.strategy_name,
            strategy_type=request.strategy_type,
            parameters=request.parameters,
            indicators=request.indicators,
            timeframes=request.timeframes,
            markets=request.markets,
            risk_management=request.risk_management,
        )

        return TradingStrategyResponse(
            strategy_id=strategy.strategy_id,
            strategy_name=strategy.strategy_name,
            strategy_type=strategy.strategy_type,
            parameters=strategy.parameters,
            indicators=strategy.indicators,
            timeframes=strategy.timeframes,
            markets=strategy.markets,
            risk_management=strategy.risk_management,
            performance_metrics=strategy.performance_metrics,
            is_active=strategy.is_active,
            created_at=strategy.created_at,
            last_updated=strategy.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating trading strategy: {e}")
        raise HTTPException(status_code=500, detail="Error creating trading strategy")


@router.get("/strategies", response_model=List[TradingStrategyResponse])
async def get_trading_strategies(
    strategy_type: Optional[str] = Query(None, description="Filter by strategy type"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get trading strategies with optional filtering
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        strategies = []
        for strategy in quant_service.trading_strategies.values():
            # Apply filters
            if strategy_type and strategy.strategy_type != strategy_type:
                continue
            if is_active is not None and strategy.is_active != is_active:
                continue

            strategies.append(
                TradingStrategyResponse(
                    strategy_id=strategy.strategy_id,
                    strategy_name=strategy.strategy_name,
                    strategy_type=strategy.strategy_type,
                    parameters=strategy.parameters,
                    indicators=strategy.indicators,
                    timeframes=strategy.timeframes,
                    markets=strategy.markets,
                    risk_management=strategy.risk_management,
                    performance_metrics=strategy.performance_metrics,
                    is_active=strategy.is_active,
                    created_at=strategy.created_at,
                    last_updated=strategy.last_updated,
                )
            )

        return strategies

    except Exception as e:
        logger.error(f"Error getting trading strategies: {e}")
        raise HTTPException(status_code=500, detail="Error getting trading strategies")


@router.get("/strategies/{strategy_id}", response_model=TradingStrategyResponse)
async def get_trading_strategy(
    strategy_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get trading strategy by ID
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        strategy = quant_service.trading_strategies.get(strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Trading strategy not found")

        return TradingStrategyResponse(
            strategy_id=strategy.strategy_id,
            strategy_name=strategy.strategy_name,
            strategy_type=strategy.strategy_type,
            parameters=strategy.parameters,
            indicators=strategy.indicators,
            timeframes=strategy.timeframes,
            markets=strategy.markets,
            risk_management=strategy.risk_management,
            performance_metrics=strategy.performance_metrics,
            is_active=strategy.is_active,
            created_at=strategy.created_at,
            last_updated=strategy.last_updated,
        )

    except Exception as e:
        logger.error(f"Error getting trading strategy: {e}")
        raise HTTPException(status_code=500, detail="Error getting trading strategy")


@router.post(
    "/strategies/{strategy_id}/signals/generate", response_model=SignalResponse
)
async def generate_signal(
    strategy_id: str,
    market_id: int = Query(..., description="Market ID for signal generation"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Generate trading signal for a strategy and market
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        signal = await quant_service.generate_signal(strategy_id, market_id)

        return SignalResponse(
            signal_id=signal.signal_id,
            strategy_id=signal.strategy_id,
            market_id=signal.market_id,
            signal_type=signal.signal_type,
            strength=signal.strength,
            confidence=signal.confidence,
            price_target=signal.price_target,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            reasoning=signal.reasoning,
            indicators=signal.indicators,
            created_at=signal.created_at,
            expires_at=signal.expires_at,
        )

    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        raise HTTPException(status_code=500, detail="Error generating signal")


@router.get("/signals")
async def get_trading_signals(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    market_id: Optional[int] = Query(None, description="Filter by market ID"),
    signal_type: Optional[str] = Query(None, description="Filter by signal type"),
    limit: int = Query(20, description="Number of signals to return"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get trading signals with optional filtering
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        signals = []
        for signal in quant_service.trading_signals.values():
            # Apply filters
            if strategy_id and signal.strategy_id != strategy_id:
                continue
            if market_id and signal.market_id != market_id:
                continue
            if signal_type and signal.signal_type != signal_type:
                continue

            signals.append(
                {
                    "signal_id": signal.signal_id,
                    "strategy_id": signal.strategy_id,
                    "market_id": signal.market_id,
                    "signal_type": signal.signal_type,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                    "price_target": signal.price_target,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "reasoning": signal.reasoning,
                    "indicators": signal.indicators,
                    "created_at": signal.created_at.isoformat(),
                    "expires_at": signal.expires_at.isoformat(),
                }
            )

        # Sort by creation date and limit
        signals.sort(key=lambda x: x["created_at"], reverse=True)
        signals = signals[:limit]

        return JSONResponse(
            content={
                "signals": signals,
                "count": len(signals),
                "filters": {
                    "strategy_id": strategy_id,
                    "market_id": market_id,
                    "signal_type": signal_type,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error getting trading signals: {e}")
        raise HTTPException(status_code=500, detail="Error getting trading signals")


@router.post("/strategies/{strategy_id}/backtest", response_model=BacktestResponse)
async def run_backtest(
    strategy_id: str,
    request: BacktestRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Run backtest for a trading strategy
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        backtest_result = await quant_service.run_backtest(
            strategy_id=strategy_id,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
        )

        return BacktestResponse(
            backtest_id=backtest_result.backtest_id,
            strategy_id=backtest_result.strategy_id,
            start_date=backtest_result.start_date,
            end_date=backtest_result.end_date,
            initial_capital=backtest_result.initial_capital,
            final_capital=backtest_result.final_capital,
            total_return=backtest_result.total_return,
            annualized_return=backtest_result.annualized_return,
            sharpe_ratio=backtest_result.sharpe_ratio,
            max_drawdown=backtest_result.max_drawdown,
            win_rate=backtest_result.win_rate,
            profit_factor=backtest_result.profit_factor,
            total_trades=backtest_result.total_trades,
            winning_trades=backtest_result.winning_trades,
            losing_trades=backtest_result.losing_trades,
            avg_win=backtest_result.avg_win,
            avg_loss=backtest_result.avg_loss,
            trade_history=backtest_result.trade_history,
            equity_curve=backtest_result.equity_curve,
            created_at=backtest_result.created_at,
        )

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail="Error running backtest")


@router.get("/backtests")
async def get_backtest_results(
    strategy_id: Optional[str] = Query(None, description="Filter by strategy ID"),
    limit: int = Query(20, description="Number of backtest results to return"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get backtest results with optional filtering
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        backtests = []
        for backtest in quant_service.backtest_results.values():
            # Apply filters
            if strategy_id and backtest.strategy_id != strategy_id:
                continue

            backtests.append(
                {
                    "backtest_id": backtest.backtest_id,
                    "strategy_id": backtest.strategy_id,
                    "start_date": backtest.start_date.isoformat(),
                    "end_date": backtest.end_date.isoformat(),
                    "initial_capital": backtest.initial_capital,
                    "final_capital": backtest.final_capital,
                    "total_return": backtest.total_return,
                    "annualized_return": backtest.annualized_return,
                    "sharpe_ratio": backtest.sharpe_ratio,
                    "max_drawdown": backtest.max_drawdown,
                    "win_rate": backtest.win_rate,
                    "profit_factor": backtest.profit_factor,
                    "total_trades": backtest.total_trades,
                    "winning_trades": backtest.winning_trades,
                    "losing_trades": backtest.losing_trades,
                    "avg_win": backtest.avg_win,
                    "avg_loss": backtest.avg_loss,
                    "created_at": backtest.created_at.isoformat(),
                }
            )

        # Sort by creation date and limit
        backtests.sort(key=lambda x: x["created_at"], reverse=True)
        backtests = backtests[:limit]

        return JSONResponse(
            content={
                "backtests": backtests,
                "count": len(backtests),
                "filters": {"strategy_id": strategy_id},
            }
        )

    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        raise HTTPException(status_code=500, detail="Error getting backtest results")


@router.post("/portfolio/optimize", response_model=PortfolioOptimizationResponse)
async def optimize_portfolio(
    request: PortfolioOptimizationRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Optimize portfolio allocation
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        optimization = await quant_service.optimize_portfolio(
            user_id=current_user.id,
            optimization_type=request.optimization_type,
            target_return=request.target_return,
            risk_tolerance=request.risk_tolerance,
            constraints=request.constraints,
        )

        return PortfolioOptimizationResponse(
            optimization_id=optimization.optimization_id,
            user_id=optimization.user_id,
            optimization_type=optimization.optimization_type,
            target_return=optimization.target_return,
            risk_tolerance=optimization.risk_tolerance,
            constraints=optimization.constraints,
            optimal_weights=optimization.optimal_weights,
            expected_return=optimization.expected_return,
            expected_volatility=optimization.expected_volatility,
            sharpe_ratio=optimization.sharpe_ratio,
            efficient_frontier=optimization.efficient_frontier,
            created_at=optimization.created_at,
        )

    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail="Error optimizing portfolio")


@router.get("/indicators/calculate")
async def calculate_technical_indicators(
    market_id: int = Query(..., description="Market ID"),
    indicators: List[str] = Query(..., description="List of indicators to calculate"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Calculate technical indicators for a market
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        # Get market data (this would be implemented based on your data source)
        market_data = await quant_service._get_market_data(market_id)

        if market_data.empty:
            raise HTTPException(status_code=404, detail="Market data not found")

        # Calculate indicators
        calculated_indicators = await quant_service.calculate_technical_indicators(
            market_data, indicators
        )

        # Convert to response format
        indicator_values = {}
        for indicator_name, indicator_series in calculated_indicators.items():
            if hasattr(indicator_series, "iloc"):
                indicator_values[indicator_name] = (
                    indicator_series.iloc[-1] if len(indicator_series) > 0 else None
                )
            else:
                indicator_values[indicator_name] = indicator_series

        return TechnicalIndicatorsResponse(
            market_id=market_id,
            indicators=indicator_values,
            calculated_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        raise HTTPException(
            status_code=500, detail="Error calculating technical indicators"
        )


@router.get("/portfolio/greeks")
async def get_portfolio_greeks(
    db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get portfolio Greeks for the current user
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        # This would be implemented to get actual portfolio Greeks
        # For now, returning placeholder data
        portfolio_greeks = {
            "total_delta": 0.0,
            "total_gamma": 0.0,
            "total_theta": 0.0,
            "total_vega": 0.0,
            "position_count": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return JSONResponse(content=portfolio_greeks)

    except Exception as e:
        logger.error(f"Error getting portfolio Greeks: {e}")
        raise HTTPException(status_code=500, detail="Error getting portfolio Greeks")


@router.websocket("/ws/quantitative-trading/{client_id}")
async def websocket_quantitative_trading_updates(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time quantitative trading updates
    """
    await websocket.accept()
    websocket_connections[client_id] = websocket

    try:
        # Send initial connection confirmation
        await websocket.send_text(
            json.dumps(
                {
                    "type": "quantitative_trading_connected",
                    "client_id": client_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "available_features": [
                        "strategy_updates",
                        "signal_notifications",
                        "backtest_results",
                        "portfolio_optimization",
                        "real_time_indicators",
                    ],
                }
            )
        )

        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "subscribe_strategy_updates":
                    strategy_id = message.get("strategy_id")
                    # Subscribe to strategy updates
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "strategy_subscription_confirmed",
                                "strategy_id": strategy_id,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )

                elif message.get("type") == "subscribe_signals":
                    # Subscribe to signal notifications
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "signal_subscription_confirmed",
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )

                elif message.get("type") == "get_backtest_updates":
                    strategy_id = message.get("strategy_id")
                    # Get backtest updates
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "backtest_update",
                                "strategy_id": strategy_id,
                                "timestamp": datetime.utcnow().isoformat(),
                                "data": {
                                    "update_type": "periodic",
                                    "status": "completed",
                                },
                            }
                        )
                    )

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                )

    except WebSocketDisconnect:
        logger.info(f"Quantitative trading client {client_id} disconnected")
    finally:
        if client_id in websocket_connections:
            del websocket_connections[client_id]


@router.get("/health/quantitative-trading")
async def quantitative_trading_health():
    """
    Health check for quantitative trading services
    """
    return {
        "status": "healthy",
        "service": "quantitative_trading",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "trading_strategies",
            "signal_generation",
            "backtesting",
            "portfolio_optimization",
            "technical_indicators",
            "real_time_updates",
        ],
        "metrics": {
            "active_strategies": 0,  # Would be calculated from actual data
            "total_signals": 0,
            "completed_backtests": 0,
            "portfolio_optimizations": 0,
        },
    }


@router.get("/stats/quantitative-trading")
async def get_quantitative_trading_stats(
    db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get quantitative trading statistics
    """
    try:
        redis_client = await get_redis_client()
        quant_service = await get_quantitative_trading_service(redis_client, db)

        stats = {
            "total_strategies": len(quant_service.trading_strategies),
            "active_strategies": len(
                [s for s in quant_service.trading_strategies.values() if s.is_active]
            ),
            "total_signals": len(quant_service.trading_signals),
            "total_backtests": len(quant_service.backtest_results),
            "total_optimizations": len(quant_service.portfolio_optimizations),
            "active_signals": len(
                [
                    s
                    for s in quant_service.trading_signals.values()
                    if s.expires_at > datetime.utcnow()
                ]
            ),
            "top_performing_strategies": [
                {
                    "strategy_id": strategy.strategy_id,
                    "strategy_name": strategy.strategy_name,
                    "total_return": strategy.performance_metrics.get(
                        "total_return", 0.0
                    ),
                    "sharpe_ratio": strategy.performance_metrics.get(
                        "sharpe_ratio", 0.0
                    ),
                }
                for strategy in sorted(
                    quant_service.trading_strategies.values(),
                    key=lambda s: s.performance_metrics.get("total_return", 0.0),
                    reverse=True,
                )[:5]
            ],
            "recent_signals": [
                {
                    "signal_id": signal.signal_id,
                    "strategy_id": signal.strategy_id,
                    "market_id": signal.market_id,
                    "signal_type": signal.signal_type,
                    "strength": signal.strength,
                    "confidence": signal.confidence,
                }
                for signal in sorted(
                    quant_service.trading_signals.values(),
                    key=lambda s: s.created_at,
                    reverse=True,
                )[:10]
            ],
        }

        return JSONResponse(content=stats)

    except Exception as e:
        logger.error(f"Error getting quantitative trading stats: {e}")
        raise HTTPException(
            status_code=500, detail="Error getting quantitative trading stats"
        )
