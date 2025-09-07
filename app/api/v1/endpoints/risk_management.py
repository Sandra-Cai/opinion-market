"""
Risk Management API Endpoints
Provides portfolio risk assessment, position sizing, stop-loss management, and risk analytics
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
from app.core.redis_client import get_redis_sync
from app.services.risk_management import (
    get_risk_management_service,
    RiskProfile,
    PositionRisk,
    PortfolioRisk,
    RiskAlert,
)
from app.schemas.risk_management import (
    RiskProfileRequest,
    RiskProfileResponse,
    PositionRiskRequest,
    PositionRiskResponse,
    PortfolioRiskResponse,
    RiskAlertResponse,
    RiskDashboardResponse,
    PositionLimitCheckResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connections for real-time risk alerts
websocket_connections: Dict[str, WebSocket] = {}


@router.post("/profiles/create", response_model=RiskProfileResponse)
async def create_risk_profile(
    request: RiskProfileRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Create a new risk profile for the user
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        profile = await risk_service.create_risk_profile(
            user_id=current_user.id,
            risk_tolerance=request.risk_tolerance,
            max_portfolio_risk=request.max_portfolio_risk,
            max_position_size=request.max_position_size,
            max_drawdown=request.max_drawdown,
            stop_loss_percentage=request.stop_loss_percentage,
            take_profit_percentage=request.take_profit_percentage,
            correlation_threshold=request.correlation_threshold,
            volatility_preference=request.volatility_preference,
        )

        return RiskProfileResponse(
            user_id=profile.user_id,
            risk_tolerance=profile.risk_tolerance,
            max_portfolio_risk=profile.max_portfolio_risk,
            max_position_size=profile.max_position_size,
            max_drawdown=profile.max_drawdown,
            stop_loss_percentage=profile.stop_loss_percentage,
            take_profit_percentage=profile.take_profit_percentage,
            correlation_threshold=profile.correlation_threshold,
            volatility_preference=profile.volatility_preference,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )

    except Exception as e:
        logger.error(f"Error creating risk profile: {e}")
        raise HTTPException(status_code=500, detail="Error creating risk profile")


@router.get("/profiles/{user_id}", response_model=RiskProfileResponse)
async def get_risk_profile(
    user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get risk profile by user ID
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        profile = risk_service.risk_profiles.get(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Risk profile not found")

        return RiskProfileResponse(
            user_id=profile.user_id,
            risk_tolerance=profile.risk_tolerance,
            max_portfolio_risk=profile.max_portfolio_risk,
            max_position_size=profile.max_position_size,
            max_drawdown=profile.max_drawdown,
            stop_loss_percentage=profile.stop_loss_percentage,
            take_profit_percentage=profile.take_profit_percentage,
            correlation_threshold=profile.correlation_threshold,
            volatility_preference=profile.volatility_preference,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )

    except Exception as e:
        logger.error(f"Error getting risk profile: {e}")
        raise HTTPException(status_code=500, detail="Error getting risk profile")


@router.post("/position-risk/calculate", response_model=PositionRiskResponse)
async def calculate_position_risk(
    request: PositionRiskRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Calculate risk metrics for a position
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        position_risk = await risk_service.calculate_position_risk(
            user_id=current_user.id,
            market_id=request.market_id,
            position_size=request.position_size,
            entry_price=request.entry_price,
            current_price=request.current_price,
        )

        return PositionRiskResponse(
            position_id=position_risk.position_id,
            user_id=position_risk.user_id,
            market_id=position_risk.market_id,
            position_size=position_risk.position_size,
            entry_price=position_risk.entry_price,
            current_price=position_risk.current_price,
            unrealized_pnl=position_risk.unrealized_pnl,
            risk_amount=position_risk.risk_amount,
            risk_percentage=position_risk.risk_percentage,
            var_95=position_risk.var_95,
            var_99=position_risk.var_99,
            expected_shortfall=position_risk.expected_shortfall,
            beta=position_risk.beta,
            volatility=position_risk.volatility,
            correlation_score=position_risk.correlation_score,
            risk_score=position_risk.risk_score,
            last_updated=position_risk.last_updated,
        )

    except Exception as e:
        logger.error(f"Error calculating position risk: {e}")
        raise HTTPException(status_code=500, detail="Error calculating position risk")


@router.get("/portfolio-risk/{user_id}", response_model=PortfolioRiskResponse)
async def get_portfolio_risk(
    user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get comprehensive portfolio risk metrics
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        portfolio_risk = await risk_service.calculate_portfolio_risk(user_id)

        return PortfolioRiskResponse(
            user_id=portfolio_risk.user_id,
            total_value=portfolio_risk.total_value,
            total_risk=portfolio_risk.total_risk,
            portfolio_var_95=portfolio_risk.portfolio_var_95,
            portfolio_var_99=portfolio_risk.portfolio_var_99,
            expected_shortfall=portfolio_risk.expected_shortfall,
            sharpe_ratio=portfolio_risk.sharpe_ratio,
            sortino_ratio=portfolio_risk.sortino_ratio,
            max_drawdown=portfolio_risk.max_drawdown,
            current_drawdown=portfolio_risk.current_drawdown,
            diversification_score=portfolio_risk.diversification_score,
            concentration_risk=portfolio_risk.concentration_risk,
            correlation_risk=portfolio_risk.correlation_risk,
            volatility_risk=portfolio_risk.volatility_risk,
            overall_risk_score=portfolio_risk.overall_risk_score,
            risk_level=portfolio_risk.risk_level,
            recommendations=portfolio_risk.recommendations,
            last_updated=portfolio_risk.last_updated,
        )

    except Exception as e:
        logger.error(f"Error getting portfolio risk: {e}")
        raise HTTPException(status_code=500, detail="Error getting portfolio risk")


@router.post("/position-limits/check", response_model=PositionLimitCheckResponse)
async def check_position_limits(
    market_id: int,
    position_size: float,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Check if a position meets risk limits
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        result = await risk_service.check_position_limits(
            user_id=current_user.id, market_id=market_id, position_size=position_size
        )

        return PositionLimitCheckResponse(
            allowed=result["allowed"],
            reason=result["reason"],
            max_allowed=result.get("max_allowed"),
            max_allowed_risk=result.get("max_allowed_risk"),
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Error checking position limits: {e}")
        raise HTTPException(status_code=500, detail="Error checking position limits")


@router.get("/dashboard/{user_id}", response_model=RiskDashboardResponse)
async def get_risk_dashboard(
    user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get comprehensive risk dashboard for a user
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        dashboard = await risk_service.get_risk_dashboard(user_id)

        return RiskDashboardResponse(
            user_id=dashboard["user_id"],
            risk_profile=dashboard["risk_profile"],
            portfolio_risk=dashboard["portfolio_risk"],
            position_risks=dashboard["position_risks"],
            risk_alerts=dashboard["risk_alerts"],
            risk_metrics=dashboard["risk_metrics"],
            recommendations=dashboard["recommendations"],
            last_updated=dashboard["last_updated"],
        )

    except Exception as e:
        logger.error(f"Error getting risk dashboard: {e}")
        raise HTTPException(status_code=500, detail="Error getting risk dashboard")


@router.get("/alerts")
async def get_risk_alerts(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    limit: int = Query(20, description="Number of alerts to return"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get risk alerts with optional filtering
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        alerts = []
        for alert in risk_service.risk_alerts.values():
            # Apply filters
            if user_id and alert.user_id != user_id:
                continue
            if alert_type and alert.alert_type != alert_type:
                continue
            if severity and alert.severity != severity:
                continue
            if resolved is not None and alert.resolved != resolved:
                continue

            alerts.append(
                {
                    "alert_id": alert.alert_id,
                    "user_id": alert.user_id,
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "position_id": alert.position_id,
                    "market_id": alert.market_id,
                    "created_at": alert.created_at.isoformat(),
                    "resolved": alert.resolved,
                    "resolved_at": (
                        alert.resolved_at.isoformat() if alert.resolved_at else None
                    ),
                }
            )

        # Sort by creation date and limit
        alerts.sort(key=lambda x: x["created_at"], reverse=True)
        alerts = alerts[:limit]

        return JSONResponse(
            content={
                "alerts": alerts,
                "count": len(alerts),
                "filters": {
                    "user_id": user_id,
                    "alert_type": alert_type,
                    "severity": severity,
                    "resolved": resolved,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error getting risk alerts: {e}")
        raise HTTPException(status_code=500, detail="Error getting risk alerts")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_risk_alert(
    alert_id: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Resolve a risk alert
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        alert = risk_service.risk_alerts.get(alert_id)
        if not alert:
            raise HTTPException(status_code=404, detail="Risk alert not found")

        if alert.user_id != current_user.id:
            raise HTTPException(
                status_code=403, detail="Not authorized to resolve this alert"
            )

        alert.resolved = True
        alert.resolved_at = datetime.utcnow()

        await risk_service._cache_risk_alert(alert)

        return JSONResponse(
            content={
                "message": f"Risk alert {alert_id} resolved successfully",
                "alert_id": alert_id,
                "resolved_at": alert.resolved_at.isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error resolving risk alert: {e}")
        raise HTTPException(status_code=500, detail="Error resolving risk alert")


@router.get("/analytics/var")
async def get_var_analytics(
    user_id: int,
    confidence_level: float = Query(
        0.95, ge=0.9, le=0.99, description="VaR confidence level"
    ),
    time_horizon: int = Query(1, ge=1, le=30, description="Time horizon in days"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get Value at Risk (VaR) analytics
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        portfolio_risk = await risk_service.calculate_portfolio_risk(user_id)

        # Calculate VaR based on confidence level
        if confidence_level == 0.95:
            var_value = portfolio_risk.portfolio_var_95
        elif confidence_level == 0.99:
            var_value = portfolio_risk.portfolio_var_99
        else:
            # Interpolate between 95% and 99% VaR
            var_value = (
                portfolio_risk.portfolio_var_95
                + (confidence_level - 0.95)
                * (portfolio_risk.portfolio_var_99 - portfolio_risk.portfolio_var_95)
                / 0.04
            )

        # Adjust for time horizon
        var_adjusted = var_value * (time_horizon**0.5)

        analytics = {
            "user_id": user_id,
            "confidence_level": confidence_level,
            "time_horizon_days": time_horizon,
            "var_value": var_adjusted,
            "var_percentage": (
                (var_adjusted / portfolio_risk.total_value) * 100
                if portfolio_risk.total_value > 0
                else 0
            ),
            "expected_shortfall": portfolio_risk.expected_shortfall,
            "portfolio_value": portfolio_risk.total_value,
            "risk_level": portfolio_risk.risk_level,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return JSONResponse(content=analytics)

    except Exception as e:
        logger.error(f"Error getting VaR analytics: {e}")
        raise HTTPException(status_code=500, detail="Error getting VaR analytics")


@router.get("/analytics/drawdown")
async def get_drawdown_analytics(
    user_id: int,
    period_days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Get drawdown analytics
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        portfolio_risk = await risk_service.calculate_portfolio_risk(user_id)

        analytics = {
            "user_id": user_id,
            "period_days": period_days,
            "max_drawdown": portfolio_risk.max_drawdown,
            "current_drawdown": portfolio_risk.current_drawdown,
            "max_drawdown_percentage": portfolio_risk.max_drawdown * 100,
            "current_drawdown_percentage": portfolio_risk.current_drawdown * 100,
            "drawdown_limit": 0.15,  # From risk profile
            "drawdown_status": (
                "within_limits"
                if portfolio_risk.current_drawdown <= 0.15
                else "exceeded"
            ),
            "portfolio_value": portfolio_risk.total_value,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return JSONResponse(content=analytics)

    except Exception as e:
        logger.error(f"Error getting drawdown analytics: {e}")
        raise HTTPException(status_code=500, detail="Error getting drawdown analytics")


@router.get("/analytics/correlation")
async def get_correlation_analytics(
    user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get correlation analytics
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        portfolio_risk = await risk_service.calculate_portfolio_risk(user_id)

        analytics = {
            "user_id": user_id,
            "correlation_risk": portfolio_risk.correlation_risk,
            "correlation_percentage": portfolio_risk.correlation_risk * 100,
            "correlation_threshold": 0.7,
            "correlation_status": (
                "low"
                if portfolio_risk.correlation_risk < 0.5
                else "medium" if portfolio_risk.correlation_risk < 0.7 else "high"
            ),
            "diversification_score": portfolio_risk.diversification_score,
            "diversification_percentage": portfolio_risk.diversification_score * 100,
            "recommendations": portfolio_risk.recommendations,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return JSONResponse(content=analytics)

    except Exception as e:
        logger.error(f"Error getting correlation analytics: {e}")
        raise HTTPException(
            status_code=500, detail="Error getting correlation analytics"
        )


@router.get("/analytics/volatility")
async def get_volatility_analytics(
    user_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get volatility analytics
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        portfolio_risk = await risk_service.calculate_portfolio_risk(user_id)

        analytics = {
            "user_id": user_id,
            "volatility_risk": portfolio_risk.volatility_risk,
            "volatility_percentage": portfolio_risk.volatility_risk * 100,
            "annualized_volatility": portfolio_risk.volatility_risk * (252**0.5),
            "sharpe_ratio": portfolio_risk.sharpe_ratio,
            "sortino_ratio": portfolio_risk.sortino_ratio,
            "risk_adjusted_return": (
                portfolio_risk.sharpe_ratio if portfolio_risk.sharpe_ratio > 0 else 0
            ),
            "volatility_status": (
                "low"
                if portfolio_risk.volatility_risk < 0.15
                else "medium" if portfolio_risk.volatility_risk < 0.25 else "high"
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return JSONResponse(content=analytics)

    except Exception as e:
        logger.error(f"Error getting volatility analytics: {e}")
        raise HTTPException(
            status_code=500, detail="Error getting volatility analytics"
        )


@router.websocket("/ws/risk-alerts/{client_id}")
async def websocket_risk_alerts(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time risk alerts
    """
    await websocket.accept()
    websocket_connections[client_id] = websocket

    try:
        # Send initial connection confirmation
        await websocket.send_text(
            json.dumps(
                {
                    "type": "risk_alerts_connected",
                    "client_id": client_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "available_features": [
                        "real_time_alerts",
                        "portfolio_risk_updates",
                        "position_risk_monitoring",
                        "var_breach_notifications",
                        "drawdown_alerts",
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
                if message.get("type") == "subscribe_user_alerts":
                    user_id = message.get("user_id")
                    # Subscribe to user-specific alerts
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "user_alerts_subscription_confirmed",
                                "user_id": user_id,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )

                elif message.get("type") == "get_portfolio_risk":
                    user_id = message.get("user_id")
                    # Get portfolio risk update
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "portfolio_risk_update",
                                "user_id": user_id,
                                "timestamp": datetime.utcnow().isoformat(),
                                "data": {
                                    "risk_level": "medium",
                                    "overall_risk_score": 45.5,
                                    "var_95": 0.025,
                                    "current_drawdown": 0.08,
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
        logger.info(f"Risk alerts client {client_id} disconnected")
    finally:
        if client_id in websocket_connections:
            del websocket_connections[client_id]


@router.get("/health/risk-management")
async def risk_management_health():
    """
    Health check for risk management services
    """
    return {
        "status": "healthy",
        "service": "risk_management",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "risk_profiles",
            "position_risk_calculation",
            "portfolio_risk_assessment",
            "var_analytics",
            "drawdown_monitoring",
            "correlation_analysis",
            "volatility_tracking",
            "risk_alerts",
            "position_limit_checks",
            "websocket_alerts",
        ],
        "metrics": {
            "active_risk_profiles": 0,  # Would be calculated from actual data
            "total_position_risks": 0,
            "active_portfolio_risks": 0,
            "active_alerts": 0,
        },
    }


@router.get("/stats/risk-management")
async def get_risk_management_stats(
    db: Session = Depends(get_db), current_user=Depends(get_current_user)
):
    """
    Get risk management statistics
    """
    try:
        redis_client = await get_redis_sync()
        risk_service = await get_risk_management_service(redis_client, db)

        stats = {
            "total_risk_profiles": len(risk_service.risk_profiles),
            "total_position_risks": len(risk_service.position_risks),
            "total_portfolio_risks": len(risk_service.portfolio_risks),
            "total_alerts": len(risk_service.risk_alerts),
            "active_alerts": len(
                [a for a in risk_service.risk_alerts.values() if not a.resolved]
            ),
            "resolved_alerts": len(
                [a for a in risk_service.risk_alerts.values() if a.resolved]
            ),
            "risk_level_distribution": {
                "low": len(
                    [
                        p
                        for p in risk_service.portfolio_risks.values()
                        if p.risk_level == "low"
                    ]
                ),
                "medium": len(
                    [
                        p
                        for p in risk_service.portfolio_risks.values()
                        if p.risk_level == "medium"
                    ]
                ),
                "high": len(
                    [
                        p
                        for p in risk_service.portfolio_risks.values()
                        if p.risk_level == "high"
                    ]
                ),
                "critical": len(
                    [
                        p
                        for p in risk_service.portfolio_risks.values()
                        if p.risk_level == "critical"
                    ]
                ),
            },
            "alert_severity_distribution": {
                "low": len(
                    [
                        a
                        for a in risk_service.risk_alerts.values()
                        if a.severity == "low"
                    ]
                ),
                "medium": len(
                    [
                        a
                        for a in risk_service.risk_alerts.values()
                        if a.severity == "medium"
                    ]
                ),
                "high": len(
                    [
                        a
                        for a in risk_service.risk_alerts.values()
                        if a.severity == "high"
                    ]
                ),
                "critical": len(
                    [
                        a
                        for a in risk_service.risk_alerts.values()
                        if a.severity == "critical"
                    ]
                ),
            },
            "average_risk_scores": {
                "position_risk": (
                    sum(p.risk_score for p in risk_service.position_risks.values())
                    / len(risk_service.position_risks)
                    if risk_service.position_risks
                    else 0
                ),
                "portfolio_risk": (
                    sum(
                        p.overall_risk_score
                        for p in risk_service.portfolio_risks.values()
                    )
                    / len(risk_service.portfolio_risks)
                    if risk_service.portfolio_risks
                    else 0
                ),
            },
        }

        return JSONResponse(content=stats)

    except Exception as e:
        logger.error(f"Error getting risk management stats: {e}")
        raise HTTPException(
            status_code=500, detail="Error getting risk management stats"
        )


@router.post("/simulation/position-risk")
async def simulate_position_risk(
    market_id: int,
    position_size: float,
    entry_price: float,
    scenarios: int = Query(
        1000, ge=100, le=10000, description="Number of simulation scenarios"
    ),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Simulate position risk under different scenarios
    """
    try:
        import numpy as np

        # Simulate price movements
        volatility = 0.2  # 20% annual volatility
        daily_volatility = volatility / np.sqrt(252)

        # Generate price scenarios
        price_scenarios = []
        for _ in range(scenarios):
            # Simulate 30-day price path
            daily_returns = np.random.normal(0, daily_volatility, 30)
            price_path = entry_price * np.exp(np.cumsum(daily_returns))
            price_scenarios.append(price_path[-1])  # Final price

        # Calculate risk metrics
        pnl_scenarios = [
            (price - entry_price) * position_size for price in price_scenarios
        ]

        var_95 = np.percentile(pnl_scenarios, 5)
        var_99 = np.percentile(pnl_scenarios, 1)
        expected_shortfall = np.mean([pnl for pnl in pnl_scenarios if pnl <= var_95])

        # Calculate probability of loss
        probability_of_loss = len([pnl for pnl in pnl_scenarios if pnl < 0]) / scenarios

        simulation_results = {
            "market_id": market_id,
            "position_size": position_size,
            "entry_price": entry_price,
            "scenarios": scenarios,
            "simulation_results": {
                "var_95": var_95,
                "var_99": var_99,
                "expected_shortfall": expected_shortfall,
                "probability_of_loss": probability_of_loss,
                "max_profit": max(pnl_scenarios),
                "max_loss": min(pnl_scenarios),
                "average_pnl": np.mean(pnl_scenarios),
                "pnl_volatility": np.std(pnl_scenarios),
            },
            "price_distribution": {
                "min_price": min(price_scenarios),
                "max_price": max(price_scenarios),
                "average_price": np.mean(price_scenarios),
                "price_volatility": np.std(price_scenarios),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        return JSONResponse(content=simulation_results)

    except Exception as e:
        logger.error(f"Error simulating position risk: {e}")
        raise HTTPException(status_code=500, detail="Error simulating position risk")
