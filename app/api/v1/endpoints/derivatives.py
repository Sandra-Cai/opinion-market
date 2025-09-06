"""
Derivatives Trading API Endpoints
Advanced derivatives trading, pricing, and risk management endpoints
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import redis.asyncio as redis
from pydantic import BaseModel, Field
import json
import uuid

from app.core.database import get_db
from app.core.redis_client import get_redis_client
from app.services.derivatives_trading import (
    get_derivatives_trading_service,
    DerivativesTradingService,
)
from app.services.derivatives_risk_management import (
    get_derivatives_risk_management_service,
    DerivativesRiskManagementService,
)
from app.schemas.derivatives import (
    DerivativeCreate,
    DerivativeResponse,
    DerivativePriceResponse,
    DerivativePositionCreate,
    DerivativePositionResponse,
    DerivativeOrderCreate,
    DerivativeOrderResponse,
    OptionPriceRequest,
    OptionPriceResponse,
    VolatilitySurfaceResponse,
    GreeksResponse,
    RiskLimitCreate,
    RiskLimitResponse,
    RiskMetricResponse,
    StressTestRequest,
    StressTestResponse,
    RiskReportResponse,
    PortfolioRiskResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket, user_id: int):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if (
            user_id in self.user_connections
            and websocket in self.user_connections[user_id]
        ):
            self.user_connections[user_id].remove(websocket)

    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except:
                    pass

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass


manager = ConnectionManager()


# Dependency injection
async def get_derivatives_service(
    redis_client: redis.Redis = Depends(get_redis_client), db: Session = Depends(get_db)
) -> DerivativesTradingService:
    return await get_derivatives_trading_service(redis_client, db)


async def get_risk_service(
    redis_client: redis.Redis = Depends(get_redis_client), db: Session = Depends(get_db)
) -> DerivativesRiskManagementService:
    return await get_derivatives_risk_management_service(redis_client, db)


# Derivatives Management Endpoints
@router.post("/derivatives", response_model=DerivativeResponse)
async def create_derivative(
    derivative_data: DerivativeCreate,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Create a new derivative instrument"""
    try:
        derivative = await derivatives_service.create_derivative(
            symbol=derivative_data.symbol,
            derivative_type=derivative_data.derivative_type,
            underlying_asset=derivative_data.underlying_asset,
            strike_price=derivative_data.strike_price,
            expiration_date=derivative_data.expiration_date,
            option_type=derivative_data.option_type,
            exercise_style=derivative_data.exercise_style,
            contract_size=derivative_data.contract_size,
            multiplier=derivative_data.multiplier,
            currency=derivative_data.currency,
            exchange=derivative_data.exchange,
        )

        return DerivativeResponse(
            derivative_id=derivative.derivative_id,
            symbol=derivative.symbol,
            derivative_type=derivative.derivative_type.value,
            underlying_asset=derivative.underlying_asset,
            strike_price=derivative.strike_price,
            expiration_date=derivative.expiration_date,
            option_type=(
                derivative.option_type.value if derivative.option_type else None
            ),
            exercise_style=(
                derivative.exercise_style.value if derivative.exercise_style else None
            ),
            contract_size=derivative.contract_size,
            multiplier=derivative.multiplier,
            currency=derivative.currency,
            exchange=derivative.exchange,
            is_active=derivative.is_active,
            created_at=derivative.created_at,
            last_updated=derivative.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating derivative: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/derivatives/{derivative_id}", response_model=DerivativeResponse)
async def get_derivative(
    derivative_id: str,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Get derivative by ID"""
    try:
        derivative = derivatives_service.derivatives.get(derivative_id)
        if not derivative:
            raise HTTPException(status_code=404, detail="Derivative not found")

        return DerivativeResponse(
            derivative_id=derivative.derivative_id,
            symbol=derivative.symbol,
            derivative_type=derivative.derivative_type.value,
            underlying_asset=derivative.underlying_asset,
            strike_price=derivative.strike_price,
            expiration_date=derivative.expiration_date,
            option_type=(
                derivative.option_type.value if derivative.option_type else None
            ),
            exercise_style=(
                derivative.exercise_style.value if derivative.exercise_style else None
            ),
            contract_size=derivative.contract_size,
            multiplier=derivative.multiplier,
            currency=derivative.currency,
            exchange=derivative.exchange,
            is_active=derivative.is_active,
            created_at=derivative.created_at,
            last_updated=derivative.last_updated,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting derivative: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/derivatives", response_model=List[DerivativeResponse])
async def list_derivatives(
    derivative_type: Optional[str] = None,
    underlying_asset: Optional[str] = None,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """List derivatives with optional filtering"""
    try:
        derivatives = list(derivatives_service.derivatives.values())

        # Apply filters
        if derivative_type:
            derivatives = [
                d for d in derivatives if d.derivative_type.value == derivative_type
            ]
        if underlying_asset:
            derivatives = [
                d for d in derivatives if d.underlying_asset == underlying_asset
            ]

        return [
            DerivativeResponse(
                derivative_id=d.derivative_id,
                symbol=d.symbol,
                derivative_type=d.derivative_type.value,
                underlying_asset=d.underlying_asset,
                strike_price=d.strike_price,
                expiration_date=d.expiration_date,
                option_type=d.option_type.value if d.option_type else None,
                exercise_style=d.exercise_style.value if d.exercise_style else None,
                contract_size=d.contract_size,
                multiplier=d.multiplier,
                currency=d.currency,
                exchange=d.exchange,
                is_active=d.is_active,
                created_at=d.created_at,
                last_updated=d.last_updated,
            )
            for d in derivatives
        ]

    except Exception as e:
        logger.error(f"Error listing derivatives: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Pricing Endpoints
@router.post("/derivatives/{derivative_id}/price", response_model=OptionPriceResponse)
async def calculate_option_price(
    derivative_id: str,
    price_request: OptionPriceRequest,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Calculate option price using Black-Scholes model"""
    try:
        result = await derivatives_service.calculate_option_price(
            derivative_id=derivative_id,
            underlying_price=price_request.underlying_price,
            risk_free_rate=price_request.risk_free_rate,
            dividend_yield=price_request.dividend_yield,
            volatility=price_request.volatility,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        greeks = result.get("greeks")
        greeks_response = None
        if greeks:
            greeks_response = GreeksResponse(
                delta=greeks.delta,
                gamma=greeks.gamma,
                theta=greeks.theta,
                vega=greeks.vega,
                rho=greeks.rho,
                vanna=greeks.vanna,
                volga=greeks.volga,
                charm=greeks.charm,
            )

        return OptionPriceResponse(
            derivative_id=result["derivative_id"],
            theoretical_price=result["theoretical_price"],
            greeks=greeks_response,
            underlying_price=result["underlying_price"],
            strike_price=result["strike_price"],
            time_to_expiry=result["time_to_expiry"],
            risk_free_rate=result["risk_free_rate"],
            dividend_yield=result["dividend_yield"],
            volatility=result["volatility"],
            option_type=result["option_type"],
            exercise_style=result["exercise_style"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating option price: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/derivatives/{derivative_id}/price", response_model=DerivativePriceResponse
)
async def get_derivative_price(
    derivative_id: str,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Get current derivative price"""
    try:
        price_data = await derivatives_service.get_derivative_price(derivative_id)
        if not price_data:
            raise HTTPException(status_code=404, detail="Price data not found")

        greeks_response = None
        if price_data.greeks:
            greeks_response = GreeksResponse(
                delta=price_data.greeks.delta,
                gamma=price_data.greeks.gamma,
                theta=price_data.greeks.theta,
                vega=price_data.greeks.vega,
                rho=price_data.greeks.rho,
                vanna=price_data.greeks.vanna,
                volga=price_data.greeks.volga,
                charm=price_data.greeks.charm,
            )

        return DerivativePriceResponse(
            derivative_id=price_data.derivative_id,
            timestamp=price_data.timestamp,
            bid_price=price_data.bid_price,
            ask_price=price_data.ask_price,
            mid_price=price_data.mid_price,
            last_price=price_data.last_price,
            volume=price_data.volume,
            open_interest=price_data.open_interest,
            implied_volatility=price_data.implied_volatility,
            greeks=greeks_response,
            theoretical_price=price_data.theoretical_price,
            price_source=price_data.price_source,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting derivative price: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Volatility Surface Endpoints
@router.get(
    "/volatility-surfaces/{underlying_asset}", response_model=VolatilitySurfaceResponse
)
async def get_volatility_surface(
    underlying_asset: str,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Get volatility surface for underlying asset"""
    try:
        volatility_surface = await derivatives_service.get_volatility_surface(
            underlying_asset
        )
        if not volatility_surface:
            raise HTTPException(status_code=404, detail="Volatility surface not found")

        return VolatilitySurfaceResponse(
            underlying_asset=volatility_surface.underlying_asset,
            timestamp=volatility_surface.timestamp,
            strikes=volatility_surface.strikes,
            expirations=[exp.isoformat() for exp in volatility_surface.expirations],
            implied_volatilities=volatility_surface.implied_volatilities,
            risk_free_rate=volatility_surface.risk_free_rate,
            dividend_yield=volatility_surface.dividend_yield,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting volatility surface: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Position Management Endpoints
@router.post("/positions", response_model=DerivativePositionResponse)
async def create_derivative_position(
    position_data: DerivativePositionCreate,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Create a derivative position"""
    try:
        position = await derivatives_service.create_derivative_position(
            user_id=position_data.user_id,
            derivative_id=position_data.derivative_id,
            quantity=position_data.quantity,
            average_price=position_data.average_price,
        )

        return DerivativePositionResponse(
            position_id=position.position_id,
            user_id=position.user_id,
            derivative_id=position.derivative_id,
            quantity=position.quantity,
            average_price=position.average_price,
            current_price=position.current_price,
            unrealized_pnl=position.unrealized_pnl,
            realized_pnl=position.realized_pnl,
            margin_required=position.margin_required,
            delta_exposure=position.delta_exposure,
            gamma_exposure=position.gamma_exposure,
            theta_exposure=position.theta_exposure,
            vega_exposure=position.vega_exposure,
            created_at=position.created_at,
            last_updated=position.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating derivative position: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/users/{user_id}/positions", response_model=List[DerivativePositionResponse]
)
async def get_user_positions(
    user_id: int,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Get user's derivative positions"""
    try:
        positions = derivatives_service.derivative_positions.get(user_id, [])

        return [
            DerivativePositionResponse(
                position_id=p.position_id,
                user_id=p.user_id,
                derivative_id=p.derivative_id,
                quantity=p.quantity,
                average_price=p.average_price,
                current_price=p.current_price,
                unrealized_pnl=p.unrealized_pnl,
                realized_pnl=p.realized_pnl,
                margin_required=p.margin_required,
                delta_exposure=p.delta_exposure,
                gamma_exposure=p.gamma_exposure,
                theta_exposure=p.theta_exposure,
                vega_exposure=p.vega_exposure,
                created_at=p.created_at,
                last_updated=p.last_updated,
            )
            for p in positions
        ]

    except Exception as e:
        logger.error(f"Error getting user positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Order Management Endpoints
@router.post("/orders", response_model=DerivativeOrderResponse)
async def place_derivative_order(
    order_data: DerivativeOrderCreate,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Place a derivative order"""
    try:
        order = await derivatives_service.place_derivative_order(
            user_id=order_data.user_id,
            derivative_id=order_data.derivative_id,
            order_type=order_data.order_type,
            side=order_data.side,
            quantity=order_data.quantity,
            price=order_data.price,
            stop_price=order_data.stop_price,
            time_in_force=order_data.time_in_force,
        )

        return DerivativeOrderResponse(
            order_id=order.order_id,
            user_id=order.user_id,
            derivative_id=order.derivative_id,
            order_type=order.order_type,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            time_in_force=order.time_in_force,
            status=order.status,
            filled_quantity=order.filled_quantity,
            average_fill_price=order.average_fill_price,
            commission=order.commission,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )

    except Exception as e:
        logger.error(f"Error placing derivative order: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/users/{user_id}/orders", response_model=List[DerivativeOrderResponse])
async def get_user_orders(
    user_id: int,
    status: Optional[str] = None,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Get user's derivative orders"""
    try:
        orders = derivatives_service.derivative_orders.get(user_id, [])

        # Apply status filter
        if status:
            orders = [o for o in orders if o.status == status]

        return [
            DerivativeOrderResponse(
                order_id=o.order_id,
                user_id=o.user_id,
                derivative_id=o.derivative_id,
                order_type=o.order_type,
                side=o.side,
                quantity=o.quantity,
                price=o.price,
                stop_price=o.stop_price,
                time_in_force=o.time_in_force,
                status=o.status,
                filled_quantity=o.filled_quantity,
                average_fill_price=o.average_fill_price,
                commission=o.commission,
                created_at=o.created_at,
                updated_at=o.updated_at,
            )
            for o in orders
        ]

    except Exception as e:
        logger.error(f"Error getting user orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Portfolio Greeks Endpoints
@router.get("/users/{user_id}/portfolio-greeks")
async def get_portfolio_greeks(
    user_id: int,
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Get portfolio Greeks for user"""
    try:
        portfolio_greeks = await derivatives_service.calculate_portfolio_greeks(user_id)

        return {
            "user_id": user_id,
            "portfolio_greeks": portfolio_greeks,
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        logger.error(f"Error getting portfolio Greeks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Risk Management Endpoints
@router.post("/risk-limits", response_model=RiskLimitResponse)
async def create_risk_limit(
    limit_data: RiskLimitCreate,
    risk_service: DerivativesRiskManagementService = Depends(get_risk_service),
):
    """Create a risk limit"""
    try:
        from app.services.derivatives_risk_management import RiskType

        risk_limit = await risk_service.create_risk_limit(
            user_id=limit_data.user_id,
            risk_type=RiskType(limit_data.risk_type),
            limit_name=limit_data.limit_name,
            limit_value=limit_data.limit_value,
            limit_type=limit_data.limit_type,
            time_horizon=limit_data.time_horizon,
        )

        return RiskLimitResponse(
            limit_id=risk_limit.limit_id,
            user_id=risk_limit.user_id,
            risk_type=risk_limit.risk_type.value,
            limit_name=risk_limit.limit_name,
            limit_value=risk_limit.limit_value,
            current_value=risk_limit.current_value,
            limit_type=risk_limit.limit_type,
            time_horizon=risk_limit.time_horizon,
            is_active=risk_limit.is_active,
            breach_count=risk_limit.breach_count,
            last_breach=risk_limit.last_breach,
            created_at=risk_limit.created_at,
            last_updated=risk_limit.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating risk limit: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/users/{user_id}/risk-limits", response_model=List[RiskLimitResponse])
async def get_user_risk_limits(
    user_id: int,
    risk_service: DerivativesRiskManagementService = Depends(get_risk_service),
):
    """Get user's risk limits"""
    try:
        risk_limits = risk_service.risk_limits.get(user_id, [])

        return [
            RiskLimitResponse(
                limit_id=rl.limit_id,
                user_id=rl.user_id,
                risk_type=rl.risk_type.value,
                limit_name=rl.limit_name,
                limit_value=rl.limit_value,
                current_value=rl.current_value,
                limit_type=rl.limit_type,
                time_horizon=rl.time_horizon,
                is_active=rl.is_active,
                breach_count=rl.breach_count,
                last_breach=rl.last_breach,
                created_at=rl.created_at,
                last_updated=rl.last_updated,
            )
            for rl in risk_limits
        ]

    except Exception as e:
        logger.error(f"Error getting user risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/var", response_model=Dict[str, Any])
async def calculate_var(
    user_id: int,
    confidence_level: float = 0.95,
    time_horizon: int = 1,
    method: str = "historical",
    risk_service: DerivativesRiskManagementService = Depends(get_risk_service),
):
    """Calculate Value at Risk (VaR) for user"""
    try:
        var_result = await risk_service.calculate_var(
            user_id=user_id,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            method=method,
        )

        return var_result

    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/users/{user_id}/stress-tests", response_model=StressTestResponse)
async def run_stress_test(
    user_id: int,
    stress_test_request: StressTestRequest,
    risk_service: DerivativesRiskManagementService = Depends(get_risk_service),
):
    """Run stress test for user"""
    try:
        from app.services.derivatives_risk_management import StressTestType

        stress_test = await risk_service.run_stress_test(
            user_id=user_id,
            test_type=StressTestType(stress_test_request.test_type),
            test_name=stress_test_request.test_name,
            scenarios=stress_test_request.scenarios,
        )

        return StressTestResponse(
            test_id=stress_test.test_id,
            user_id=stress_test.user_id,
            test_type=stress_test.test_type.value,
            test_name=stress_test.test_name,
            portfolio_value=stress_test.portfolio_value,
            stress_scenarios=stress_test.stress_scenarios,
            results=stress_test.results,
            max_loss=stress_test.max_loss,
            var_95=stress_test.var_95,
            var_99=stress_test.var_99,
            expected_shortfall=stress_test.expected_shortfall,
            created_at=stress_test.created_at,
        )

    except Exception as e:
        logger.error(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/risk-report", response_model=RiskReportResponse)
async def generate_risk_report(
    user_id: int,
    report_type: str = "comprehensive",
    risk_service: DerivativesRiskManagementService = Depends(get_risk_service),
):
    """Generate risk report for user"""
    try:
        risk_report = await risk_service.generate_risk_report(
            user_id=user_id, report_type=report_type
        )

        return RiskReportResponse(
            report_id=risk_report.report_id,
            user_id=risk_report.user_id,
            report_type=risk_report.report_type,
            report_date=risk_report.report_date,
            summary=risk_report.summary,
            risk_metrics=[
                RiskMetricResponse(
                    metric_id=rm.metric_id,
                    user_id=rm.user_id,
                    risk_type=rm.risk_type.value,
                    metric_name=rm.metric_name,
                    metric_value=rm.metric_value,
                    threshold=rm.threshold,
                    is_breached=rm.is_breached,
                    confidence_level=rm.confidence_level,
                    time_horizon=rm.time_horizon,
                    calculation_method=rm.calculation_method,
                    timestamp=rm.timestamp,
                )
                for rm in risk_report.risk_metrics
            ],
            risk_limits=[
                RiskLimitResponse(
                    limit_id=rl.limit_id,
                    user_id=rl.user_id,
                    risk_type=rl.risk_type.value,
                    limit_name=rl.limit_name,
                    limit_value=rl.limit_value,
                    current_value=rl.current_value,
                    limit_type=rl.limit_type,
                    time_horizon=rl.time_horizon,
                    is_active=rl.is_active,
                    breach_count=rl.breach_count,
                    last_breach=rl.last_breach,
                    created_at=rl.created_at,
                    last_updated=rl.last_updated,
                )
                for rl in risk_report.risk_limits
            ],
            stress_tests=[
                StressTestResponse(
                    test_id=st.test_id,
                    user_id=st.user_id,
                    test_type=st.test_type.value,
                    test_name=st.test_name,
                    portfolio_value=st.portfolio_value,
                    stress_scenarios=st.stress_scenarios,
                    results=st.results,
                    max_loss=st.max_loss,
                    var_95=st.var_95,
                    var_99=st.var_99,
                    expected_shortfall=st.expected_shortfall,
                    created_at=st.created_at,
                )
                for st in risk_report.stress_tests
            ],
            recommendations=risk_report.recommendations,
            created_at=risk_report.created_at,
        )

    except Exception as e:
        logger.error(f"Error generating risk report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/portfolio-risk", response_model=PortfolioRiskResponse)
async def get_portfolio_risk(
    user_id: int,
    risk_service: DerivativesRiskManagementService = Depends(get_risk_service),
):
    """Get portfolio risk metrics for user"""
    try:
        portfolio_risks = risk_service.portfolio_risks.get(user_id, [])
        if not portfolio_risks:
            raise HTTPException(status_code=404, detail="No portfolio risk data found")

        latest_risk = portfolio_risks[-1]

        return PortfolioRiskResponse(
            user_id=latest_risk.user_id,
            timestamp=latest_risk.timestamp,
            total_exposure=latest_risk.total_exposure,
            net_exposure=latest_risk.net_exposure,
            gross_exposure=latest_risk.gross_exposure,
            leverage=latest_risk.leverage,
            var_95=latest_risk.var_95,
            var_99=latest_risk.var_99,
            expected_shortfall=latest_risk.expected_shortfall,
            max_drawdown=latest_risk.max_drawdown,
            sharpe_ratio=latest_risk.sharpe_ratio,
            sortino_ratio=latest_risk.sortino_ratio,
            calmar_ratio=latest_risk.calmar_ratio,
            portfolio_greeks=latest_risk.portfolio_greeks,
            concentration_risk=latest_risk.concentration_risk,
            correlation_risk=latest_risk.correlation_risk,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting portfolio risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket Endpoints
@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    """WebSocket endpoint for real-time derivatives data"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Keep connection alive and send periodic updates
            await asyncio.sleep(30)

            # Send portfolio Greeks update
            try:
                derivatives_service = await get_derivatives_service()
                portfolio_greeks = await derivatives_service.calculate_portfolio_greeks(
                    user_id
                )

                message = {
                    "type": "portfolio_greeks",
                    "data": portfolio_greeks,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                await manager.send_personal_message(json.dumps(message), user_id)

            except Exception as e:
                logger.error(f"Error sending WebSocket update: {e}")

    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, user_id)


# Analytics Endpoints
@router.get("/analytics/derivatives-summary")
async def get_derivatives_summary(
    derivatives_service: DerivativesTradingService = Depends(get_derivatives_service),
):
    """Get derivatives trading summary"""
    try:
        total_derivatives = len(derivatives_service.derivatives)
        active_derivatives = len(
            [d for d in derivatives_service.derivatives.values() if d.is_active]
        )

        # Count by type
        type_counts = {}
        for derivative in derivatives_service.derivatives.values():
            type_name = derivative.derivative_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        # Count by underlying
        underlying_counts = {}
        for derivative in derivatives_service.derivatives.values():
            underlying = derivative.underlying_asset
            underlying_counts[underlying] = underlying_counts.get(underlying, 0) + 1

        return {
            "total_derivatives": total_derivatives,
            "active_derivatives": active_derivatives,
            "type_distribution": type_counts,
            "underlying_distribution": underlying_counts,
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        logger.error(f"Error getting derivatives summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/risk-summary")
async def get_risk_summary(
    risk_service: DerivativesRiskManagementService = Depends(get_risk_service),
):
    """Get risk management summary"""
    try:
        total_users = len(risk_service.risk_limits)
        total_limits = sum(len(limits) for limits in risk_service.risk_limits.values())
        active_limits = sum(
            len([l for l in limits if l.is_active])
            for limits in risk_service.risk_limits.values()
        )

        # Count breaches
        total_breaches = sum(
            sum(l.breach_count for l in limits)
            for limits in risk_service.risk_limits.values()
        )

        return {
            "total_users": total_users,
            "total_risk_limits": total_limits,
            "active_risk_limits": active_limits,
            "total_breaches": total_breaches,
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        logger.error(f"Error getting risk summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
