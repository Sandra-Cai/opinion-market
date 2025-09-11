"""
Derivatives Risk Management Service
Advanced risk management for derivatives trading
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis as redis_sync
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import math
from enum import Enum
import uuid
from scipy.stats import norm
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class RiskType(Enum):
    """Risk types"""

    MARKET = "market"
    CREDIT = "credit"
    LIQUIDITY = "liquidity"
    OPERATIONAL = "operational"
    MODEL = "model"
    CONCENTRATION = "concentration"
    BASIS = "basis"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    GAMMA = "gamma"
    THETA = "theta"
    VEGA = "vega"


class RiskLevel(Enum):
    """Risk levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StressTestType(Enum):
    """Stress test types"""

    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    SCENARIO = "scenario"
    SENSITIVITY = "sensitivity"


@dataclass
class RiskLimit:
    """Risk limit"""

    limit_id: str
    user_id: int
    risk_type: RiskType
    limit_name: str
    limit_value: float
    current_value: float
    limit_type: str  # 'absolute', 'percentage', 'var'
    time_horizon: str  # 'intraday', 'daily', 'weekly', 'monthly'
    is_active: bool
    breach_count: int
    last_breach: Optional[datetime]
    created_at: datetime
    last_updated: datetime


@dataclass
class RiskMetric:
    """Risk metric"""

    metric_id: str
    user_id: int
    risk_type: RiskType
    metric_name: str
    metric_value: float
    threshold: float
    is_breached: bool
    confidence_level: float
    time_horizon: str
    calculation_method: str
    timestamp: datetime


@dataclass
class StressTestResult:
    """Stress test result"""

    test_id: str
    user_id: int
    test_type: StressTestType
    test_name: str
    portfolio_value: float
    stress_scenarios: List[Dict[str, Any]]
    results: Dict[str, Any]
    max_loss: float
    var_95: float
    var_99: float
    expected_shortfall: float
    created_at: datetime


@dataclass
class RiskReport:
    """Risk report"""

    report_id: str
    user_id: int
    report_type: str
    report_date: datetime
    summary: Dict[str, Any]
    risk_metrics: List[RiskMetric]
    risk_limits: List[RiskLimit]
    stress_tests: List[StressTestResult]
    recommendations: List[str]
    created_at: datetime


@dataclass
class PortfolioRisk:
    """Portfolio risk metrics"""

    user_id: int
    timestamp: datetime
    total_exposure: float
    net_exposure: float
    gross_exposure: float
    leverage: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    portfolio_greeks: Dict[str, float]
    concentration_risk: Dict[str, float]
    correlation_risk: Dict[str, float]


class DerivativesRiskManagementService:
    """Comprehensive Derivatives Risk Management Service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session

        # Risk management
        self.risk_limits: Dict[str, List[RiskLimit]] = defaultdict(list)
        self.risk_metrics: Dict[str, List[RiskMetric]] = defaultdict(list)
        self.stress_tests: Dict[str, List[StressTestResult]] = defaultdict(list)
        self.risk_reports: Dict[str, List[RiskReport]] = defaultdict(list)
        self.portfolio_risks: Dict[str, List[PortfolioRisk]] = defaultdict(list)

        # Risk calculations
        self.var_models: Dict[str, Any] = {}
        self.correlation_matrices: Dict[str, np.ndarray] = {}
        self.volatility_estimates: Dict[str, float] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize the Derivatives Risk Management Service"""
        logger.info("Initializing Derivatives Risk Management Service")

        # Initialize risk models
        await self._initialize_risk_models()

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._monitor_risk_limits()),
            asyncio.create_task(self._calculate_risk_metrics()),
            asyncio.create_task(self._run_stress_tests()),
            asyncio.create_task(self._generate_risk_reports()),
            asyncio.create_task(self._update_portfolio_risks()),
        ]

        logger.info("Derivatives Risk Management Service initialized successfully")

    async def create_risk_limit(
        self,
        user_id: int,
        risk_type: RiskType,
        limit_name: str,
        limit_value: float,
        limit_type: str = "absolute",
        time_horizon: str = "daily",
    ) -> RiskLimit:
        """Create a risk limit"""
        try:
            limit_id = f"LIMIT_{uuid.uuid4().hex[:8]}"

            limit = RiskLimit(
                limit_id=limit_id,
                user_id=user_id,
                risk_type=risk_type,
                limit_name=limit_name,
                limit_value=limit_value,
                current_value=0.0,
                limit_type=limit_type,
                time_horizon=time_horizon,
                is_active=True,
                breach_count=0,
                last_breach=None,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.risk_limits[user_id].append(limit)

            logger.info(f"Created risk limit {limit_id}")
            return limit

        except Exception as e:
            logger.error(f"Error creating risk limit: {e}")
            raise

    async def calculate_var(
        self,
        user_id: int,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        method: str = "historical",
    ) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR)"""
        try:
            # Get portfolio positions
            positions = await self._get_user_positions(user_id)
            if not positions:
                return {
                    "var": 0.0,
                    "confidence_level": confidence_level,
                    "time_horizon": time_horizon,
                }

            # Get historical returns
            returns = await self._get_historical_returns(positions)

            if method == "historical":
                var = await self._calculate_historical_var(
                    returns, confidence_level, time_horizon
                )
            elif method == "parametric":
                var = await self._calculate_parametric_var(
                    returns, confidence_level, time_horizon
                )
            elif method == "monte_carlo":
                var = await self._calculate_monte_carlo_var(
                    positions, confidence_level, time_horizon
                )
            else:
                var = await self._calculate_historical_var(
                    returns, confidence_level, time_horizon
                )

            # Calculate Expected Shortfall
            expected_shortfall = await self._calculate_expected_shortfall(
                returns, confidence_level, time_horizon
            )

            return {
                "var": var,
                "expected_shortfall": expected_shortfall,
                "confidence_level": confidence_level,
                "time_horizon": time_horizon,
                "method": method,
                "portfolio_value": sum(pos["value"] for pos in positions),
                "timestamp": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {"error": str(e)}

    async def run_stress_test(
        self,
        user_id: int,
        test_type: StressTestType,
        test_name: str,
        scenarios: List[Dict[str, Any]],
    ) -> StressTestResult:
        """Run stress test"""
        try:
            test_id = f"STRESS_{uuid.uuid4().hex[:8]}"

            # Get current portfolio
            positions = await self._get_user_positions(user_id)
            current_value = sum(pos["value"] for pos in positions)

            # Run stress scenarios
            stress_results = []
            max_loss = 0.0

            for scenario in scenarios:
                scenario_result = await self._run_stress_scenario(positions, scenario)
                stress_results.append(scenario_result)
                max_loss = min(max_loss, scenario_result["loss"])

            # Calculate VaR and Expected Shortfall
            losses = [result["loss"] for result in stress_results]
            var_95 = np.percentile(losses, 5)
            var_99 = np.percentile(losses, 1)
            expected_shortfall = np.mean([loss for loss in losses if loss <= var_95])

            result = StressTestResult(
                test_id=test_id,
                user_id=user_id,
                test_type=test_type,
                test_name=test_name,
                portfolio_value=current_value,
                stress_scenarios=scenarios,
                results={
                    "scenario_results": stress_results,
                    "losses": losses,
                    "max_loss": max_loss,
                },
                max_loss=max_loss,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                created_at=datetime.utcnow(),
            )

            self.stress_tests[user_id].append(result)

            logger.info(f"Completed stress test {test_id}")
            return result

        except Exception as e:
            logger.error(f"Error running stress test: {e}")
            raise

    async def calculate_portfolio_greeks_risk(self, user_id: int) -> Dict[str, Any]:
        """Calculate portfolio Greeks risk"""
        try:
            positions = await self._get_user_positions(user_id)

            total_delta = 0.0
            total_gamma = 0.0
            total_theta = 0.0
            total_vega = 0.0

            for position in positions:
                if "greeks" in position:
                    greeks = position["greeks"]
                    quantity = position["quantity"]

                    total_delta += quantity * greeks.get("delta", 0)
                    total_gamma += quantity * greeks.get("gamma", 0)
                    total_theta += quantity * greeks.get("theta", 0)
                    total_vega += quantity * greeks.get("vega", 0)

            # Calculate Greeks risk
            delta_risk = abs(total_delta) * 0.01  # 1% underlying move
            gamma_risk = abs(total_gamma) * (0.01**2)  # 1% underlying move squared
            theta_risk = abs(total_theta) * (1 / 365)  # 1 day time decay
            vega_risk = abs(total_vega) * 0.01  # 1% volatility change

            return {
                "total_delta": total_delta,
                "total_gamma": total_gamma,
                "total_theta": total_theta,
                "total_vega": total_vega,
                "delta_risk": delta_risk,
                "gamma_risk": gamma_risk,
                "theta_risk": theta_risk,
                "vega_risk": vega_risk,
                "total_greeks_risk": delta_risk + gamma_risk + theta_risk + vega_risk,
                "timestamp": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio Greeks risk: {e}")
            return {"error": str(e)}

    async def calculate_concentration_risk(self, user_id: int) -> Dict[str, Any]:
        """Calculate concentration risk"""
        try:
            positions = await self._get_user_positions(user_id)
            if not positions:
                return {"concentration_risk": 0.0, "herfindahl_index": 0.0}

            # Calculate position weights
            total_value = sum(pos["value"] for pos in positions)
            weights = [pos["value"] / total_value for pos in positions]

            # Calculate Herfindahl-Hirschman Index
            hhi = sum(w**2 for w in weights)

            # Calculate concentration risk metrics
            max_weight = max(weights)
            top_5_weight = sum(sorted(weights, reverse=True)[:5])
            top_10_weight = sum(sorted(weights, reverse=True)[:10])

            # Calculate effective number of positions
            effective_positions = 1 / hhi if hhi > 0 else 0

            return {
                "herfindahl_index": hhi,
                "max_weight": max_weight,
                "top_5_weight": top_5_weight,
                "top_10_weight": top_10_weight,
                "effective_positions": effective_positions,
                "concentration_risk": hhi,  # Higher HHI = higher concentration
                "timestamp": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return {"error": str(e)}

    async def calculate_correlation_risk(self, user_id: int) -> Dict[str, Any]:
        """Calculate correlation risk"""
        try:
            positions = await self._get_user_positions(user_id)
            if len(positions) < 2:
                return {"correlation_risk": 0.0, "avg_correlation": 0.0}

            # Get correlation matrix
            symbols = [pos["symbol"] for pos in positions]
            correlation_matrix = await self._get_correlation_matrix(symbols)

            if correlation_matrix is None:
                return {"correlation_risk": 0.0, "avg_correlation": 0.0}

            # Calculate weighted average correlation
            weights = [pos["value"] for pos in positions]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            avg_correlation = 0.0
            correlation_count = 0

            for i in range(len(correlation_matrix)):
                for j in range(i + 1, len(correlation_matrix)):
                    correlation = correlation_matrix[i, j]
                    weight = weights[i] * weights[j]
                    avg_correlation += correlation * weight
                    correlation_count += 1

            if correlation_count > 0:
                avg_correlation /= correlation_count

            # Calculate correlation risk
            correlation_risk = avg_correlation * 0.5  # Simplified risk measure

            return {
                "avg_correlation": avg_correlation,
                "correlation_risk": correlation_risk,
                "max_correlation": float(
                    np.max(
                        correlation_matrix[
                            np.triu_indices_from(correlation_matrix, k=1)
                        ]
                    )
                ),
                "min_correlation": float(
                    np.min(
                        correlation_matrix[
                            np.triu_indices_from(correlation_matrix, k=1)
                        ]
                    )
                ),
                "timestamp": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return {"error": str(e)}

    async def generate_risk_report(
        self, user_id: int, report_type: str = "comprehensive"
    ) -> RiskReport:
        """Generate risk report"""
        try:
            report_id = f"REPORT_{uuid.uuid4().hex[:8]}"

            # Calculate risk metrics
            var_result = await self.calculate_var(user_id)
            greeks_risk = await self.calculate_portfolio_greeks_risk(user_id)
            concentration_risk = await self.calculate_concentration_risk(user_id)
            correlation_risk = await self.calculate_correlation_risk(user_id)

            # Get risk limits
            risk_limits = self.risk_limits.get(user_id, [])

            # Get recent stress tests
            recent_stress_tests = self.stress_tests.get(user_id, [])[
                -5:
            ]  # Last 5 tests

            # Create risk metrics
            risk_metrics = []

            # VaR metric
            if "var" in var_result:
                risk_metrics.append(
                    RiskMetric(
                        metric_id=f"METRIC_{uuid.uuid4().hex[:8]}",
                        user_id=user_id,
                        risk_type=RiskType.MARKET,
                        metric_name="Value at Risk (95%)",
                        metric_value=var_result["var"],
                        threshold=10000.0,  # Example threshold
                        is_breached=var_result["var"] > 10000.0,
                        confidence_level=0.95,
                        time_horizon="daily",
                        calculation_method="historical",
                        timestamp=datetime.utcnow(),
                    )
                )

            # Greeks risk metric
            if "total_greeks_risk" in greeks_risk:
                risk_metrics.append(
                    RiskMetric(
                        metric_id=f"METRIC_{uuid.uuid4().hex[:8]}",
                        user_id=user_id,
                        risk_type=RiskType.GAMMA,
                        metric_name="Greeks Risk",
                        metric_value=greeks_risk["total_greeks_risk"],
                        threshold=5000.0,  # Example threshold
                        is_breached=greeks_risk["total_greeks_risk"] > 5000.0,
                        confidence_level=0.95,
                        time_horizon="daily",
                        calculation_method="analytical",
                        timestamp=datetime.utcnow(),
                    )
                )

            # Create summary
            summary = {
                "total_positions": len(await self._get_user_positions(user_id)),
                "portfolio_value": var_result.get("portfolio_value", 0),
                "var_95": var_result.get("var", 0),
                "expected_shortfall": var_result.get("expected_shortfall", 0),
                "greeks_risk": greeks_risk.get("total_greeks_risk", 0),
                "concentration_risk": concentration_risk.get("concentration_risk", 0),
                "correlation_risk": correlation_risk.get("correlation_risk", 0),
                "active_limits": len([l for l in risk_limits if l.is_active]),
                "breached_limits": len(
                    [l for l in risk_limits if l.current_value > l.limit_value]
                ),
                "risk_level": self._determine_risk_level(
                    var_result, greeks_risk, concentration_risk
                ),
            }

            # Generate recommendations
            recommendations = await self._generate_risk_recommendations(
                var_result, greeks_risk, concentration_risk, correlation_risk
            )

            report = RiskReport(
                report_id=report_id,
                user_id=user_id,
                report_type=report_type,
                report_date=datetime.utcnow(),
                summary=summary,
                risk_metrics=risk_metrics,
                risk_limits=risk_limits,
                stress_tests=recent_stress_tests,
                recommendations=recommendations,
                created_at=datetime.utcnow(),
            )

            self.risk_reports[user_id].append(report)

            logger.info(f"Generated risk report {report_id}")
            return report

        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            raise

    # Background tasks
    async def _monitor_risk_limits(self):
        """Monitor risk limits"""
        while True:
            try:
                for user_id, limits in self.risk_limits.items():
                    for limit in limits:
                        if limit.is_active:
                            await self._check_risk_limit(limit)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error monitoring risk limits: {e}")
                await asyncio.sleep(300)

    async def _check_risk_limit(self, limit: RiskLimit):
        """Check individual risk limit"""
        try:
            # Calculate current value based on risk type
            if limit.risk_type == RiskType.MARKET:
                var_result = await self.calculate_var(limit.user_id)
                limit.current_value = var_result.get("var", 0)
            elif limit.risk_type == RiskType.GAMMA:
                greeks_risk = await self.calculate_portfolio_greeks_risk(limit.user_id)
                limit.current_value = greeks_risk.get("total_greeks_risk", 0)
            elif limit.risk_type == RiskType.CONCENTRATION:
                concentration_risk = await self.calculate_concentration_risk(
                    limit.user_id
                )
                limit.current_value = concentration_risk.get("concentration_risk", 0)

            # Check for breach
            if limit.current_value > limit.limit_value:
                limit.breach_count += 1
                limit.last_breach = datetime.utcnow()
                logger.warning(
                    f"Risk limit breached: {limit.limit_name} - {limit.current_value} > {limit.limit_value}"
                )

            limit.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error checking risk limit: {e}")

    async def _calculate_risk_metrics(self):
        """Calculate risk metrics for all users"""
        while True:
            try:
                for user_id in self.risk_limits.keys():
                    await self._update_user_risk_metrics(user_id)

                await asyncio.sleep(300)  # Calculate every 5 minutes

            except Exception as e:
                logger.error(f"Error calculating risk metrics: {e}")
                await asyncio.sleep(600)

    async def _update_user_risk_metrics(self, user_id: int):
        """Update risk metrics for a user"""
        try:
            # Calculate various risk metrics
            var_result = await self.calculate_var(user_id)
            greeks_risk = await self.calculate_portfolio_greeks_risk(user_id)
            concentration_risk = await self.calculate_concentration_risk(user_id)
            correlation_risk = await self.calculate_correlation_risk(user_id)

            # Create portfolio risk record
            portfolio_risk = PortfolioRisk(
                user_id=user_id,
                timestamp=datetime.utcnow(),
                total_exposure=var_result.get("portfolio_value", 0),
                net_exposure=0,  # Simplified
                gross_exposure=var_result.get("portfolio_value", 0),
                leverage=1.0,  # Simplified
                var_95=var_result.get("var", 0),
                var_99=var_result.get("var", 0) * 1.3,  # Approximate
                expected_shortfall=var_result.get("expected_shortfall", 0),
                max_drawdown=0,  # Would need historical data
                sharpe_ratio=0,  # Would need historical data
                sortino_ratio=0,  # Would need historical data
                calmar_ratio=0,  # Would need historical data
                portfolio_greeks={
                    "delta": greeks_risk.get("total_delta", 0),
                    "gamma": greeks_risk.get("total_gamma", 0),
                    "theta": greeks_risk.get("total_theta", 0),
                    "vega": greeks_risk.get("total_vega", 0),
                },
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
            )

            self.portfolio_risks[user_id].append(portfolio_risk)

            # Keep only recent records
            if len(self.portfolio_risks[user_id]) > 1000:
                self.portfolio_risks[user_id] = self.portfolio_risks[user_id][-1000:]

        except Exception as e:
            logger.error(f"Error updating user risk metrics: {e}")

    async def _run_stress_tests(self):
        """Run periodic stress tests"""
        while True:
            try:
                for user_id in self.risk_limits.keys():
                    await self._run_user_stress_tests(user_id)

                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                logger.error(f"Error running stress tests: {e}")
                await asyncio.sleep(7200)

    async def _run_user_stress_tests(self, user_id: int):
        """Run stress tests for a user"""
        try:
            # Define stress scenarios
            scenarios = [
                {
                    "name": "Market Crash",
                    "description": "20% market decline",
                    "market_shock": -0.20,
                    "volatility_shock": 0.50,
                },
                {
                    "name": "Volatility Spike",
                    "description": "Volatility doubles",
                    "market_shock": 0.0,
                    "volatility_shock": 1.0,
                },
                {
                    "name": "Interest Rate Shock",
                    "description": "2% rate increase",
                    "rate_shock": 0.02,
                    "market_shock": -0.05,
                },
            ]

            # Run stress test
            await self.run_stress_test(
                user_id, StressTestType.SCENARIO, "Daily Stress Test", scenarios
            )

        except Exception as e:
            logger.error(f"Error running user stress tests: {e}")

    async def _generate_risk_reports(self):
        """Generate periodic risk reports"""
        while True:
            try:
                for user_id in self.risk_limits.keys():
                    # Generate daily report
                    await self.generate_risk_report(user_id, "daily")

                await asyncio.sleep(86400)  # Generate daily

            except Exception as e:
                logger.error(f"Error generating risk reports: {e}")
                await asyncio.sleep(172800)

    async def _update_portfolio_risks(self):
        """Update portfolio risk metrics"""
        while True:
            try:
                for user_id in self.risk_limits.keys():
                    await self._update_user_risk_metrics(user_id)

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error updating portfolio risks: {e}")
                await asyncio.sleep(600)

    # Risk calculation methods
    async def _calculate_historical_var(
        self, returns: List[float], confidence_level: float, time_horizon: int
    ) -> float:
        """Calculate historical VaR"""
        try:
            if not returns:
                return 0.0

            # Sort returns
            sorted_returns = sorted(returns)

            # Calculate VaR
            var_index = int((1 - confidence_level) * len(sorted_returns))
            var = -sorted_returns[var_index] if var_index < len(sorted_returns) else 0.0

            # Scale for time horizon
            var *= np.sqrt(time_horizon)

            return var

        except Exception as e:
            logger.error(f"Error calculating historical VaR: {e}")
            return 0.0

    async def _calculate_parametric_var(
        self, returns: List[float], confidence_level: float, time_horizon: int
    ) -> float:
        """Calculate parametric VaR"""
        try:
            if not returns:
                return 0.0

            # Calculate mean and standard deviation
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            # Calculate VaR using normal distribution
            z_score = norm.ppf(1 - confidence_level)
            var = -(mean_return + z_score * std_return)

            # Scale for time horizon
            var *= np.sqrt(time_horizon)

            return var

        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}")
            return 0.0

    async def _calculate_monte_carlo_var(
        self,
        positions: List[Dict[str, Any]],
        confidence_level: float,
        time_horizon: int,
    ) -> float:
        """Calculate Monte Carlo VaR"""
        try:
            if not positions:
                return 0.0

            # Generate random scenarios
            n_scenarios = 10000
            portfolio_returns = []

            for _ in range(n_scenarios):
                scenario_return = 0.0
                for position in positions:
                    # Generate random return for position
                    position_return = np.random.normal(0, 0.02)  # 2% daily volatility
                    scenario_return += position["value"] * position_return

                portfolio_returns.append(scenario_return)

            # Calculate VaR
            sorted_returns = sorted(portfolio_returns)
            var_index = int((1 - confidence_level) * len(sorted_returns))
            var = -sorted_returns[var_index] if var_index < len(sorted_returns) else 0.0

            return var

        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0

    async def _calculate_expected_shortfall(
        self, returns: List[float], confidence_level: float, time_horizon: int
    ) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        try:
            if not returns:
                return 0.0

            # Calculate VaR first
            var = await self._calculate_historical_var(
                returns, confidence_level, time_horizon
            )

            # Calculate expected shortfall
            tail_returns = [r for r in returns if -r >= var]
            expected_shortfall = np.mean(tail_returns) if tail_returns else 0.0

            return -expected_shortfall

        except Exception as e:
            logger.error(f"Error calculating expected shortfall: {e}")
            return 0.0

    async def _run_stress_scenario(
        self, positions: List[Dict[str, Any]], scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a stress scenario"""
        try:
            total_loss = 0.0

            for position in positions:
                # Apply market shock
                market_shock = scenario.get("market_shock", 0.0)
                volatility_shock = scenario.get("volatility_shock", 0.0)
                rate_shock = scenario.get("rate_shock", 0.0)

                # Calculate position loss
                position_value = position["value"]
                position_loss = position_value * market_shock

                # Add Greeks impact
                if "greeks" in position:
                    greeks = position["greeks"]
                    quantity = position["quantity"]

                    # Delta impact
                    delta_impact = quantity * greeks.get("delta", 0) * market_shock

                    # Gamma impact
                    gamma_impact = (
                        0.5 * quantity * greeks.get("gamma", 0) * (market_shock**2)
                    )

                    # Theta impact (1 day)
                    theta_impact = quantity * greeks.get("theta", 0) * (1 / 365)

                    # Vega impact
                    vega_impact = quantity * greeks.get("vega", 0) * volatility_shock

                    position_loss += (
                        delta_impact + gamma_impact + theta_impact + vega_impact
                    )

                total_loss += position_loss

            return {
                "scenario_name": scenario.get("name", "Unknown"),
                "loss": total_loss,
                "positions_affected": len(positions),
            }

        except Exception as e:
            logger.error(f"Error running stress scenario: {e}")
            return {"scenario_name": "Error", "loss": 0.0, "positions_affected": 0}

    # Helper methods
    async def _get_user_positions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get user positions (simplified)"""
        try:
            # This would typically come from the derivatives trading service
            # For now, return sample positions
            return [
                {
                    "symbol": "AAPL_C_150",
                    "quantity": 100,
                    "value": 15000,
                    "greeks": {"delta": 0.5, "gamma": 0.01, "theta": -0.1, "vega": 0.2},
                },
                {
                    "symbol": "GOOGL_P_2800",
                    "quantity": -50,
                    "value": -14000,
                    "greeks": {
                        "delta": -0.3,
                        "gamma": 0.005,
                        "theta": -0.05,
                        "vega": 0.15,
                    },
                },
            ]

        except Exception as e:
            logger.error(f"Error getting user positions: {e}")
            return []

    async def _get_historical_returns(
        self, positions: List[Dict[str, Any]]
    ) -> List[float]:
        """Get historical returns for positions"""
        try:
            # Generate sample historical returns
            n_days = 252  # 1 year
            returns = []

            for _ in range(n_days):
                daily_return = 0.0
                for position in positions:
                    # Generate random return
                    position_return = np.random.normal(0, 0.02)
                    daily_return += position["value"] * position_return
                returns.append(daily_return)

            return returns

        except Exception as e:
            logger.error(f"Error getting historical returns: {e}")
            return []

    async def _get_correlation_matrix(self, symbols: List[str]) -> Optional[np.ndarray]:
        """Get correlation matrix for symbols"""
        try:
            # Generate sample correlation matrix
            n = len(symbols)
            correlation_matrix = np.random.uniform(-0.5, 0.8, (n, n))
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)

            return correlation_matrix

        except Exception as e:
            logger.error(f"Error getting correlation matrix: {e}")
            return None

    def _determine_risk_level(
        self,
        var_result: Dict[str, Any],
        greeks_risk: Dict[str, Any],
        concentration_risk: Dict[str, Any],
    ) -> str:
        """Determine overall risk level"""
        try:
            var = var_result.get("var", 0)
            greeks = greeks_risk.get("total_greeks_risk", 0)
            concentration = concentration_risk.get("concentration_risk", 0)

            # Simple risk scoring
            risk_score = 0

            if var > 20000:
                risk_score += 3
            elif var > 10000:
                risk_score += 2
            elif var > 5000:
                risk_score += 1

            if greeks > 10000:
                risk_score += 3
            elif greeks > 5000:
                risk_score += 2
            elif greeks > 2000:
                risk_score += 1

            if concentration > 0.5:
                risk_score += 2
            elif concentration > 0.3:
                risk_score += 1

            if risk_score >= 6:
                return "critical"
            elif risk_score >= 4:
                return "high"
            elif risk_score >= 2:
                return "medium"
            else:
                return "low"

        except Exception as e:
            logger.error(f"Error determining risk level: {e}")
            return "medium"

    async def _generate_risk_recommendations(
        self,
        var_result: Dict[str, Any],
        greeks_risk: Dict[str, Any],
        concentration_risk: Dict[str, Any],
        correlation_risk: Dict[str, Any],
    ) -> List[str]:
        """Generate risk recommendations"""
        try:
            recommendations = []

            # VaR recommendations
            var = var_result.get("var", 0)
            if var > 20000:
                recommendations.append(
                    "Consider reducing portfolio exposure - VaR exceeds $20,000"
                )
            elif var > 10000:
                recommendations.append(
                    "Monitor VaR closely - approaching high risk threshold"
                )

            # Greeks recommendations
            greeks = greeks_risk.get("total_greeks_risk", 0)
            if greeks > 10000:
                recommendations.append(
                    "High Greeks risk detected - consider hedging strategies"
                )

            # Concentration recommendations
            concentration = concentration_risk.get("concentration_risk", 0)
            if concentration > 0.5:
                recommendations.append(
                    "High concentration risk - consider diversifying positions"
                )

            # Correlation recommendations
            correlation = correlation_risk.get("correlation_risk", 0)
            if correlation > 0.3:
                recommendations.append(
                    "High correlation risk - consider uncorrelated assets"
                )

            if not recommendations:
                recommendations.append(
                    "Portfolio risk levels are within acceptable ranges"
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating risk recommendations: {e}")
            return ["Unable to generate recommendations"]

    async def _initialize_risk_models(self):
        """Initialize risk models"""
        try:
            # Initialize VaR models
            self.var_models = {
                "historical": "Historical Simulation",
                "parametric": "Parametric (Normal)",
                "monte_carlo": "Monte Carlo Simulation",
            }

            logger.info("Initialized risk models")

        except Exception as e:
            logger.error(f"Error initializing risk models: {e}")


# Factory function
async def get_derivatives_risk_management_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> DerivativesRiskManagementService:
    """Get Derivatives Risk Management Service instance"""
    service = DerivativesRiskManagementService(redis_client, db_session)
    await service.initialize()
    return service
