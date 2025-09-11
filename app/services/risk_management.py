"""
Risk Management Service
Provides portfolio risk assessment, position sizing, stop-loss management, and risk analytics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis as redis_sync
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json

logger = logging.getLogger(__name__)


@dataclass
class RiskProfile:
    """User risk profile"""

    user_id: int
    risk_tolerance: str  # 'conservative', 'moderate', 'aggressive'
    max_portfolio_risk: float  # Maximum portfolio risk percentage
    max_position_size: float  # Maximum position size percentage
    max_drawdown: float  # Maximum acceptable drawdown
    stop_loss_percentage: float  # Default stop loss percentage
    take_profit_percentage: float  # Default take profit percentage
    correlation_threshold: float  # Maximum correlation between positions
    volatility_preference: str  # 'low', 'medium', 'high'
    created_at: datetime
    updated_at: datetime


@dataclass
class PositionRisk:
    """Position risk assessment"""

    position_id: str
    user_id: int
    market_id: int
    position_size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    risk_amount: float
    risk_percentage: float
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    expected_shortfall: float
    beta: float  # Market beta
    volatility: float
    correlation_score: float
    risk_score: float  # Overall risk score (0-100)
    last_updated: datetime


@dataclass
class PortfolioRisk:
    """Portfolio risk assessment"""

    user_id: int
    total_value: float
    total_risk: float
    portfolio_var_95: float
    portfolio_var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    diversification_score: float
    concentration_risk: float
    correlation_risk: float
    volatility_risk: float
    overall_risk_score: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    recommendations: List[str]
    last_updated: datetime


@dataclass
class RiskAlert:
    """Risk alert"""

    alert_id: str
    user_id: int
    alert_type: (
        str  # 'position_limit', 'drawdown', 'correlation', 'volatility', 'var_breach'
    )
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    current_value: float
    threshold_value: float
    position_id: Optional[str] = None
    market_id: Optional[int] = None
    created_at: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class RiskManagementService:
    """Comprehensive risk management service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.risk_profiles: Dict[int, RiskProfile] = {}
        self.position_risks: Dict[str, PositionRisk] = {}
        self.portfolio_risks: Dict[int, PortfolioRisk] = {}
        self.risk_alerts: Dict[str, RiskAlert] = {}

        # Risk metrics history
        self.risk_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.price_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Risk thresholds
        self.risk_thresholds = {
            "position_limit": 0.05,  # 5% max position size
            "portfolio_risk": 0.02,  # 2% max portfolio risk
            "drawdown_limit": 0.15,  # 15% max drawdown
            "correlation_limit": 0.7,  # 70% max correlation
            "var_limit_95": 0.03,  # 3% VaR limit
            "var_limit_99": 0.05,  # 5% VaR limit
        }

    async def initialize(self):
        """Initialize the risk management service"""
        logger.info("Initializing Risk Management Service")

        # Load existing data
        await self._load_risk_profiles()
        await self._load_position_risks()
        await self._load_portfolio_risks()

        # Start background tasks
        asyncio.create_task(self._monitor_risk_metrics())
        asyncio.create_task(self._update_risk_assessments())
        asyncio.create_task(self._check_risk_alerts())
        asyncio.create_task(self._cleanup_resolved_alerts())

        logger.info("Risk Management Service initialized successfully")

    async def create_risk_profile(
        self,
        user_id: int,
        risk_tolerance: str = "moderate",
        max_portfolio_risk: float = 0.02,
        max_position_size: float = 0.05,
        max_drawdown: float = 0.15,
        stop_loss_percentage: float = 0.10,
        take_profit_percentage: float = 0.20,
        correlation_threshold: float = 0.7,
        volatility_preference: str = "medium",
    ) -> RiskProfile:
        """Create a new risk profile for a user"""
        try:
            profile = RiskProfile(
                user_id=user_id,
                risk_tolerance=risk_tolerance,
                max_portfolio_risk=max_portfolio_risk,
                max_position_size=max_position_size,
                max_drawdown=max_drawdown,
                stop_loss_percentage=stop_loss_percentage,
                take_profit_percentage=take_profit_percentage,
                correlation_threshold=correlation_threshold,
                volatility_preference=volatility_preference,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            self.risk_profiles[user_id] = profile
            await self._cache_risk_profile(profile)

            logger.info(f"Created risk profile for user {user_id}")
            return profile

        except Exception as e:
            logger.error(f"Error creating risk profile: {e}")
            raise

    async def calculate_position_risk(
        self,
        user_id: int,
        market_id: int,
        position_size: float,
        entry_price: float,
        current_price: float,
    ) -> PositionRisk:
        """Calculate risk metrics for a position"""
        try:
            position_id = f"pos_{user_id}_{market_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # Calculate basic metrics
            unrealized_pnl = (current_price - entry_price) * position_size
            risk_amount = position_size * entry_price
            risk_percentage = risk_amount / await self._get_portfolio_value(user_id)

            # Calculate VaR and Expected Shortfall
            price_history = await self._get_price_history(market_id)
            if len(price_history) > 10:
                returns = np.diff(np.log(price_history))
                var_95 = np.percentile(returns, 5) * risk_amount
                var_99 = np.percentile(returns, 1) * risk_amount
                expected_shortfall = (
                    np.mean(returns[returns <= np.percentile(returns, 5)]) * risk_amount
                )
            else:
                var_95 = risk_amount * 0.02  # Default 2% VaR
                var_99 = risk_amount * 0.05  # Default 5% VaR
                expected_shortfall = risk_amount * 0.025  # Default 2.5% ES

            # Calculate beta and volatility
            market_returns = await self._get_market_returns(market_id)
            if len(market_returns) > 10:
                beta = self._calculate_beta(returns, market_returns)
                volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            else:
                beta = 1.0
                volatility = 0.2  # Default 20% volatility

            # Calculate correlation score
            correlation_score = await self._calculate_correlation_score(
                user_id, market_id
            )

            # Calculate overall risk score
            risk_score = self._calculate_position_risk_score(
                risk_percentage, var_95, volatility, correlation_score, beta
            )

            position_risk = PositionRisk(
                position_id=position_id,
                user_id=user_id,
                market_id=market_id,
                position_size=position_size,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                risk_amount=risk_amount,
                risk_percentage=risk_percentage,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                beta=beta,
                volatility=volatility,
                correlation_score=correlation_score,
                risk_score=risk_score,
                last_updated=datetime.utcnow(),
            )

            self.position_risks[position_id] = position_risk
            await self._cache_position_risk(position_risk)

            return position_risk

        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            raise

    async def calculate_portfolio_risk(self, user_id: int) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Get user positions
            positions = await self._get_user_positions(user_id)
            if not positions:
                return await self._create_empty_portfolio_risk(user_id)

            # Calculate portfolio metrics
            total_value = sum(pos["value"] for pos in positions)
            total_risk = sum(pos["risk_amount"] for pos in positions)

            # Calculate portfolio VaR
            portfolio_returns = await self._calculate_portfolio_returns(
                user_id, positions
            )
            if len(portfolio_returns) > 10:
                portfolio_var_95 = np.percentile(portfolio_returns, 5) * total_value
                portfolio_var_99 = np.percentile(portfolio_returns, 1) * total_value
                expected_shortfall = (
                    np.mean(
                        portfolio_returns[
                            portfolio_returns <= np.percentile(portfolio_returns, 5)
                        ]
                    )
                    * total_value
                )
            else:
                portfolio_var_95 = total_value * 0.02
                portfolio_var_99 = total_value * 0.05
                expected_shortfall = total_value * 0.025

            # Calculate risk-adjusted returns
            sharpe_ratio = await self._calculate_sharpe_ratio(
                user_id, portfolio_returns
            )
            sortino_ratio = await self._calculate_sortino_ratio(
                user_id, portfolio_returns
            )

            # Calculate drawdown
            max_drawdown, current_drawdown = await self._calculate_drawdown(user_id)

            # Calculate diversification and concentration metrics
            diversification_score = self._calculate_diversification_score(positions)
            concentration_risk = self._calculate_concentration_risk(
                positions, total_value
            )
            correlation_risk = await self._calculate_correlation_risk(
                user_id, positions
            )
            volatility_risk = self._calculate_volatility_risk(portfolio_returns)

            # Calculate overall risk score
            overall_risk_score = self._calculate_portfolio_risk_score(
                portfolio_var_95,
                max_drawdown,
                concentration_risk,
                correlation_risk,
                volatility_risk,
            )

            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk_score)

            # Generate recommendations
            recommendations = self._generate_risk_recommendations(
                positions,
                portfolio_var_95,
                max_drawdown,
                concentration_risk,
                correlation_risk,
            )

            portfolio_risk = PortfolioRisk(
                user_id=user_id,
                total_value=total_value,
                total_risk=total_risk,
                portfolio_var_95=portfolio_var_95,
                portfolio_var_99=portfolio_var_99,
                expected_shortfall=expected_shortfall,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                diversification_score=diversification_score,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                volatility_risk=volatility_risk,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                recommendations=recommendations,
                last_updated=datetime.utcnow(),
            )

            self.portfolio_risks[user_id] = portfolio_risk
            await self._cache_portfolio_risk(portfolio_risk)

            return portfolio_risk

        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            raise

    async def check_position_limits(
        self, user_id: int, market_id: int, position_size: float
    ) -> Dict[str, Any]:
        """Check if a position meets risk limits"""
        try:
            risk_profile = self.risk_profiles.get(user_id)
            if not risk_profile:
                return {"allowed": True, "reason": "No risk profile found"}

            portfolio_value = await self._get_portfolio_value(user_id)
            position_value = position_size * await self._get_current_price(market_id)
            position_percentage = (
                position_value / portfolio_value if portfolio_value > 0 else 0
            )

            # Check position size limit
            if position_percentage > risk_profile.max_position_size:
                return {
                    "allowed": False,
                    "reason": f"Position size {position_percentage:.2%} exceeds limit {risk_profile.max_position_size:.2%}",
                    "max_allowed": portfolio_value * risk_profile.max_position_size,
                }

            # Check portfolio risk limit
            current_portfolio_risk = await self._get_current_portfolio_risk(user_id)
            new_portfolio_risk = current_portfolio_risk + position_value

            if new_portfolio_risk / portfolio_value > risk_profile.max_portfolio_risk:
                return {
                    "allowed": False,
                    "reason": f"Portfolio risk would exceed limit {risk_profile.max_portfolio_risk:.2%}",
                    "max_allowed_risk": portfolio_value
                    * risk_profile.max_portfolio_risk,
                }

            return {"allowed": True, "reason": "Position within risk limits"}

        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
            return {"allowed": False, "reason": f"Error checking limits: {str(e)}"}

    async def create_risk_alert(
        self,
        user_id: int,
        alert_type: str,
        severity: str,
        message: str,
        current_value: float,
        threshold_value: float,
        position_id: Optional[str] = None,
        market_id: Optional[int] = None,
    ) -> RiskAlert:
        """Create a risk alert"""
        try:
            alert_id = f"alert_{user_id}_{alert_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            alert = RiskAlert(
                alert_id=alert_id,
                user_id=user_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                current_value=current_value,
                threshold_value=threshold_value,
                position_id=position_id,
                market_id=market_id,
                created_at=datetime.utcnow(),
            )

            self.risk_alerts[alert_id] = alert
            await self._cache_risk_alert(alert)

            logger.warning(f"Risk alert created: {alert_id} - {message}")
            return alert

        except Exception as e:
            logger.error(f"Error creating risk alert: {e}")
            raise

    async def get_risk_dashboard(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive risk dashboard for a user"""
        try:
            # Get risk profile
            risk_profile = self.risk_profiles.get(user_id)

            # Get portfolio risk
            portfolio_risk = await self.calculate_portfolio_risk(user_id)

            # Get active alerts
            active_alerts = [
                alert
                for alert in self.risk_alerts.values()
                if alert.user_id == user_id and not alert.resolved
            ]

            # Get position risks
            position_risks = [
                pos for pos in self.position_risks.values() if pos.user_id == user_id
            ]

            # Calculate risk metrics
            risk_metrics = {
                "total_positions": len(position_risks),
                "high_risk_positions": len(
                    [p for p in position_risks if p.risk_score > 70]
                ),
                "medium_risk_positions": len(
                    [p for p in position_risks if 30 <= p.risk_score <= 70]
                ),
                "low_risk_positions": len(
                    [p for p in position_risks if p.risk_score < 30]
                ),
                "active_alerts": len(active_alerts),
                "critical_alerts": len(
                    [a for a in active_alerts if a.severity == "critical"]
                ),
                "high_alerts": len([a for a in active_alerts if a.severity == "high"]),
            }

            dashboard = {
                "user_id": user_id,
                "risk_profile": {
                    "risk_tolerance": (
                        risk_profile.risk_tolerance if risk_profile else "moderate"
                    ),
                    "max_portfolio_risk": (
                        risk_profile.max_portfolio_risk if risk_profile else 0.02
                    ),
                    "max_position_size": (
                        risk_profile.max_position_size if risk_profile else 0.05
                    ),
                    "max_drawdown": risk_profile.max_drawdown if risk_profile else 0.15,
                },
                "portfolio_risk": {
                    "total_value": portfolio_risk.total_value,
                    "total_risk": portfolio_risk.total_risk,
                    "portfolio_var_95": portfolio_risk.portfolio_var_95,
                    "portfolio_var_99": portfolio_risk.portfolio_var_99,
                    "expected_shortfall": portfolio_risk.expected_shortfall,
                    "sharpe_ratio": portfolio_risk.sharpe_ratio,
                    "sortino_ratio": portfolio_risk.sortino_ratio,
                    "max_drawdown": portfolio_risk.max_drawdown,
                    "current_drawdown": portfolio_risk.current_drawdown,
                    "diversification_score": portfolio_risk.diversification_score,
                    "concentration_risk": portfolio_risk.concentration_risk,
                    "correlation_risk": portfolio_risk.correlation_risk,
                    "volatility_risk": portfolio_risk.volatility_risk,
                    "overall_risk_score": portfolio_risk.overall_risk_score,
                    "risk_level": portfolio_risk.risk_level,
                },
                "position_risks": [
                    {
                        "position_id": pos.position_id,
                        "market_id": pos.market_id,
                        "position_size": pos.position_size,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "risk_percentage": pos.risk_percentage,
                        "var_95": pos.var_95,
                        "beta": pos.beta,
                        "volatility": pos.volatility,
                        "risk_score": pos.risk_score,
                    }
                    for pos in position_risks
                ],
                "risk_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value,
                        "created_at": alert.created_at.isoformat(),
                    }
                    for alert in active_alerts
                ],
                "risk_metrics": risk_metrics,
                "recommendations": portfolio_risk.recommendations,
                "last_updated": datetime.utcnow().isoformat(),
            }

            return dashboard

        except Exception as e:
            logger.error(f"Error getting risk dashboard: {e}")
            raise

    def _calculate_position_risk_score(
        self,
        risk_percentage: float,
        var_95: float,
        volatility: float,
        correlation_score: float,
        beta: float,
    ) -> float:
        """Calculate overall risk score for a position"""
        try:
            # Weighted risk factors
            risk_factors = {
                "size": risk_percentage * 100,  # Position size weight
                "var": abs(var_95) * 100,  # VaR weight
                "volatility": volatility * 100,  # Volatility weight
                "correlation": correlation_score * 100,  # Correlation weight
                "beta": abs(beta - 1) * 50,  # Beta deviation weight
            }

            # Calculate weighted score
            weights = {
                "size": 0.3,
                "var": 0.25,
                "volatility": 0.2,
                "correlation": 0.15,
                "beta": 0.1,
            }
            risk_score = sum(
                risk_factors[factor] * weights[factor] for factor in risk_factors
            )

            return min(100, max(0, risk_score))

        except Exception as e:
            logger.error(f"Error calculating position risk score: {e}")
            return 50.0  # Default medium risk

    def _calculate_portfolio_risk_score(
        self,
        portfolio_var_95: float,
        max_drawdown: float,
        concentration_risk: float,
        correlation_risk: float,
        volatility_risk: float,
    ) -> float:
        """Calculate overall portfolio risk score"""
        try:
            # Normalize risk factors
            var_score = min(100, abs(portfolio_var_95) * 1000)
            drawdown_score = max_drawdown * 100
            concentration_score = concentration_risk * 100
            correlation_score = correlation_risk * 100
            volatility_score = volatility_risk * 100

            # Weighted average
            weights = {
                "var": 0.3,
                "drawdown": 0.25,
                "concentration": 0.2,
                "correlation": 0.15,
                "volatility": 0.1,
            }

            risk_score = (
                var_score * weights["var"]
                + drawdown_score * weights["drawdown"]
                + concentration_score * weights["concentration"]
                + correlation_score * weights["correlation"]
                + volatility_score * weights["volatility"]
            )

            return min(100, max(0, risk_score))

        except Exception as e:
            logger.error(f"Error calculating portfolio risk score: {e}")
            return 50.0

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level based on score"""
        if risk_score < 25:
            return "low"
        elif risk_score < 50:
            return "medium"
        elif risk_score < 75:
            return "high"
        else:
            return "critical"

    def _generate_risk_recommendations(
        self,
        positions: List[Dict],
        portfolio_var_95: float,
        max_drawdown: float,
        concentration_risk: float,
        correlation_risk: float,
    ) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        try:
            # Portfolio diversification
            if len(positions) < 3:
                recommendations.append(
                    "Consider diversifying across more markets to reduce concentration risk"
                )

            # Position size recommendations
            large_positions = [
                p for p in positions if p.get("risk_percentage", 0) > 0.05
            ]
            if large_positions:
                recommendations.append(
                    "Consider reducing position sizes to limit individual position risk"
                )

            # VaR recommendations
            if portfolio_var_95 > 0.03:
                recommendations.append(
                    "Portfolio VaR is high - consider reducing overall exposure"
                )

            # Drawdown recommendations
            if max_drawdown > 0.15:
                recommendations.append(
                    "Maximum drawdown exceeded - review risk management strategy"
                )

            # Correlation recommendations
            if correlation_risk > 0.7:
                recommendations.append(
                    "High correlation between positions - consider uncorrelated markets"
                )

            # Concentration recommendations
            if concentration_risk > 0.5:
                recommendations.append(
                    "High concentration risk - consider rebalancing portfolio"
                )

            if not recommendations:
                recommendations.append("Portfolio risk is well-managed")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations at this time"]

    async def _monitor_risk_metrics(self):
        """Monitor risk metrics continuously"""
        while True:
            try:
                # Update risk metrics for all users
                for user_id in self.risk_profiles.keys():
                    await self.calculate_portfolio_risk(user_id)

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error monitoring risk metrics: {e}")
                await asyncio.sleep(600)

    async def _update_risk_assessments(self):
        """Update risk assessments"""
        while True:
            try:
                # Update position risks
                for position_id, position_risk in self.position_risks.items():
                    current_price = await self._get_current_price(
                        position_risk.market_id
                    )
                    if current_price != position_risk.current_price:
                        # Recalculate position risk
                        await self.calculate_position_risk(
                            position_risk.user_id,
                            position_risk.market_id,
                            position_risk.position_size,
                            position_risk.entry_price,
                            current_price,
                        )

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Error updating risk assessments: {e}")
                await asyncio.sleep(120)

    async def _check_risk_alerts(self):
        """Check for risk alert conditions"""
        while True:
            try:
                for user_id, portfolio_risk in self.portfolio_risks.items():
                    risk_profile = self.risk_profiles.get(user_id)
                    if not risk_profile:
                        continue

                    # Check drawdown alert
                    if portfolio_risk.current_drawdown > risk_profile.max_drawdown:
                        await self.create_risk_alert(
                            user_id,
                            "drawdown",
                            "high",
                            f"Current drawdown {portfolio_risk.current_drawdown:.2%} exceeds limit {risk_profile.max_drawdown:.2%}",
                            portfolio_risk.current_drawdown,
                            risk_profile.max_drawdown,
                        )

                    # Check VaR alert
                    if (
                        portfolio_risk.portfolio_var_95
                        > self.risk_thresholds["var_limit_95"]
                    ):
                        await self.create_risk_alert(
                            user_id,
                            "var_breach",
                            "medium",
                            f"Portfolio VaR {portfolio_risk.portfolio_var_95:.2%} exceeds limit {self.risk_thresholds['var_limit_95']:.2%}",
                            portfolio_risk.portfolio_var_95,
                            self.risk_thresholds["var_limit_95"],
                        )

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error checking risk alerts: {e}")
                await asyncio.sleep(600)

    async def _cleanup_resolved_alerts(self):
        """Clean up resolved alerts"""
        while True:
            try:
                resolved_alerts = [
                    alert_id
                    for alert_id, alert in self.risk_alerts.items()
                    if alert.resolved
                    and (datetime.utcnow() - alert.resolved_at).days > 7
                ]

                for alert_id in resolved_alerts:
                    del self.risk_alerts[alert_id]
                    await self.redis.delete(f"risk_alert:{alert_id}")

                await asyncio.sleep(3600)  # Cleanup every hour

            except Exception as e:
                logger.error(f"Error cleaning up resolved alerts: {e}")
                await asyncio.sleep(7200)

    # Helper methods (implementations would depend on your data models)
    async def _load_risk_profiles(self):
        """Load risk profiles from database"""
        pass

    async def _load_position_risks(self):
        """Load position risks from database"""
        pass

    async def _load_portfolio_risks(self):
        """Load portfolio risks from database"""
        pass

    async def _get_portfolio_value(self, user_id: int) -> float:
        """Get portfolio value for user"""
        return 10000.0  # Placeholder

    async def _get_price_history(self, market_id: int) -> List[float]:
        """Get price history for market"""
        return [100.0 + i * 0.1 for i in range(100)]  # Placeholder

    async def _get_market_returns(self, market_id: int) -> List[float]:
        """Get market returns"""
        return [0.001 * np.random.randn() for _ in range(100)]  # Placeholder

    def _calculate_beta(
        self, returns: List[float], market_returns: List[float]
    ) -> float:
        """Calculate beta"""
        if len(returns) != len(market_returns) or len(returns) < 10:
            return 1.0
        return np.cov(returns, market_returns)[0, 1] / np.var(market_returns)

    async def _calculate_correlation_score(self, user_id: int, market_id: int) -> float:
        """Calculate correlation with existing positions"""
        return 0.3  # Placeholder

    async def _get_user_positions(self, user_id: int) -> List[Dict]:
        """Get user positions"""
        return []  # Placeholder

    async def _calculate_portfolio_returns(
        self, user_id: int, positions: List[Dict]
    ) -> List[float]:
        """Calculate portfolio returns"""
        return [0.001 * np.random.randn() for _ in range(100)]  # Placeholder

    async def _calculate_sharpe_ratio(
        self, user_id: int, returns: List[float]
    ) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 10:
            return 0.0
        return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

    async def _calculate_sortino_ratio(
        self, user_id: int, returns: List[float]
    ) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 10:
            return 0.0
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return np.mean(returns) if np.mean(returns) > 0 else 0.0
        downside_deviation = np.std(negative_returns)
        return np.mean(returns) / downside_deviation if downside_deviation > 0 else 0.0

    async def _calculate_drawdown(self, user_id: int) -> Tuple[float, float]:
        """Calculate maximum and current drawdown"""
        return 0.05, 0.02  # Placeholder

    def _calculate_diversification_score(self, positions: List[Dict]) -> float:
        """Calculate diversification score"""
        if len(positions) <= 1:
            return 0.0
        return min(1.0, len(positions) / 10.0)  # Normalize to 0-1

    def _calculate_concentration_risk(
        self, positions: List[Dict], total_value: float
    ) -> float:
        """Calculate concentration risk"""
        if not positions or total_value == 0:
            return 0.0
        max_position_value = max(pos.get("value", 0) for pos in positions)
        return max_position_value / total_value

    async def _calculate_correlation_risk(
        self, user_id: int, positions: List[Dict]
    ) -> float:
        """Calculate correlation risk"""
        return 0.3  # Placeholder

    def _calculate_volatility_risk(self, returns: List[float]) -> float:
        """Calculate volatility risk"""
        if len(returns) < 10:
            return 0.2
        return np.std(returns) * np.sqrt(252)

    async def _create_empty_portfolio_risk(self, user_id: int) -> PortfolioRisk:
        """Create empty portfolio risk"""
        return PortfolioRisk(
            user_id=user_id,
            total_value=0.0,
            total_risk=0.0,
            portfolio_var_95=0.0,
            portfolio_var_99=0.0,
            expected_shortfall=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            diversification_score=0.0,
            concentration_risk=0.0,
            correlation_risk=0.0,
            volatility_risk=0.0,
            overall_risk_score=0.0,
            risk_level="low",
            recommendations=["No positions to analyze"],
            last_updated=datetime.utcnow(),
        )

    async def _get_current_price(self, market_id: int) -> float:
        """Get current price for market"""
        return 100.0  # Placeholder

    async def _get_current_portfolio_risk(self, user_id: int) -> float:
        """Get current portfolio risk"""
        return 0.0  # Placeholder

    # Caching methods
    async def _cache_risk_profile(self, profile: RiskProfile):
        """Cache risk profile"""
        try:
            cache_key = f"risk_profile:{profile.user_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "risk_tolerance": profile.risk_tolerance,
                        "max_portfolio_risk": profile.max_portfolio_risk,
                        "max_position_size": profile.max_position_size,
                        "max_drawdown": profile.max_drawdown,
                        "stop_loss_percentage": profile.stop_loss_percentage,
                        "take_profit_percentage": profile.take_profit_percentage,
                        "correlation_threshold": profile.correlation_threshold,
                        "volatility_preference": profile.volatility_preference,
                        "created_at": profile.created_at.isoformat(),
                        "updated_at": profile.updated_at.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching risk profile: {e}")

    async def _cache_position_risk(self, position_risk: PositionRisk):
        """Cache position risk"""
        try:
            cache_key = f"position_risk:{position_risk.position_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(
                    {
                        "user_id": position_risk.user_id,
                        "market_id": position_risk.market_id,
                        "position_size": position_risk.position_size,
                        "entry_price": position_risk.entry_price,
                        "current_price": position_risk.current_price,
                        "unrealized_pnl": position_risk.unrealized_pnl,
                        "risk_amount": position_risk.risk_amount,
                        "risk_percentage": position_risk.risk_percentage,
                        "var_95": position_risk.var_95,
                        "var_99": position_risk.var_99,
                        "expected_shortfall": position_risk.expected_shortfall,
                        "beta": position_risk.beta,
                        "volatility": position_risk.volatility,
                        "correlation_score": position_risk.correlation_score,
                        "risk_score": position_risk.risk_score,
                        "last_updated": position_risk.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching position risk: {e}")

    async def _cache_portfolio_risk(self, portfolio_risk: PortfolioRisk):
        """Cache portfolio risk"""
        try:
            cache_key = f"portfolio_risk:{portfolio_risk.user_id}"
            await self.redis.setex(
                cache_key,
                1800,  # 30 minutes TTL
                json.dumps(
                    {
                        "total_value": portfolio_risk.total_value,
                        "total_risk": portfolio_risk.total_risk,
                        "portfolio_var_95": portfolio_risk.portfolio_var_95,
                        "portfolio_var_99": portfolio_risk.portfolio_var_99,
                        "expected_shortfall": portfolio_risk.expected_shortfall,
                        "sharpe_ratio": portfolio_risk.sharpe_ratio,
                        "sortino_ratio": portfolio_risk.sortino_ratio,
                        "max_drawdown": portfolio_risk.max_drawdown,
                        "current_drawdown": portfolio_risk.current_drawdown,
                        "diversification_score": portfolio_risk.diversification_score,
                        "concentration_risk": portfolio_risk.concentration_risk,
                        "correlation_risk": portfolio_risk.correlation_risk,
                        "volatility_risk": portfolio_risk.volatility_risk,
                        "overall_risk_score": portfolio_risk.overall_risk_score,
                        "risk_level": portfolio_risk.risk_level,
                        "recommendations": portfolio_risk.recommendations,
                        "last_updated": portfolio_risk.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching portfolio risk: {e}")

    async def _cache_risk_alert(self, alert: RiskAlert):
        """Cache risk alert"""
        try:
            cache_key = f"risk_alert:{alert.alert_id}"
            await self.redis.setex(
                cache_key,
                86400,  # 24 hours TTL
                json.dumps(
                    {
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
                ),
            )
        except Exception as e:
            logger.error(f"Error caching risk alert: {e}")


# Factory function
async def get_risk_management_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> RiskManagementService:
    """Get risk management service instance"""
    service = RiskManagementService(redis_client, db_session)
    await service.initialize()
    return service
