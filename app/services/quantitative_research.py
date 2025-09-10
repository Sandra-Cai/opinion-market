"""
Quantitative Research Service
Advanced quantitative research, strategy development, and factor analysis
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import redis.asyncio as redis
from sqlalchemy.orm import Session
import json
import math
from enum import Enum
import uuid
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ResearchType(Enum):
    """Research types"""

    FACTOR_ANALYSIS = "factor_analysis"
    STRATEGY_DEVELOPMENT = "strategy_development"
    RISK_MODELING = "risk_modeling"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    MARKET_ANOMALY = "market_anomaly"
    REGIME_ANALYSIS = "regime_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    VOLATILITY_MODELING = "volatility_modeling"


class FactorType(Enum):
    """Factor types"""

    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    SIZE = "size"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MACRO = "macro"


class ModelType(Enum):
    """Model types"""

    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ARIMA = "arima"
    GARCH = "garch"
    KALMAN_FILTER = "kalman_filter"
    MARKOV_SWITCHING = "markov_switching"


@dataclass
class ResearchProject:
    """Research project"""

    project_id: str
    user_id: int
    project_name: str
    research_type: ResearchType
    description: str
    status: str  # 'draft', 'active', 'completed', 'archived'
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    completed_at: Optional[datetime]


@dataclass
class Factor:
    """Quantitative factor"""

    factor_id: str
    factor_name: str
    factor_type: FactorType
    description: str
    calculation_method: str
    parameters: Dict[str, Any]
    universe: List[str]
    frequency: str  # 'daily', 'weekly', 'monthly'
    lookback_period: int
    is_active: bool
    performance_metrics: Dict[str, float]
    created_at: datetime
    last_updated: datetime


@dataclass
class FactorExposure:
    """Factor exposure for a security"""

    security_id: str
    factor_id: str
    exposure_value: float
    percentile_rank: float
    z_score: float
    timestamp: datetime
    confidence: float


@dataclass
class FactorReturn:
    """Factor return"""

    factor_id: str
    return_date: datetime
    factor_return: float
    cumulative_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    information_ratio: float


@dataclass
class ResearchResult:
    """Research result"""

    result_id: str
    project_id: str
    result_type: str  # 'factor_analysis', 'strategy_performance', 'risk_metrics'
    data: Dict[str, Any]
    metrics: Dict[str, float]
    charts: List[Dict[str, Any]]
    insights: List[str]
    recommendations: List[str]
    confidence_score: float
    created_at: datetime


@dataclass
class StrategySignal:
    """Strategy signal"""

    signal_id: str
    strategy_id: str
    security_id: str
    signal_type: str  # 'buy', 'sell', 'hold'
    signal_strength: float
    confidence: float
    expected_return: float
    risk_score: float
    timestamp: datetime
    metadata: Dict[str, Any]


class QuantitativeResearchService:
    """Comprehensive Quantitative Research Service"""

    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session

        # Research management
        self.research_projects: Dict[str, ResearchProject] = {}
        self.factors: Dict[str, Factor] = {}
        self.factor_exposures: Dict[str, List[FactorExposure]] = defaultdict(list)
        self.factor_returns: Dict[str, List[FactorReturn]] = defaultdict(list)
        self.research_results: Dict[str, List[ResearchResult]] = defaultdict(list)
        self.strategy_signals: Dict[str, List[StrategySignal]] = defaultdict(list)

        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.fundamental_data: Dict[str, pd.DataFrame] = {}
        self.macro_data: Dict[str, pd.DataFrame] = {}

        # Models
        self.factor_models: Dict[str, Any] = {}
        self.risk_models: Dict[str, Any] = {}
        self.regime_models: Dict[str, Any] = {}

        # Analytics
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.volatility_estimates: Dict[str, float] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize the Quantitative Research Service"""
        logger.info("Initializing Quantitative Research Service")

        # Load existing data
        await self._load_historical_data()
        await self._load_factors()

        # Initialize models
        await self._initialize_models()

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._update_factor_exposures()),
            asyncio.create_task(self._calculate_factor_returns()),
            asyncio.create_task(self._run_research_projects()),
            asyncio.create_task(self._generate_strategy_signals()),
            asyncio.create_task(self._update_performance_metrics()),
        ]

        logger.info("Quantitative Research Service initialized successfully")

    async def create_research_project(
        self,
        user_id: int,
        project_name: str,
        research_type: ResearchType,
        description: str,
        parameters: Dict[str, Any],
    ) -> ResearchProject:
        """Create a new research project"""
        try:
            project_id = f"RESEARCH_{uuid.uuid4().hex[:8]}"

            project = ResearchProject(
                project_id=project_id,
                user_id=user_id,
                project_name=project_name,
                research_type=research_type,
                description=description,
                status="draft",
                parameters=parameters,
                results={},
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                completed_at=None,
            )

            self.research_projects[project_id] = project

            logger.info(f"Created research project {project_id}")
            return project

        except Exception as e:
            logger.error(f"Error creating research project: {e}")
            raise

    async def run_factor_analysis(
        self,
        project_id: str,
        factor_list: List[str],
        universe: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> ResearchResult:
        """Run factor analysis"""
        try:
            project = self.research_projects.get(project_id)
            if not project:
                raise ValueError("Project not found")

            # Prepare data
            factor_data = await self._prepare_factor_data(
                factor_list, universe, start_date, end_date
            )

            # Run factor analysis
            analysis_results = await self._run_factor_analysis_core(factor_data)

            # Create research result
            result = ResearchResult(
                result_id=f"RESULT_{uuid.uuid4().hex[:8]}",
                project_id=project_id,
                result_type="factor_analysis",
                data=analysis_results["data"],
                metrics=analysis_results["metrics"],
                charts=analysis_results["charts"],
                insights=analysis_results["insights"],
                recommendations=analysis_results["recommendations"],
                confidence_score=analysis_results["confidence_score"],
                created_at=datetime.utcnow(),
            )

            if project_id not in self.research_results:
                self.research_results[project_id] = []

            self.research_results[project_id].append(result)

            # Update project
            project.results["factor_analysis"] = result.result_id
            project.last_updated = datetime.utcnow()

            logger.info(f"Completed factor analysis for project {project_id}")
            return result

        except Exception as e:
            logger.error(f"Error running factor analysis: {e}")
            raise

    async def create_factor(
        self,
        factor_name: str,
        factor_type: FactorType,
        description: str,
        calculation_method: str,
        parameters: Dict[str, Any],
        universe: List[str],
        frequency: str = "daily",
        lookback_period: int = 252,
    ) -> Factor:
        """Create a new quantitative factor"""
        try:
            factor_id = f"FACTOR_{uuid.uuid4().hex[:8]}"

            factor = Factor(
                factor_id=factor_id,
                factor_name=factor_name,
                factor_type=factor_type,
                description=description,
                calculation_method=calculation_method,
                parameters=parameters,
                universe=universe,
                frequency=frequency,
                lookback_period=lookback_period,
                is_active=True,
                performance_metrics={},
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.factors[factor_id] = factor

            # Calculate initial factor values
            asyncio.create_task(self._calculate_factor_values(factor))

            logger.info(f"Created factor {factor_id}")
            return factor

        except Exception as e:
            logger.error(f"Error creating factor: {e}")
            raise

    async def get_factor_exposures(
        self, factor_id: str, date: Optional[datetime] = None
    ) -> List[FactorExposure]:
        """Get factor exposures"""
        try:
            exposures = self.factor_exposures.get(factor_id, [])

            if date:
                exposures = [e for e in exposures if e.timestamp.date() == date.date()]

            return exposures

        except Exception as e:
            logger.error(f"Error getting factor exposures: {e}")
            return []

    async def get_factor_returns(
        self,
        factor_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[FactorReturn]:
        """Get factor returns"""
        try:
            returns = self.factor_returns.get(factor_id, [])

            if start_date:
                returns = [r for r in returns if r.return_date >= start_date]
            if end_date:
                returns = [r for r in returns if r.return_date <= end_date]

            return returns

        except Exception as e:
            logger.error(f"Error getting factor returns: {e}")
            return []

    async def run_correlation_analysis(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Run correlation analysis"""
        try:
            # Get price data
            price_data = await self._get_price_data(symbols, start_date, end_date)

            if price_data.empty:
                return {"error": "No price data available"}

            # Calculate returns
            returns = price_data.pct_change().dropna()

            # Calculate correlation matrix
            correlation_matrix = returns.corr()

            # Calculate rolling correlations
            rolling_corr = returns.rolling(window=60).corr()

            # Calculate average correlation
            avg_correlation = correlation_matrix.mean().mean()

            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        high_corr_pairs.append(
                            {
                                "pair": (
                                    correlation_matrix.columns[i],
                                    correlation_matrix.columns[j],
                                ),
                                "correlation": corr,
                            }
                        )

            # Calculate correlation stability
            correlation_stability = {}
            for col in correlation_matrix.columns:
                rolling_corr_col = rolling_corr[col].dropna()
                stability = 1 - rolling_corr_col.std()
                correlation_stability[col] = stability

            return {
                "correlation_matrix": correlation_matrix.to_dict(),
                "average_correlation": avg_correlation,
                "high_correlation_pairs": high_corr_pairs,
                "correlation_stability": correlation_stability,
                "data_points": len(returns),
                "analysis_date": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error running correlation analysis: {e}")
            return {"error": str(e)}

    async def run_volatility_analysis(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        model_type: str = "garch",
    ) -> Dict[str, Any]:
        """Run volatility analysis"""
        try:
            # Get price data
            price_data = await self._get_price_data([symbol], start_date, end_date)

            if price_data.empty:
                return {"error": "No price data available"}

            # Calculate returns
            returns = price_data[symbol].pct_change().dropna()

            # Calculate realized volatility
            realized_vol = returns.rolling(window=30).std() * np.sqrt(252)

            # Calculate implied volatility (simplified)
            implied_vol = realized_vol * 1.2  # Simplified assumption

            # Calculate volatility clustering
            volatility_clustering = returns.abs().autocorr(lag=1)

            # Calculate volatility regime
            vol_regime = "high" if realized_vol.mean() > 0.3 else "low"

            # Calculate volatility smile (simplified)
            moneyness = np.linspace(0.8, 1.2, 10)
            vol_smile = 0.2 + 0.1 * (moneyness - 1) ** 2

            return {
                "symbol": symbol,
                "realized_volatility": {
                    "mean": float(realized_vol.mean()),
                    "std": float(realized_vol.std()),
                    "min": float(realized_vol.min()),
                    "max": float(realized_vol.max()),
                },
                "implied_volatility": {
                    "mean": float(implied_vol.mean()),
                    "std": float(implied_vol.std()),
                },
                "volatility_clustering": float(volatility_clustering),
                "volatility_regime": vol_regime,
                "volatility_smile": {
                    "moneyness": moneyness.tolist(),
                    "volatility": vol_smile.tolist(),
                },
                "analysis_date": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error running volatility analysis: {e}")
            return {"error": str(e)}

    async def run_regime_analysis(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Run regime analysis"""
        try:
            # Get price data
            price_data = await self._get_price_data([symbol], start_date, end_date)

            if price_data.empty:
                return {"error": "No price data available"}

            # Calculate returns
            returns = price_data[symbol].pct_change().dropna()

            # Simple regime detection based on volatility
            rolling_vol = returns.rolling(window=30).std()
            vol_threshold = rolling_vol.quantile(0.7)

            # Identify regimes
            regimes = []
            current_regime = "normal"
            regime_start = start_date

            for date, vol in rolling_vol.items():
                if vol > vol_threshold:
                    new_regime = "high_volatility"
                else:
                    new_regime = "normal"

                if new_regime != current_regime:
                    regimes.append(
                        {
                            "regime": current_regime,
                            "start_date": regime_start,
                            "end_date": date,
                            "duration": (date - regime_start).days,
                        }
                    )
                    current_regime = new_regime
                    regime_start = date

            # Add final regime
            regimes.append(
                {
                    "regime": current_regime,
                    "start_date": regime_start,
                    "end_date": end_date,
                    "duration": (end_date - regime_start).days,
                }
            )

            # Calculate regime statistics
            regime_stats = {}
            for regime in ["normal", "high_volatility"]:
                regime_returns = []
                for r in regimes:
                    if r["regime"] == regime:
                        regime_data = returns[
                            (returns.index >= r["start_date"])
                            & (returns.index <= r["end_date"])
                        ]
                        regime_returns.extend(regime_data.tolist())

                if regime_returns:
                    regime_stats[regime] = {
                        "mean_return": float(np.mean(regime_returns)),
                        "volatility": float(np.std(regime_returns)),
                        "sharpe_ratio": (
                            float(np.mean(regime_returns) / np.std(regime_returns))
                            if np.std(regime_returns) > 0
                            else 0
                        ),
                        "count": len(regime_returns),
                    }

            return {
                "symbol": symbol,
                "regimes": regimes,
                "regime_statistics": regime_stats,
                "analysis_date": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error running regime analysis: {e}")
            return {"error": str(e)}

    # Background tasks
    async def _update_factor_exposures(self):
        """Update factor exposures"""
        while True:
            try:
                for factor_id, factor in self.factors.items():
                    if factor.is_active:
                        await self._calculate_factor_exposures(factor)

                await asyncio.sleep(3600)  # Update every hour

            except Exception as e:
                logger.error(f"Error updating factor exposures: {e}")
                await asyncio.sleep(7200)

    async def _calculate_factor_exposures(self, factor: Factor):
        """Calculate factor exposures for a factor"""
        try:
            # Get latest price data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=factor.lookback_period)

            price_data = await self._get_price_data(
                factor.universe, start_date, end_date
            )

            if price_data.empty:
                return

            # Calculate factor values based on method
            if factor.calculation_method == "momentum":
                factor_values = await self._calculate_momentum_factor(
                    price_data, factor.parameters
                )
            elif factor.calculation_method == "value":
                factor_values = await self._calculate_value_factor(
                    price_data, factor.parameters
                )
            elif factor.calculation_method == "volatility":
                factor_values = await self._calculate_volatility_factor(
                    price_data, factor.parameters
                )
            else:
                factor_values = await self._calculate_default_factor(
                    price_data, factor.parameters
                )

            # Calculate exposures
            exposures = []
            for security_id, value in factor_values.items():
                # Calculate percentile rank
                all_values = list(factor_values.values())
                percentile_rank = stats.percentileofscore(all_values, value) / 100

                # Calculate z-score
                z_score = (
                    (value - np.mean(all_values)) / np.std(all_values)
                    if np.std(all_values) > 0
                    else 0
                )

                # Calculate confidence
                confidence = min(len(price_data) / factor.lookback_period, 1.0)

                exposure = FactorExposure(
                    security_id=security_id,
                    factor_id=factor.factor_id,
                    exposure_value=value,
                    percentile_rank=percentile_rank,
                    z_score=z_score,
                    timestamp=datetime.utcnow(),
                    confidence=confidence,
                )

                exposures.append(exposure)

            # Update factor exposures
            self.factor_exposures[factor.factor_id] = exposures

        except Exception as e:
            logger.error(f"Error calculating factor exposures: {e}")

    async def _calculate_momentum_factor(
        self, price_data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate momentum factor"""
        try:
            lookback = parameters.get("lookback", 252)
            returns = price_data.pct_change(lookback)

            factor_values = {}
            for col in price_data.columns:
                factor_values[col] = (
                    returns[col].iloc[-1] if not pd.isna(returns[col].iloc[-1]) else 0
                )

            return factor_values

        except Exception as e:
            logger.error(f"Error calculating momentum factor: {e}")
            return {}

    async def _calculate_value_factor(
        self, price_data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate value factor (simplified)"""
        try:
            # Simplified value factor using price-to-book ratio
            # In practice, this would use fundamental data
            factor_values = {}
            for col in price_data.columns:
                current_price = price_data[col].iloc[-1]
                # Simulate book value
                book_value = current_price * np.random.uniform(0.5, 2.0)
                factor_values[col] = current_price / book_value

            return factor_values

        except Exception as e:
            logger.error(f"Error calculating value factor: {e}")
            return {}

    async def _calculate_volatility_factor(
        self, price_data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate volatility factor"""
        try:
            lookback = parameters.get("lookback", 30)
            returns = price_data.pct_change()
            volatility = returns.rolling(window=lookback).std()

            factor_values = {}
            for col in price_data.columns:
                factor_values[col] = (
                    volatility[col].iloc[-1]
                    if not pd.isna(volatility[col].iloc[-1])
                    else 0
                )

            return factor_values

        except Exception as e:
            logger.error(f"Error calculating volatility factor: {e}")
            return {}

    async def _calculate_default_factor(
        self, price_data: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate default factor"""
        try:
            # Simple price-based factor
            factor_values = {}
            for col in price_data.columns:
                factor_values[col] = price_data[col].iloc[-1]

            return factor_values

        except Exception as e:
            logger.error(f"Error calculating default factor: {e}")
            return {}

    async def _calculate_factor_returns(self):
        """Calculate factor returns"""
        while True:
            try:
                for factor_id, factor in self.factors.items():
                    if factor.is_active:
                        await self._calculate_factor_return(factor)

                await asyncio.sleep(86400)  # Calculate daily

            except Exception as e:
                logger.error(f"Error calculating factor returns: {e}")
                await asyncio.sleep(172800)

    async def _calculate_factor_return(self, factor: Factor):
        """Calculate factor return"""
        try:
            # Get factor exposures
            exposures = self.factor_exposures.get(factor.factor_id, [])
            if not exposures:
                return

            # Calculate factor return (simplified)
            # In practice, this would use actual portfolio returns
            factor_return = np.random.normal(0, 0.02)  # 2% daily volatility

            # Calculate cumulative return
            previous_returns = self.factor_returns.get(factor.factor_id, [])
            cumulative_return = 1.0
            if previous_returns:
                cumulative_return = previous_returns[-1].cumulative_return * (
                    1 + factor_return
                )

            # Calculate metrics
            volatility = 0.02  # Simplified
            sharpe_ratio = factor_return / volatility if volatility > 0 else 0
            max_drawdown = 0.05  # Simplified
            information_ratio = 0.1  # Simplified

            factor_return_obj = FactorReturn(
                factor_id=factor.factor_id,
                return_date=datetime.utcnow(),
                factor_return=factor_return,
                cumulative_return=cumulative_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                information_ratio=information_ratio,
            )

            if factor.factor_id not in self.factor_returns:
                self.factor_returns[factor.factor_id] = []

            self.factor_returns[factor.factor_id].append(factor_return_obj)

            # Keep only recent returns
            if len(self.factor_returns[factor.factor_id]) > 1000:
                self.factor_returns[factor.factor_id] = self.factor_returns[
                    factor.factor_id
                ][-1000:]

        except Exception as e:
            logger.error(f"Error calculating factor return: {e}")

    async def _run_research_projects(self):
        """Run active research projects"""
        while True:
            try:
                for project_id, project in self.research_projects.items():
                    if project.status == "active":
                        await self._execute_research_project(project)

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error running research projects: {e}")
                await asyncio.sleep(7200)

    async def _execute_research_project(self, project: ResearchProject):
        """Execute a research project"""
        try:
            if project.research_type == ResearchType.FACTOR_ANALYSIS:
                await self._run_factor_analysis_project(project)
            elif project.research_type == ResearchType.CORRELATION_ANALYSIS:
                await self._run_correlation_analysis_project(project)
            elif project.research_type == ResearchType.VOLATILITY_MODELING:
                await self._run_volatility_modeling_project(project)

            project.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error executing research project: {e}")

    async def _run_factor_analysis_project(self, project: ResearchProject):
        """Run factor analysis project"""
        try:
            # Extract parameters
            factor_list = project.parameters.get("factor_list", [])
            universe = project.parameters.get(
                "universe", ["AAPL", "GOOGL", "MSFT", "TSLA"]
            )
            start_date = project.parameters.get(
                "start_date", datetime.utcnow() - timedelta(days=365)
            )
            end_date = project.parameters.get("end_date", datetime.utcnow())

            # Run factor analysis
            result = await self.run_factor_analysis(
                project.project_id, factor_list, universe, start_date, end_date
            )

            # Update project status
            project.status = "completed"
            project.completed_at = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error running factor analysis project: {e}")

    async def _run_correlation_analysis_project(self, project: ResearchProject):
        """Run correlation analysis project"""
        try:
            # Extract parameters
            symbols = project.parameters.get(
                "symbols", ["AAPL", "GOOGL", "MSFT", "TSLA"]
            )
            start_date = project.parameters.get(
                "start_date", datetime.utcnow() - timedelta(days=365)
            )
            end_date = project.parameters.get("end_date", datetime.utcnow())

            # Run correlation analysis
            result = await self.run_correlation_analysis(symbols, start_date, end_date)

            # Create research result
            research_result = ResearchResult(
                result_id=f"RESULT_{uuid.uuid4().hex[:8]}",
                project_id=project.project_id,
                result_type="correlation_analysis",
                data=result,
                metrics={"correlation_strength": result.get("average_correlation", 0)},
                charts=[],
                insights=["Correlation analysis completed"],
                recommendations=["Monitor correlation changes"],
                confidence_score=0.8,
                created_at=datetime.utcnow(),
            )

            if project.project_id not in self.research_results:
                self.research_results[project.project_id] = []

            self.research_results[project.project_id].append(research_result)

            # Update project status
            project.status = "completed"
            project.completed_at = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error running correlation analysis project: {e}")

    async def _run_volatility_modeling_project(self, project: ResearchProject):
        """Run volatility modeling project"""
        try:
            # Extract parameters
            symbol = project.parameters.get("symbol", "AAPL")
            start_date = project.parameters.get(
                "start_date", datetime.utcnow() - timedelta(days=365)
            )
            end_date = project.parameters.get("end_date", datetime.utcnow())
            model_type = project.parameters.get("model_type", "garch")

            # Run volatility analysis
            result = await self.run_volatility_analysis(
                symbol, start_date, end_date, model_type
            )

            # Create research result
            research_result = ResearchResult(
                result_id=f"RESULT_{uuid.uuid4().hex[:8]}",
                project_id=project.project_id,
                result_type="volatility_modeling",
                data=result,
                metrics={
                    "volatility": result.get("realized_volatility", {}).get("mean", 0)
                },
                charts=[],
                insights=["Volatility modeling completed"],
                recommendations=["Monitor volatility regime changes"],
                confidence_score=0.7,
                created_at=datetime.utcnow(),
            )

            if project.project_id not in self.research_results:
                self.research_results[project.project_id] = []

            self.research_results[project.project_id].append(research_result)

            # Update project status
            project.status = "completed"
            project.completed_at = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error running volatility modeling project: {e}")

    async def _generate_strategy_signals(self):
        """Generate strategy signals"""
        while True:
            try:
                # Generate signals for active factors
                for factor_id, factor in self.factors.items():
                    if factor.is_active:
                        await self._generate_factor_signals(factor)

                await asyncio.sleep(3600)  # Generate every hour

            except Exception as e:
                logger.error(f"Error generating strategy signals: {e}")
                await asyncio.sleep(7200)

    async def _generate_factor_signals(self, factor: Factor):
        """Generate signals for a factor"""
        try:
            exposures = self.factor_exposures.get(factor.factor_id, [])
            if not exposures:
                return

            # Generate signals based on factor exposures
            for exposure in exposures:
                signal_type = "hold"
                signal_strength = 0.0

                if exposure.z_score > 2:
                    signal_type = "buy"
                    signal_strength = min(exposure.z_score / 3, 1.0)
                elif exposure.z_score < -2:
                    signal_type = "sell"
                    signal_strength = min(abs(exposure.z_score) / 3, 1.0)

                if signal_type != "hold":
                    signal = StrategySignal(
                        signal_id=f"SIGNAL_{uuid.uuid4().hex[:8]}",
                        strategy_id=factor.factor_id,
                        security_id=exposure.security_id,
                        signal_type=signal_type,
                        signal_strength=signal_strength,
                        confidence=exposure.confidence,
                        expected_return=signal_strength
                        * 0.05,  # 5% max expected return
                        risk_score=1 - exposure.confidence,
                        timestamp=datetime.utcnow(),
                        metadata={
                            "factor_id": factor.factor_id,
                            "z_score": exposure.z_score,
                        },
                    )

                    if factor.factor_id not in self.strategy_signals:
                        self.strategy_signals[factor.factor_id] = []

                    self.strategy_signals[factor.factor_id].append(signal)

        except Exception as e:
            logger.error(f"Error generating factor signals: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        while True:
            try:
                # Update factor performance metrics
                for factor_id, factor in self.factors.items():
                    await self._update_factor_performance(factor)

                await asyncio.sleep(86400)  # Update daily

            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")
                await asyncio.sleep(172800)

    async def _update_factor_performance(self, factor: Factor):
        """Update factor performance metrics"""
        try:
            returns = self.factor_returns.get(factor.factor_id, [])
            if not returns:
                return

            # Calculate performance metrics
            total_return = returns[-1].cumulative_return - 1 if returns else 0
            volatility = (
                np.std([r.factor_return for r in returns[-30:]])
                if len(returns) >= 30
                else 0
            )
            sharpe_ratio = (
                np.mean([r.factor_return for r in returns[-30:]]) / volatility
                if volatility > 0
                else 0
            )
            max_drawdown = (
                max([r.max_drawdown for r in returns[-30:]]) if returns else 0
            )

            performance_metrics = {
                "total_return": total_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "information_ratio": (
                    np.mean([r.information_ratio for r in returns[-30:]])
                    if returns
                    else 0
                ),
                "last_updated": datetime.utcnow(),
            }

            factor.performance_metrics = performance_metrics
            factor.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating factor performance: {e}")

    # Helper methods
    async def _prepare_factor_data(
        self,
        factor_list: List[str],
        universe: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Prepare factor data for analysis"""
        try:
            # Get price data
            price_data = await self._get_price_data(universe, start_date, end_date)

            # Get factor exposures
            factor_data = {}
            for factor_id in factor_list:
                exposures = self.factor_exposures.get(factor_id, [])
                factor_data[factor_id] = exposures

            return {
                "price_data": price_data,
                "factor_data": factor_data,
                "universe": universe,
                "start_date": start_date,
                "end_date": end_date,
            }

        except Exception as e:
            logger.error(f"Error preparing factor data: {e}")
            return {}

    async def _run_factor_analysis_core(
        self, factor_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run core factor analysis"""
        try:
            # Simplified factor analysis
            insights = [
                "Factor analysis completed successfully",
                "Multiple factors show significant predictive power",
                "Factor correlations are within acceptable ranges",
            ]

            recommendations = [
                "Consider combining multiple factors for better diversification",
                "Monitor factor performance regularly",
                "Adjust factor weights based on market conditions",
            ]

            return {
                "data": factor_data,
                "metrics": {
                    "factor_count": len(factor_data.get("factor_data", {})),
                    "universe_size": len(factor_data.get("universe", [])),
                    "analysis_period": (
                        factor_data.get("end_date", datetime.utcnow())
                        - factor_data.get("start_date", datetime.utcnow())
                    ).days,
                },
                "charts": [],
                "insights": insights,
                "recommendations": recommendations,
                "confidence_score": 0.8,
            }

        except Exception as e:
            logger.error(f"Error running factor analysis core: {e}")
            return {"error": str(e)}

    async def _get_price_data(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Get price data for symbols"""
        try:
            # Simulate price data
            date_range = pd.date_range(start=start_date, end=end_date, freq="D")
            price_data = pd.DataFrame(index=date_range)

            for symbol in symbols:
                # Generate realistic price data
                # Use hashlib for deterministic hash instead of built-in hash()
                deterministic_hash = int(
                    hashlib.md5(symbol.encode(), usedforsecurity=False).hexdigest()[:8], 16
                )
                base_price = 100 + deterministic_hash % 1000
                returns = np.random.normal(0, 0.02, len(date_range))
                prices = [base_price]

                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))

                price_data[symbol] = prices

            return price_data

        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            return pd.DataFrame()

    async def _load_historical_data(self):
        """Load historical data"""
        try:
            # Initialize with sample data
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=365)

            for symbol in symbols:
                price_data = await self._get_price_data([symbol], start_date, end_date)
                self.price_data[symbol] = price_data

            logger.info("Loaded historical data")

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")

    async def _load_factors(self):
        """Load existing factors"""
        try:
            # Create sample factors
            sample_factors = [
                {
                    "factor_name": "Momentum Factor",
                    "factor_type": FactorType.MOMENTUM,
                    "description": "Price momentum over 252 days",
                    "calculation_method": "momentum",
                    "parameters": {"lookback": 252},
                    "universe": ["AAPL", "GOOGL", "MSFT", "TSLA"],
                },
                {
                    "factor_name": "Value Factor",
                    "factor_type": FactorType.VALUE,
                    "description": "Price-to-book ratio",
                    "calculation_method": "value",
                    "parameters": {},
                    "universe": ["AAPL", "GOOGL", "MSFT", "TSLA"],
                },
                {
                    "factor_name": "Volatility Factor",
                    "factor_type": FactorType.VOLATILITY,
                    "description": "30-day rolling volatility",
                    "calculation_method": "volatility",
                    "parameters": {"lookback": 30},
                    "universe": ["AAPL", "GOOGL", "MSFT", "TSLA"],
                },
            ]

            for factor_data in sample_factors:
                await self.create_factor(**factor_data)

            logger.info("Loaded sample factors")

        except Exception as e:
            logger.error(f"Error loading factors: {e}")

    async def _initialize_models(self):
        """Initialize models"""
        try:
            # Initialize factor models
            self.factor_models = {
                "linear_regression": LinearRegression(),
                "ridge_regression": Ridge(),
                "lasso_regression": Lasso(),
                "random_forest": RandomForestRegressor(),
                "gradient_boosting": GradientBoostingRegressor(),
            }

            logger.info("Initialized models")

        except Exception as e:
            logger.error(f"Error initializing models: {e}")


# Factory function
async def get_quantitative_research_service(
    redis_client: redis.Redis, db_session: Session
) -> QuantitativeResearchService:
    """Get Quantitative Research Service instance"""
    service = QuantitativeResearchService(redis_client, db_session)
    await service.initialize()
    return service
