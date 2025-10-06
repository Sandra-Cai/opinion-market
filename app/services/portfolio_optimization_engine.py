"""
Portfolio Optimization Engine
AI-powered portfolio management with advanced optimization algorithms
"""

import asyncio
import logging
import time
import json
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_VAR = "minimize_var"
    MAXIMIZE_UTILITY = "maximize_utility"
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"

class RebalancingFrequency(Enum):
    """Rebalancing frequencies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    DYNAMIC = "dynamic"

class RiskModel(Enum):
    """Risk models"""
    HISTORICAL = "historical"
    FACTOR = "factor"
    MONTE_CARLO = "monte_carlo"
    GARCH = "garch"
    COPULA = "copula"

@dataclass
class Asset:
    """Asset information"""
    symbol: str
    name: str
    asset_type: str  # stock, bond, crypto, commodity, etc.
    expected_return: float
    volatility: float
    beta: float
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Portfolio:
    """Portfolio definition"""
    portfolio_id: str
    name: str
    description: str
    assets: List[Asset]
    weights: List[float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    beta: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional Value at Risk 95%
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_sector_weight: float = 0.4
    max_country_weight: float = 0.5
    max_single_asset_weight: float = 0.1
    min_assets: int = 5
    max_assets: int = 50
    target_return: Optional[float] = None
    max_volatility: Optional[float] = None
    max_turnover: float = 0.2  # 20% max turnover per rebalancing

@dataclass
class RebalancingSignal:
    """Rebalancing signal"""
    signal_id: str
    portfolio_id: str
    current_weights: List[float]
    target_weights: List[float]
    rebalancing_amounts: List[float]
    expected_improvement: float
    transaction_costs: float
    urgency: float  # 0.0 to 1.0
    reason: str
    created_at: datetime = field(default_factory=datetime.now)

class PortfolioOptimizationEngine:
    """Advanced Portfolio Optimization Engine"""
    
    def __init__(self):
        self.engine_id = f"portfolio_opt_{secrets.token_hex(8)}"
        self.is_running = False
        
        # Portfolio data
        self.portfolios: Dict[str, Portfolio] = {}
        self.assets: Dict[str, Asset] = {}
        self.rebalancing_signals: Dict[str, RebalancingSignal] = {}
        
        # Market data
        self.price_data: Dict[str, List[float]] = {}
        self.returns_data: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        
        # Optimization parameters
        self.risk_free_rate: float = 0.02  # 2% risk-free rate
        self.transaction_cost: float = 0.001  # 0.1% transaction cost
        self.optimization_objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE
        self.rebalancing_frequency: RebalancingFrequency = RebalancingFrequency.MONTHLY
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        
        # Processing tasks
        self.optimization_task: Optional[asyncio.Task] = None
        self.rebalancing_task: Optional[asyncio.Task] = None
        self.performance_tracking_task: Optional[asyncio.Task] = None
        
        logger.info(f"Portfolio Optimization Engine {self.engine_id} initialized")

    async def start_portfolio_optimization_engine(self):
        """Start the portfolio optimization engine"""
        if self.is_running:
            return
        
        logger.info("Starting Portfolio Optimization Engine...")
        
        # Initialize assets and market data
        await self._initialize_assets()
        await self._initialize_market_data()
        
        # Start processing tasks
        self.is_running = True
        
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        self.rebalancing_task = asyncio.create_task(self._rebalancing_loop())
        self.performance_tracking_task = asyncio.create_task(self._performance_tracking_loop())
        
        logger.info("Portfolio Optimization Engine started")

    async def stop_portfolio_optimization_engine(self):
        """Stop the portfolio optimization engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping Portfolio Optimization Engine...")
        
        self.is_running = False
        
        # Cancel all processing tasks
        tasks = [
            self.optimization_task,
            self.rebalancing_task,
            self.performance_tracking_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Portfolio Optimization Engine stopped")

    async def _initialize_assets(self):
        """Initialize asset universe"""
        assets_data = [
            {"symbol": "AAPL", "name": "Apple Inc.", "asset_type": "stock", "sector": "technology", "country": "US"},
            {"symbol": "MSFT", "name": "Microsoft Corp.", "asset_type": "stock", "sector": "technology", "country": "US"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "asset_type": "stock", "sector": "technology", "country": "US"},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "asset_type": "stock", "sector": "consumer_discretionary", "country": "US"},
            {"symbol": "TSLA", "name": "Tesla Inc.", "asset_type": "stock", "sector": "automotive", "country": "US"},
            {"symbol": "BTC", "name": "Bitcoin", "asset_type": "crypto", "sector": "cryptocurrency", "country": "global"},
            {"symbol": "ETH", "name": "Ethereum", "asset_type": "crypto", "sector": "cryptocurrency", "country": "global"},
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "asset_type": "etf", "sector": "diversified", "country": "US"},
            {"symbol": "GLD", "name": "SPDR Gold Trust", "asset_type": "commodity", "sector": "precious_metals", "country": "global"},
            {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "asset_type": "bond", "sector": "fixed_income", "country": "US"},
        ]
        
        for asset_data in assets_data:
            # Generate mock financial metrics
            expected_return = np.random.uniform(0.05, 0.15)  # 5-15% expected return
            volatility = np.random.uniform(0.15, 0.35)  # 15-35% volatility
            beta = np.random.uniform(0.5, 1.5)  # 0.5-1.5 beta
            
            asset = Asset(
                symbol=asset_data["symbol"],
                name=asset_data["name"],
                asset_type=asset_data["asset_type"],
                expected_return=expected_return,
                volatility=volatility,
                beta=beta,
                sector=asset_data["sector"],
                country=asset_data["country"],
                metadata=asset_data
            )
            
            self.assets[asset.symbol] = asset
        
        logger.info(f"Initialized {len(self.assets)} assets")

    async def _initialize_market_data(self):
        """Initialize market data"""
        # Generate mock historical price data
        np.random.seed(42)  # For reproducible results
        n_days = 252  # 1 year of trading days
        
        for symbol, asset in self.assets.items():
            # Generate price series with drift and volatility
            returns = np.random.normal(asset.expected_return/252, asset.volatility/np.sqrt(252), n_days)
            prices = [100]  # Starting price
            
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            self.price_data[symbol] = prices
            self.returns_data[symbol] = returns
        
        # Calculate correlation matrix
        returns_df = pd.DataFrame(self.returns_data)
        self.correlation_matrix = returns_df.corr().values
        
        logger.info("Initialized market data and correlation matrix")

    async def _optimization_loop(self):
        """Main optimization loop"""
        while self.is_running:
            try:
                # Optimize all portfolios
                for portfolio_id, portfolio in self.portfolios.items():
                    await self._optimize_portfolio(portfolio_id)
                
                await asyncio.sleep(3600)  # Optimize every hour
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)

    async def _rebalancing_loop(self):
        """Rebalancing loop"""
        while self.is_running:
            try:
                # Check for rebalancing opportunities
                for portfolio_id, portfolio in self.portfolios.items():
                    await self._check_rebalancing_opportunity(portfolio_id)
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in rebalancing loop: {e}")
                await asyncio.sleep(60)

    async def _performance_tracking_loop(self):
        """Performance tracking loop"""
        while self.is_running:
            try:
                # Update performance metrics
                for portfolio_id, portfolio in self.portfolios.items():
                    await self._update_portfolio_performance(portfolio_id)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(60)

    async def _optimize_portfolio(self, portfolio_id: str):
        """Optimize a specific portfolio"""
        try:
            if portfolio_id not in self.portfolios:
                return
            
            portfolio = self.portfolios[portfolio_id]
            asset_symbols = [asset.symbol for asset in portfolio.assets]
            
            # Get current weights
            current_weights = np.array(portfolio.weights)
            
            # Define optimization constraints
            constraints = self._get_optimization_constraints(portfolio)
            
            # Run optimization
            if self.optimization_objective == OptimizationObjective.MAXIMIZE_SHARPE:
                optimal_weights = await self._maximize_sharpe_ratio(asset_symbols, constraints)
            elif self.optimization_objective == OptimizationObjective.MINIMIZE_RISK:
                optimal_weights = await self._minimize_risk(asset_symbols, constraints)
            elif self.optimization_objective == OptimizationObjective.MAXIMIZE_RETURN:
                optimal_weights = await self._maximize_return(asset_symbols, constraints)
            elif self.optimization_objective == OptimizationObjective.RISK_PARITY:
                optimal_weights = await self._risk_parity_optimization(asset_symbols, constraints)
            else:
                optimal_weights = current_weights
            
            # Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(asset_symbols, optimal_weights)
            
            # Update portfolio
            portfolio.weights = optimal_weights.tolist()
            portfolio.expected_return = portfolio_metrics["expected_return"]
            portfolio.volatility = portfolio_metrics["volatility"]
            portfolio.sharpe_ratio = portfolio_metrics["sharpe_ratio"]
            portfolio.beta = portfolio_metrics["beta"]
            portfolio.max_drawdown = portfolio_metrics["max_drawdown"]
            portfolio.var_95 = portfolio_metrics["var_95"]
            portfolio.cvar_95 = portfolio_metrics["cvar_95"]
            portfolio.updated_at = datetime.now()
            
            # Log optimization
            self.optimization_history.append({
                "portfolio_id": portfolio_id,
                "timestamp": datetime.now().isoformat(),
                "objective": self.optimization_objective.value,
                "old_weights": current_weights.tolist(),
                "new_weights": optimal_weights.tolist(),
                "expected_return": portfolio_metrics["expected_return"],
                "volatility": portfolio_metrics["volatility"],
                "sharpe_ratio": portfolio_metrics["sharpe_ratio"]
            })
            
            logger.info(f"Optimized portfolio {portfolio_id}: "
                       f"Return={portfolio_metrics['expected_return']:.2%}, "
                       f"Risk={portfolio_metrics['volatility']:.2%}, "
                       f"Sharpe={portfolio_metrics['sharpe_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio {portfolio_id}: {e}")

    async def _maximize_sharpe_ratio(self, asset_symbols: List[str], constraints: OptimizationConstraints) -> np.ndarray:
        """Maximize Sharpe ratio optimization"""
        try:
            n_assets = len(asset_symbols)
            
            # Expected returns
            expected_returns = np.array([self.assets[symbol].expected_return for symbol in asset_symbols])
            
            # Covariance matrix
            cov_matrix = self._calculate_covariance_matrix(asset_symbols)
            
            # Objective function (negative Sharpe ratio to minimize)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                return -sharpe_ratio
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
            
            if result.success:
                return result.x
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
                
        except Exception as e:
            logger.error(f"Error in Sharpe ratio optimization: {e}")
            return np.ones(len(asset_symbols)) / len(asset_symbols)

    async def _minimize_risk(self, asset_symbols: List[str], constraints: OptimizationConstraints) -> np.ndarray:
        """Minimize portfolio risk optimization"""
        try:
            n_assets = len(asset_symbols)
            
            # Covariance matrix
            cov_matrix = self._calculate_covariance_matrix(asset_symbols)
            
            # Objective function (portfolio variance)
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
            
            if result.success:
                return result.x
            else:
                return np.ones(n_assets) / n_assets
                
        except Exception as e:
            logger.error(f"Error in risk minimization: {e}")
            return np.ones(len(asset_symbols)) / len(asset_symbols)

    async def _maximize_return(self, asset_symbols: List[str], constraints: OptimizationConstraints) -> np.ndarray:
        """Maximize portfolio return optimization"""
        try:
            n_assets = len(asset_symbols)
            
            # Expected returns
            expected_returns = np.array([self.assets[symbol].expected_return for symbol in asset_symbols])
            
            # Objective function (negative return to minimize)
            def objective(weights):
                return -np.dot(weights, expected_returns)
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
            
            if result.success:
                return result.x
            else:
                return np.ones(n_assets) / n_assets
                
        except Exception as e:
            logger.error(f"Error in return maximization: {e}")
            return np.ones(len(asset_symbols)) / len(asset_symbols)

    async def _risk_parity_optimization(self, asset_symbols: List[str], constraints: OptimizationConstraints) -> np.ndarray:
        """Risk parity optimization"""
        try:
            n_assets = len(asset_symbols)
            
            # Covariance matrix
            cov_matrix = self._calculate_covariance_matrix(asset_symbols)
            
            # Risk parity objective
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                risk_contributions = (weights * np.dot(cov_matrix, weights)) / portfolio_vol
                target_risk = 1.0 / n_assets
                return np.sum((risk_contributions - target_risk) ** 2)
            
            # Constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
            
            if result.success:
                return result.x
            else:
                return np.ones(n_assets) / n_assets
                
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            return np.ones(len(asset_symbols)) / len(asset_symbols)

    def _calculate_covariance_matrix(self, asset_symbols: List[str]) -> np.ndarray:
        """Calculate covariance matrix for assets"""
        try:
            # Get returns data
            returns_data = []
            for symbol in asset_symbols:
                if symbol in self.returns_data:
                    returns_data.append(self.returns_data[symbol])
                else:
                    # Use asset volatility if no returns data
                    returns_data.append(np.random.normal(0, self.assets[symbol].volatility, 252))
            
            returns_df = pd.DataFrame(returns_data).T
            cov_matrix = returns_df.cov().values
            
            return cov_matrix
            
        except Exception as e:
            logger.error(f"Error calculating covariance matrix: {e}")
            # Return identity matrix as fallback
            n = len(asset_symbols)
            return np.eye(n)

    async def _calculate_portfolio_metrics(self, asset_symbols: List[str], weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            # Expected return
            expected_returns = np.array([self.assets[symbol].expected_return for symbol in asset_symbols])
            portfolio_return = np.dot(weights, expected_returns)
            
            # Volatility
            cov_matrix = self._calculate_covariance_matrix(asset_symbols)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Beta
            market_returns = np.array([self.assets[symbol].beta for symbol in asset_symbols])
            portfolio_beta = np.dot(weights, market_returns)
            
            # VaR and CVaR (simplified)
            var_95 = portfolio_volatility * 1.645  # 95% VaR
            cvar_95 = portfolio_volatility * 2.06  # 95% CVaR
            
            # Max drawdown (simplified)
            max_drawdown = portfolio_volatility * 2.0  # Rough estimate
            
            return {
                "expected_return": portfolio_return,
                "volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "beta": portfolio_beta,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "max_drawdown": max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "beta": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "max_drawdown": 0.0
            }

    def _get_optimization_constraints(self, portfolio: Portfolio) -> OptimizationConstraints:
        """Get optimization constraints for portfolio"""
        return OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.2,  # Max 20% per asset
            max_sector_weight=0.4,
            max_country_weight=0.5,
            max_single_asset_weight=0.1,
            min_assets=len(portfolio.assets),
            max_assets=len(portfolio.assets)
        )

    async def _check_rebalancing_opportunity(self, portfolio_id: str):
        """Check if portfolio needs rebalancing"""
        try:
            if portfolio_id not in self.portfolios:
                return
            
            portfolio = self.portfolios[portfolio_id]
            
            # Calculate current vs target weights deviation
            current_weights = np.array(portfolio.weights)
            target_weights = np.array(portfolio.weights)  # In real implementation, this would be the optimized weights
            
            # Calculate tracking error
            tracking_error = np.sqrt(np.sum((current_weights - target_weights) ** 2))
            
            # Check if rebalancing is needed
            rebalancing_threshold = 0.05  # 5% threshold
            
            if tracking_error > rebalancing_threshold:
                # Generate rebalancing signal
                rebalancing_amounts = target_weights - current_weights
                transaction_costs = np.sum(np.abs(rebalancing_amounts)) * self.transaction_cost
                
                signal = RebalancingSignal(
                    signal_id=f"rebal_{secrets.token_hex(8)}",
                    portfolio_id=portfolio_id,
                    current_weights=current_weights.tolist(),
                    target_weights=target_weights.tolist(),
                    rebalancing_amounts=rebalancing_amounts.tolist(),
                    expected_improvement=tracking_error * 0.1,  # Rough estimate
                    transaction_costs=transaction_costs,
                    urgency=min(tracking_error / rebalancing_threshold, 1.0),
                    reason=f"Tracking error {tracking_error:.2%} exceeds threshold {rebalancing_threshold:.2%}"
                )
                
                self.rebalancing_signals[signal.signal_id] = signal
                
                logger.info(f"Rebalancing signal generated for portfolio {portfolio_id}: "
                           f"tracking error {tracking_error:.2%}")
            
        except Exception as e:
            logger.error(f"Error checking rebalancing opportunity for {portfolio_id}: {e}")

    async def _update_portfolio_performance(self, portfolio_id: str):
        """Update portfolio performance metrics"""
        try:
            if portfolio_id not in self.portfolios:
                return
            
            portfolio = self.portfolios[portfolio_id]
            
            # Calculate current performance
            current_return = portfolio.expected_return
            current_volatility = portfolio.volatility
            current_sharpe = portfolio.sharpe_ratio
            
            # Store performance metrics
            if portfolio_id not in self.performance_metrics:
                self.performance_metrics[portfolio_id] = {
                    "returns": [],
                    "volatilities": [],
                    "sharpe_ratios": []
                }
            
            self.performance_metrics[portfolio_id]["returns"].append(current_return)
            self.performance_metrics[portfolio_id]["volatilities"].append(current_volatility)
            self.performance_metrics[portfolio_id]["sharpe_ratios"].append(current_sharpe)
            
            # Keep only last 100 data points
            for metric in self.performance_metrics[portfolio_id].values():
                if len(metric) > 100:
                    metric.pop(0)
            
        except Exception as e:
            logger.error(f"Error updating portfolio performance for {portfolio_id}: {e}")

    # Public API methods
    async def create_portfolio(self, name: str, description: str, asset_symbols: List[str], 
                             initial_weights: Optional[List[float]] = None) -> str:
        """Create a new portfolio"""
        try:
            portfolio_id = f"portfolio_{secrets.token_hex(8)}"
            
            # Get assets
            assets = [self.assets[symbol] for symbol in asset_symbols if symbol in self.assets]
            
            if len(assets) == 0:
                raise ValueError("No valid assets found")
            
            # Set initial weights
            if initial_weights is None:
                weights = np.ones(len(assets)) / len(assets)
            else:
                weights = np.array(initial_weights)
                if len(weights) != len(assets):
                    raise ValueError("Number of weights must match number of assets")
                if not np.isclose(np.sum(weights), 1.0):
                    raise ValueError("Weights must sum to 1.0")
            
            # Calculate initial metrics
            metrics = await self._calculate_portfolio_metrics(asset_symbols, weights)
            
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                name=name,
                description=description,
                assets=assets,
                weights=weights.tolist(),
                expected_return=metrics["expected_return"],
                volatility=metrics["volatility"],
                sharpe_ratio=metrics["sharpe_ratio"],
                beta=metrics["beta"],
                max_drawdown=metrics["max_drawdown"],
                var_95=metrics["var_95"],
                cvar_95=metrics["cvar_95"]
            )
            
            self.portfolios[portfolio_id] = portfolio
            
            logger.info(f"Created portfolio {portfolio_id}: {name}")
            return portfolio_id
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {e}")
            raise

    async def get_portfolio(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio information"""
        try:
            if portfolio_id not in self.portfolios:
                return None
            
            portfolio = self.portfolios[portfolio_id]
            
            return {
                "portfolio_id": portfolio.portfolio_id,
                "name": portfolio.name,
                "description": portfolio.description,
                "assets": [
                    {
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "asset_type": asset.asset_type,
                        "expected_return": asset.expected_return,
                        "volatility": asset.volatility,
                        "beta": asset.beta,
                        "sector": asset.sector,
                        "country": asset.country
                    }
                    for asset in portfolio.assets
                ],
                "weights": portfolio.weights,
                "expected_return": portfolio.expected_return,
                "volatility": portfolio.volatility,
                "sharpe_ratio": portfolio.sharpe_ratio,
                "beta": portfolio.beta,
                "max_drawdown": portfolio.max_drawdown,
                "var_95": portfolio.var_95,
                "cvar_95": portfolio.cvar_95,
                "created_at": portfolio.created_at.isoformat(),
                "updated_at": portfolio.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio {portfolio_id}: {e}")
            return None

    async def get_all_portfolios(self) -> List[Dict[str, Any]]:
        """Get all portfolios"""
        try:
            portfolios = []
            for portfolio_id in self.portfolios:
                portfolio_data = await self.get_portfolio(portfolio_id)
                if portfolio_data:
                    portfolios.append(portfolio_data)
            
            return portfolios
            
        except Exception as e:
            logger.error(f"Error getting all portfolios: {e}")
            return []

    async def get_rebalancing_signals(self) -> List[Dict[str, Any]]:
        """Get rebalancing signals"""
        try:
            signals = []
            for signal in self.rebalancing_signals.values():
                signals.append({
                    "signal_id": signal.signal_id,
                    "portfolio_id": signal.portfolio_id,
                    "current_weights": signal.current_weights,
                    "target_weights": signal.target_weights,
                    "rebalancing_amounts": signal.rebalancing_amounts,
                    "expected_improvement": signal.expected_improvement,
                    "transaction_costs": signal.transaction_costs,
                    "urgency": signal.urgency,
                    "reason": signal.reason,
                    "created_at": signal.created_at.isoformat()
                })
            
            return sorted(signals, key=lambda x: x["created_at"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting rebalancing signals: {e}")
            return []

    async def get_available_assets(self) -> List[Dict[str, Any]]:
        """Get available assets"""
        try:
            assets = []
            for asset in self.assets.values():
                assets.append({
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "asset_type": asset.asset_type,
                    "expected_return": asset.expected_return,
                    "volatility": asset.volatility,
                    "beta": asset.beta,
                    "sector": asset.sector,
                    "country": asset.country,
                    "currency": asset.currency
                })
            
            return assets
            
        except Exception as e:
            logger.error(f"Error getting available assets: {e}")
            return []

    async def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        try:
            return self.optimization_history[-50:]  # Last 50 optimizations
            
        except Exception as e:
            logger.error(f"Error getting optimization history: {e}")
            return []

    async def get_performance_metrics(self, portfolio_id: str) -> Dict[str, Any]:
        """Get performance metrics for a portfolio"""
        try:
            if portfolio_id not in self.performance_metrics:
                return {}
            
            metrics = self.performance_metrics[portfolio_id]
            
            return {
                "portfolio_id": portfolio_id,
                "returns": metrics["returns"],
                "volatilities": metrics["volatilities"],
                "sharpe_ratios": metrics["sharpe_ratios"],
                "average_return": np.mean(metrics["returns"]) if metrics["returns"] else 0,
                "average_volatility": np.mean(metrics["volatilities"]) if metrics["volatilities"] else 0,
                "average_sharpe": np.mean(metrics["sharpe_ratios"]) if metrics["sharpe_ratios"] else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics for {portfolio_id}: {e}")
            return {}

# Global instance
portfolio_optimization_engine = PortfolioOptimizationEngine()
