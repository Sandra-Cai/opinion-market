"""
Portfolio Management Service
Provides portfolio optimization, rebalancing, and risk management
"""

import asyncio
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

logger = logging.getLogger(__name__)


class PortfolioType(Enum):
    """Portfolio types"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    INCOME = "income"
    GROWTH = "growth"
    BALANCED = "balanced"
    SECTOR = "sector"
    THEMATIC = "thematic"


class RebalancingFrequency(Enum):
    """Rebalancing frequencies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"
    ON_DEMAND = "on_demand"


@dataclass
class Portfolio:
    """Portfolio definition"""
    portfolio_id: str
    user_id: int
    portfolio_name: str
    portfolio_type: PortfolioType
    description: str
    target_allocation: Dict[str, float]
    current_allocation: Dict[str, float]
    total_value: float
    cash_balance: float
    risk_profile: str
    rebalancing_frequency: RebalancingFrequency
    last_rebalanced: datetime
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class PortfolioPosition:
    """Portfolio position"""
    position_id: str
    portfolio_id: str
    asset_id: str
    asset_type: str
    quantity: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl: float
    weight: float
    target_weight: float
    created_at: datetime
    last_updated: datetime


@dataclass
class PortfolioPerformance:
    """Portfolio performance metrics"""
    portfolio_id: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    calmar_ratio: float
    current_value: float
    peak_value: float
    last_updated: datetime


@dataclass
class RebalancingEvent:
    """Portfolio rebalancing event"""
    event_id: str
    portfolio_id: str
    user_id: int
    event_type: str  # 'scheduled', 'threshold', 'manual'
    trigger_reason: str
    old_allocation: Dict[str, float]
    new_allocation: Dict[str, float]
    trades_required: List[Dict[str, Any]]
    estimated_cost: float
    status: str  # 'pending', 'executing', 'completed', 'failed'
    execution_time: Optional[datetime]
    created_at: datetime
    last_updated: datetime


@dataclass
class AssetAllocation:
    """Asset allocation strategy"""
    allocation_id: str
    portfolio_id: str
    asset_class: str
    target_percentage: float
    current_percentage: float
    min_percentage: float
    max_percentage: float
    rebalancing_threshold: float
    last_rebalanced: datetime
    created_at: datetime
    last_updated: datetime


class PortfolioManagementService:
    """Comprehensive portfolio management service"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.portfolios: Dict[str, Portfolio] = {}
        self.positions: Dict[str, List[PortfolioPosition]] = defaultdict(list)
        self.performance: Dict[str, PortfolioPerformance] = {}
        self.rebalancing_events: Dict[str, RebalancingEvent] = {}
        self.asset_allocations: Dict[str, List[AssetAllocation]] = defaultdict(list)
        
        # Portfolio data
        self.portfolio_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.position_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Risk management
        self.risk_metrics: Dict[str, Dict[str, float]] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.var_calculations: Dict[str, Dict[str, float]] = {}
        
        # Optimization
        self.optimization_results: Dict[str, Dict[str, Any]] = {}
        self.efficient_frontier: Dict[str, List[Dict[str, float]]] = {}
        
    async def initialize(self):
        """Initialize the portfolio management service"""
        logger.info("Initializing Portfolio Management Service")
        
        # Load existing data
        await self._load_portfolios()
        await self._load_positions()
        await self._load_performance()
        
        # Start background tasks
        asyncio.create_task(self._update_portfolio_values())
        asyncio.create_task(self._check_rebalancing_needs())
        asyncio.create_task(self._update_performance_metrics())
        
        logger.info("Portfolio Management Service initialized successfully")
    
    async def create_portfolio(self, user_id: int, portfolio_name: str, portfolio_type: PortfolioType,
                              description: str, target_allocation: Dict[str, float], initial_cash: float = 100000.0,
                              risk_profile: str = "moderate", rebalancing_frequency: RebalancingFrequency = RebalancingFrequency.MONTHLY) -> Portfolio:
        """Create a new portfolio"""
        try:
            portfolio_id = f"portfolio_{portfolio_type.value}_{uuid.uuid4().hex[:8]}"
            
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                user_id=user_id,
                portfolio_name=portfolio_name,
                portfolio_type=portfolio_type,
                description=description,
                target_allocation=target_allocation,
                current_allocation=target_allocation.copy(),
                total_value=initial_cash,
                cash_balance=initial_cash,
                risk_profile=risk_profile,
                rebalancing_frequency=rebalancing_frequency,
                last_rebalanced=datetime.utcnow(),
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.portfolios[portfolio_id] = portfolio
            await self._cache_portfolio(portfolio)
            
            # Create asset allocations
            await self._create_asset_allocations(portfolio_id, target_allocation)
            
            logger.info(f"Created portfolio {portfolio_name}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {e}")
            raise
    
    async def add_position(self, portfolio_id: str, asset_id: str, asset_type: str,
                          quantity: float, price: float) -> PortfolioPosition:
        """Add a position to a portfolio"""
        try:
            if portfolio_id not in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            portfolio = self.portfolios[portfolio_id]
            
            # Calculate position values
            market_value = quantity * price
            cost_basis = market_value  # For new positions
            
            # Create position
            position_id = f"position_{portfolio_id}_{asset_id}_{uuid.uuid4().hex[:8]}"
            position = PortfolioPosition(
                position_id=position_id,
                portfolio_id=portfolio_id,
                asset_id=asset_id,
                asset_type=asset_type,
                quantity=quantity,
                current_price=price,
                market_value=market_value,
                cost_basis=cost_basis,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                weight=market_value / portfolio.total_value if portfolio.total_value > 0 else 0.0,
                target_weight=portfolio.target_allocation.get(asset_id, 0.0),
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            # Add to positions
            if portfolio_id not in self.positions:
                self.positions[portfolio_id] = []
            self.positions[portfolio_id].append(position)
            
            # Update portfolio
            portfolio.total_value += market_value
            portfolio.cash_balance -= market_value
            portfolio.current_allocation[asset_id] = position.weight
            portfolio.last_updated = datetime.utcnow()
            
            await self._cache_portfolio(portfolio)
            await self._cache_position(position)
            
            logger.info(f"Added position {asset_id} to portfolio {portfolio_id}")
            return position
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            raise
    
    async def rebalance_portfolio(self, portfolio_id: str, rebalancing_type: str = "threshold") -> RebalancingEvent:
        """Rebalance a portfolio"""
        try:
            if portfolio_id not in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            portfolio = self.portfolios[portfolio_id]
            
            # Check if rebalancing is needed
            if not await self._needs_rebalancing(portfolio_id):
                raise ValueError(f"Portfolio {portfolio_id} does not need rebalancing")
            
            # Calculate new allocation
            new_allocation = await self._calculate_optimal_allocation(portfolio_id)
            
            # Calculate trades required
            trades_required = await self._calculate_rebalancing_trades(portfolio_id, new_allocation)
            
            # Create rebalancing event
            event_id = f"rebalance_{portfolio_id}_{uuid.uuid4().hex[:8]}"
            event = RebalancingEvent(
                event_id=event_id,
                portfolio_id=portfolio_id,
                user_id=portfolio.user_id,
                event_type=rebalancing_type,
                trigger_reason="Portfolio drift exceeded threshold",
                old_allocation=portfolio.current_allocation.copy(),
                new_allocation=new_allocation,
                trades_required=trades_required,
                estimated_cost=sum(trade.get('estimated_cost', 0) for trade in trades_required),
                status='pending',
                execution_time=None,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.rebalancing_events[event_id] = event
            await self._cache_rebalancing_event(event)
            
            # Execute rebalancing
            asyncio.create_task(self._execute_rebalancing(event))
            
            logger.info(f"Started rebalancing for portfolio {portfolio_id}")
            return event
            
        except Exception as e:
            logger.error(f"Error starting rebalancing: {e}")
            raise
    
    async def optimize_portfolio(self, portfolio_id: str, optimization_method: str = "markowitz",
                               risk_free_rate: float = 0.02, target_return: Optional[float] = None,
                               target_risk: Optional[float] = None) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        try:
            if portfolio_id not in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            portfolio = self.portfolios[portfolio_id]
            positions = self.positions.get(portfolio_id, [])
            
            if not positions:
                raise ValueError(f"No positions found for portfolio {portfolio_id}")
            
            # Get historical data for optimization
            historical_data = await self._get_historical_data(portfolio_id)
            
            if optimization_method == "markowitz":
                result = await self._optimize_markowitz(portfolio_id, historical_data, risk_free_rate, target_return, target_risk)
            elif optimization_method == "risk_parity":
                result = await self._optimize_risk_parity(portfolio_id, historical_data)
            elif optimization_method == "black_litterman":
                result = await self._optimize_black_litterman(portfolio_id, historical_data)
            else:
                raise ValueError(f"Unsupported optimization method: {optimization_method}")
            
            # Store optimization result
            self.optimization_results[portfolio_id] = result
            
            logger.info(f"Completed portfolio optimization for {portfolio_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            raise
    
    async def get_portfolio_performance(self, portfolio_id: str) -> PortfolioPerformance:
        """Get portfolio performance metrics"""
        try:
            if portfolio_id not in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            performance = self.performance.get(portfolio_id)
            
            if not performance:
                # Calculate performance metrics
                performance = await self._calculate_portfolio_performance(portfolio_id)
                self.performance[portfolio_id] = performance
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            raise
    
    async def get_efficient_frontier(self, portfolio_id: str, risk_free_rate: float = 0.02) -> List[Dict[str, float]]:
        """Get efficient frontier for portfolio"""
        try:
            if portfolio_id not in self.portfolios:
                raise ValueError(f"Portfolio {portfolio_id} not found")
            
            if portfolio_id in self.efficient_frontier:
                return self.efficient_frontier[portfolio_id]
            
            # Calculate efficient frontier
            positions = self.positions.get(portfolio_id, [])
            if not positions:
                return []
            
            # Get historical data
            historical_data = await self._get_historical_data(portfolio_id)
            
            # Calculate efficient frontier points
            frontier_points = await self._calculate_efficient_frontier(portfolio_id, historical_data, risk_free_rate)
            
            self.efficient_frontier[portfolio_id] = frontier_points
            return frontier_points
            
        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return []
    
    async def _needs_rebalancing(self, portfolio_id: str) -> bool:
        """Check if portfolio needs rebalancing"""
        try:
            portfolio = self.portfolios[portfolio_id]
            positions = self.positions.get(portfolio_id, [])
            
            if not positions:
                return False
            
            # Check if any position exceeds rebalancing threshold
            for position in positions:
                target_weight = portfolio.target_allocation.get(position.asset_id, 0.0)
                current_weight = position.weight
                
                if abs(current_weight - target_weight) > 0.05:  # 5% threshold
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rebalancing needs: {e}")
            return False
    
    async def _calculate_optimal_allocation(self, portfolio_id: str) -> Dict[str, float]:
        """Calculate optimal portfolio allocation"""
        try:
            portfolio = self.portfolios[portfolio_id]
            positions = self.positions.get(portfolio_id, [])
            
            # Simple rebalancing to target allocation
            # In practice, this would use more sophisticated optimization
            optimal_allocation = {}
            
            for asset_id, target_weight in portfolio.target_allocation.items():
                optimal_allocation[asset_id] = target_weight
            
            return optimal_allocation
            
        except Exception as e:
            logger.error(f"Error calculating optimal allocation: {e}")
            return {}
    
    async def _calculate_rebalancing_trades(self, portfolio_id: str, new_allocation: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate trades required for rebalancing"""
        try:
            portfolio = self.portfolios[portfolio_id]
            positions = self.positions.get(portfolio_id, [])
            
            trades = []
            
            for asset_id, target_weight in new_allocation.items():
                current_position = next((p for p in positions if p.asset_id == asset_id), None)
                current_weight = current_position.weight if current_position else 0.0
                
                if abs(target_weight - current_weight) > 0.01:  # 1% threshold
                    target_value = portfolio.total_value * target_weight
                    current_value = portfolio.total_value * current_weight
                    
                    if target_value > current_value:
                        # Need to buy
                        trade = {
                            'asset_id': asset_id,
                            'action': 'buy',
                            'quantity': (target_value - current_value) / current_position.current_price if current_position else 0,
                            'estimated_cost': target_value - current_value
                        }
                    else:
                        # Need to sell
                        trade = {
                            'asset_id': asset_id,
                            'action': 'sell',
                            'quantity': (current_value - target_value) / current_position.current_price if current_position else 0,
                            'estimated_cost': 0  # Selling doesn't cost money
                        }
                    
                    trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing trades: {e}")
            return []
    
    async def _execute_rebalancing(self, event: RebalancingEvent):
        """Execute portfolio rebalancing"""
        try:
            event.status = 'executing'
            event.last_updated = datetime.utcnow()
            await self._cache_rebalancing_event(event)
            
            portfolio = self.portfolios[event.portfolio_id]
            
            # Execute trades
            for trade in event.trades_required:
                try:
                    if trade['action'] == 'buy':
                        # Simulate buying
                        await asyncio.sleep(1)  # Simulate execution time
                    elif trade['action'] == 'sell':
                        # Simulate selling
                        await asyncio.sleep(1)  # Simulate execution time
                    
                except Exception as e:
                    logger.error(f"Error executing trade: {e}")
            
            # Update portfolio allocation
            portfolio.current_allocation = event.new_allocation.copy()
            portfolio.last_rebalanced = datetime.utcnow()
            portfolio.last_updated = datetime.utcnow()
            
            await self._cache_portfolio(portfolio)
            
            # Update event status
            event.status = 'completed'
            event.execution_time = datetime.utcnow()
            event.last_updated = datetime.utcnow()
            await self._cache_rebalancing_event(event)
            
            logger.info(f"Completed rebalancing for portfolio {event.portfolio_id}")
            
        except Exception as e:
            logger.error(f"Error executing rebalancing: {e}")
            event.status = 'failed'
            event.last_updated = datetime.utcnow()
            await self._cache_rebalancing_event(event)
    
    async def _optimize_markowitz(self, portfolio_id: str, historical_data: pd.DataFrame,
                                 risk_free_rate: float, target_return: Optional[float],
                                 target_risk: Optional[float]) -> Dict[str, Any]:
        """Optimize portfolio using Markowitz mean-variance optimization"""
        try:
            # Simulate Markowitz optimization
            # In practice, this would use scipy.optimize or similar
            
            result = {
                'method': 'markowitz',
                'optimal_weights': {},
                'expected_return': 0.08,
                'expected_risk': 0.15,
                'sharpe_ratio': 0.4,
                'optimization_status': 'success'
            }
            
            # Calculate optimal weights based on historical data
            positions = self.positions.get(portfolio_id, [])
            for position in positions:
                result['optimal_weights'][position.asset_id] = np.random.uniform(0.1, 0.3)
            
            # Normalize weights
            total_weight = sum(result['optimal_weights'].values())
            for asset_id in result['optimal_weights']:
                result['optimal_weights'][asset_id] /= total_weight
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Markowitz optimization: {e}")
            raise
    
    async def _optimize_risk_parity(self, portfolio_id: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio using risk parity approach"""
        try:
            # Simulate risk parity optimization
            result = {
                'method': 'risk_parity',
                'optimal_weights': {},
                'risk_contribution': {},
                'optimization_status': 'success'
            }
            
            positions = self.positions.get(portfolio_id, [])
            for position in positions:
                result['optimal_weights'][position.asset_id] = 1.0 / len(positions)
                result['risk_contribution'][position.asset_id] = 1.0 / len(positions)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {e}")
            raise
    
    async def _optimize_black_litterman(self, portfolio_id: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize portfolio using Black-Litterman model"""
        try:
            # Simulate Black-Litterman optimization
            result = {
                'method': 'black_litterman',
                'optimal_weights': {},
                'view_confidence': {},
                'optimization_status': 'success'
            }
            
            positions = self.positions.get(portfolio_id, [])
            for position in positions:
                result['optimal_weights'][position.asset_id] = np.random.uniform(0.1, 0.3)
                result['view_confidence'][position.asset_id] = np.random.uniform(0.5, 0.9)
            
            # Normalize weights
            total_weight = sum(result['optimal_weights'].values())
            for asset_id in result['optimal_weights']:
                result['optimal_weights'][asset_id] /= total_weight
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {e}")
            raise
    
    async def _calculate_efficient_frontier(self, portfolio_id: str, historical_data: pd.DataFrame,
                                          risk_free_rate: float) -> List[Dict[str, float]]:
        """Calculate efficient frontier points"""
        try:
            # Simulate efficient frontier calculation
            frontier_points = []
            
            for i in range(10):
                risk = 0.05 + i * 0.02  # 5% to 23% risk
                return_rate = risk_free_rate + (risk - 0.05) * 0.8  # Sharpe ratio of 0.8
                
                frontier_points.append({
                    'risk': risk,
                    'return': return_rate,
                    'sharpe_ratio': (return_rate - risk_free_rate) / risk
                })
            
            return frontier_points
            
        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return []
    
    async def _calculate_portfolio_performance(self, portfolio_id: str) -> PortfolioPerformance:
        """Calculate portfolio performance metrics"""
        try:
            portfolio = self.portfolios[portfolio_id]
            positions = self.positions.get(portfolio_id, [])
            
            if not positions:
                return PortfolioPerformance(
                    portfolio_id=portfolio_id,
                    total_return=0.0,
                    annualized_return=0.0,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    max_drawdown=0.0,
                    volatility=0.0,
                    beta=1.0,
                    alpha=0.0,
                    tracking_error=0.0,
                    information_ratio=0.0,
                    calmar_ratio=0.0,
                    current_value=portfolio.total_value,
                    peak_value=portfolio.total_value,
                    last_updated=datetime.utcnow()
                )
            
            # Calculate basic metrics
            total_return = (portfolio.total_value - 100000) / 100000  # Assuming 100k initial
            annualized_return = total_return * 12  # Simplified annualization
            
            # Simulate other metrics
            sharpe_ratio = np.random.uniform(0.5, 1.5)
            sortino_ratio = sharpe_ratio * 1.1
            max_drawdown = np.random.uniform(0.05, 0.25)
            volatility = np.random.uniform(0.1, 0.3)
            beta = np.random.uniform(0.8, 1.2)
            alpha = np.random.uniform(-0.02, 0.02)
            
            performance = PortfolioPerformance(
                portfolio_id=portfolio_id,
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                alpha=alpha,
                tracking_error=volatility * 0.1,
                information_ratio=alpha / (volatility * 0.1) if volatility > 0 else 0,
                calmar_ratio=annualized_return / max_drawdown if max_drawdown > 0 else 0,
                current_value=portfolio.total_value,
                peak_value=portfolio.total_value * 1.1,  # Simulated peak
                last_updated=datetime.utcnow()
            )
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            raise
    
    async def _get_historical_data(self, portfolio_id: str) -> pd.DataFrame:
        """Get historical data for portfolio optimization"""
        try:
            # Simulate historical data
            # In practice, this would fetch from market data service
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            data = {}
            
            positions = self.positions.get(portfolio_id, [])
            for position in positions:
                # Generate simulated price data
                base_price = position.current_price
                returns = np.random.normal(0, 0.02, len(dates))
                prices = [base_price * (1 + np.cumsum(returns))]
                data[position.asset_id] = prices[0]
            
            return pd.DataFrame(data, index=dates)
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    async def _create_asset_allocations(self, portfolio_id: str, target_allocation: Dict[str, float]):
        """Create asset allocation records"""
        try:
            allocations = []
            
            for asset_id, target_percentage in target_allocation.items():
                allocation = AssetAllocation(
                    allocation_id=f"allocation_{portfolio_id}_{asset_id}_{uuid.uuid4().hex[:8]}",
                    portfolio_id=portfolio_id,
                    asset_class=asset_id,
                    target_percentage=target_percentage,
                    current_percentage=target_percentage,
                    min_percentage=max(0, target_percentage - 0.05),
                    max_percentage=min(1, target_percentage + 0.05),
                    rebalancing_threshold=0.05,
                    last_rebalanced=datetime.utcnow(),
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                )
                
                allocations.append(allocation)
            
            self.asset_allocations[portfolio_id] = allocations
            
        except Exception as e:
            logger.error(f"Error creating asset allocations: {e}")
    
    # Background tasks
    async def _update_portfolio_values(self):
        """Update portfolio values and positions"""
        while True:
            try:
                # Update portfolio values
                for portfolio_id, portfolio in self.portfolios.items():
                    if portfolio.is_active:
                        positions = self.positions.get(portfolio_id, [])
                        
                        # Update position values
                        total_value = portfolio.cash_balance
                        for position in positions:
                            # Simulate price changes
                            price_change = np.random.normal(0, 0.01)
                            position.current_price *= (1 + price_change)
                            position.market_value = position.quantity * position.current_price
                            position.unrealized_pnl = position.market_value - position.cost_basis
                            position.weight = position.market_value / (total_value + sum(p.market_value for p in positions))
                            position.last_updated = datetime.utcnow()
                            
                            total_value += position.market_value
                            
                            await self._cache_position(position)
                        
                        # Update portfolio
                        portfolio.total_value = total_value
                        portfolio.last_updated = datetime.utcnow()
                        await self._cache_portfolio(portfolio)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error updating portfolio values: {e}")
                await asyncio.sleep(600)
    
    async def _check_rebalancing_needs(self):
        """Check if portfolios need rebalancing"""
        while True:
            try:
                for portfolio_id, portfolio in self.portfolios.items():
                    if portfolio.is_active and await self._needs_rebalancing(portfolio_id):
                        logger.info(f"Portfolio {portfolio_id} needs rebalancing")
                        
                        # Check if it's time for scheduled rebalancing
                        if portfolio.rebalancing_frequency != RebalancingFrequency.ON_DEMAND:
                            days_since_rebalance = (datetime.utcnow() - portfolio.last_rebalanced).days
                            
                            if portfolio.rebalancing_frequency == RebalancingFrequency.DAILY and days_since_rebalance >= 1:
                                await self.rebalance_portfolio(portfolio_id, "scheduled")
                            elif portfolio.rebalancing_frequency == RebalancingFrequency.WEEKLY and days_since_rebalance >= 7:
                                await self.rebalance_portfolio(portfolio_id, "scheduled")
                            elif portfolio.rebalancing_frequency == RebalancingFrequency.MONTHLY and days_since_rebalance >= 30:
                                await self.rebalance_portfolio(portfolio_id, "scheduled")
                            elif portfolio.rebalancing_frequency == RebalancingFrequency.QUARTERLY and days_since_rebalance >= 90:
                                await self.rebalance_portfolio(portfolio_id, "scheduled")
                            elif portfolio.rebalancing_frequency == RebalancingFrequency.SEMI_ANNUALLY and days_since_rebalance >= 180:
                                await self.rebalance_portfolio(portfolio_id, "scheduled")
                            elif portfolio.rebalancing_frequency == RebalancingFrequency.ANNUALLY and days_since_rebalance >= 365:
                                await self.rebalance_portfolio(portfolio_id, "scheduled")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error checking rebalancing needs: {e}")
                await asyncio.sleep(7200)
    
    async def _update_performance_metrics(self):
        """Update portfolio performance metrics"""
        while True:
            try:
                for portfolio_id in self.portfolios:
                    if portfolio_id in self.performance:
                        performance = self.performance[portfolio_id]
                        performance.last_updated = datetime.utcnow()
                        
                        # Store performance history
                        self.performance_history[portfolio_id].append({
                            'timestamp': datetime.utcnow().isoformat(),
                            'total_return': performance.total_return,
                            'sharpe_ratio': performance.sharpe_ratio,
                            'volatility': performance.volatility
                        })
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")
                await asyncio.sleep(7200)
    
    # Helper methods
    async def _load_portfolios(self):
        """Load portfolios from database"""
        pass
    
    async def _load_positions(self):
        """Load positions from database"""
        pass
    
    async def _load_performance(self):
        """Load performance from database"""
        pass
    
    # Caching methods
    async def _cache_portfolio(self, portfolio: Portfolio):
        """Cache portfolio"""
        try:
            cache_key = f"portfolio:{portfolio.portfolio_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'user_id': portfolio.user_id,
                    'portfolio_name': portfolio.portfolio_name,
                    'portfolio_type': portfolio.portfolio_type.value,
                    'description': portfolio.description,
                    'target_allocation': portfolio.target_allocation,
                    'current_allocation': portfolio.current_allocation,
                    'total_value': portfolio.total_value,
                    'cash_balance': portfolio.cash_balance,
                    'risk_profile': portfolio.risk_profile,
                    'rebalancing_frequency': portfolio.rebalancing_frequency.value,
                    'last_rebalanced': portfolio.last_rebalanced.isoformat(),
                    'is_active': portfolio.is_active,
                    'created_at': portfolio.created_at.isoformat(),
                    'last_updated': portfolio.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching portfolio: {e}")
    
    async def _cache_position(self, position: PortfolioPosition):
        """Cache position"""
        try:
            cache_key = f"position:{position.position_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps({
                    'portfolio_id': position.portfolio_id,
                    'asset_id': position.asset_id,
                    'asset_type': position.asset_type,
                    'quantity': position.quantity,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'cost_basis': position.cost_basis,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'weight': position.weight,
                    'target_weight': position.target_weight,
                    'created_at': position.created_at.isoformat(),
                    'last_updated': position.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching position: {e}")
    
    async def _cache_rebalancing_event(self, event: RebalancingEvent):
        """Cache rebalancing event"""
        try:
            cache_key = f"rebalancing:{event.event_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps({
                    'portfolio_id': event.portfolio_id,
                    'user_id': event.user_id,
                    'event_type': event.event_type,
                    'trigger_reason': event.trigger_reason,
                    'old_allocation': event.old_allocation,
                    'new_allocation': event.new_allocation,
                    'trades_required': event.trades_required,
                    'estimated_cost': event.estimated_cost,
                    'status': event.status,
                    'execution_time': event.execution_time.isoformat() if event.execution_time else None,
                    'created_at': event.created_at.isoformat(),
                    'last_updated': event.last_updated.isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Error caching rebalancing event: {e}")


# Factory function
async def get_portfolio_management_service(redis_client: redis.Redis, db_session: Session) -> PortfolioManagementService:
    """Get portfolio management service instance"""
    service = PortfolioManagementService(redis_client, db_session)
    await service.initialize()
    return service
