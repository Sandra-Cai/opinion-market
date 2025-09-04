"""
Wealth Management Service
Advanced wealth management and financial planning
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
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ClientType(Enum):
    """Client types"""
    INDIVIDUAL = "individual"
    FAMILY = "family"
    CORPORATE = "corporate"
    INSTITUTIONAL = "institutional"
    ULTRA_HIGH_NET_WORTH = "ultra_high_net_worth"
    HIGH_NET_WORTH = "high_net_worth"
    MASS_AFFLUENT = "mass_affluent"


class RiskProfile(Enum):
    """Risk profiles"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    BALANCED = "balanced"
    GROWTH = "growth"
    AGGRESSIVE = "aggressive"


class InvestmentObjective(Enum):
    """Investment objectives"""
    CAPITAL_PRESERVATION = "capital_preservation"
    INCOME_GENERATION = "income_generation"
    GROWTH = "growth"
    TOTAL_RETURN = "total_return"
    TAX_EFFICIENCY = "tax_efficiency"
    INFLATION_PROTECTION = "inflation_protection"


class LifeStage(Enum):
    """Life stages"""
    YOUNG_PROFESSIONAL = "young_professional"
    ESTABLISHED_CAREER = "established_career"
    PRE_RETIREMENT = "pre_retirement"
    RETIREMENT = "retirement"
    LEGACY_PLANNING = "legacy_planning"


@dataclass
class Client:
    """Client profile"""
    client_id: str
    user_id: int
    client_type: ClientType
    first_name: str
    last_name: str
    email: str
    phone: str
    date_of_birth: datetime
    risk_profile: RiskProfile
    investment_objective: InvestmentObjective
    life_stage: LifeStage
    annual_income: float
    net_worth: float
    investable_assets: float
    risk_tolerance: float
    time_horizon: int
    liquidity_needs: float
    tax_bracket: float
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class FinancialGoal:
    """Financial goal"""
    goal_id: str
    client_id: str
    goal_name: str
    goal_type: str  # 'retirement', 'education', 'purchase', 'legacy', 'income'
    target_amount: float
    current_amount: float
    target_date: datetime
    priority: int
    is_achievable: bool
    required_monthly_contribution: float
    expected_return: float
    risk_level: str
    status: str  # 'active', 'achieved', 'paused', 'cancelled'
    created_at: datetime
    last_updated: datetime


@dataclass
class AssetAllocation:
    """Asset allocation"""
    allocation_id: str
    client_id: str
    asset_class: str
    target_percentage: float
    current_percentage: float
    current_value: float
    target_value: float
    rebalance_threshold: float
    last_rebalanced: datetime
    created_at: datetime
    last_updated: datetime


@dataclass
class InvestmentRecommendation:
    """Investment recommendation"""
    recommendation_id: str
    client_id: str
    recommendation_type: str  # 'buy', 'sell', 'hold', 'rebalance'
    asset_class: str
    security_name: str
    symbol: str
    current_weight: float
    recommended_weight: float
    expected_return: float
    risk_score: float
    rationale: str
    priority: int
    status: str  # 'pending', 'approved', 'rejected', 'implemented'
    created_at: datetime
    last_updated: datetime


@dataclass
class Portfolio:
    """Portfolio"""
    portfolio_id: str
    client_id: str
    portfolio_name: str
    portfolio_type: str  # 'taxable', 'tax_deferred', 'tax_free'
    total_value: float
    cash_balance: float
    invested_value: float
    unrealized_gain_loss: float
    realized_gain_loss: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    created_at: datetime
    last_updated: datetime


@dataclass
class RebalancingEvent:
    """Rebalancing event"""
    rebalance_id: str
    client_id: str
    rebalance_date: datetime
    rebalance_type: str  # 'scheduled', 'threshold', 'drift', 'manual'
    total_value: float
    trades: List[Dict[str, Any]]
    transaction_costs: float
    tax_impact: float
    drift_amount: float
    status: str  # 'pending', 'completed', 'failed'
    created_at: datetime


@dataclass
class TaxOptimization:
    """Tax optimization"""
    optimization_id: str
    client_id: str
    optimization_type: str  # 'harvesting', 'location', 'timing', 'structure'
    description: str
    potential_savings: float
    implementation_cost: float
    net_benefit: float
    risk_level: str
    time_horizon: str
    status: str  # 'identified', 'recommended', 'implemented', 'monitoring'
    created_at: datetime
    last_updated: datetime


class WealthManagementService:
    """Comprehensive Wealth Management Service"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        
        # Client management
        self.clients: Dict[str, Client] = {}
        self.financial_goals: Dict[str, List[FinancialGoal]] = defaultdict(list)
        self.asset_allocations: Dict[str, List[AssetAllocation]] = defaultdict(list)
        self.investment_recommendations: Dict[str, List[InvestmentRecommendation]] = defaultdict(list)
        self.portfolios: Dict[str, List[Portfolio]] = defaultdict(list)
        self.rebalancing_events: Dict[str, List[RebalancingEvent]] = defaultdict(list)
        self.tax_optimizations: Dict[str, List[TaxOptimization]] = defaultdict(list)
        
        # Analytics
        self.portfolio_analytics: Dict[str, Dict[str, Any]] = {}
        self.risk_analytics: Dict[str, Dict[str, float]] = {}
        self.performance_analytics: Dict[str, Dict[str, float]] = {}
        
        # Market data
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Dict[str, float] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize the Wealth Management Service"""
        logger.info("Initializing Wealth Management Service")
        
        # Load sample clients
        await self._load_sample_clients()
        
        # Initialize market data
        await self._initialize_market_data()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._monitor_portfolios()),
            asyncio.create_task(self._check_rebalancing()),
            asyncio.create_task(self._update_performance()),
            asyncio.create_task(self._identify_tax_opportunities()),
            asyncio.create_task(self._update_goals_progress())
        ]
        
        logger.info("Wealth Management Service initialized successfully")
    
    async def create_client(self, user_id: int, client_type: ClientType,
                          first_name: str, last_name: str, email: str,
                          phone: str, date_of_birth: datetime,
                          risk_profile: RiskProfile,
                          investment_objective: InvestmentObjective,
                          life_stage: LifeStage, annual_income: float,
                          net_worth: float, investable_assets: float,
                          risk_tolerance: float, time_horizon: int,
                          liquidity_needs: float, tax_bracket: float) -> Client:
        """Create a new client"""
        try:
            client_id = f"CLIENT_{uuid.uuid4().hex[:8]}"
            
            client = Client(
                client_id=client_id,
                user_id=user_id,
                client_type=client_type,
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone,
                date_of_birth=date_of_birth,
                risk_profile=risk_profile,
                investment_objective=investment_objective,
                life_stage=life_stage,
                annual_income=annual_income,
                net_worth=net_worth,
                investable_assets=investable_assets,
                risk_tolerance=risk_tolerance,
                time_horizon=time_horizon,
                liquidity_needs=liquidity_needs,
                tax_bracket=tax_bracket,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.clients[client_id] = client
            
            # Create initial asset allocation
            await self._create_initial_asset_allocation(client)
            
            # Create initial portfolio
            await self._create_initial_portfolio(client)
            
            logger.info(f"Created client {client_id}")
            return client
            
        except Exception as e:
            logger.error(f"Error creating client: {e}")
            raise
    
    async def create_financial_goal(self, client_id: str, goal_name: str,
                                  goal_type: str, target_amount: float,
                                  target_date: datetime, priority: int,
                                  expected_return: float = 0.08) -> FinancialGoal:
        """Create a financial goal"""
        try:
            client = self.clients.get(client_id)
            if not client:
                raise ValueError("Client not found")
            
            goal_id = f"GOAL_{uuid.uuid4().hex[:8]}"
            
            # Calculate required monthly contribution
            years_to_goal = (target_date - datetime.utcnow()).days / 365.0
            monthly_rate = expected_return / 12
            months_to_goal = years_to_goal * 12
            
            if months_to_goal > 0:
                required_contribution = target_amount / ((1 + monthly_rate) ** months_to_goal - 1) * monthly_rate
            else:
                required_contribution = 0
            
            goal = FinancialGoal(
                goal_id=goal_id,
                client_id=client_id,
                goal_name=goal_name,
                goal_type=goal_type,
                target_amount=target_amount,
                current_amount=0.0,
                target_date=target_date,
                priority=priority,
                is_achievable=required_contribution <= client.annual_income * 0.3 / 12,  # 30% of income
                required_monthly_contribution=required_contribution,
                expected_return=expected_return,
                risk_level=client.risk_profile.value,
                status='active',
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.financial_goals[client_id].append(goal)
            
            logger.info(f"Created financial goal {goal_id}")
            return goal
            
        except Exception as e:
            logger.error(f"Error creating financial goal: {e}")
            raise
    
    async def create_asset_allocation(self, client_id: str, asset_class: str,
                                    target_percentage: float,
                                    rebalance_threshold: float = 0.05) -> AssetAllocation:
        """Create asset allocation"""
        try:
            client = self.clients.get(client_id)
            if not client:
                raise ValueError("Client not found")
            
            allocation_id = f"ALLOC_{uuid.uuid4().hex[:8]}"
            
            # Calculate current and target values
            total_assets = client.investable_assets
            target_value = total_assets * target_percentage
            current_value = target_value  # Initial allocation
            
            allocation = AssetAllocation(
                allocation_id=allocation_id,
                client_id=client_id,
                asset_class=asset_class,
                target_percentage=target_percentage,
                current_percentage=target_percentage,
                current_value=current_value,
                target_value=target_value,
                rebalance_threshold=rebalance_threshold,
                last_rebalanced=datetime.utcnow(),
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.asset_allocations[client_id].append(allocation)
            
            logger.info(f"Created asset allocation {allocation_id}")
            return allocation
            
        except Exception as e:
            logger.error(f"Error creating asset allocation: {e}")
            raise
    
    async def create_investment_recommendation(self, client_id: str,
                                            recommendation_type: str,
                                            asset_class: str, security_name: str,
                                            symbol: str, current_weight: float,
                                            recommended_weight: float,
                                            expected_return: float,
                                            risk_score: float, rationale: str,
                                            priority: int) -> InvestmentRecommendation:
        """Create investment recommendation"""
        try:
            client = self.clients.get(client_id)
            if not client:
                raise ValueError("Client not found")
            
            recommendation_id = f"REC_{uuid.uuid4().hex[:8]}"
            
            recommendation = InvestmentRecommendation(
                recommendation_id=recommendation_id,
                client_id=client_id,
                recommendation_type=recommendation_type,
                asset_class=asset_class,
                security_name=security_name,
                symbol=symbol,
                current_weight=current_weight,
                recommended_weight=recommended_weight,
                expected_return=expected_return,
                risk_score=risk_score,
                rationale=rationale,
                priority=priority,
                status='pending',
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.investment_recommendations[client_id].append(recommendation)
            
            logger.info(f"Created investment recommendation {recommendation_id}")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error creating investment recommendation: {e}")
            raise
    
    async def create_portfolio(self, client_id: str, portfolio_name: str,
                             portfolio_type: str) -> Portfolio:
        """Create a portfolio"""
        try:
            client = self.clients.get(client_id)
            if not client:
                raise ValueError("Client not found")
            
            portfolio_id = f"PORT_{uuid.uuid4().hex[:8]}"
            
            portfolio = Portfolio(
                portfolio_id=portfolio_id,
                client_id=client_id,
                portfolio_name=portfolio_name,
                portfolio_type=portfolio_type,
                total_value=0.0,
                cash_balance=0.0,
                invested_value=0.0,
                unrealized_gain_loss=0.0,
                realized_gain_loss=0.0,
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                beta=0.0,
                alpha=0.0,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.portfolios[client_id].append(portfolio)
            
            logger.info(f"Created portfolio {portfolio_id}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {e}")
            raise
    
    async def rebalance_portfolio(self, client_id: str, rebalance_type: str = 'scheduled') -> RebalancingEvent:
        """Rebalance portfolio"""
        try:
            client = self.clients.get(client_id)
            if not client:
                raise ValueError("Client not found")
            
            allocations = self.asset_allocations.get(client_id, [])
            if not allocations:
                raise ValueError("No asset allocations found")
            
            rebalance_id = f"REBAL_{uuid.uuid4().hex[:8]}"
            
            # Calculate total portfolio value
            total_value = sum(alloc.current_value for alloc in allocations)
            
            # Generate rebalancing trades
            trades = []
            total_transaction_costs = 0.0
            total_tax_impact = 0.0
            total_drift = 0.0
            
            for allocation in allocations:
                target_value = total_value * allocation.target_percentage
                current_value = allocation.current_value
                drift = abs(current_value - target_value)
                total_drift += drift
                
                if drift > total_value * allocation.rebalance_threshold:
                    trade_amount = target_value - current_value
                    transaction_cost = abs(trade_amount) * 0.001  # 0.1% transaction cost
                    tax_impact = max(0, trade_amount) * client.tax_bracket  # Tax on gains
                    
                    trades.append({
                        'asset_class': allocation.asset_class,
                        'current_value': current_value,
                        'target_value': target_value,
                        'trade_amount': trade_amount,
                        'transaction_cost': transaction_cost,
                        'tax_impact': tax_impact
                    })
                    
                    total_transaction_costs += transaction_cost
                    total_tax_impact += tax_impact
                    
                    # Update allocation
                    allocation.current_value = target_value
                    allocation.current_percentage = (target_value / total_value) * 100
                    allocation.last_rebalanced = datetime.utcnow()
                    allocation.last_updated = datetime.utcnow()
            
            rebalancing_event = RebalancingEvent(
                rebalance_id=rebalance_id,
                client_id=client_id,
                rebalance_date=datetime.utcnow(),
                rebalance_type=rebalance_type,
                total_value=total_value,
                trades=trades,
                transaction_costs=total_transaction_costs,
                tax_impact=total_tax_impact,
                drift_amount=total_drift,
                status='completed',
                created_at=datetime.utcnow()
            )
            
            self.rebalancing_events[client_id].append(rebalancing_event)
            
            logger.info(f"Completed portfolio rebalancing {rebalance_id}")
            return rebalancing_event
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            raise
    
    async def identify_tax_optimization(self, client_id: str,
                                      optimization_type: str,
                                      description: str,
                                      potential_savings: float,
                                      implementation_cost: float,
                                      risk_level: str,
                                      time_horizon: str) -> TaxOptimization:
        """Identify tax optimization opportunity"""
        try:
            client = self.clients.get(client_id)
            if not client:
                raise ValueError("Client not found")
            
            optimization_id = f"TAX_{uuid.uuid4().hex[:8]}"
            
            net_benefit = potential_savings - implementation_cost
            
            optimization = TaxOptimization(
                optimization_id=optimization_id,
                client_id=client_id,
                optimization_type=optimization_type,
                description=description,
                potential_savings=potential_savings,
                implementation_cost=implementation_cost,
                net_benefit=net_benefit,
                risk_level=risk_level,
                time_horizon=time_horizon,
                status='identified',
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.tax_optimizations[client_id].append(optimization)
            
            logger.info(f"Identified tax optimization {optimization_id}")
            return optimization
            
        except Exception as e:
            logger.error(f"Error identifying tax optimization: {e}")
            raise
    
    async def get_portfolio_analytics(self, client_id: str) -> Dict[str, Any]:
        """Get portfolio analytics for a client"""
        try:
            client = self.clients.get(client_id)
            if not client:
                return {'error': 'Client not found'}
            
            portfolios = self.portfolios.get(client_id, [])
            allocations = self.asset_allocations.get(client_id, [])
            goals = self.financial_goals.get(client_id, [])
            
            # Calculate portfolio metrics
            total_value = sum(port.total_value for port in portfolios)
            total_return = sum(port.total_return for port in portfolios)
            total_volatility = np.mean([port.volatility for port in portfolios]) if portfolios else 0
            total_sharpe = np.mean([port.sharpe_ratio for port in portfolios]) if portfolios else 0
            
            # Asset allocation analysis
            allocation_analysis = {}
            for allocation in allocations:
                allocation_analysis[allocation.asset_class] = {
                    'target_percentage': allocation.target_percentage,
                    'current_percentage': allocation.current_percentage,
                    'drift': allocation.current_percentage - allocation.target_percentage,
                    'value': allocation.current_value
                }
            
            # Goals analysis
            goals_analysis = {}
            for goal in goals:
                if goal.status == 'active':
                    years_to_goal = (goal.target_date - datetime.utcnow()).days / 365.0
                    progress = goal.current_amount / goal.target_amount if goal.target_amount > 0 else 0
                    on_track = progress >= (1 - years_to_goal / goal.time_horizon) if hasattr(goal, 'time_horizon') else True
                    
                    goals_analysis[goal.goal_name] = {
                        'target_amount': goal.target_amount,
                        'current_amount': goal.current_amount,
                        'progress': progress,
                        'on_track': on_track,
                        'required_contribution': goal.required_monthly_contribution
                    }
            
            # Risk analysis
            risk_analysis = await self._calculate_risk_metrics(client, allocations)
            
            # Performance analysis
            performance_analysis = await self._calculate_performance_metrics(portfolios)
            
            analytics = {
                'client_id': client_id,
                'total_value': total_value,
                'total_return': total_return,
                'total_volatility': total_volatility,
                'total_sharpe': total_sharpe,
                'allocation_analysis': allocation_analysis,
                'goals_analysis': goals_analysis,
                'risk_analysis': risk_analysis,
                'performance_analysis': performance_analysis,
                'number_of_portfolios': len(portfolios),
                'number_of_goals': len(goals),
                'timestamp': datetime.utcnow()
            }
            
            self.portfolio_analytics[client_id] = analytics
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting portfolio analytics: {e}")
            return {'error': str(e)}
    
    # Background tasks
    async def _monitor_portfolios(self):
        """Monitor portfolios"""
        while True:
            try:
                for client_id, portfolios in self.portfolios.items():
                    for portfolio in portfolios:
                        await self._update_portfolio_metrics(portfolio)
                
                await asyncio.sleep(3600)  # Monitor every hour
                
            except Exception as e:
                logger.error(f"Error monitoring portfolios: {e}")
                await asyncio.sleep(7200)
    
    async def _update_portfolio_metrics(self, portfolio: Portfolio):
        """Update portfolio metrics"""
        try:
            # Simulate portfolio performance
            daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily return, 2% volatility
            portfolio.total_value *= (1 + daily_return)
            portfolio.total_return = (portfolio.total_value - portfolio.invested_value) / portfolio.invested_value if portfolio.invested_value > 0 else 0
            
            # Update other metrics
            portfolio.volatility = 0.15  # 15% annual volatility
            portfolio.sharpe_ratio = portfolio.total_return / portfolio.volatility if portfolio.volatility > 0 else 0
            portfolio.beta = 1.0  # Market beta
            portfolio.alpha = portfolio.total_return - 0.08  # 8% market return assumption
            
            portfolio.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    async def _check_rebalancing(self):
        """Check for rebalancing needs"""
        while True:
            try:
                for client_id, allocations in self.asset_allocations.items():
                    await self._check_rebalancing_needs(client_id, allocations)
                
                await asyncio.sleep(86400)  # Check daily
                
            except Exception as e:
                logger.error(f"Error checking rebalancing: {e}")
                await asyncio.sleep(172800)
    
    async def _check_rebalancing_needs(self, client_id: str, allocations: List[AssetAllocation]):
        """Check rebalancing needs for a client"""
        try:
            total_value = sum(alloc.current_value for alloc in allocations)
            if total_value == 0:
                return
            
            needs_rebalancing = False
            for allocation in allocations:
                current_percentage = (allocation.current_value / total_value) * 100
                drift = abs(current_percentage - allocation.target_percentage)
                if drift > allocation.rebalance_threshold * 100:
                    needs_rebalancing = True
                    break
            
            if needs_rebalancing:
                await self.rebalance_portfolio(client_id, 'threshold')
            
        except Exception as e:
            logger.error(f"Error checking rebalancing needs: {e}")
    
    async def _update_performance(self):
        """Update performance metrics"""
        while True:
            try:
                for client_id in self.clients.keys():
                    await self.get_portfolio_analytics(client_id)
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error updating performance: {e}")
                await asyncio.sleep(7200)
    
    async def _identify_tax_opportunities(self):
        """Identify tax optimization opportunities"""
        while True:
            try:
                for client_id, client in self.clients.items():
                    if client.is_active:
                        await self._scan_tax_opportunities(client)
                
                await asyncio.sleep(86400)  # Scan daily
                
            except Exception as e:
                logger.error(f"Error identifying tax opportunities: {e}")
                await asyncio.sleep(172800)
    
    async def _scan_tax_opportunities(self, client: Client):
        """Scan for tax optimization opportunities"""
        try:
            # Simulate tax opportunity identification
            if np.random.random() < 0.1:  # 10% chance of finding opportunity
                opportunity_types = ['harvesting', 'location', 'timing', 'structure']
                opportunity_type = np.random.choice(opportunity_types)
                
                potential_savings = client.investable_assets * np.random.uniform(0.01, 0.05)  # 1-5% of assets
                implementation_cost = potential_savings * 0.1  # 10% of savings
                
                await self.identify_tax_optimization(
                    client.client_id,
                    opportunity_type,
                    f'Tax {opportunity_type} optimization opportunity',
                    potential_savings,
                    implementation_cost,
                    'low',
                    'short_term'
                )
            
        except Exception as e:
            logger.error(f"Error scanning tax opportunities: {e}")
    
    async def _update_goals_progress(self):
        """Update goals progress"""
        while True:
            try:
                for client_id, goals in self.financial_goals.items():
                    for goal in goals:
                        if goal.status == 'active':
                            await self._update_goal_progress(goal)
                
                await asyncio.sleep(86400)  # Update daily
                
            except Exception as e:
                logger.error(f"Error updating goals progress: {e}")
                await asyncio.sleep(172800)
    
    async def _update_goal_progress(self, goal: FinancialGoal):
        """Update goal progress"""
        try:
            # Simulate goal progress
            monthly_contribution = goal.required_monthly_contribution
            monthly_return = goal.expected_return / 12
            
            # Add monthly contribution and return
            goal.current_amount = goal.current_amount * (1 + monthly_return) + monthly_contribution
            
            # Check if goal is achieved
            if goal.current_amount >= goal.target_amount:
                goal.status = 'achieved'
            
            goal.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating goal progress: {e}")
    
    # Helper methods
    async def _create_initial_asset_allocation(self, client: Client):
        """Create initial asset allocation based on risk profile"""
        try:
            # Define asset allocation based on risk profile
            allocations = {
                RiskProfile.CONSERVATIVE: {
                    'Stocks': 0.30,
                    'Bonds': 0.50,
                    'Cash': 0.15,
                    'Alternatives': 0.05
                },
                RiskProfile.MODERATE: {
                    'Stocks': 0.50,
                    'Bonds': 0.35,
                    'Cash': 0.10,
                    'Alternatives': 0.05
                },
                RiskProfile.BALANCED: {
                    'Stocks': 0.60,
                    'Bonds': 0.30,
                    'Cash': 0.05,
                    'Alternatives': 0.05
                },
                RiskProfile.GROWTH: {
                    'Stocks': 0.70,
                    'Bonds': 0.20,
                    'Cash': 0.05,
                    'Alternatives': 0.05
                },
                RiskProfile.AGGRESSIVE: {
                    'Stocks': 0.80,
                    'Bonds': 0.10,
                    'Cash': 0.05,
                    'Alternatives': 0.05
                }
            }
            
            client_allocation = allocations.get(client.risk_profile, allocations[RiskProfile.BALANCED])
            
            for asset_class, percentage in client_allocation.items():
                await self.create_asset_allocation(
                    client.client_id,
                    asset_class,
                    percentage,
                    0.05  # 5% rebalance threshold
                )
            
        except Exception as e:
            logger.error(f"Error creating initial asset allocation: {e}")
    
    async def _create_initial_portfolio(self, client: Client):
        """Create initial portfolio"""
        try:
            await self.create_portfolio(
                client.client_id,
                'Main Portfolio',
                'taxable'
            )
            
        except Exception as e:
            logger.error(f"Error creating initial portfolio: {e}")
    
    async def _calculate_risk_metrics(self, client: Client, allocations: List[AssetAllocation]) -> Dict[str, float]:
        """Calculate risk metrics"""
        try:
            # Simplified risk metrics
            total_value = sum(alloc.current_value for alloc in allocations)
            
            # Concentration risk
            max_allocation = max(alloc.current_percentage for alloc in allocations) if allocations else 0
            concentration_risk = max_allocation / 100
            
            # Diversification score
            diversification_score = len(allocations) / 10  # Max 10 asset classes
            
            # Risk-adjusted return
            expected_return = 0.08  # 8% expected return
            volatility = 0.15  # 15% volatility
            risk_adjusted_return = expected_return / volatility if volatility > 0 else 0
            
            return {
                'concentration_risk': concentration_risk,
                'diversification_score': diversification_score,
                'risk_adjusted_return': risk_adjusted_return,
                'expected_return': expected_return,
                'volatility': volatility,
                'var_95': total_value * 0.05,  # 5% VaR
                'max_drawdown': 0.20  # 20% max drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    async def _calculate_performance_metrics(self, portfolios: List[Portfolio]) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            if not portfolios:
                return {}
            
            # Aggregate performance metrics
            total_return = np.mean([port.total_return for port in portfolios])
            total_volatility = np.mean([port.volatility for port in portfolios])
            total_sharpe = np.mean([port.sharpe_ratio for port in portfolios])
            total_alpha = np.mean([port.alpha for port in portfolios])
            total_beta = np.mean([port.beta for port in portfolios])
            
            return {
                'total_return': total_return,
                'volatility': total_volatility,
                'sharpe_ratio': total_sharpe,
                'alpha': total_alpha,
                'beta': total_beta,
                'information_ratio': total_alpha / total_volatility if total_volatility > 0 else 0,
                'treynor_ratio': total_return / total_beta if total_beta > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    async def _load_sample_clients(self):
        """Load sample clients"""
        try:
            sample_clients = [
                {
                    'user_id': 1,
                    'client_type': ClientType.HIGH_NET_WORTH,
                    'first_name': 'John',
                    'last_name': 'Smith',
                    'email': 'john.smith@email.com',
                    'phone': '+1-555-0123',
                    'date_of_birth': datetime.utcnow() - timedelta(days=365*45),
                    'risk_profile': RiskProfile.BALANCED,
                    'investment_objective': InvestmentObjective.GROWTH,
                    'life_stage': LifeStage.ESTABLISHED_CAREER,
                    'annual_income': 500000,
                    'net_worth': 5000000,
                    'investable_assets': 3000000,
                    'risk_tolerance': 0.6,
                    'time_horizon': 20,
                    'liquidity_needs': 100000,
                    'tax_bracket': 0.35
                },
                {
                    'user_id': 2,
                    'client_type': ClientType.ULTRA_HIGH_NET_WORTH,
                    'first_name': 'Sarah',
                    'last_name': 'Johnson',
                    'email': 'sarah.johnson@email.com',
                    'phone': '+1-555-0124',
                    'date_of_birth': datetime.utcnow() - timedelta(days=365*55),
                    'risk_profile': RiskProfile.GROWTH,
                    'investment_objective': InvestmentObjective.TOTAL_RETURN,
                    'life_stage': LifeStage.PRE_RETIREMENT,
                    'annual_income': 1000000,
                    'net_worth': 25000000,
                    'investable_assets': 20000000,
                    'risk_tolerance': 0.7,
                    'time_horizon': 15,
                    'liquidity_needs': 500000,
                    'tax_bracket': 0.37
                }
            ]
            
            for client_data in sample_clients:
                await self.create_client(**client_data)
            
            logger.info("Loaded sample clients")
            
        except Exception as e:
            logger.error(f"Error loading sample clients: {e}")
    
    async def _initialize_market_data(self):
        """Initialize market data"""
        try:
            # Initialize benchmark data
            self.benchmark_data = {
                'S&P_500': 4000.0,
                'Bond_Index': 105.0,
                'Cash_Rate': 0.05,
                'Inflation_Rate': 0.03
            }
            
            logger.info("Initialized market data")
            
        except Exception as e:
            logger.error(f"Error initializing market data: {e}")


# Factory function
async def get_wealth_management_service(redis_client: redis.Redis, db_session: Session) -> WealthManagementService:
    """Get Wealth Management Service instance"""
    service = WealthManagementService(redis_client, db_session)
    await service.initialize()
    return service
