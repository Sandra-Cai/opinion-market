"""
Alternative Investments Service
Advanced alternative investments and private markets management
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


class AlternativeInvestmentType(Enum):
    """Alternative investment types"""
    PRIVATE_EQUITY = "private_equity"
    VENTURE_CAPITAL = "venture_capital"
    REAL_ESTATE = "real_estate"
    HEDGE_FUND = "hedge_fund"
    PRIVATE_DEBT = "private_debt"
    INFRASTRUCTURE = "infrastructure"
    COMMODITIES = "commodities"
    ART_COLLECTIBLES = "art_collectibles"
    CRYPTOCURRENCY = "cryptocurrency"
    PRECIOUS_METALS = "precious_metals"
    TIMBER = "timber"
    AGRICULTURE = "agriculture"


class InvestmentStage(Enum):
    """Investment stages"""
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    GROWTH = "growth"
    LATE_STAGE = "late_stage"
    PRE_IPO = "pre_ipo"
    MATURE = "mature"


class RiskLevel(Enum):
    """Risk levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"


class LiquidityProfile(Enum):
    """Liquidity profiles"""
    HIGHLY_LIQUID = "highly_liquid"
    LIQUID = "liquid"
    ILLIQUID = "illiquid"
    VERY_ILLIQUID = "very_illiquid"


@dataclass
class AlternativeInvestment:
    """Alternative investment"""
    investment_id: str
    name: str
    investment_type: AlternativeInvestmentType
    description: str
    manager: str
    inception_date: datetime
    target_size: float
    committed_capital: float
    called_capital: float
    distributed_capital: float
    net_asset_value: float
    currency: str
    risk_level: RiskLevel
    liquidity_profile: LiquidityProfile
    minimum_investment: float
    management_fee: float
    performance_fee: float
    hurdle_rate: float
    preferred_return: float
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class InvestmentCommitment:
    """Investment commitment"""
    commitment_id: str
    user_id: int
    investment_id: str
    commitment_amount: float
    committed_date: datetime
    called_amount: float
    distributed_amount: float
    remaining_commitment: float
    status: str  # 'active', 'fully_called', 'fully_distributed'
    created_at: datetime
    last_updated: datetime


@dataclass
class CapitalCall:
    """Capital call"""
    call_id: str
    investment_id: str
    call_number: int
    call_date: datetime
    due_date: datetime
    call_amount: float
    call_percentage: float
    purpose: str
    status: str  # 'pending', 'paid', 'overdue', 'cancelled'
    paid_amount: float
    outstanding_amount: float
    created_at: datetime


@dataclass
class Distribution:
    """Distribution"""
    distribution_id: str
    investment_id: str
    distribution_number: int
    distribution_date: datetime
    distribution_amount: float
    distribution_percentage: float
    distribution_type: str  # 'dividend', 'capital_gain', 'return_of_capital'
    tax_status: str  # 'ordinary', 'capital_gain', 'tax_free'
    created_at: datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    investment_id: str
    period: str
    net_irr: float
    net_multiple: float
    gross_irr: float
    gross_multiple: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    jensen_alpha: float
    treynor_ratio: float
    information_ratio: float
    tracking_error: float
    beta: float
    correlation: float
    created_at: datetime


@dataclass
class Valuation:
    """Investment valuation"""
    valuation_id: str
    investment_id: str
    valuation_date: datetime
    fair_value: float
    cost_basis: float
    unrealized_gain_loss: float
    realized_gain_loss: float
    total_gain_loss: float
    valuation_method: str
    valuation_firm: str
    confidence_level: float
    notes: str
    created_at: datetime


@dataclass
class DueDiligence:
    """Due diligence record"""
    dd_id: str
    investment_id: str
    dd_type: str  # 'financial', 'legal', 'operational', 'environmental', 'social', 'governance'
    dd_date: datetime
    conducted_by: str
    findings: str
    recommendations: str
    risk_assessment: str
    score: float
    status: str  # 'completed', 'in_progress', 'pending'
    created_at: datetime


class AlternativeInvestmentsService:
    """Comprehensive Alternative Investments Service"""
    
    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        
        # Investment management
        self.investments: Dict[str, AlternativeInvestment] = {}
        self.commitments: Dict[str, List[InvestmentCommitment]] = defaultdict(list)
        self.capital_calls: Dict[str, List[CapitalCall]] = defaultdict(list)
        self.distributions: Dict[str, List[Distribution]] = defaultdict(list)
        self.performance_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.valuations: Dict[str, List[Valuation]] = defaultdict(list)
        self.due_diligence: Dict[str, List[DueDiligence]] = defaultdict(list)
        
        # Analytics
        self.portfolio_analytics: Dict[str, Dict[str, Any]] = {}
        self.risk_metrics: Dict[str, Dict[str, float]] = {}
        self.performance_attribution: Dict[str, Dict[str, float]] = {}
        
        # Market data
        self.benchmark_data: Dict[str, pd.DataFrame] = {}
        self.market_indices: Dict[str, float] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize the Alternative Investments Service"""
        logger.info("Initializing Alternative Investments Service")
        
        # Load sample investments
        await self._load_sample_investments()
        
        # Initialize market data
        await self._initialize_market_data()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._update_valuations()),
            asyncio.create_task(self._calculate_performance_metrics()),
            asyncio.create_task(self._process_capital_calls()),
            asyncio.create_task(self._process_distributions()),
            asyncio.create_task(self._update_portfolio_analytics())
        ]
        
        logger.info("Alternative Investments Service initialized successfully")
    
    async def create_investment(self, name: str, investment_type: AlternativeInvestmentType,
                              description: str, manager: str, inception_date: datetime,
                              target_size: float, currency: str = 'USD',
                              risk_level: RiskLevel = RiskLevel.MODERATE,
                              liquidity_profile: LiquidityProfile = LiquidityProfile.ILLIQUID,
                              minimum_investment: float = 1000000.0,
                              management_fee: float = 0.02, performance_fee: float = 0.20,
                              hurdle_rate: float = 0.08, preferred_return: float = 0.08) -> AlternativeInvestment:
        """Create a new alternative investment"""
        try:
            investment_id = f"ALT_{uuid.uuid4().hex[:8]}"
            
            investment = AlternativeInvestment(
                investment_id=investment_id,
                name=name,
                investment_type=investment_type,
                description=description,
                manager=manager,
                inception_date=inception_date,
                target_size=target_size,
                committed_capital=0.0,
                called_capital=0.0,
                distributed_capital=0.0,
                net_asset_value=0.0,
                currency=currency,
                risk_level=risk_level,
                liquidity_profile=liquidity_profile,
                minimum_investment=minimum_investment,
                management_fee=management_fee,
                performance_fee=performance_fee,
                hurdle_rate=hurdle_rate,
                preferred_return=preferred_return,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.investments[investment_id] = investment
            
            logger.info(f"Created alternative investment {investment_id}")
            return investment
            
        except Exception as e:
            logger.error(f"Error creating investment: {e}")
            raise
    
    async def create_commitment(self, user_id: int, investment_id: str,
                              commitment_amount: float) -> InvestmentCommitment:
        """Create an investment commitment"""
        try:
            investment = self.investments.get(investment_id)
            if not investment:
                raise ValueError("Investment not found")
            
            if commitment_amount < investment.minimum_investment:
                raise ValueError(f"Commitment amount below minimum: {investment.minimum_investment}")
            
            commitment_id = f"COMMIT_{uuid.uuid4().hex[:8]}"
            
            commitment = InvestmentCommitment(
                commitment_id=commitment_id,
                user_id=user_id,
                investment_id=investment_id,
                commitment_amount=commitment_amount,
                committed_date=datetime.utcnow(),
                called_amount=0.0,
                distributed_amount=0.0,
                remaining_commitment=commitment_amount,
                status='active',
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow()
            )
            
            self.commitments[user_id].append(commitment)
            
            # Update investment committed capital
            investment.committed_capital += commitment_amount
            investment.last_updated = datetime.utcnow()
            
            logger.info(f"Created commitment {commitment_id}")
            return commitment
            
        except Exception as e:
            logger.error(f"Error creating commitment: {e}")
            raise
    
    async def create_capital_call(self, investment_id: str, call_amount: float,
                                call_percentage: float, purpose: str,
                                due_date: Optional[datetime] = None) -> CapitalCall:
        """Create a capital call"""
        try:
            investment = self.investments.get(investment_id)
            if not investment:
                raise ValueError("Investment not found")
            
            # Get next call number
            existing_calls = self.capital_calls.get(investment_id, [])
            call_number = len(existing_calls) + 1
            
            call_id = f"CALL_{uuid.uuid4().hex[:8]}"
            
            if due_date is None:
                due_date = datetime.utcnow() + timedelta(days=30)
            
            call = CapitalCall(
                call_id=call_id,
                investment_id=investment_id,
                call_number=call_number,
                call_date=datetime.utcnow(),
                due_date=due_date,
                call_amount=call_amount,
                call_percentage=call_percentage,
                purpose=purpose,
                status='pending',
                paid_amount=0.0,
                outstanding_amount=call_amount,
                created_at=datetime.utcnow()
            )
            
            self.capital_calls[investment_id].append(call)
            
            # Update investment called capital
            investment.called_capital += call_amount
            investment.last_updated = datetime.utcnow()
            
            logger.info(f"Created capital call {call_id}")
            return call
            
        except Exception as e:
            logger.error(f"Error creating capital call: {e}")
            raise
    
    async def create_distribution(self, investment_id: str, distribution_amount: float,
                                distribution_percentage: float, distribution_type: str,
                                tax_status: str = 'capital_gain') -> Distribution:
        """Create a distribution"""
        try:
            investment = self.investments.get(investment_id)
            if not investment:
                raise ValueError("Investment not found")
            
            # Get next distribution number
            existing_distributions = self.distributions.get(investment_id, [])
            distribution_number = len(existing_distributions) + 1
            
            distribution_id = f"DIST_{uuid.uuid4().hex[:8]}"
            
            distribution = Distribution(
                distribution_id=distribution_id,
                investment_id=investment_id,
                distribution_number=distribution_number,
                distribution_date=datetime.utcnow(),
                distribution_amount=distribution_amount,
                distribution_percentage=distribution_percentage,
                distribution_type=distribution_type,
                tax_status=tax_status,
                created_at=datetime.utcnow()
            )
            
            self.distributions[investment_id].append(distribution)
            
            # Update investment distributed capital
            investment.distributed_capital += distribution_amount
            investment.last_updated = datetime.utcnow()
            
            logger.info(f"Created distribution {distribution_id}")
            return distribution
            
        except Exception as e:
            logger.error(f"Error creating distribution: {e}")
            raise
    
    async def calculate_performance_metrics(self, investment_id: str,
                                          period: str = 'total') -> PerformanceMetrics:
        """Calculate performance metrics for an investment"""
        try:
            investment = self.investments.get(investment_id)
            if not investment:
                raise ValueError("Investment not found")
            
            # Get capital calls and distributions
            calls = self.capital_calls.get(investment_id, [])
            distributions = self.distributions.get(investment_id, [])
            
            if not calls and not distributions:
                # No activity yet
                return PerformanceMetrics(
                    investment_id=investment_id,
                    period=period,
                    net_irr=0.0,
                    net_multiple=1.0,
                    gross_irr=0.0,
                    gross_multiple=1.0,
                    total_return=0.0,
                    annualized_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    calmar_ratio=0.0,
                    jensen_alpha=0.0,
                    treynor_ratio=0.0,
                    information_ratio=0.0,
                    tracking_error=0.0,
                    beta=0.0,
                    correlation=0.0,
                    created_at=datetime.utcnow()
                )
            
            # Calculate cash flows
            cash_flows = []
            dates = []
            
            # Add capital calls (negative cash flows)
            for call in calls:
                if call.status == 'paid':
                    cash_flows.append(-call.paid_amount)
                    dates.append(call.call_date)
            
            # Add distributions (positive cash flows)
            for dist in distributions:
                cash_flows.append(dist.distribution_amount)
                dates.append(dist.distribution_date)
            
            # Add current NAV (positive cash flow)
            if investment.net_asset_value > 0:
                cash_flows.append(investment.net_asset_value)
                dates.append(datetime.utcnow())
            
            # Calculate metrics
            total_invested = sum(call.paid_amount for call in calls if call.status == 'paid')
            total_distributed = sum(dist.distribution_amount for dist in distributions)
            current_nav = investment.net_asset_value
            
            # Net IRR calculation
            net_irr = await self._calculate_irr(cash_flows, dates)
            
            # Net multiple
            net_multiple = (total_distributed + current_nav) / total_invested if total_invested > 0 else 1.0
            
            # Gross metrics (before fees)
            gross_irr = net_irr * 1.1  # Simplified assumption
            gross_multiple = net_multiple * 1.1  # Simplified assumption
            
            # Other metrics
            total_return = net_multiple - 1.0
            years = (datetime.utcnow() - investment.inception_date).days / 365.0
            annualized_return = (net_multiple ** (1/years)) - 1 if years > 0 else 0.0
            
            # Risk metrics (simplified)
            volatility = 0.15  # 15% annual volatility assumption
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
            max_drawdown = 0.20  # 20% max drawdown assumption
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
            
            # Benchmark comparison
            benchmark_return = await self._get_benchmark_return(investment.investment_type, years)
            jensen_alpha = annualized_return - benchmark_return
            beta = 0.8  # Beta assumption
            correlation = 0.6  # Correlation assumption
            tracking_error = 0.10  # 10% tracking error
            information_ratio = jensen_alpha / tracking_error if tracking_error > 0 else 0.0
            treynor_ratio = annualized_return / beta if beta > 0 else 0.0
            
            metrics = PerformanceMetrics(
                investment_id=investment_id,
                period=period,
                net_irr=net_irr,
                net_multiple=net_multiple,
                gross_irr=gross_irr,
                gross_multiple=gross_multiple,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                jensen_alpha=jensen_alpha,
                treynor_ratio=treynor_ratio,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                beta=beta,
                correlation=correlation,
                created_at=datetime.utcnow()
            )
            
            self.performance_metrics[investment_id].append(metrics)
            
            logger.info(f"Calculated performance metrics for {investment_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    async def create_valuation(self, investment_id: str, fair_value: float,
                             valuation_method: str, valuation_firm: str,
                             confidence_level: float = 0.8,
                             notes: str = "") -> Valuation:
        """Create a valuation"""
        try:
            investment = self.investments.get(investment_id)
            if not investment:
                raise ValueError("Investment not found")
            
            valuation_id = f"VAL_{uuid.uuid4().hex[:8]}"
            
            # Calculate cost basis and gains/losses
            calls = self.capital_calls.get(investment_id, [])
            distributions = self.distributions.get(investment_id, [])
            
            cost_basis = sum(call.paid_amount for call in calls if call.status == 'paid')
            realized_gain_loss = sum(dist.distribution_amount for dist in distributions) - cost_basis
            unrealized_gain_loss = fair_value - (cost_basis - sum(dist.distribution_amount for dist in distributions))
            total_gain_loss = realized_gain_loss + unrealized_gain_loss
            
            valuation = Valuation(
                valuation_id=valuation_id,
                investment_id=investment_id,
                valuation_date=datetime.utcnow(),
                fair_value=fair_value,
                cost_basis=cost_basis,
                unrealized_gain_loss=unrealized_gain_loss,
                realized_gain_loss=realized_gain_loss,
                total_gain_loss=total_gain_loss,
                valuation_method=valuation_method,
                valuation_firm=valuation_firm,
                confidence_level=confidence_level,
                notes=notes,
                created_at=datetime.utcnow()
            )
            
            self.valuations[investment_id].append(valuation)
            
            # Update investment NAV
            investment.net_asset_value = fair_value
            investment.last_updated = datetime.utcnow()
            
            logger.info(f"Created valuation {valuation_id}")
            return valuation
            
        except Exception as e:
            logger.error(f"Error creating valuation: {e}")
            raise
    
    async def create_due_diligence(self, investment_id: str, dd_type: str,
                                 conducted_by: str, findings: str,
                                 recommendations: str, risk_assessment: str,
                                 score: float) -> DueDiligence:
        """Create due diligence record"""
        try:
            investment = self.investments.get(investment_id)
            if not investment:
                raise ValueError("Investment not found")
            
            dd_id = f"DD_{uuid.uuid4().hex[:8]}"
            
            due_diligence = DueDiligence(
                dd_id=dd_id,
                investment_id=investment_id,
                dd_type=dd_type,
                dd_date=datetime.utcnow(),
                conducted_by=conducted_by,
                findings=findings,
                recommendations=recommendations,
                risk_assessment=risk_assessment,
                score=score,
                status='completed',
                created_at=datetime.utcnow()
            )
            
            self.due_diligence[investment_id].append(due_diligence)
            
            logger.info(f"Created due diligence {dd_id}")
            return due_diligence
            
        except Exception as e:
            logger.error(f"Error creating due diligence: {e}")
            raise
    
    async def get_portfolio_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get portfolio analytics for a user"""
        try:
            user_commitments = self.commitments.get(user_id, [])
            if not user_commitments:
                return {'error': 'No commitments found'}
            
            # Calculate portfolio metrics
            total_committed = sum(c.commitment_amount for c in user_commitments)
            total_called = sum(c.called_amount for c in user_commitments)
            total_distributed = sum(c.distributed_amount for c in user_commitments)
            total_nav = sum(self.investments[c.investment_id].net_asset_value for c in user_commitments)
            
            # Calculate portfolio performance
            portfolio_irr = await self._calculate_portfolio_irr(user_commitments)
            portfolio_multiple = (total_distributed + total_nav) / total_called if total_called > 0 else 1.0
            
            # Asset allocation
            asset_allocation = {}
            for commitment in user_commitments:
                investment = self.investments[commitment.investment_id]
                investment_type = investment.investment_type.value
                if investment_type not in asset_allocation:
                    asset_allocation[investment_type] = 0
                asset_allocation[investment_type] += commitment.commitment_amount
            
            # Risk metrics
            risk_metrics = await self._calculate_portfolio_risk_metrics(user_commitments)
            
            # Performance attribution
            performance_attribution = await self._calculate_performance_attribution(user_commitments)
            
            analytics = {
                'user_id': user_id,
                'total_committed': total_committed,
                'total_called': total_called,
                'total_distributed': total_distributed,
                'total_nav': total_nav,
                'portfolio_irr': portfolio_irr,
                'portfolio_multiple': portfolio_multiple,
                'asset_allocation': asset_allocation,
                'risk_metrics': risk_metrics,
                'performance_attribution': performance_attribution,
                'number_of_investments': len(user_commitments),
                'timestamp': datetime.utcnow()
            }
            
            self.portfolio_analytics[user_id] = analytics
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting portfolio analytics: {e}")
            return {'error': str(e)}
    
    # Background tasks
    async def _update_valuations(self):
        """Update valuations periodically"""
        while True:
            try:
                for investment_id, investment in self.investments.items():
                    if investment.is_active:
                        await self._update_investment_valuation(investment)
                
                await asyncio.sleep(86400)  # Update daily
                
            except Exception as e:
                logger.error(f"Error updating valuations: {e}")
                await asyncio.sleep(172800)
    
    async def _update_investment_valuation(self, investment: AlternativeInvestment):
        """Update valuation for an investment"""
        try:
            # Simulate valuation update
            current_nav = investment.net_asset_value
            if current_nav == 0:
                # Initial valuation
                new_nav = investment.called_capital * np.random.uniform(0.8, 1.2)
            else:
                # Update existing valuation
                change = np.random.normal(0, 0.02)  # 2% daily volatility
                new_nav = current_nav * (1 + change)
            
            # Create valuation record
            await self.create_valuation(
                investment.investment_id,
                new_nav,
                'mark_to_market',
                'Internal Valuation',
                0.8,
                'Automated valuation update'
            )
            
        except Exception as e:
            logger.error(f"Error updating investment valuation: {e}")
    
    async def _calculate_performance_metrics(self):
        """Calculate performance metrics for all investments"""
        while True:
            try:
                for investment_id in self.investments.keys():
                    await self.calculate_performance_metrics(investment_id)
                
                await asyncio.sleep(3600)  # Calculate every hour
                
            except Exception as e:
                logger.error(f"Error calculating performance metrics: {e}")
                await asyncio.sleep(7200)
    
    async def _process_capital_calls(self):
        """Process capital calls"""
        while True:
            try:
                for investment_id, calls in self.capital_calls.items():
                    for call in calls:
                        if call.status == 'pending':
                            await self._process_capital_call(call)
                
                await asyncio.sleep(3600)  # Process every hour
                
            except Exception as e:
                logger.error(f"Error processing capital calls: {e}")
                await asyncio.sleep(7200)
    
    async def _process_capital_call(self, call: CapitalCall):
        """Process a capital call"""
        try:
            # Simulate capital call processing
            if datetime.utcnow() > call.due_date:
                # Mark as overdue
                call.status = 'overdue'
            else:
                # Simulate payment
                payment_probability = 0.95  # 95% payment rate
                if np.random.random() < payment_probability:
                    call.paid_amount = call.call_amount
                    call.outstanding_amount = 0
                    call.status = 'paid'
                    
                    # Update commitments
                    for user_id, commitments in self.commitments.items():
                        for commitment in commitments:
                            if commitment.investment_id == call.investment_id:
                                commitment.called_amount += call.call_amount * (commitment.commitment_amount / self.investments[call.investment_id].committed_capital)
                                commitment.remaining_commitment = commitment.commitment_amount - commitment.called_amount
                                commitment.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error processing capital call: {e}")
    
    async def _process_distributions(self):
        """Process distributions"""
        while True:
            try:
                # Simulate distribution processing
                for investment_id, investment in self.investments.items():
                    if investment.is_active and investment.net_asset_value > 0:
                        # Random distribution probability
                        if np.random.random() < 0.01:  # 1% daily probability
                            distribution_amount = investment.net_asset_value * np.random.uniform(0.05, 0.20)
                            await self.create_distribution(
                                investment_id,
                                distribution_amount,
                                0.1,  # 10% distribution
                                'dividend',
                                'capital_gain'
                            )
                
                await asyncio.sleep(86400)  # Process daily
                
            except Exception as e:
                logger.error(f"Error processing distributions: {e}")
                await asyncio.sleep(172800)
    
    async def _update_portfolio_analytics(self):
        """Update portfolio analytics"""
        while True:
            try:
                for user_id in self.commitments.keys():
                    await self.get_portfolio_analytics(user_id)
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error updating portfolio analytics: {e}")
                await asyncio.sleep(7200)
    
    # Helper methods
    async def _calculate_irr(self, cash_flows: List[float], dates: List[datetime]) -> float:
        """Calculate Internal Rate of Return"""
        try:
            if len(cash_flows) < 2:
                return 0.0
            
            # Convert dates to years from first date
            first_date = dates[0]
            years = [(date - first_date).days / 365.0 for date in dates]
            
            # Use numpy's IRR calculation
            try:
                irr = np.irr(cash_flows)
                return irr if not np.isnan(irr) else 0.0
            except:
                # Fallback to simple calculation
                total_return = sum(cash_flows)
                total_invested = sum(cf for cf in cash_flows if cf < 0)
                if total_invested != 0:
                    return (total_return / abs(total_invested)) - 1
                return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating IRR: {e}")
            return 0.0
    
    async def _get_benchmark_return(self, investment_type: AlternativeInvestmentType, years: float) -> float:
        """Get benchmark return for investment type"""
        try:
            # Simplified benchmark returns
            benchmark_returns = {
                AlternativeInvestmentType.PRIVATE_EQUITY: 0.12,
                AlternativeInvestmentType.VENTURE_CAPITAL: 0.15,
                AlternativeInvestmentType.REAL_ESTATE: 0.08,
                AlternativeInvestmentType.HEDGE_FUND: 0.10,
                AlternativeInvestmentType.PRIVATE_DEBT: 0.07,
                AlternativeInvestmentType.INFRASTRUCTURE: 0.09,
                AlternativeInvestmentType.COMMODITIES: 0.06,
                AlternativeInvestmentType.ART_COLLECTIBLES: 0.05,
                AlternativeInvestmentType.CRYPTOCURRENCY: 0.20,
                AlternativeInvestmentType.PRECIOUS_METALS: 0.04,
                AlternativeInvestmentType.TIMBER: 0.08,
                AlternativeInvestmentType.AGRICULTURE: 0.07
            }
            
            base_return = benchmark_returns.get(investment_type, 0.10)
            return base_return * years
            
        except Exception as e:
            logger.error(f"Error getting benchmark return: {e}")
            return 0.10 * years
    
    async def _calculate_portfolio_irr(self, commitments: List[InvestmentCommitment]) -> float:
        """Calculate portfolio IRR"""
        try:
            all_cash_flows = []
            all_dates = []
            
            for commitment in commitments:
                investment = self.investments[commitment.investment_id]
                
                # Add capital calls
                calls = self.capital_calls.get(commitment.investment_id, [])
                for call in calls:
                    if call.status == 'paid':
                        all_cash_flows.append(-call.paid_amount * (commitment.commitment_amount / investment.committed_capital))
                        all_dates.append(call.call_date)
                
                # Add distributions
                distributions = self.distributions.get(commitment.investment_id, [])
                for dist in distributions:
                    all_cash_flows.append(dist.distribution_amount * (commitment.commitment_amount / investment.committed_capital))
                    all_dates.append(dist.distribution_date)
                
                # Add current NAV
                if investment.net_asset_value > 0:
                    all_cash_flows.append(investment.net_asset_value * (commitment.commitment_amount / investment.committed_capital))
                    all_dates.append(datetime.utcnow())
            
            return await self._calculate_irr(all_cash_flows, all_dates)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio IRR: {e}")
            return 0.0
    
    async def _calculate_portfolio_risk_metrics(self, commitments: List[InvestmentCommitment]) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        try:
            # Simplified risk metrics
            total_committed = sum(c.commitment_amount for c in commitments)
            
            # Concentration risk
            max_commitment = max(c.commitment_amount for c in commitments) if commitments else 0
            concentration_risk = max_commitment / total_committed if total_committed > 0 else 0
            
            # Diversification
            unique_types = len(set(self.investments[c.investment_id].investment_type for c in commitments))
            diversification_score = unique_types / len(AlternativeInvestmentType)
            
            # Liquidity risk
            illiquid_commitments = sum(c.commitment_amount for c in commitments 
                                    if self.investments[c.investment_id].liquidity_profile in 
                                    [LiquidityProfile.ILLIQUID, LiquidityProfile.VERY_ILLIQUID])
            liquidity_risk = illiquid_commitments / total_committed if total_committed > 0 else 0
            
            return {
                'concentration_risk': concentration_risk,
                'diversification_score': diversification_score,
                'liquidity_risk': liquidity_risk,
                'number_of_investments': len(commitments),
                'unique_asset_types': unique_types
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {}
    
    async def _calculate_performance_attribution(self, commitments: List[InvestmentCommitment]) -> Dict[str, float]:
        """Calculate performance attribution"""
        try:
            attribution = {}
            total_committed = sum(c.commitment_amount for c in commitments)
            
            for commitment in commitments:
                investment = self.investments[commitment.investment_id]
                investment_type = investment.investment_type.value
                
                if investment_type not in attribution:
                    attribution[investment_type] = 0
                
                # Calculate contribution
                weight = commitment.commitment_amount / total_committed if total_committed > 0 else 0
                performance = await self.calculate_performance_metrics(commitment.investment_id)
                contribution = weight * performance.annualized_return
                attribution[investment_type] += contribution
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error calculating performance attribution: {e}")
            return {}
    
    async def _load_sample_investments(self):
        """Load sample alternative investments"""
        try:
            sample_investments = [
                {
                    'name': 'Tech Growth Fund III',
                    'investment_type': AlternativeInvestmentType.PRIVATE_EQUITY,
                    'description': 'Growth equity fund focused on technology companies',
                    'manager': 'Tech Capital Partners',
                    'inception_date': datetime.utcnow() - timedelta(days=365*2),
                    'target_size': 500000000,
                    'risk_level': RiskLevel.AGGRESSIVE,
                    'liquidity_profile': LiquidityProfile.ILLIQUID,
                    'minimum_investment': 5000000,
                    'management_fee': 0.02,
                    'performance_fee': 0.20,
                    'hurdle_rate': 0.08
                },
                {
                    'name': 'Venture Capital Fund V',
                    'investment_type': AlternativeInvestmentType.VENTURE_CAPITAL,
                    'description': 'Early-stage venture capital fund',
                    'manager': 'Innovation Ventures',
                    'inception_date': datetime.utcnow() - timedelta(days=365*3),
                    'target_size': 200000000,
                    'risk_level': RiskLevel.SPECULATIVE,
                    'liquidity_profile': LiquidityProfile.VERY_ILLIQUID,
                    'minimum_investment': 2000000,
                    'management_fee': 0.025,
                    'performance_fee': 0.25,
                    'hurdle_rate': 0.10
                },
                {
                    'name': 'Real Estate Income Fund',
                    'investment_type': AlternativeInvestmentType.REAL_ESTATE,
                    'description': 'Commercial real estate income fund',
                    'manager': 'Property Partners',
                    'inception_date': datetime.utcnow() - timedelta(days=365*4),
                    'target_size': 1000000000,
                    'risk_level': RiskLevel.MODERATE,
                    'liquidity_profile': LiquidityProfile.ILLIQUID,
                    'minimum_investment': 10000000,
                    'management_fee': 0.015,
                    'performance_fee': 0.15,
                    'hurdle_rate': 0.06
                },
                {
                    'name': 'Infrastructure Fund II',
                    'investment_type': AlternativeInvestmentType.INFRASTRUCTURE,
                    'description': 'Core infrastructure investments',
                    'manager': 'Infrastructure Capital',
                    'inception_date': datetime.utcnow() - timedelta(days=365*5),
                    'target_size': 2000000000,
                    'risk_level': RiskLevel.CONSERVATIVE,
                    'liquidity_profile': LiquidityProfile.ILLIQUID,
                    'minimum_investment': 25000000,
                    'management_fee': 0.012,
                    'performance_fee': 0.12,
                    'hurdle_rate': 0.05
                }
            ]
            
            for investment_data in sample_investments:
                await self.create_investment(**investment_data)
            
            logger.info("Loaded sample alternative investments")
            
        except Exception as e:
            logger.error(f"Error loading sample investments: {e}")
    
    async def _initialize_market_data(self):
        """Initialize market data"""
        try:
            # Initialize benchmark data
            self.market_indices = {
                'S&P_500': 4000.0,
                'NASDAQ': 12000.0,
                'REIT_Index': 1000.0,
                'Commodity_Index': 200.0,
                'Bond_Index': 105.0
            }
            
            logger.info("Initialized market data")
            
        except Exception as e:
            logger.error(f"Error initializing market data: {e}")


# Factory function
async def get_alternative_investments_service(redis_client: redis.Redis, db_session: Session) -> AlternativeInvestmentsService:
    """Get Alternative Investments Service instance"""
    service = AlternativeInvestmentsService(redis_client, db_session)
    await service.initialize()
    return service
