"""
Private Markets Service
Advanced private markets and direct investment management
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

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class PrivateMarketType(Enum):
    """Private market types"""

    DIRECT_INVESTMENT = "direct_investment"
    CO_INVESTMENT = "co_investment"
    SECONDARY = "secondary"
    PRIMARIES = "primaries"
    FUND_OF_FUNDS = "fund_of_funds"
    SPV = "spv"
    JOINT_VENTURE = "joint_venture"
    STRATEGIC_PARTNERSHIP = "strategic_partnership"


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
    TURNAROUND = "turnaround"
    DISTRESSED = "distressed"


class DealStatus(Enum):
    """Deal status"""

    PIPELINE = "pipeline"
    UNDER_REVIEW = "under_review"
    DUE_DILIGENCE = "due_diligence"
    NEGOTIATION = "negotiation"
    APPROVED = "approved"
    CLOSED = "closed"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class CompanyStage(Enum):
    """Company stages"""

    STARTUP = "startup"
    EARLY_STAGE = "early_stage"
    GROWTH_STAGE = "growth_stage"
    MATURE = "mature"
    DECLINING = "declining"
    TURNAROUND = "turnaround"
    DISTRESSED = "distressed"


@dataclass
class PrivateCompany:
    """Private company"""

    company_id: str
    name: str
    industry: str
    sector: str
    stage: CompanyStage
    description: str
    headquarters: str
    founded_date: datetime
    employees: int
    revenue: float
    ebitda: float
    valuation: float
    last_funding_round: str
    total_funding: float
    investors: List[str]
    website: str
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class InvestmentOpportunity:
    """Investment opportunity"""

    opportunity_id: str
    company_id: str
    opportunity_type: PrivateMarketType
    investment_stage: InvestmentStage
    deal_size: float
    minimum_investment: float
    maximum_investment: float
    valuation: float
    ownership_percentage: float
    deal_status: DealStatus
    expected_close_date: datetime
    lead_investor: str
    co_investors: List[str]
    deal_terms: Dict[str, Any]
    due_diligence_status: str
    risk_assessment: str
    expected_return: float
    holding_period: int
    exit_strategy: str
    created_at: datetime
    last_updated: datetime


@dataclass
class DirectInvestment:
    """Direct investment"""

    investment_id: str
    user_id: int
    opportunity_id: str
    investment_amount: float
    ownership_percentage: float
    investment_date: datetime
    cost_basis: float
    current_value: float
    unrealized_gain_loss: float
    realized_gain_loss: float
    total_gain_loss: float
    irr: float
    multiple: float
    status: str  # 'active', 'exited', 'written_off'
    exit_date: Optional[datetime]
    exit_value: Optional[float]
    created_at: datetime
    last_updated: datetime


@dataclass
class DealFlow:
    """Deal flow"""

    deal_id: str
    company_id: str
    opportunity_id: str
    source: str
    deal_size: float
    stage: InvestmentStage
    status: DealStatus
    probability: float
    expected_close_date: datetime
    lead_investor: str
    co_investors: List[str]
    deal_terms: Dict[str, Any]
    notes: str
    created_at: datetime
    last_updated: datetime


@dataclass
class PortfolioCompany:
    """Portfolio company"""

    company_id: str
    investment_id: str
    user_id: int
    investment_date: datetime
    investment_amount: float
    ownership_percentage: float
    board_seat: bool
    board_observer: bool
    voting_rights: bool
    anti_dilution: bool
    liquidation_preference: bool
    drag_along: bool
    tag_along: bool
    information_rights: bool
    current_value: float
    unrealized_gain_loss: float
    irr: float
    multiple: float
    status: str
    created_at: datetime
    last_updated: datetime


@dataclass
class ValuationUpdate:
    """Valuation update"""

    valuation_id: str
    company_id: str
    valuation_date: datetime
    valuation_method: str
    fair_value: float
    previous_value: float
    change_percentage: float
    valuation_firm: str
    confidence_level: float
    key_assumptions: List[str]
    risk_factors: List[str]
    notes: str
    created_at: datetime


@dataclass
class ExitEvent:
    """Exit event"""

    exit_id: str
    company_id: str
    investment_id: str
    exit_type: str  # 'ipo', 'acquisition', 'merger', 'liquidation', 'buyback'
    exit_date: datetime
    exit_value: float
    proceeds: float
    multiple: float
    irr: float
    holding_period: int
    exit_valuation: float
    acquirer: str
    exit_terms: Dict[str, Any]
    created_at: datetime


class PrivateMarketsService:
    """Comprehensive Private Markets Service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session

        # Private markets management
        self.companies: Dict[str, PrivateCompany] = {}
        self.opportunities: Dict[str, InvestmentOpportunity] = {}
        self.direct_investments: Dict[str, List[DirectInvestment]] = defaultdict(list)
        self.deal_flow: Dict[str, List[DealFlow]] = defaultdict(list)
        self.portfolio_companies: Dict[str, List[PortfolioCompany]] = defaultdict(list)
        self.valuations: Dict[str, List[ValuationUpdate]] = defaultdict(list)
        self.exit_events: Dict[str, List[ExitEvent]] = defaultdict(list)

        # Analytics
        self.portfolio_analytics: Dict[str, Dict[str, Any]] = {}
        self.deal_analytics: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}

        # Market data
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.industry_metrics: Dict[str, Dict[str, float]] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize the Private Markets Service"""
        logger.info("Initializing Private Markets Service")

        # Load sample data
        await self._load_sample_companies()
        await self._load_sample_opportunities()

        # Initialize market data
        await self._initialize_market_data()

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._update_valuations()),
            asyncio.create_task(self._process_deal_flow()),
            asyncio.create_task(self._update_portfolio_analytics()),
            asyncio.create_task(self._monitor_exit_opportunities()),
            asyncio.create_task(self._update_performance_metrics()),
        ]

        logger.info("Private Markets Service initialized successfully")

    async def create_company(
        self,
        name: str,
        industry: str,
        sector: str,
        stage: CompanyStage,
        description: str,
        headquarters: str,
        founded_date: datetime,
        employees: int,
        revenue: float,
        ebitda: float,
        valuation: float,
        last_funding_round: str,
        total_funding: float,
        investors: List[str],
        website: str = "",
    ) -> PrivateCompany:
        """Create a new private company"""
        try:
            company_id = f"COMP_{uuid.uuid4().hex[:8]}"

            company = PrivateCompany(
                company_id=company_id,
                name=name,
                industry=industry,
                sector=sector,
                stage=stage,
                description=description,
                headquarters=headquarters,
                founded_date=founded_date,
                employees=employees,
                revenue=revenue,
                ebitda=ebitda,
                valuation=valuation,
                last_funding_round=last_funding_round,
                total_funding=total_funding,
                investors=investors,
                website=website,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.companies[company_id] = company

            logger.info(f"Created company {company_id}")
            return company

        except Exception as e:
            logger.error(f"Error creating company: {e}")
            raise

    async def create_investment_opportunity(
        self,
        company_id: str,
        opportunity_type: PrivateMarketType,
        investment_stage: InvestmentStage,
        deal_size: float,
        minimum_investment: float,
        maximum_investment: float,
        valuation: float,
        ownership_percentage: float,
        expected_close_date: datetime,
        lead_investor: str,
        co_investors: List[str],
        deal_terms: Dict[str, Any],
        expected_return: float,
        holding_period: int,
        exit_strategy: str,
    ) -> InvestmentOpportunity:
        """Create an investment opportunity"""
        try:
            company = self.companies.get(company_id)
            if not company:
                raise ValueError("Company not found")

            opportunity_id = f"OPP_{uuid.uuid4().hex[:8]}"

            opportunity = InvestmentOpportunity(
                opportunity_id=opportunity_id,
                company_id=company_id,
                opportunity_type=opportunity_type,
                investment_stage=investment_stage,
                deal_size=deal_size,
                minimum_investment=minimum_investment,
                maximum_investment=maximum_investment,
                valuation=valuation,
                ownership_percentage=ownership_percentage,
                deal_status=DealStatus.PIPELINE,
                expected_close_date=expected_close_date,
                lead_investor=lead_investor,
                co_investors=co_investors,
                deal_terms=deal_terms,
                due_diligence_status="pending",
                risk_assessment="medium",
                expected_return=expected_return,
                holding_period=holding_period,
                exit_strategy=exit_strategy,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.opportunities[opportunity_id] = opportunity

            logger.info(f"Created investment opportunity {opportunity_id}")
            return opportunity

        except Exception as e:
            logger.error(f"Error creating investment opportunity: {e}")
            raise

    async def make_direct_investment(
        self, user_id: int, opportunity_id: str, investment_amount: float
    ) -> DirectInvestment:
        """Make a direct investment"""
        try:
            opportunity = self.opportunities.get(opportunity_id)
            if not opportunity:
                raise ValueError("Investment opportunity not found")

            if investment_amount < opportunity.minimum_investment:
                raise ValueError(
                    f"Investment amount below minimum: {opportunity.minimum_investment}"
                )

            if investment_amount > opportunity.maximum_investment:
                raise ValueError(
                    f"Investment amount above maximum: {opportunity.maximum_investment}"
                )

            investment_id = f"INV_{uuid.uuid4().hex[:8]}"

            # Calculate ownership percentage
            ownership_percentage = (investment_amount / opportunity.valuation) * 100

            investment = DirectInvestment(
                investment_id=investment_id,
                user_id=user_id,
                opportunity_id=opportunity_id,
                investment_amount=investment_amount,
                ownership_percentage=ownership_percentage,
                investment_date=datetime.utcnow(),
                cost_basis=investment_amount,
                current_value=investment_amount,  # Initial value
                unrealized_gain_loss=0.0,
                realized_gain_loss=0.0,
                total_gain_loss=0.0,
                irr=0.0,
                multiple=1.0,
                status="active",
                exit_date=None,
                exit_value=None,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.direct_investments[user_id].append(investment)

            # Create portfolio company record
            await self._create_portfolio_company(user_id, investment, opportunity)

            # Update opportunity status
            opportunity.deal_status = DealStatus.CLOSED
            opportunity.last_updated = datetime.utcnow()

            logger.info(f"Made direct investment {investment_id}")
            return investment

        except Exception as e:
            logger.error(f"Error making direct investment: {e}")
            raise

    async def create_deal_flow(
        self,
        company_id: str,
        source: str,
        deal_size: float,
        stage: InvestmentStage,
        probability: float,
        expected_close_date: datetime,
        lead_investor: str,
        co_investors: List[str],
        deal_terms: Dict[str, Any],
        notes: str = "",
    ) -> DealFlow:
        """Create deal flow entry"""
        try:
            company = self.companies.get(company_id)
            if not company:
                raise ValueError("Company not found")

            deal_id = f"DEAL_{uuid.uuid4().hex[:8]}"

            deal = DealFlow(
                deal_id=deal_id,
                company_id=company_id,
                opportunity_id="",  # Will be set when opportunity is created
                source=source,
                deal_size=deal_size,
                stage=stage,
                status=DealStatus.PIPELINE,
                probability=probability,
                expected_close_date=expected_close_date,
                lead_investor=lead_investor,
                co_investors=co_investors,
                deal_terms=deal_terms,
                notes=notes,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.deal_flow[company_id].append(deal)

            logger.info(f"Created deal flow {deal_id}")
            return deal

        except Exception as e:
            logger.error(f"Error creating deal flow: {e}")
            raise

    async def update_valuation(
        self,
        company_id: str,
        valuation_method: str,
        fair_value: float,
        valuation_firm: str,
        confidence_level: float = 0.8,
        key_assumptions: List[str] = None,
        risk_factors: List[str] = None,
        notes: str = "",
    ) -> ValuationUpdate:
        """Update company valuation"""
        try:
            company = self.companies.get(company_id)
            if not company:
                raise ValueError("Company not found")

            valuation_id = f"VAL_{uuid.uuid4().hex[:8]}"

            # Calculate change percentage
            previous_value = company.valuation
            change_percentage = (
                ((fair_value - previous_value) / previous_value) * 100
                if previous_value > 0
                else 0
            )

            valuation = ValuationUpdate(
                valuation_id=valuation_id,
                company_id=company_id,
                valuation_date=datetime.utcnow(),
                valuation_method=valuation_method,
                fair_value=fair_value,
                previous_value=previous_value,
                change_percentage=change_percentage,
                valuation_firm=valuation_firm,
                confidence_level=confidence_level,
                key_assumptions=key_assumptions or [],
                risk_factors=risk_factors or [],
                notes=notes,
                created_at=datetime.utcnow(),
            )

            self.valuations[company_id].append(valuation)

            # Update company valuation
            company.valuation = fair_value
            company.last_updated = datetime.utcnow()

            # Update portfolio company values
            await self._update_portfolio_valuations(company_id, fair_value)

            logger.info(f"Updated valuation {valuation_id}")
            return valuation

        except Exception as e:
            logger.error(f"Error updating valuation: {e}")
            raise

    async def create_exit_event(
        self,
        company_id: str,
        investment_id: str,
        exit_type: str,
        exit_date: datetime,
        exit_value: float,
        proceeds: float,
        acquirer: str = "",
        exit_terms: Dict[str, Any] = None,
    ) -> ExitEvent:
        """Create exit event"""
        try:
            company = self.companies.get(company_id)
            if not company:
                raise ValueError("Company not found")

            # Find investment
            investment = None
            for user_id, investments in self.direct_investments.items():
                for inv in investments:
                    if inv.investment_id == investment_id:
                        investment = inv
                        break
                if investment:
                    break

            if not investment:
                raise ValueError("Investment not found")

            exit_id = f"EXIT_{uuid.uuid4().hex[:8]}"

            # Calculate metrics
            holding_period = (exit_date - investment.investment_date).days / 365.0
            multiple = (
                proceeds / investment.cost_basis if investment.cost_basis > 0 else 1.0
            )
            irr = await self._calculate_irr(
                [-investment.cost_basis, proceeds],
                [investment.investment_date, exit_date],
            )

            exit_event = ExitEvent(
                exit_id=exit_id,
                company_id=company_id,
                investment_id=investment_id,
                exit_type=exit_type,
                exit_date=exit_date,
                exit_value=exit_value,
                proceeds=proceeds,
                multiple=multiple,
                irr=irr,
                holding_period=holding_period,
                exit_valuation=exit_value,
                acquirer=acquirer,
                exit_terms=exit_terms or {},
                created_at=datetime.utcnow(),
            )

            self.exit_events[company_id].append(exit_event)

            # Update investment
            investment.status = "exited"
            investment.exit_date = exit_date
            investment.exit_value = exit_value
            investment.realized_gain_loss = proceeds - investment.cost_basis
            investment.total_gain_loss = investment.realized_gain_loss
            investment.irr = irr
            investment.multiple = multiple
            investment.last_updated = datetime.utcnow()

            logger.info(f"Created exit event {exit_id}")
            return exit_event

        except Exception as e:
            logger.error(f"Error creating exit event: {e}")
            raise

    async def get_portfolio_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get portfolio analytics for a user"""
        try:
            investments = self.direct_investments.get(user_id, [])
            if not investments:
                return {"error": "No investments found"}

            # Calculate portfolio metrics
            total_invested = sum(inv.cost_basis for inv in investments)
            total_current_value = sum(inv.current_value for inv in investments)
            total_realized_gain = sum(inv.realized_gain_loss for inv in investments)
            total_unrealized_gain = sum(inv.unrealized_gain_loss for inv in investments)

            # Calculate portfolio IRR
            portfolio_irr = await self._calculate_portfolio_irr(investments)

            # Calculate portfolio multiple
            portfolio_multiple = (
                (total_current_value + total_realized_gain) / total_invested
                if total_invested > 0
                else 1.0
            )

            # Asset allocation by stage
            stage_allocation = {}
            for investment in investments:
                opportunity = self.opportunities.get(investment.opportunity_id)
                if opportunity:
                    stage = opportunity.investment_stage.value
                    if stage not in stage_allocation:
                        stage_allocation[stage] = 0
                    stage_allocation[stage] += investment.current_value

            # Industry allocation
            industry_allocation = {}
            for investment in investments:
                opportunity = self.opportunities.get(investment.opportunity_id)
                if opportunity:
                    company = self.companies.get(opportunity.company_id)
                    if company:
                        industry = company.industry
                        if industry not in industry_allocation:
                            industry_allocation[industry] = 0
                        industry_allocation[industry] += investment.current_value

            # Performance metrics
            performance_metrics = await self._calculate_performance_metrics(investments)

            analytics = {
                "user_id": user_id,
                "total_invested": total_invested,
                "total_current_value": total_current_value,
                "total_realized_gain": total_realized_gain,
                "total_unrealized_gain": total_unrealized_gain,
                "total_gain_loss": total_realized_gain + total_unrealized_gain,
                "portfolio_irr": portfolio_irr,
                "portfolio_multiple": portfolio_multiple,
                "stage_allocation": stage_allocation,
                "industry_allocation": industry_allocation,
                "performance_metrics": performance_metrics,
                "number_of_investments": len(investments),
                "active_investments": len(
                    [inv for inv in investments if inv.status == "active"]
                ),
                "exited_investments": len(
                    [inv for inv in investments if inv.status == "exited"]
                ),
                "timestamp": datetime.utcnow(),
            }

            self.portfolio_analytics[user_id] = analytics

            return analytics

        except Exception as e:
            logger.error(f"Error getting portfolio analytics: {e}")
            return {"error": str(e)}

    # Background tasks
    async def _update_valuations(self):
        """Update valuations periodically"""
        while True:
            try:
                for company_id, company in self.companies.items():
                    if company.is_active:
                        await self._update_company_valuation(company)

                await asyncio.sleep(86400)  # Update daily

            except Exception as e:
                logger.error(f"Error updating valuations: {e}")
                await asyncio.sleep(172800)

    async def _update_company_valuation(self, company: PrivateCompany):
        """Update company valuation"""
        try:
            # Simulate valuation update
            current_valuation = company.valuation
            if current_valuation == 0:
                # Initial valuation
                new_valuation = company.revenue * np.random.uniform(
                    2, 8
                )  # Revenue multiple
            else:
                # Update existing valuation
                change = np.random.normal(0, 0.05)  # 5% daily volatility
                new_valuation = current_valuation * (1 + change)

            # Create valuation update
            await self.update_valuation(
                company.company_id,
                "mark_to_market",
                new_valuation,
                "Internal Valuation",
                0.8,
                ["Revenue growth", "Market conditions"],
                ["Competition", "Regulatory risk"],
                "Automated valuation update",
            )

        except Exception as e:
            logger.error(f"Error updating company valuation: {e}")

    async def _process_deal_flow(self):
        """Process deal flow"""
        while True:
            try:
                for company_id, deals in self.deal_flow.items():
                    for deal in deals:
                        if deal.status == DealStatus.PIPELINE:
                            await self._process_deal(deal)

                await asyncio.sleep(3600)  # Process every hour

            except Exception as e:
                logger.error(f"Error processing deal flow: {e}")
                await asyncio.sleep(7200)

    async def _process_deal(self, deal: DealFlow):
        """Process a deal"""
        try:
            # Simulate deal progression
            if np.random.random() < 0.1:  # 10% chance of progression
                if deal.status == DealStatus.PIPELINE:
                    deal.status = DealStatus.UNDER_REVIEW
                elif deal.status == DealStatus.UNDER_REVIEW:
                    deal.status = DealStatus.DUE_DILIGENCE
                elif deal.status == DealStatus.DUE_DILIGENCE:
                    deal.status = DealStatus.NEGOTIATION
                elif deal.status == DealStatus.NEGOTIATION:
                    if np.random.random() < 0.7:  # 70% success rate
                        deal.status = DealStatus.APPROVED
                        # Create investment opportunity
                        await self._create_opportunity_from_deal(deal)
                    else:
                        deal.status = DealStatus.REJECTED

                deal.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error processing deal: {e}")

    async def _create_opportunity_from_deal(self, deal: DealFlow):
        """Create investment opportunity from deal"""
        try:
            opportunity = await self.create_investment_opportunity(
                company_id=deal.company_id,
                opportunity_type=PrivateMarketType.DIRECT_INVESTMENT,
                investment_stage=deal.stage,
                deal_size=deal.deal_size,
                minimum_investment=deal.deal_size * 0.1,  # 10% minimum
                maximum_investment=deal.deal_size * 0.5,  # 50% maximum
                valuation=deal.deal_size * 10,  # 10x deal size valuation
                ownership_percentage=5.0,  # 5% ownership
                expected_close_date=deal.expected_close_date,
                lead_investor=deal.lead_investor,
                co_investors=deal.co_investors,
                deal_terms=deal.deal_terms,
                expected_return=0.25,  # 25% expected return
                holding_period=5,  # 5 year holding period
                exit_strategy="IPO or Acquisition",
            )

            # Update deal with opportunity ID
            deal.opportunity_id = opportunity.opportunity_id
            deal.status = DealStatus.CLOSED

        except Exception as e:
            logger.error(f"Error creating opportunity from deal: {e}")

    async def _update_portfolio_analytics(self):
        """Update portfolio analytics"""
        while True:
            try:
                for user_id in self.direct_investments.keys():
                    await self.get_portfolio_analytics(user_id)

                await asyncio.sleep(3600)  # Update every hour

            except Exception as e:
                logger.error(f"Error updating portfolio analytics: {e}")
                await asyncio.sleep(7200)

    async def _monitor_exit_opportunities(self):
        """Monitor exit opportunities"""
        while True:
            try:
                for company_id, company in self.companies.items():
                    if company.is_active:
                        await self._check_exit_opportunities(company)

                await asyncio.sleep(86400)  # Check daily

            except Exception as e:
                logger.error(f"Error monitoring exit opportunities: {e}")
                await asyncio.sleep(172800)

    async def _check_exit_opportunities(self, company: PrivateCompany):
        """Check for exit opportunities"""
        try:
            # Simulate exit opportunity
            if np.random.random() < 0.01:  # 1% daily probability
                # Find investments in this company
                for user_id, investments in self.direct_investments.items():
                    for investment in investments:
                        if investment.status == "active":
                            opportunity = self.opportunities.get(
                                investment.opportunity_id
                            )
                            if (
                                opportunity
                                and opportunity.company_id == company.company_id
                            ):
                                # Simulate exit
                                exit_value = (
                                    investment.current_value
                                    * np.random.uniform(1.5, 3.0)
                                )
                                proceeds = (
                                    exit_value * investment.ownership_percentage / 100
                                )

                                await self.create_exit_event(
                                    company.company_id,
                                    investment.investment_id,
                                    "acquisition",
                                    datetime.utcnow(),
                                    exit_value,
                                    proceeds,
                                    "Strategic Acquirer",
                                    {"cash": proceeds, "stock": 0},
                                )

        except Exception as e:
            logger.error(f"Error checking exit opportunities: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        while True:
            try:
                for user_id, investments in self.direct_investments.items():
                    for investment in investments:
                        if investment.status == "active":
                            await self._update_investment_performance(investment)

                await asyncio.sleep(3600)  # Update every hour

            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")
                await asyncio.sleep(7200)

    async def _update_investment_performance(self, investment: DirectInvestment):
        """Update investment performance"""
        try:
            # Update current value based on company valuation
            opportunity = self.opportunities.get(investment.opportunity_id)
            if opportunity:
                company = self.companies.get(opportunity.company_id)
                if company:
                    new_value = (
                        company.valuation * investment.ownership_percentage / 100
                    )
                    investment.current_value = new_value
                    investment.unrealized_gain_loss = new_value - investment.cost_basis
                    investment.total_gain_loss = (
                        investment.realized_gain_loss + investment.unrealized_gain_loss
                    )

                    # Calculate IRR
                    if investment.status == "active":
                        investment.irr = await self._calculate_irr(
                            [-investment.cost_basis, investment.current_value],
                            [investment.investment_date, datetime.utcnow()],
                        )
                        investment.multiple = (
                            investment.current_value / investment.cost_basis
                        )

                    investment.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating investment performance: {e}")

    # Helper methods
    async def _create_portfolio_company(
        self,
        user_id: int,
        investment: DirectInvestment,
        opportunity: InvestmentOpportunity,
    ):
        """Create portfolio company record"""
        try:
            portfolio_company = PortfolioCompany(
                company_id=opportunity.company_id,
                investment_id=investment.investment_id,
                user_id=user_id,
                investment_date=investment.investment_date,
                investment_amount=investment.investment_amount,
                ownership_percentage=investment.ownership_percentage,
                board_seat=investment.ownership_percentage > 10,  # Board seat if >10%
                board_observer=investment.ownership_percentage > 5,  # Observer if >5%
                voting_rights=True,
                anti_dilution=True,
                liquidation_preference=True,
                drag_along=True,
                tag_along=True,
                information_rights=True,
                current_value=investment.current_value,
                unrealized_gain_loss=investment.unrealized_gain_loss,
                irr=investment.irr,
                multiple=investment.multiple,
                status=investment.status,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.portfolio_companies[user_id].append(portfolio_company)

        except Exception as e:
            logger.error(f"Error creating portfolio company: {e}")

    async def _update_portfolio_valuations(self, company_id: str, new_valuation: float):
        """Update portfolio company valuations"""
        try:
            for user_id, portfolio_companies in self.portfolio_companies.items():
                for portfolio_company in portfolio_companies:
                    if portfolio_company.company_id == company_id:
                        # Update portfolio company value
                        portfolio_company.current_value = (
                            new_valuation * portfolio_company.ownership_percentage / 100
                        )
                        portfolio_company.unrealized_gain_loss = (
                            portfolio_company.current_value
                            - portfolio_company.investment_amount
                        )
                        portfolio_company.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating portfolio valuations: {e}")

    async def _calculate_irr(
        self, cash_flows: List[float], dates: List[datetime]
    ) -> float:
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

    async def _calculate_portfolio_irr(
        self, investments: List[DirectInvestment]
    ) -> float:
        """Calculate portfolio IRR"""
        try:
            all_cash_flows = []
            all_dates = []

            for investment in investments:
                # Add initial investment
                all_cash_flows.append(-investment.cost_basis)
                all_dates.append(investment.investment_date)

                # Add current value or exit value
                if investment.status == "exited" and investment.exit_value:
                    all_cash_flows.append(investment.exit_value)
                    all_dates.append(investment.exit_date)
                elif investment.status == "active":
                    all_cash_flows.append(investment.current_value)
                    all_dates.append(datetime.utcnow())

            return await self._calculate_irr(all_cash_flows, all_dates)

        except Exception as e:
            logger.error(f"Error calculating portfolio IRR: {e}")
            return 0.0

    async def _calculate_performance_metrics(
        self, investments: List[DirectInvestment]
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            if not investments:
                return {}

            # Calculate various metrics
            total_invested = sum(inv.cost_basis for inv in investments)
            total_current_value = sum(inv.current_value for inv in investments)
            total_realized_gain = sum(inv.realized_gain_loss for inv in investments)
            total_unrealized_gain = sum(inv.unrealized_gain_loss for inv in investments)

            # Performance metrics
            total_return = (
                (total_current_value + total_realized_gain - total_invested)
                / total_invested
                if total_invested > 0
                else 0
            )
            realized_return = (
                total_realized_gain / total_invested if total_invested > 0 else 0
            )
            unrealized_return = (
                total_unrealized_gain / total_invested if total_invested > 0 else 0
            )

            # Risk metrics (simplified)
            volatility = 0.20  # 20% volatility assumption
            sharpe_ratio = total_return / volatility if volatility > 0 else 0
            max_drawdown = 0.30  # 30% max drawdown assumption

            return {
                "total_return": total_return,
                "realized_return": realized_return,
                "unrealized_return": unrealized_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": (
                    len([inv for inv in investments if inv.total_gain_loss > 0])
                    / len(investments)
                    if investments
                    else 0
                ),
                "avg_holding_period": np.mean(
                    [
                        (datetime.utcnow() - inv.investment_date).days / 365.0
                        for inv in investments
                    ]
                ),
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    async def _load_sample_companies(self):
        """Load sample companies"""
        try:
            sample_companies = [
                {
                    "name": "TechCorp Inc.",
                    "industry": "Technology",
                    "sector": "Software",
                    "stage": CompanyStage.GROWTH_STAGE,
                    "description": "AI-powered enterprise software company",
                    "headquarters": "San Francisco, CA",
                    "founded_date": datetime.utcnow() - timedelta(days=365 * 5),
                    "employees": 150,
                    "revenue": 25000000,
                    "ebitda": 5000000,
                    "valuation": 200000000,
                    "last_funding_round": "Series B",
                    "total_funding": 50000000,
                    "investors": ["Sequoia Capital", "Andreessen Horowitz"],
                    "website": "https://techcorp.com",
                },
                {
                    "name": "BioMed Solutions",
                    "industry": "Healthcare",
                    "sector": "Biotechnology",
                    "stage": CompanyStage.EARLY_STAGE,
                    "description": "Personalized medicine platform",
                    "headquarters": "Boston, MA",
                    "founded_date": datetime.utcnow() - timedelta(days=365 * 3),
                    "employees": 45,
                    "revenue": 5000000,
                    "ebitda": -2000000,
                    "valuation": 75000000,
                    "last_funding_round": "Series A",
                    "total_funding": 20000000,
                    "investors": ["Kleiner Perkins", "GV"],
                    "website": "https://biomedsolutions.com",
                },
                {
                    "name": "GreenEnergy Corp",
                    "industry": "Energy",
                    "sector": "Renewable Energy",
                    "stage": CompanyStage.MATURE,
                    "description": "Solar and wind energy solutions",
                    "headquarters": "Austin, TX",
                    "founded_date": datetime.utcnow() - timedelta(days=365 * 8),
                    "employees": 300,
                    "revenue": 100000000,
                    "ebitda": 20000000,
                    "valuation": 500000000,
                    "last_funding_round": "Growth",
                    "total_funding": 150000000,
                    "investors": ["TPG Growth", "KKR"],
                    "website": "https://greenenergy.com",
                },
            ]

            for company_data in sample_companies:
                await self.create_company(**company_data)

            logger.info("Loaded sample companies")

        except Exception as e:
            logger.error(f"Error loading sample companies: {e}")

    async def _load_sample_opportunities(self):
        """Load sample investment opportunities"""
        try:
            company_ids = list(self.companies.keys())
            if not company_ids:
                return

            sample_opportunities = [
                {
                    "company_id": company_ids[0],
                    "opportunity_type": PrivateMarketType.DIRECT_INVESTMENT,
                    "investment_stage": InvestmentStage.GROWTH,
                    "deal_size": 50000000,
                    "minimum_investment": 5000000,
                    "maximum_investment": 25000000,
                    "valuation": 200000000,
                    "ownership_percentage": 5.0,
                    "expected_close_date": datetime.utcnow() + timedelta(days=90),
                    "lead_investor": "Sequoia Capital",
                    "co_investors": ["Andreessen Horowitz", "Accel"],
                    "deal_terms": {
                        "liquidation_preference": 1.0,
                        "anti_dilution": True,
                    },
                    "expected_return": 0.30,
                    "holding_period": 5,
                    "exit_strategy": "IPO or Strategic Acquisition",
                },
                {
                    "company_id": company_ids[1],
                    "opportunity_type": PrivateMarketType.CO_INVESTMENT,
                    "investment_stage": InvestmentStage.SERIES_A,
                    "deal_size": 20000000,
                    "minimum_investment": 2000000,
                    "maximum_investment": 10000000,
                    "valuation": 75000000,
                    "ownership_percentage": 3.0,
                    "expected_close_date": datetime.utcnow() + timedelta(days=60),
                    "lead_investor": "Kleiner Perkins",
                    "co_investors": ["GV", "First Round"],
                    "deal_terms": {
                        "liquidation_preference": 1.5,
                        "anti_dilution": True,
                    },
                    "expected_return": 0.40,
                    "holding_period": 7,
                    "exit_strategy": "IPO or Acquisition",
                },
            ]

            for opportunity_data in sample_opportunities:
                await self.create_investment_opportunity(**opportunity_data)

            logger.info("Loaded sample investment opportunities")

        except Exception as e:
            logger.error(f"Error loading sample opportunities: {e}")

    async def _initialize_market_data(self):
        """Initialize market data"""
        try:
            # Initialize industry metrics
            self.industry_metrics = {
                "Technology": {
                    "avg_valuation_multiple": 8.5,
                    "avg_revenue_growth": 0.25,
                    "avg_ebitda_margin": 0.15,
                },
                "Healthcare": {
                    "avg_valuation_multiple": 12.0,
                    "avg_revenue_growth": 0.20,
                    "avg_ebitda_margin": 0.10,
                },
                "Energy": {
                    "avg_valuation_multiple": 6.0,
                    "avg_revenue_growth": 0.15,
                    "avg_ebitda_margin": 0.20,
                },
            }

            logger.info("Initialized market data")

        except Exception as e:
            logger.error(f"Error initializing market data: {e}")


# Factory function
async def get_private_markets_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> PrivateMarketsService:
    """Get Private Markets Service instance"""
    service = PrivateMarketsService(redis_client, db_session)
    await service.initialize()
    return service
