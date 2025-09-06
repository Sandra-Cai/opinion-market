"""
Financial Planning Service
Advanced financial planning and advisory services
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


class PlanningType(Enum):
    """Planning types"""

    RETIREMENT = "retirement"
    EDUCATION = "education"
    ESTATE = "estate"
    TAX = "tax"
    INSURANCE = "insurance"
    CASH_FLOW = "cash_flow"
    INVESTMENT = "investment"
    COMPREHENSIVE = "comprehensive"


class ScenarioType(Enum):
    """Scenario types"""

    BASE_CASE = "base_case"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    STRESS_TEST = "stress_test"


class RecommendationType(Enum):
    """Recommendation types"""

    INVESTMENT = "investment"
    INSURANCE = "insurance"
    TAX = "tax"
    ESTATE = "estate"
    CASH_FLOW = "cash_flow"
    RETIREMENT = "retirement"


@dataclass
class FinancialPlan:
    """Financial plan"""

    plan_id: str
    client_id: str
    plan_type: PlanningType
    plan_name: str
    description: str
    target_date: datetime
    target_amount: float
    current_amount: float
    required_contribution: float
    expected_return: float
    risk_level: str
    probability_of_success: float
    scenarios: Dict[str, Any]
    recommendations: List[str]
    status: str  # 'draft', 'active', 'completed', 'archived'
    created_at: datetime
    last_updated: datetime


@dataclass
class CashFlowAnalysis:
    """Cash flow analysis"""

    analysis_id: str
    client_id: str
    analysis_date: datetime
    monthly_income: float
    monthly_expenses: float
    monthly_savings: float
    annual_income: float
    annual_expenses: float
    annual_savings: float
    savings_rate: float
    emergency_fund_months: float
    debt_to_income_ratio: float
    cash_flow_health_score: float
    recommendations: List[str]
    created_at: datetime


@dataclass
class RetirementAnalysis:
    """Retirement analysis"""

    analysis_id: str
    client_id: str
    current_age: int
    retirement_age: int
    life_expectancy: int
    current_savings: float
    annual_contribution: float
    employer_match: float
    expected_return: float
    inflation_rate: float
    retirement_income_needed: float
    projected_retirement_savings: float
    income_replacement_ratio: float
    probability_of_success: float
    shortfall_amount: float
    recommendations: List[str]
    created_at: datetime


@dataclass
class EstatePlan:
    """Estate plan"""

    plan_id: str
    client_id: str
    total_estate_value: float
    liquid_assets: float
    illiquid_assets: float
    estimated_taxes: float
    beneficiaries: List[Dict[str, Any]]
    trusts: List[Dict[str, Any]]
    will_status: str
    power_of_attorney: bool
    healthcare_directive: bool
    tax_optimization_opportunities: List[str]
    recommendations: List[str]
    created_at: datetime
    last_updated: datetime


@dataclass
class InsuranceAnalysis:
    """Insurance analysis"""

    analysis_id: str
    client_id: str
    life_insurance_needed: float
    current_life_insurance: float
    life_insurance_gap: float
    disability_insurance_needed: float
    current_disability_insurance: float
    disability_insurance_gap: float
    long_term_care_needed: float
    current_long_term_care: float
    long_term_care_gap: float
    property_insurance_adequate: bool
    liability_insurance_adequate: bool
    recommendations: List[str]
    created_at: datetime


@dataclass
class TaxStrategy:
    """Tax strategy"""

    strategy_id: str
    client_id: str
    strategy_type: str  # 'optimization', 'harvesting', 'location', 'timing'
    description: str
    potential_savings: float
    implementation_cost: float
    net_benefit: float
    risk_level: str
    time_horizon: str
    requirements: List[str]
    status: str  # 'identified', 'recommended', 'implemented'
    created_at: datetime
    last_updated: datetime


class FinancialPlanningService:
    """Comprehensive Financial Planning Service"""

    def __init__(self, redis_client: redis.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session

        # Planning management
        self.financial_plans: Dict[str, List[FinancialPlan]] = defaultdict(list)
        self.cash_flow_analyses: Dict[str, List[CashFlowAnalysis]] = defaultdict(list)
        self.retirement_analyses: Dict[str, List[RetirementAnalysis]] = defaultdict(
            list
        )
        self.estate_plans: Dict[str, List[EstatePlan]] = defaultdict(list)
        self.insurance_analyses: Dict[str, List[InsuranceAnalysis]] = defaultdict(list)
        self.tax_strategies: Dict[str, List[TaxStrategy]] = defaultdict(list)

        # Analytics
        self.planning_analytics: Dict[str, Dict[str, Any]] = {}
        self.scenario_analyses: Dict[str, Dict[str, Any]] = {}

        # Market data
        self.planning_assumptions: Dict[str, float] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize the Financial Planning Service"""
        logger.info("Initializing Financial Planning Service")

        # Initialize planning assumptions
        await self._initialize_planning_assumptions()

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._update_plans()),
            asyncio.create_task(self._monitor_goals()),
            asyncio.create_task(self._identify_opportunities()),
        ]

        logger.info("Financial Planning Service initialized successfully")

    async def create_financial_plan(
        self,
        client_id: str,
        plan_type: PlanningType,
        plan_name: str,
        description: str,
        target_date: datetime,
        target_amount: float,
        current_amount: float,
        expected_return: float = 0.08,
    ) -> FinancialPlan:
        """Create a financial plan"""
        try:
            plan_id = f"PLAN_{uuid.uuid4().hex[:8]}"

            # Calculate required contribution
            years_to_target = (target_date - datetime.utcnow()).days / 365.0
            monthly_rate = expected_return / 12
            months_to_target = years_to_target * 12

            if months_to_target > 0:
                required_contribution = (
                    (target_amount - current_amount)
                    / ((1 + monthly_rate) ** months_to_target - 1)
                    * monthly_rate
                )
            else:
                required_contribution = 0

            # Calculate probability of success
            probability_of_success = await self._calculate_success_probability(
                current_amount,
                required_contribution,
                target_amount,
                expected_return,
                years_to_target,
            )

            # Generate scenarios
            scenarios = await self._generate_scenarios(
                current_amount,
                required_contribution,
                target_amount,
                expected_return,
                years_to_target,
            )

            # Generate recommendations
            recommendations = await self._generate_recommendations(
                plan_type, current_amount, target_amount, required_contribution
            )

            plan = FinancialPlan(
                plan_id=plan_id,
                client_id=client_id,
                plan_type=plan_type,
                plan_name=plan_name,
                description=description,
                target_date=target_date,
                target_amount=target_amount,
                current_amount=current_amount,
                required_contribution=required_contribution,
                expected_return=expected_return,
                risk_level="moderate",
                probability_of_success=probability_of_success,
                scenarios=scenarios,
                recommendations=recommendations,
                status="draft",
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.financial_plans[client_id].append(plan)

            logger.info(f"Created financial plan {plan_id}")
            return plan

        except Exception as e:
            logger.error(f"Error creating financial plan: {e}")
            raise

    async def create_cash_flow_analysis(
        self, client_id: str, monthly_income: float, monthly_expenses: float
    ) -> CashFlowAnalysis:
        """Create cash flow analysis"""
        try:
            analysis_id = f"CF_{uuid.uuid4().hex[:8]}"

            # Calculate metrics
            monthly_savings = monthly_income - monthly_expenses
            annual_income = monthly_income * 12
            annual_expenses = monthly_expenses * 12
            annual_savings = monthly_savings * 12
            savings_rate = monthly_savings / monthly_income if monthly_income > 0 else 0

            # Calculate emergency fund months
            emergency_fund_months = (
                (monthly_expenses * 6) / monthly_savings if monthly_savings > 0 else 0
            )

            # Calculate debt-to-income ratio (simplified)
            debt_to_income_ratio = 0.3  # 30% assumption

            # Calculate cash flow health score
            cash_flow_health_score = await self._calculate_cash_flow_health_score(
                savings_rate, emergency_fund_months, debt_to_income_ratio
            )

            # Generate recommendations
            recommendations = await self._generate_cash_flow_recommendations(
                savings_rate, emergency_fund_months, debt_to_income_ratio
            )

            analysis = CashFlowAnalysis(
                analysis_id=analysis_id,
                client_id=client_id,
                analysis_date=datetime.utcnow(),
                monthly_income=monthly_income,
                monthly_expenses=monthly_expenses,
                monthly_savings=monthly_savings,
                annual_income=annual_income,
                annual_expenses=annual_expenses,
                annual_savings=annual_savings,
                savings_rate=savings_rate,
                emergency_fund_months=emergency_fund_months,
                debt_to_income_ratio=debt_to_income_ratio,
                cash_flow_health_score=cash_flow_health_score,
                recommendations=recommendations,
                created_at=datetime.utcnow(),
            )

            self.cash_flow_analyses[client_id].append(analysis)

            logger.info(f"Created cash flow analysis {analysis_id}")
            return analysis

        except Exception as e:
            logger.error(f"Error creating cash flow analysis: {e}")
            raise

    async def create_retirement_analysis(
        self,
        client_id: str,
        current_age: int,
        retirement_age: int,
        current_savings: float,
        annual_contribution: float,
        employer_match: float = 0.0,
        expected_return: float = 0.07,
        inflation_rate: float = 0.03,
        retirement_income_needed: float = 0.0,
    ) -> RetirementAnalysis:
        """Create retirement analysis"""
        try:
            analysis_id = f"RET_{uuid.uuid4().hex[:8]}"

            # Calculate life expectancy
            life_expectancy = 85  # Default assumption

            # Calculate total annual contribution
            total_annual_contribution = annual_contribution + employer_match

            # Calculate years to retirement
            years_to_retirement = retirement_age - current_age

            # Calculate projected retirement savings
            projected_savings = await self._calculate_retirement_savings(
                current_savings,
                total_annual_contribution,
                expected_return,
                years_to_retirement,
            )

            # Calculate retirement income needed
            if retirement_income_needed == 0:
                retirement_income_needed = (
                    current_savings * 0.8
                )  # 80% replacement ratio assumption

            # Calculate income replacement ratio
            income_replacement_ratio = (
                retirement_income_needed / (current_savings * 12)
                if current_savings > 0
                else 0
            )

            # Calculate probability of success
            probability_of_success = (
                await self._calculate_retirement_success_probability(
                    projected_savings,
                    retirement_income_needed,
                    life_expectancy - retirement_age,
                )
            )

            # Calculate shortfall
            shortfall_amount = max(0, retirement_income_needed - projected_savings)

            # Generate recommendations
            recommendations = await self._generate_retirement_recommendations(
                projected_savings,
                retirement_income_needed,
                shortfall_amount,
                probability_of_success,
            )

            analysis = RetirementAnalysis(
                analysis_id=analysis_id,
                client_id=client_id,
                current_age=current_age,
                retirement_age=retirement_age,
                life_expectancy=life_expectancy,
                current_savings=current_savings,
                annual_contribution=annual_contribution,
                employer_match=employer_match,
                expected_return=expected_return,
                inflation_rate=inflation_rate,
                retirement_income_needed=retirement_income_needed,
                projected_retirement_savings=projected_savings,
                income_replacement_ratio=income_replacement_ratio,
                probability_of_success=probability_of_success,
                shortfall_amount=shortfall_amount,
                recommendations=recommendations,
                created_at=datetime.utcnow(),
            )

            self.retirement_analyses[client_id].append(analysis)

            logger.info(f"Created retirement analysis {analysis_id}")
            return analysis

        except Exception as e:
            logger.error(f"Error creating retirement analysis: {e}")
            raise

    async def create_estate_plan(
        self,
        client_id: str,
        total_estate_value: float,
        liquid_assets: float,
        illiquid_assets: float,
        beneficiaries: List[Dict[str, Any]],
        will_status: str = "none",
    ) -> EstatePlan:
        """Create estate plan"""
        try:
            plan_id = f"ESTATE_{uuid.uuid4().hex[:8]}"

            # Calculate estimated taxes
            estimated_taxes = await self._calculate_estate_taxes(total_estate_value)

            # Generate tax optimization opportunities
            tax_optimization_opportunities = (
                await self._identify_estate_tax_opportunities(
                    total_estate_value, liquid_assets, illiquid_assets
                )
            )

            # Generate recommendations
            recommendations = await self._generate_estate_recommendations(
                total_estate_value,
                estimated_taxes,
                will_status,
                tax_optimization_opportunities,
            )

            plan = EstatePlan(
                plan_id=plan_id,
                client_id=client_id,
                total_estate_value=total_estate_value,
                liquid_assets=liquid_assets,
                illiquid_assets=illiquid_assets,
                estimated_taxes=estimated_taxes,
                beneficiaries=beneficiaries,
                trusts=[],
                will_status=will_status,
                power_of_attorney=False,
                healthcare_directive=False,
                tax_optimization_opportunities=tax_optimization_opportunities,
                recommendations=recommendations,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.estate_plans[client_id].append(plan)

            logger.info(f"Created estate plan {plan_id}")
            return plan

        except Exception as e:
            logger.error(f"Error creating estate plan: {e}")
            raise

    async def create_insurance_analysis(
        self,
        client_id: str,
        annual_income: float,
        dependents: int,
        current_life_insurance: float = 0.0,
        current_disability_insurance: float = 0.0,
        current_long_term_care: float = 0.0,
    ) -> InsuranceAnalysis:
        """Create insurance analysis"""
        try:
            analysis_id = f"INS_{uuid.uuid4().hex[:8]}"

            # Calculate insurance needs
            life_insurance_needed = await self._calculate_life_insurance_needed(
                annual_income, dependents
            )
            disability_insurance_needed = annual_income * 0.6  # 60% of income
            long_term_care_needed = 500000  # $500k assumption

            # Calculate gaps
            life_insurance_gap = max(0, life_insurance_needed - current_life_insurance)
            disability_insurance_gap = max(
                0, disability_insurance_needed - current_disability_insurance
            )
            long_term_care_gap = max(0, long_term_care_needed - current_long_term_care)

            # Generate recommendations
            recommendations = await self._generate_insurance_recommendations(
                life_insurance_gap, disability_insurance_gap, long_term_care_gap
            )

            analysis = InsuranceAnalysis(
                analysis_id=analysis_id,
                client_id=client_id,
                life_insurance_needed=life_insurance_needed,
                current_life_insurance=current_life_insurance,
                life_insurance_gap=life_insurance_gap,
                disability_insurance_needed=disability_insurance_needed,
                current_disability_insurance=current_disability_insurance,
                disability_insurance_gap=disability_insurance_gap,
                long_term_care_needed=long_term_care_needed,
                current_long_term_care=current_long_term_care,
                long_term_care_gap=long_term_care_gap,
                property_insurance_adequate=True,
                liability_insurance_adequate=True,
                recommendations=recommendations,
                created_at=datetime.utcnow(),
            )

            self.insurance_analyses[client_id].append(analysis)

            logger.info(f"Created insurance analysis {analysis_id}")
            return analysis

        except Exception as e:
            logger.error(f"Error creating insurance analysis: {e}")
            raise

    async def create_tax_strategy(
        self,
        client_id: str,
        strategy_type: str,
        description: str,
        potential_savings: float,
        implementation_cost: float,
        risk_level: str,
        time_horizon: str,
        requirements: List[str],
    ) -> TaxStrategy:
        """Create tax strategy"""
        try:
            strategy_id = f"TAX_{uuid.uuid4().hex[:8]}"

            net_benefit = potential_savings - implementation_cost

            strategy = TaxStrategy(
                strategy_id=strategy_id,
                client_id=client_id,
                strategy_type=strategy_type,
                description=description,
                potential_savings=potential_savings,
                implementation_cost=implementation_cost,
                net_benefit=net_benefit,
                risk_level=risk_level,
                time_horizon=time_horizon,
                requirements=requirements,
                status="identified",
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.tax_strategies[client_id].append(strategy)

            logger.info(f"Created tax strategy {strategy_id}")
            return strategy

        except Exception as e:
            logger.error(f"Error creating tax strategy: {e}")
            raise

    # Background tasks
    async def _update_plans(self):
        """Update financial plans"""
        while True:
            try:
                for client_id, plans in self.financial_plans.items():
                    for plan in plans:
                        if plan.status == "active":
                            await self._update_plan_progress(plan)

                await asyncio.sleep(86400)  # Update daily

            except Exception as e:
                logger.error(f"Error updating plans: {e}")
                await asyncio.sleep(172800)

    async def _update_plan_progress(self, plan: FinancialPlan):
        """Update plan progress"""
        try:
            # Simulate plan progress
            monthly_contribution = plan.required_contribution
            monthly_return = plan.expected_return / 12

            # Add monthly contribution and return
            plan.current_amount = (
                plan.current_amount * (1 + monthly_return) + monthly_contribution
            )

            # Check if plan is achieved
            if plan.current_amount >= plan.target_amount:
                plan.status = "completed"

            plan.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating plan progress: {e}")

    async def _monitor_goals(self):
        """Monitor financial goals"""
        while True:
            try:
                # Monitor goal progress and generate alerts
                for client_id, plans in self.financial_plans.items():
                    for plan in plans:
                        if plan.status == "active":
                            await self._check_goal_progress(plan)

                await asyncio.sleep(86400)  # Monitor daily

            except Exception as e:
                logger.error(f"Error monitoring goals: {e}")
                await asyncio.sleep(172800)

    async def _check_goal_progress(self, plan: FinancialPlan):
        """Check goal progress"""
        try:
            # Check if goal is on track
            years_to_target = (plan.target_date - datetime.utcnow()).days / 365.0
            if years_to_target > 0:
                required_growth = (
                    (plan.target_amount - plan.current_amount) / plan.current_amount
                    if plan.current_amount > 0
                    else 0
                )
                required_annual_return = (
                    (required_growth / years_to_target) if years_to_target > 0 else 0
                )

                if (
                    required_annual_return > plan.expected_return * 1.5
                ):  # 50% above expected
                    # Goal may be at risk
                    logger.warning(
                        f"Goal {plan.plan_id} may be at risk - required return: {required_annual_return:.2%}"
                    )

        except Exception as e:
            logger.error(f"Error checking goal progress: {e}")

    async def _identify_opportunities(self):
        """Identify planning opportunities"""
        while True:
            try:
                # Identify tax optimization opportunities
                for client_id in self.financial_plans.keys():
                    if np.random.random() < 0.1:  # 10% chance
                        await self._identify_tax_opportunities(client_id)

                await asyncio.sleep(86400)  # Identify daily

            except Exception as e:
                logger.error(f"Error identifying opportunities: {e}")
                await asyncio.sleep(172800)

    async def _identify_tax_opportunities(self, client_id: str):
        """Identify tax optimization opportunities"""
        try:
            # Simulate tax opportunity identification
            opportunity_types = ["harvesting", "location", "timing", "structure"]
            opportunity_type = np.random.choice(opportunity_types)

            potential_savings = np.random.uniform(1000, 10000)
            implementation_cost = potential_savings * 0.1

            await self.create_tax_strategy(
                client_id,
                opportunity_type,
                f"Tax {opportunity_type} optimization opportunity",
                potential_savings,
                implementation_cost,
                "low",
                "short_term",
                ["Review current tax situation", "Implement strategy"],
            )

        except Exception as e:
            logger.error(f"Error identifying tax opportunities: {e}")

    # Helper methods
    async def _calculate_success_probability(
        self,
        current_amount: float,
        required_contribution: float,
        target_amount: float,
        expected_return: float,
        years: float,
    ) -> float:
        """Calculate probability of success"""
        try:
            # Simplified probability calculation
            if years <= 0:
                return 1.0 if current_amount >= target_amount else 0.0

            # Monte Carlo simulation
            n_simulations = 1000
            successes = 0

            for _ in range(n_simulations):
                # Simulate returns
                annual_returns = np.random.normal(expected_return, 0.15, int(years))
                final_amount = current_amount

                for year in range(int(years)):
                    final_amount = (
                        final_amount * (1 + annual_returns[year])
                        + required_contribution * 12
                    )

                if final_amount >= target_amount:
                    successes += 1

            return successes / n_simulations

        except Exception as e:
            logger.error(f"Error calculating success probability: {e}")
            return 0.5

    async def _generate_scenarios(
        self,
        current_amount: float,
        required_contribution: float,
        target_amount: float,
        expected_return: float,
        years: float,
    ) -> Dict[str, Any]:
        """Generate scenarios"""
        try:
            scenarios = {}

            # Base case
            base_case_amount = (
                current_amount * (1 + expected_return) ** years
                + required_contribution * 12 * years
            )
            scenarios["base_case"] = {
                "final_amount": base_case_amount,
                "success": base_case_amount >= target_amount,
                "shortfall": max(0, target_amount - base_case_amount),
            }

            # Optimistic case
            optimistic_return = expected_return + 0.02
            optimistic_amount = (
                current_amount * (1 + optimistic_return) ** years
                + required_contribution * 12 * years
            )
            scenarios["optimistic"] = {
                "final_amount": optimistic_amount,
                "success": optimistic_amount >= target_amount,
                "shortfall": max(0, target_amount - optimistic_amount),
            }

            # Pessimistic case
            pessimistic_return = expected_return - 0.02
            pessimistic_amount = (
                current_amount * (1 + pessimistic_return) ** years
                + required_contribution * 12 * years
            )
            scenarios["pessimistic"] = {
                "final_amount": pessimistic_amount,
                "success": pessimistic_amount >= target_amount,
                "shortfall": max(0, target_amount - pessimistic_amount),
            }

            return scenarios

        except Exception as e:
            logger.error(f"Error generating scenarios: {e}")
            return {}

    async def _generate_recommendations(
        self,
        plan_type: PlanningType,
        current_amount: float,
        target_amount: float,
        required_contribution: float,
    ) -> List[str]:
        """Generate recommendations"""
        try:
            recommendations = []

            if plan_type == PlanningType.RETIREMENT:
                recommendations.extend(
                    [
                        "Maximize employer 401(k) match",
                        "Consider Roth IRA contributions",
                        "Review asset allocation for age-appropriate risk",
                    ]
                )
            elif plan_type == PlanningType.EDUCATION:
                recommendations.extend(
                    [
                        "Consider 529 education savings plan",
                        "Explore Coverdell ESA options",
                        "Review financial aid implications",
                    ]
                )
            elif plan_type == PlanningType.ESTATE:
                recommendations.extend(
                    [
                        "Create or update will",
                        "Consider trust structures",
                        "Review beneficiary designations",
                    ]
                )

            # General recommendations
            if required_contribution > current_amount * 0.2:
                recommendations.append(
                    "Consider increasing income or reducing expenses"
                )

            if current_amount < target_amount * 0.1:
                recommendations.append(
                    "Start saving immediately to benefit from compound growth"
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []

    async def _calculate_cash_flow_health_score(
        self,
        savings_rate: float,
        emergency_fund_months: float,
        debt_to_income_ratio: float,
    ) -> float:
        """Calculate cash flow health score"""
        try:
            # Weighted scoring
            savings_score = min(savings_rate * 100, 100)  # Max 100
            emergency_score = min(emergency_fund_months * 10, 100)  # Max 100
            debt_score = max(0, 100 - debt_to_income_ratio * 100)  # Lower is better

            # Weighted average
            health_score = (
                savings_score * 0.4 + emergency_score * 0.3 + debt_score * 0.3
            )

            return min(max(health_score, 0), 100)

        except Exception as e:
            logger.error(f"Error calculating cash flow health score: {e}")
            return 50.0

    async def _generate_cash_flow_recommendations(
        self,
        savings_rate: float,
        emergency_fund_months: float,
        debt_to_income_ratio: float,
    ) -> List[str]:
        """Generate cash flow recommendations"""
        try:
            recommendations = []

            if savings_rate < 0.1:
                recommendations.append(
                    "Increase savings rate to at least 10% of income"
                )

            if emergency_fund_months < 3:
                recommendations.append(
                    "Build emergency fund to cover 3-6 months of expenses"
                )

            if debt_to_income_ratio > 0.4:
                recommendations.append("Reduce debt-to-income ratio to below 40%")

            if not recommendations:
                recommendations.append(
                    "Cash flow is healthy - maintain current practices"
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating cash flow recommendations: {e}")
            return []

    async def _calculate_retirement_savings(
        self,
        current_savings: float,
        annual_contribution: float,
        expected_return: float,
        years: int,
    ) -> float:
        """Calculate retirement savings"""
        try:
            # Future value calculation
            future_value = current_savings * (1 + expected_return) ** years

            # Add annual contributions
            if annual_contribution > 0 and years > 0:
                contribution_future_value = annual_contribution * (
                    ((1 + expected_return) ** years - 1) / expected_return
                )
                future_value += contribution_future_value

            return future_value

        except Exception as e:
            logger.error(f"Error calculating retirement savings: {e}")
            return current_savings

    async def _calculate_retirement_success_probability(
        self,
        projected_savings: float,
        retirement_income_needed: float,
        retirement_years: int,
    ) -> float:
        """Calculate retirement success probability"""
        try:
            if retirement_income_needed == 0:
                return 1.0

            # Simplified calculation
            annual_withdrawal = (
                retirement_income_needed / retirement_years
                if retirement_years > 0
                else 0
            )
            success_ratio = (
                projected_savings / (annual_withdrawal * retirement_years)
                if annual_withdrawal > 0
                else 0
            )

            return min(success_ratio, 1.0)

        except Exception as e:
            logger.error(f"Error calculating retirement success probability: {e}")
            return 0.5

    async def _generate_retirement_recommendations(
        self,
        projected_savings: float,
        retirement_income_needed: float,
        shortfall_amount: float,
        probability_of_success: float,
    ) -> List[str]:
        """Generate retirement recommendations"""
        try:
            recommendations = []

            if probability_of_success < 0.8:
                recommendations.append("Increase retirement contributions")
                recommendations.append(
                    "Consider working longer or part-time in retirement"
                )

            if shortfall_amount > 0:
                recommendations.append(
                    f"Address retirement shortfall of ${shortfall_amount:,.0f}"
                )

            if projected_savings > retirement_income_needed * 1.2:
                recommendations.append(
                    "Consider early retirement or increased spending"
                )

            recommendations.extend(
                [
                    "Review asset allocation for retirement timeline",
                    "Consider Roth conversion strategies",
                    "Plan for healthcare costs in retirement",
                ]
            )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating retirement recommendations: {e}")
            return []

    async def _calculate_estate_taxes(self, estate_value: float) -> float:
        """Calculate estate taxes"""
        try:
            # Simplified estate tax calculation
            exemption = 12000000  # $12M exemption
            tax_rate = 0.40  # 40% tax rate

            taxable_estate = max(0, estate_value - exemption)
            estate_tax = taxable_estate * tax_rate

            return estate_tax

        except Exception as e:
            logger.error(f"Error calculating estate taxes: {e}")
            return 0.0

    async def _identify_estate_tax_opportunities(
        self, estate_value: float, liquid_assets: float, illiquid_assets: float
    ) -> List[str]:
        """Identify estate tax opportunities"""
        try:
            opportunities = []

            if estate_value > 12000000:  # Above exemption
                opportunities.append("Consider gifting strategies to reduce estate")
                opportunities.append("Explore irrevocable trust structures")

            if illiquid_assets > liquid_assets:
                opportunities.append("Plan for liquidity needs to pay estate taxes")

            opportunities.extend(
                [
                    "Review beneficiary designations",
                    "Consider life insurance for estate liquidity",
                    "Explore charitable giving strategies",
                ]
            )

            return opportunities

        except Exception as e:
            logger.error(f"Error identifying estate tax opportunities: {e}")
            return []

    async def _generate_estate_recommendations(
        self,
        estate_value: float,
        estimated_taxes: float,
        will_status: str,
        tax_opportunities: List[str],
    ) -> List[str]:
        """Generate estate recommendations"""
        try:
            recommendations = []

            if will_status == "none":
                recommendations.append("Create a will immediately")

            if estimated_taxes > 0:
                recommendations.append(
                    f"Plan for estate taxes of ${estimated_taxes:,.0f}"
                )

            recommendations.extend(tax_opportunities)
            recommendations.extend(
                [
                    "Create power of attorney documents",
                    "Establish healthcare directives",
                    "Review and update beneficiary designations",
                ]
            )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating estate recommendations: {e}")
            return []

    async def _calculate_life_insurance_needed(
        self, annual_income: float, dependents: int
    ) -> float:
        """Calculate life insurance needed"""
        try:
            # Simplified calculation
            base_amount = annual_income * 10  # 10x income
            dependent_amount = dependents * 100000  # $100k per dependent

            return base_amount + dependent_amount

        except Exception as e:
            logger.error(f"Error calculating life insurance needed: {e}")
            return annual_income * 10

    async def _generate_insurance_recommendations(
        self, life_gap: float, disability_gap: float, long_term_care_gap: float
    ) -> List[str]:
        """Generate insurance recommendations"""
        try:
            recommendations = []

            if life_gap > 0:
                recommendations.append(
                    f"Consider ${life_gap:,.0f} in additional life insurance"
                )

            if disability_gap > 0:
                recommendations.append(
                    f"Consider ${disability_gap:,.0f} in disability insurance"
                )

            if long_term_care_gap > 0:
                recommendations.append(
                    f"Consider ${long_term_care_gap:,.0f} in long-term care insurance"
                )

            if not recommendations:
                recommendations.append("Insurance coverage appears adequate")

            return recommendations

        except Exception as e:
            logger.error(f"Error generating insurance recommendations: {e}")
            return []

    async def _initialize_planning_assumptions(self):
        """Initialize planning assumptions"""
        try:
            self.planning_assumptions = {
                "inflation_rate": 0.03,
                "expected_return_stocks": 0.10,
                "expected_return_bonds": 0.05,
                "expected_return_cash": 0.02,
                "life_expectancy": 85,
                "retirement_age": 65,
                "social_security_age": 67,
                "estate_tax_exemption": 12000000,
                "estate_tax_rate": 0.40,
            }

            logger.info("Initialized planning assumptions")

        except Exception as e:
            logger.error(f"Error initializing planning assumptions: {e}")


# Factory function
async def get_financial_planning_service(
    redis_client: redis.Redis, db_session: Session
) -> FinancialPlanningService:
    """Get Financial Planning Service instance"""
    service = FinancialPlanningService(redis_client, db_session)
    await service.initialize()
    return service
