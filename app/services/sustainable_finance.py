"""
Sustainable Finance Service
Advanced sustainable finance and green investment management
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


class GreenBondType(Enum):
    """Green bond types"""

    USE_OF_PROCEEDS = "use_of_proceeds"
    REVENUE_LINKED = "revenue_linked"
    PROJECT_LINKED = "project_linked"
    SUSTAINABILITY_LINKED = "sustainability_linked"


class ClimateRiskType(Enum):
    """Climate risk types"""

    PHYSICAL = "physical"
    TRANSITION = "transition"
    LIABILITY = "liability"


class SustainabilityFramework(Enum):
    """Sustainability frameworks"""

    EU_TAXONOMY = "eu_taxonomy"
    TCFD = "tcfd"
    SASB = "sasb"
    GRI = "gri"
    UN_SDG = "un_sdg"


class ImpactMeasurement(Enum):
    """Impact measurement types"""

    CARBON_AVOIDED = "carbon_avoided"
    RENEWABLE_ENERGY = "renewable_energy"
    WATER_SAVED = "water_saved"
    WASTE_REDUCED = "waste_reduced"
    JOBS_CREATED = "jobs_created"
    COMMUNITY_BENEFITS = "community_benefits"


@dataclass
class GreenBond:
    """Green bond"""

    bond_id: str
    issuer: str
    bond_type: GreenBondType
    issue_amount: float
    currency: str
    maturity_date: datetime
    coupon_rate: float
    use_of_proceeds: List[str]
    green_project_categories: List[str]
    third_party_verification: str
    impact_reporting: str
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class ClimateRiskAssessment:
    """Climate risk assessment"""

    assessment_id: str
    company_id: str
    risk_type: ClimateRiskType
    risk_score: float
    risk_level: str
    exposure_metrics: Dict[str, float]
    scenario_analysis: Dict[str, Any]
    mitigation_strategies: List[str]
    assessment_date: datetime
    next_assessment: datetime
    created_at: datetime


@dataclass
class SustainabilityBond:
    """Sustainability bond"""

    bond_id: str
    issuer: str
    issue_amount: float
    currency: str
    maturity_date: datetime
    coupon_rate: float
    sustainability_framework: SustainabilityFramework
    sustainability_targets: List[str]
    key_performance_indicators: List[str]
    impact_measurement: List[ImpactMeasurement]
    third_party_verification: str
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class GreenLoan:
    """Green loan"""

    loan_id: str
    borrower: str
    loan_amount: float
    currency: str
    interest_rate: float
    maturity_date: datetime
    green_project: str
    green_project_category: str
    environmental_benefits: List[str]
    impact_metrics: Dict[str, float]
    third_party_verification: str
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class CarbonCredit:
    """Carbon credit"""

    credit_id: str
    project_id: str
    project_name: str
    project_type: str
    standard: str
    vintage_year: int
    credit_amount: float
    price_per_credit: float
    total_value: float
    verification_status: str
    retirement_status: str
    environmental_benefits: List[str]
    created_at: datetime
    last_updated: datetime


@dataclass
class ImpactInvestment:
    """Impact investment"""

    investment_id: str
    user_id: int
    investment_type: str
    investment_amount: float
    target_impact: str
    impact_metrics: Dict[str, float]
    expected_return: float
    risk_level: str
    investment_date: datetime
    maturity_date: Optional[datetime]
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class SustainabilityReport:
    """Sustainability report"""

    report_id: str
    company_id: str
    report_period: str
    framework: SustainabilityFramework
    environmental_metrics: Dict[str, float]
    social_metrics: Dict[str, float]
    governance_metrics: Dict[str, float]
    sustainability_goals: List[str]
    progress_towards_goals: Dict[str, float]
    third_party_assurance: str
    report_date: datetime
    created_at: datetime


class SustainableFinanceService:
    """Comprehensive Sustainable Finance Service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session

        # Sustainable finance management
        self.green_bonds: Dict[str, List[GreenBond]] = defaultdict(list)
        self.climate_risk_assessments: Dict[str, List[ClimateRiskAssessment]] = (
            defaultdict(list)
        )
        self.sustainability_bonds: Dict[str, List[SustainabilityBond]] = defaultdict(
            list
        )
        self.green_loans: Dict[str, List[GreenLoan]] = defaultdict(list)
        self.carbon_credits: Dict[str, List[CarbonCredit]] = defaultdict(list)
        self.impact_investments: Dict[str, List[ImpactInvestment]] = defaultdict(list)
        self.sustainability_reports: Dict[str, List[SustainabilityReport]] = (
            defaultdict(list)
        )

        # Analytics
        self.sustainable_finance_analytics: Dict[str, Dict[str, Any]] = {}
        self.impact_analytics: Dict[str, Dict[str, Any]] = {}
        self.climate_risk_analytics: Dict[str, Dict[str, Any]] = {}

        # Market data
        self.green_bond_prices: Dict[str, float] = {}
        self.carbon_prices: Dict[str, float] = {}
        self.sustainability_benchmarks: Dict[str, float] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize the Sustainable Finance Service"""
        logger.info("Initializing Sustainable Finance Service")

        # Load sample data
        await self._load_sample_green_bonds()
        await self._load_sample_carbon_credits()

        # Initialize market data
        await self._initialize_market_data()

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._update_green_bond_prices()),
            asyncio.create_task(self._update_carbon_prices()),
            asyncio.create_task(self._assess_climate_risks()),
            asyncio.create_task(self._calculate_impact_metrics()),
            asyncio.create_task(self._generate_sustainability_reports()),
        ]

        logger.info("Sustainable Finance Service initialized successfully")

    async def create_green_bond(
        self,
        issuer: str,
        bond_type: GreenBondType,
        issue_amount: float,
        currency: str,
        maturity_date: datetime,
        coupon_rate: float,
        use_of_proceeds: List[str],
        green_project_categories: List[str],
        third_party_verification: str = "",
        impact_reporting: str = "",
    ) -> GreenBond:
        """Create green bond"""
        try:
            bond_id = f"GREEN_{uuid.uuid4().hex[:8]}"

            bond = GreenBond(
                bond_id=bond_id,
                issuer=issuer,
                bond_type=bond_type,
                issue_amount=issue_amount,
                currency=currency,
                maturity_date=maturity_date,
                coupon_rate=coupon_rate,
                use_of_proceeds=use_of_proceeds,
                green_project_categories=green_project_categories,
                third_party_verification=third_party_verification,
                impact_reporting=impact_reporting,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.green_bonds[issuer].append(bond)

            logger.info(f"Created green bond {bond_id}")
            return bond

        except Exception as e:
            logger.error(f"Error creating green bond: {e}")
            raise

    async def create_climate_risk_assessment(
        self,
        company_id: str,
        risk_type: ClimateRiskType,
        risk_score: float,
        exposure_metrics: Dict[str, float],
        scenario_analysis: Dict[str, Any],
        mitigation_strategies: List[str],
    ) -> ClimateRiskAssessment:
        """Create climate risk assessment"""
        try:
            assessment_id = f"CLIMATE_{uuid.uuid4().hex[:8]}"

            # Calculate risk level
            risk_level = self._calculate_risk_level(risk_score)

            assessment = ClimateRiskAssessment(
                assessment_id=assessment_id,
                company_id=company_id,
                risk_type=risk_type,
                risk_score=risk_score,
                risk_level=risk_level,
                exposure_metrics=exposure_metrics,
                scenario_analysis=scenario_analysis,
                mitigation_strategies=mitigation_strategies,
                assessment_date=datetime.utcnow(),
                next_assessment=datetime.utcnow() + timedelta(days=365),
                created_at=datetime.utcnow(),
            )

            self.climate_risk_assessments[company_id].append(assessment)

            logger.info(f"Created climate risk assessment {assessment_id}")
            return assessment

        except Exception as e:
            logger.error(f"Error creating climate risk assessment: {e}")
            raise

    async def create_sustainability_bond(
        self,
        issuer: str,
        issue_amount: float,
        currency: str,
        maturity_date: datetime,
        coupon_rate: float,
        sustainability_framework: SustainabilityFramework,
        sustainability_targets: List[str],
        key_performance_indicators: List[str],
        impact_measurement: List[ImpactMeasurement],
        third_party_verification: str = "",
    ) -> SustainabilityBond:
        """Create sustainability bond"""
        try:
            bond_id = f"SUST_{uuid.uuid4().hex[:8]}"

            bond = SustainabilityBond(
                bond_id=bond_id,
                issuer=issuer,
                issue_amount=issue_amount,
                currency=currency,
                maturity_date=maturity_date,
                coupon_rate=coupon_rate,
                sustainability_framework=sustainability_framework,
                sustainability_targets=sustainability_targets,
                key_performance_indicators=key_performance_indicators,
                impact_measurement=impact_measurement,
                third_party_verification=third_party_verification,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.sustainability_bonds[issuer].append(bond)

            logger.info(f"Created sustainability bond {bond_id}")
            return bond

        except Exception as e:
            logger.error(f"Error creating sustainability bond: {e}")
            raise

    async def create_green_loan(
        self,
        borrower: str,
        loan_amount: float,
        currency: str,
        interest_rate: float,
        maturity_date: datetime,
        green_project: str,
        green_project_category: str,
        environmental_benefits: List[str],
        impact_metrics: Dict[str, float],
        third_party_verification: str = "",
    ) -> GreenLoan:
        """Create green loan"""
        try:
            loan_id = f"LOAN_{uuid.uuid4().hex[:8]}"

            loan = GreenLoan(
                loan_id=loan_id,
                borrower=borrower,
                loan_amount=loan_amount,
                currency=currency,
                interest_rate=interest_rate,
                maturity_date=maturity_date,
                green_project=green_project,
                green_project_category=green_project_category,
                environmental_benefits=environmental_benefits,
                impact_metrics=impact_metrics,
                third_party_verification=third_party_verification,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.green_loans[borrower].append(loan)

            logger.info(f"Created green loan {loan_id}")
            return loan

        except Exception as e:
            logger.error(f"Error creating green loan: {e}")
            raise

    async def create_carbon_credit(
        self,
        project_id: str,
        project_name: str,
        project_type: str,
        standard: str,
        vintage_year: int,
        credit_amount: float,
        price_per_credit: float,
        verification_status: str = "verified",
        retirement_status: str = "available",
        environmental_benefits: List[str] = None,
    ) -> CarbonCredit:
        """Create carbon credit"""
        try:
            credit_id = f"CARBON_{uuid.uuid4().hex[:8]}"

            total_value = credit_amount * price_per_credit

            credit = CarbonCredit(
                credit_id=credit_id,
                project_id=project_id,
                project_name=project_name,
                project_type=project_type,
                standard=standard,
                vintage_year=vintage_year,
                credit_amount=credit_amount,
                price_per_credit=price_per_credit,
                total_value=total_value,
                verification_status=verification_status,
                retirement_status=retirement_status,
                environmental_benefits=environmental_benefits or [],
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.carbon_credits[project_id].append(credit)

            logger.info(f"Created carbon credit {credit_id}")
            return credit

        except Exception as e:
            logger.error(f"Error creating carbon credit: {e}")
            raise

    async def create_impact_investment(
        self,
        user_id: int,
        investment_type: str,
        investment_amount: float,
        target_impact: str,
        impact_metrics: Dict[str, float],
        expected_return: float,
        risk_level: str,
        maturity_date: Optional[datetime] = None,
    ) -> ImpactInvestment:
        """Create impact investment"""
        try:
            investment_id = f"IMPACT_{uuid.uuid4().hex[:8]}"

            investment = ImpactInvestment(
                investment_id=investment_id,
                user_id=user_id,
                investment_type=investment_type,
                investment_amount=investment_amount,
                target_impact=target_impact,
                impact_metrics=impact_metrics,
                expected_return=expected_return,
                risk_level=risk_level,
                investment_date=datetime.utcnow(),
                maturity_date=maturity_date,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.impact_investments[user_id].append(investment)

            logger.info(f"Created impact investment {investment_id}")
            return investment

        except Exception as e:
            logger.error(f"Error creating impact investment: {e}")
            raise

    async def create_sustainability_report(
        self,
        company_id: str,
        report_period: str,
        framework: SustainabilityFramework,
        environmental_metrics: Dict[str, float],
        social_metrics: Dict[str, float],
        governance_metrics: Dict[str, float],
        sustainability_goals: List[str],
        progress_towards_goals: Dict[str, float],
        third_party_assurance: str = "",
    ) -> SustainabilityReport:
        """Create sustainability report"""
        try:
            report_id = f"REPORT_{uuid.uuid4().hex[:8]}"

            report = SustainabilityReport(
                report_id=report_id,
                company_id=company_id,
                report_period=report_period,
                framework=framework,
                environmental_metrics=environmental_metrics,
                social_metrics=social_metrics,
                governance_metrics=governance_metrics,
                sustainability_goals=sustainability_goals,
                progress_towards_goals=progress_towards_goals,
                third_party_assurance=third_party_assurance,
                report_date=datetime.utcnow(),
                created_at=datetime.utcnow(),
            )

            self.sustainability_reports[company_id].append(report)

            logger.info(f"Created sustainability report {report_id}")
            return report

        except Exception as e:
            logger.error(f"Error creating sustainability report: {e}")
            raise

    async def get_sustainable_finance_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get sustainable finance analytics for a user"""
        try:
            impact_investments = self.impact_investments.get(user_id, [])

            if not impact_investments:
                return {"error": "No impact investments found"}

            # Calculate portfolio metrics
            total_investment = sum(inv.investment_amount for inv in impact_investments)
            avg_expected_return = np.mean(
                [inv.expected_return for inv in impact_investments]
            )

            # Calculate impact metrics
            total_carbon_avoided = sum(
                inv.impact_metrics.get("carbon_avoided", 0)
                for inv in impact_investments
            )
            total_renewable_energy = sum(
                inv.impact_metrics.get("renewable_energy", 0)
                for inv in impact_investments
            )
            total_water_saved = sum(
                inv.impact_metrics.get("water_saved", 0) for inv in impact_investments
            )
            total_waste_reduced = sum(
                inv.impact_metrics.get("waste_reduced", 0) for inv in impact_investments
            )
            total_jobs_created = sum(
                int(inv.impact_metrics.get("jobs_created", 0))
                for inv in impact_investments
            )
            total_community_benefits = sum(
                inv.impact_metrics.get("community_benefits", 0)
                for inv in impact_investments
            )

            # Calculate impact by investment type
            impact_by_type = {}
            for investment in impact_investments:
                investment_type = investment.investment_type
                if investment_type not in impact_by_type:
                    impact_by_type[investment_type] = {
                        "total_investment": 0,
                        "carbon_avoided": 0,
                        "renewable_energy": 0,
                        "jobs_created": 0,
                    }

                impact_by_type[investment_type][
                    "total_investment"
                ] += investment.investment_amount
                impact_by_type[investment_type][
                    "carbon_avoided"
                ] += investment.impact_metrics.get("carbon_avoided", 0)
                impact_by_type[investment_type][
                    "renewable_energy"
                ] += investment.impact_metrics.get("renewable_energy", 0)
                impact_by_type[investment_type][
                    "jobs_created"
                ] += investment.impact_metrics.get("jobs_created", 0)

            # Calculate sustainability goals progress
            sustainability_progress = await self._calculate_sustainability_progress(
                impact_investments
            )

            # Calculate risk distribution
            risk_distribution = {}
            for investment in impact_investments:
                risk_level = investment.risk_level
                if risk_level not in risk_distribution:
                    risk_distribution[risk_level] = 0
                risk_distribution[risk_level] += investment.investment_amount

            analytics = {
                "user_id": user_id,
                "total_investment": total_investment,
                "number_of_investments": len(impact_investments),
                "avg_expected_return": avg_expected_return,
                "total_carbon_avoided": total_carbon_avoided,
                "total_renewable_energy": total_renewable_energy,
                "total_water_saved": total_water_saved,
                "total_waste_reduced": total_waste_reduced,
                "total_jobs_created": total_jobs_created,
                "total_community_benefits": total_community_benefits,
                "impact_by_type": impact_by_type,
                "sustainability_progress": sustainability_progress,
                "risk_distribution": risk_distribution,
                "timestamp": datetime.utcnow(),
            }

            self.sustainable_finance_analytics[user_id] = analytics

            return analytics

        except Exception as e:
            logger.error(f"Error getting sustainable finance analytics: {e}")
            return {"error": str(e)}

    # Background tasks
    async def _update_green_bond_prices(self):
        """Update green bond prices"""
        while True:
            try:
                for issuer, bonds in self.green_bonds.items():
                    for bond in bonds:
                        if bond.is_active:
                            await self._update_bond_price(bond)

                await asyncio.sleep(3600)  # Update every hour

            except Exception as e:
                logger.error(f"Error updating green bond prices: {e}")
                await asyncio.sleep(7200)

    async def _update_bond_price(self, bond: GreenBond):
        """Update bond price"""
        try:
            # Simulate price update
            price_change = np.random.normal(0, 0.01)  # 1% volatility
            current_price = self.green_bond_prices.get(bond.bond_id, 100.0)
            new_price = current_price * (1 + price_change)

            self.green_bond_prices[bond.bond_id] = new_price
            bond.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating bond price: {e}")

    async def _update_carbon_prices(self):
        """Update carbon prices"""
        while True:
            try:
                # Update carbon credit prices
                for project_id, credits in self.carbon_credits.items():
                    for credit in credits:
                        if credit.retirement_status == "available":
                            await self._update_carbon_credit_price(credit)

                await asyncio.sleep(3600)  # Update every hour

            except Exception as e:
                logger.error(f"Error updating carbon prices: {e}")
                await asyncio.sleep(7200)

    async def _update_carbon_credit_price(self, credit: CarbonCredit):
        """Update carbon credit price"""
        try:
            # Simulate price update
            price_change = np.random.normal(0, 0.05)  # 5% volatility
            new_price = credit.price_per_credit * (1 + price_change)

            credit.price_per_credit = new_price
            credit.total_value = credit.credit_amount * new_price
            credit.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating carbon credit price: {e}")

    async def _assess_climate_risks(self):
        """Assess climate risks"""
        while True:
            try:
                # Simulate climate risk assessment
                for company_id in self.climate_risk_assessments.keys():
                    if np.random.random() < 0.1:  # 10% chance
                        await self._create_climate_risk_assessment(company_id)

                await asyncio.sleep(86400)  # Assess daily

            except Exception as e:
                logger.error(f"Error assessing climate risks: {e}")
                await asyncio.sleep(172800)

    async def _create_climate_risk_assessment(self, company_id: str):
        """Create climate risk assessment"""
        try:
            risk_types = [
                ClimateRiskType.PHYSICAL,
                ClimateRiskType.TRANSITION,
                ClimateRiskType.LIABILITY,
            ]
            risk_type = np.random.choice(risk_types)

            risk_score = np.random.uniform(0, 100)
            exposure_metrics = {
                "carbon_intensity": np.random.uniform(0, 2),
                "water_stress": np.random.uniform(0, 1),
                "physical_risk_exposure": np.random.uniform(0, 1),
            }

            scenario_analysis = {
                "2_degree_scenario": {"risk_score": risk_score * 0.8},
                "4_degree_scenario": {"risk_score": risk_score * 1.2},
                "net_zero_scenario": {"risk_score": risk_score * 0.6},
            }

            mitigation_strategies = [
                "Implement renewable energy",
                "Improve energy efficiency",
                "Develop climate adaptation plans",
            ]

            await self.create_climate_risk_assessment(
                company_id,
                risk_type,
                risk_score,
                exposure_metrics,
                scenario_analysis,
                mitigation_strategies,
            )

        except Exception as e:
            logger.error(f"Error creating climate risk assessment: {e}")

    async def _calculate_impact_metrics(self):
        """Calculate impact metrics"""
        while True:
            try:
                for user_id, investments in self.impact_investments.items():
                    for investment in investments:
                        if investment.is_active:
                            await self._update_investment_impact_metrics(investment)

                await asyncio.sleep(3600)  # Calculate every hour

            except Exception as e:
                logger.error(f"Error calculating impact metrics: {e}")
                await asyncio.sleep(7200)

    async def _update_investment_impact_metrics(self, investment: ImpactInvestment):
        """Update investment impact metrics"""
        try:
            # Simulate impact metric updates
            investment.impact_metrics.update(
                {
                    "carbon_avoided": investment.investment_amount
                    * 0.2,  # 0.2 tons per $1000
                    "renewable_energy": investment.investment_amount
                    * 0.3,  # 0.3 kWh per $1000
                    "water_saved": investment.investment_amount
                    * 0.1,  # 0.1 gallons per $1000
                    "waste_reduced": investment.investment_amount
                    * 0.05,  # 0.05 tons per $1000
                    "jobs_created": int(
                        investment.investment_amount / 50000
                    ),  # 1 job per $50k
                    "community_benefits": investment.investment_amount
                    * 0.1,  # 10% community benefit
                }
            )

            investment.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating investment impact metrics: {e}")

    async def _generate_sustainability_reports(self):
        """Generate sustainability reports"""
        while True:
            try:
                # Generate annual sustainability reports
                if (
                    datetime.utcnow().month == 1 and datetime.utcnow().day == 1
                ):  # January 1st
                    for company_id in self.sustainability_reports.keys():
                        await self._create_annual_sustainability_report(company_id)

                await asyncio.sleep(86400)  # Check daily

            except Exception as e:
                logger.error(f"Error generating sustainability reports: {e}")
                await asyncio.sleep(172800)

    async def _create_annual_sustainability_report(self, company_id: str):
        """Create annual sustainability report"""
        try:
            # Simulate sustainability metrics
            environmental_metrics = {
                "carbon_emissions": np.random.uniform(1000, 10000),
                "renewable_energy_share": np.random.uniform(0.2, 0.8),
                "water_usage": np.random.uniform(10000, 100000),
                "waste_generated": np.random.uniform(1000, 10000),
            }

            social_metrics = {
                "employee_satisfaction": np.random.uniform(0.7, 0.9),
                "diversity_index": np.random.uniform(0.6, 0.8),
                "safety_incidents": np.random.randint(0, 10),
                "community_investment": np.random.uniform(100000, 1000000),
            }

            governance_metrics = {
                "board_independence": np.random.uniform(0.6, 0.9),
                "executive_compensation_ratio": np.random.uniform(50, 200),
                "transparency_score": np.random.uniform(0.7, 0.9),
                "ethics_violations": np.random.randint(0, 3),
            }

            sustainability_goals = [
                "Reduce carbon emissions by 50% by 2030",
                "Achieve 100% renewable energy by 2025",
                "Zero waste to landfill by 2025",
                "Increase diversity to 50% by 2030",
            ]

            progress_towards_goals = {
                "carbon_reduction": np.random.uniform(0.2, 0.8),
                "renewable_energy": np.random.uniform(0.3, 0.9),
                "waste_reduction": np.random.uniform(0.1, 0.7),
                "diversity_improvement": np.random.uniform(0.2, 0.6),
            }

            await self.create_sustainability_report(
                company_id,
                "2024",
                SustainabilityFramework.GRI,
                environmental_metrics,
                social_metrics,
                governance_metrics,
                sustainability_goals,
                progress_towards_goals,
                "Third Party Verified",
            )

        except Exception as e:
            logger.error(f"Error creating annual sustainability report: {e}")

    # Helper methods
    def _calculate_risk_level(self, risk_score: float) -> str:
        """Calculate risk level"""
        if risk_score >= 80:
            return "very_high"
        elif risk_score >= 60:
            return "high"
        elif risk_score >= 40:
            return "medium"
        elif risk_score >= 20:
            return "low"
        else:
            return "very_low"

    async def _calculate_sustainability_progress(
        self, investments: List[ImpactInvestment]
    ) -> Dict[str, float]:
        """Calculate sustainability progress"""
        try:
            progress = {
                "carbon_reduction": 0.0,
                "renewable_energy_adoption": 0.0,
                "water_conservation": 0.0,
                "waste_reduction": 0.0,
                "social_impact": 0.0,
                "overall_sustainability": 0.0,
            }

            if not investments:
                return progress

            total_investment = sum(inv.investment_amount for inv in investments)

            progress["carbon_reduction"] = (
                sum(inv.impact_metrics.get("carbon_avoided", 0) for inv in investments)
                / total_investment
                * 1000
            )
            progress["renewable_energy_adoption"] = (
                sum(
                    inv.impact_metrics.get("renewable_energy", 0) for inv in investments
                )
                / total_investment
                * 100
            )
            progress["water_conservation"] = (
                sum(inv.impact_metrics.get("water_saved", 0) for inv in investments)
                / total_investment
                * 100
            )
            progress["waste_reduction"] = (
                sum(inv.impact_metrics.get("waste_reduced", 0) for inv in investments)
                / total_investment
                * 100
            )
            progress["social_impact"] = (
                sum(inv.impact_metrics.get("jobs_created", 0) for inv in investments)
                / total_investment
                * 100000
            )

            # Calculate overall sustainability score
            progress["overall_sustainability"] = np.mean(
                [
                    progress["carbon_reduction"],
                    progress["renewable_energy_adoption"],
                    progress["water_conservation"],
                    progress["waste_reduction"],
                    progress["social_impact"],
                ]
            )

            return progress

        except Exception as e:
            logger.error(f"Error calculating sustainability progress: {e}")
            return {}

    async def _load_sample_green_bonds(self):
        """Load sample green bonds"""
        try:
            sample_bonds = [
                {
                    "issuer": "Green Energy Corp",
                    "bond_type": GreenBondType.USE_OF_PROCEEDS,
                    "issue_amount": 500000000,
                    "currency": "USD",
                    "maturity_date": datetime.utcnow() + timedelta(days=365 * 10),
                    "coupon_rate": 0.035,
                    "use_of_proceeds": [
                        "Renewable energy projects",
                        "Energy efficiency improvements",
                    ],
                    "green_project_categories": [
                        "Renewable Energy",
                        "Energy Efficiency",
                    ],
                    "third_party_verification": "CICERO",
                    "impact_reporting": "Annual impact report",
                },
                {
                    "issuer": "Sustainable Infrastructure Ltd",
                    "bond_type": GreenBondType.PROJECT_LINKED,
                    "issue_amount": 300000000,
                    "currency": "EUR",
                    "maturity_date": datetime.utcnow() + timedelta(days=365 * 7),
                    "coupon_rate": 0.025,
                    "use_of_proceeds": ["Green buildings", "Sustainable transport"],
                    "green_project_categories": [
                        "Green Buildings",
                        "Clean Transportation",
                    ],
                    "third_party_verification": "Sustainalytics",
                    "impact_reporting": "Quarterly impact report",
                },
            ]

            for bond_data in sample_bonds:
                await self.create_green_bond(**bond_data)

            logger.info("Loaded sample green bonds")

        except Exception as e:
            logger.error(f"Error loading sample green bonds: {e}")

    async def _load_sample_carbon_credits(self):
        """Load sample carbon credits"""
        try:
            sample_credits = [
                {
                    "project_id": "FOREST_001",
                    "project_name": "Amazon Rainforest Protection",
                    "project_type": "REDD+",
                    "standard": "VCS",
                    "vintage_year": 2023,
                    "credit_amount": 10000,
                    "price_per_credit": 15.50,
                    "verification_status": "verified",
                    "retirement_status": "available",
                    "environmental_benefits": [
                        "Biodiversity protection",
                        "Carbon sequestration",
                    ],
                },
                {
                    "project_id": "RENEWABLE_001",
                    "project_name": "Wind Farm Development",
                    "project_type": "Renewable Energy",
                    "standard": "Gold Standard",
                    "vintage_year": 2023,
                    "credit_amount": 5000,
                    "price_per_credit": 12.75,
                    "verification_status": "verified",
                    "retirement_status": "available",
                    "environmental_benefits": [
                        "Clean energy generation",
                        "Emission reduction",
                    ],
                },
            ]

            for credit_data in sample_credits:
                await self.create_carbon_credit(**credit_data)

            logger.info("Loaded sample carbon credits")

        except Exception as e:
            logger.error(f"Error loading sample carbon credits: {e}")

    async def _initialize_market_data(self):
        """Initialize market data"""
        try:
            # Initialize green bond prices
            for issuer, bonds in self.green_bonds.items():
                for bond in bonds:
                    self.green_bond_prices[bond.bond_id] = 100.0  # Par value

            # Initialize carbon prices
            self.carbon_prices = {
                "EU_ETS": 85.50,
                "California_Cap_Trade": 28.75,
                "Voluntary_Market": 12.25,
            }

            # Initialize sustainability benchmarks
            self.sustainability_benchmarks = {
                "green_bond_yield_spread": -0.05,  # 5 bps greenium
                "carbon_price_volatility": 0.15,
                "sustainability_premium": 0.02,
            }

            logger.info("Initialized market data")

        except Exception as e:
            logger.error(f"Error initializing market data: {e}")


# Factory function
async def get_sustainable_finance_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> SustainableFinanceService:
    """Get Sustainable Finance Service instance"""
    service = SustainableFinanceService(redis_client, db_session)
    await service.initialize()
    return service
