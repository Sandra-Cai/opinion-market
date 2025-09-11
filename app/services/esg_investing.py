"""
ESG Investing Service
Advanced ESG investing and sustainable finance management
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
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class ESGCategory(Enum):
    """ESG categories"""

    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"


class ESGScore(Enum):
    """ESG score levels"""

    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    VERY_POOR = "very_poor"


class SustainabilityFramework(Enum):
    """Sustainability frameworks"""

    GRI = "gri"
    SASB = "sasb"
    TCFD = "tcfd"
    CDP = "cdp"
    UN_SDG = "un_sdg"
    EU_TAXONOMY = "eu_taxonomy"


class ImpactType(Enum):
    """Impact types"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class ESGScore:
    """ESG score"""

    score_id: str
    company_id: str
    overall_score: float
    environmental_score: float
    social_score: float
    governance_score: float
    score_grade: str
    framework: SustainabilityFramework
    data_quality: float
    coverage_percentage: float
    last_updated: datetime
    created_at: datetime


@dataclass
class ESGFactor:
    """ESG factor"""

    factor_id: str
    category: ESGCategory
    factor_name: str
    description: str
    weight: float
    measurement_unit: str
    data_source: str
    is_material: bool
    impact_type: ImpactType
    created_at: datetime


@dataclass
class ESGImpact:
    """ESG impact"""

    impact_id: str
    company_id: str
    factor_id: str
    impact_value: float
    impact_unit: str
    impact_type: ImpactType
    measurement_date: datetime
    baseline_value: float
    target_value: float
    progress_percentage: float
    created_at: datetime


@dataclass
class SustainableInvestment:
    """Sustainable investment"""

    investment_id: str
    user_id: int
    company_id: str
    investment_amount: float
    esg_score: float
    sustainability_rating: str
    impact_metrics: Dict[str, float]
    alignment_score: float
    risk_score: float
    expected_return: float
    investment_date: datetime
    created_at: datetime


@dataclass
class ESGPortfolio:
    """ESG portfolio"""

    portfolio_id: str
    user_id: int
    portfolio_name: str
    total_value: float
    esg_score: float
    carbon_footprint: float
    impact_metrics: Dict[str, float]
    sustainability_rating: str
    risk_score: float
    expected_return: float
    benchmark_comparison: Dict[str, float]
    created_at: datetime
    last_updated: datetime


@dataclass
class ImpactReport:
    """Impact report"""

    report_id: str
    user_id: int
    report_period: str
    total_investment: float
    environmental_impact: Dict[str, float]
    social_impact: Dict[str, float]
    governance_impact: Dict[str, float]
    carbon_avoided: float
    jobs_created: int
    community_benefits: float
    sustainability_goals_achieved: List[str]
    created_at: datetime


@dataclass
class ESGAlert:
    """ESG alert"""

    alert_id: str
    company_id: str
    alert_type: str
    severity: str
    description: str
    impact_score: float
    recommendation: str
    alert_date: datetime
    status: str
    created_at: datetime


class ESGInvestingService:
    """Comprehensive ESG Investing Service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session

        # ESG management
        self.esg_scores: Dict[str, List[ESGScore]] = defaultdict(list)
        self.esg_factors: Dict[str, ESGFactor] = {}
        self.esg_impacts: Dict[str, List[ESGImpact]] = defaultdict(list)
        self.sustainable_investments: Dict[str, List[SustainableInvestment]] = (
            defaultdict(list)
        )
        self.esg_portfolios: Dict[str, List[ESGPortfolio]] = defaultdict(list)
        self.impact_reports: Dict[str, List[ImpactReport]] = defaultdict(list)
        self.esg_alerts: Dict[str, List[ESGAlert]] = defaultdict(list)

        # Analytics
        self.esg_analytics: Dict[str, Dict[str, Any]] = {}
        self.impact_analytics: Dict[str, Dict[str, Any]] = {}
        self.sustainability_metrics: Dict[str, Dict[str, float]] = {}

        # Market data
        self.esg_data: Dict[str, pd.DataFrame] = {}
        self.sustainability_benchmarks: Dict[str, float] = {}

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize the ESG Investing Service"""
        logger.info("Initializing ESG Investing Service")

        # Load ESG factors
        await self._load_esg_factors()

        # Initialize sustainability benchmarks
        await self._initialize_sustainability_benchmarks()

        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._update_esg_scores()),
            asyncio.create_task(self._monitor_esg_alerts()),
            asyncio.create_task(self._calculate_impact_metrics()),
            asyncio.create_task(self._generate_impact_reports()),
            asyncio.create_task(self._update_portfolio_esg_metrics()),
        ]

        logger.info("ESG Investing Service initialized successfully")

    async def create_esg_score(
        self,
        company_id: str,
        overall_score: float,
        environmental_score: float,
        social_score: float,
        governance_score: float,
        framework: SustainabilityFramework,
        data_quality: float = 0.8,
        coverage_percentage: float = 0.9,
    ) -> ESGScore:
        """Create ESG score"""
        try:
            score_id = f"ESG_{uuid.uuid4().hex[:8]}"

            # Calculate score grade
            score_grade = self._calculate_score_grade(overall_score)

            score = ESGScore(
                score_id=score_id,
                company_id=company_id,
                overall_score=overall_score,
                environmental_score=environmental_score,
                social_score=social_score,
                governance_score=governance_score,
                score_grade=score_grade,
                framework=framework,
                data_quality=data_quality,
                coverage_percentage=coverage_percentage,
                last_updated=datetime.utcnow(),
                created_at=datetime.utcnow(),
            )

            self.esg_scores[company_id].append(score)

            logger.info(f"Created ESG score {score_id}")
            return score

        except Exception as e:
            logger.error(f"Error creating ESG score: {e}")
            raise

    async def create_esg_factor(
        self,
        category: ESGCategory,
        factor_name: str,
        description: str,
        weight: float,
        measurement_unit: str,
        data_source: str,
        is_material: bool = True,
        impact_type: ImpactType = ImpactType.NEUTRAL,
    ) -> ESGFactor:
        """Create ESG factor"""
        try:
            factor_id = f"FACTOR_{uuid.uuid4().hex[:8]}"

            factor = ESGFactor(
                factor_id=factor_id,
                category=category,
                factor_name=factor_name,
                description=description,
                weight=weight,
                measurement_unit=measurement_unit,
                data_source=data_source,
                is_material=is_material,
                impact_type=impact_type,
                created_at=datetime.utcnow(),
            )

            self.esg_factors[factor_id] = factor

            logger.info(f"Created ESG factor {factor_id}")
            return factor

        except Exception as e:
            logger.error(f"Error creating ESG factor: {e}")
            raise

    async def create_esg_impact(
        self,
        company_id: str,
        factor_id: str,
        impact_value: float,
        impact_unit: str,
        impact_type: ImpactType,
        baseline_value: float = 0.0,
        target_value: float = 0.0,
    ) -> ESGImpact:
        """Create ESG impact"""
        try:
            impact_id = f"IMPACT_{uuid.uuid4().hex[:8]}"

            # Calculate progress percentage
            progress_percentage = 0.0
            if target_value > 0 and baseline_value >= 0:
                if impact_type == ImpactType.POSITIVE:
                    progress_percentage = (
                        (impact_value - baseline_value)
                        / (target_value - baseline_value)
                        * 100
                    )
                else:
                    progress_percentage = (
                        (baseline_value - impact_value)
                        / (baseline_value - target_value)
                        * 100
                    )

            impact = ESGImpact(
                impact_id=impact_id,
                company_id=company_id,
                factor_id=factor_id,
                impact_value=impact_value,
                impact_unit=impact_unit,
                impact_type=impact_type,
                measurement_date=datetime.utcnow(),
                baseline_value=baseline_value,
                target_value=target_value,
                progress_percentage=progress_percentage,
                created_at=datetime.utcnow(),
            )

            self.esg_impacts[company_id].append(impact)

            logger.info(f"Created ESG impact {impact_id}")
            return impact

        except Exception as e:
            logger.error(f"Error creating ESG impact: {e}")
            raise

    async def create_sustainable_investment(
        self,
        user_id: int,
        company_id: str,
        investment_amount: float,
        esg_score: float,
        sustainability_rating: str,
        impact_metrics: Dict[str, float],
        alignment_score: float,
        risk_score: float,
        expected_return: float,
    ) -> SustainableInvestment:
        """Create sustainable investment"""
        try:
            investment_id = f"SUST_{uuid.uuid4().hex[:8]}"

            investment = SustainableInvestment(
                investment_id=investment_id,
                user_id=user_id,
                company_id=company_id,
                investment_amount=investment_amount,
                esg_score=esg_score,
                sustainability_rating=sustainability_rating,
                impact_metrics=impact_metrics,
                alignment_score=alignment_score,
                risk_score=risk_score,
                expected_return=expected_return,
                investment_date=datetime.utcnow(),
                created_at=datetime.utcnow(),
            )

            self.sustainable_investments[user_id].append(investment)

            logger.info(f"Created sustainable investment {investment_id}")
            return investment

        except Exception as e:
            logger.error(f"Error creating sustainable investment: {e}")
            raise

    async def create_esg_portfolio(
        self, user_id: int, portfolio_name: str
    ) -> ESGPortfolio:
        """Create ESG portfolio"""
        try:
            portfolio_id = f"ESG_PORT_{uuid.uuid4().hex[:8]}"

            # Get user's sustainable investments
            investments = self.sustainable_investments.get(user_id, [])

            # Calculate portfolio metrics
            total_value = sum(inv.investment_amount for inv in investments)
            esg_score = (
                np.mean([inv.esg_score for inv in investments]) if investments else 0.0
            )
            carbon_footprint = sum(
                inv.impact_metrics.get("carbon_footprint", 0) for inv in investments
            )

            # Calculate impact metrics
            impact_metrics = await self._calculate_portfolio_impact_metrics(investments)

            # Calculate sustainability rating
            sustainability_rating = self._calculate_sustainability_rating(esg_score)

            # Calculate risk and return
            risk_score = (
                np.mean([inv.risk_score for inv in investments]) if investments else 0.0
            )
            expected_return = (
                np.mean([inv.expected_return for inv in investments])
                if investments
                else 0.0
            )

            # Calculate benchmark comparison
            benchmark_comparison = await self._calculate_benchmark_comparison(
                esg_score, expected_return
            )

            portfolio = ESGPortfolio(
                portfolio_id=portfolio_id,
                user_id=user_id,
                portfolio_name=portfolio_name,
                total_value=total_value,
                esg_score=esg_score,
                carbon_footprint=carbon_footprint,
                impact_metrics=impact_metrics,
                sustainability_rating=sustainability_rating,
                risk_score=risk_score,
                expected_return=expected_return,
                benchmark_comparison=benchmark_comparison,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.esg_portfolios[user_id].append(portfolio)

            logger.info(f"Created ESG portfolio {portfolio_id}")
            return portfolio

        except Exception as e:
            logger.error(f"Error creating ESG portfolio: {e}")
            raise

    async def create_impact_report(
        self, user_id: int, report_period: str
    ) -> ImpactReport:
        """Create impact report"""
        try:
            report_id = f"IMPACT_REPORT_{uuid.uuid4().hex[:8]}"

            # Get user's investments
            investments = self.sustainable_investments.get(user_id, [])

            # Calculate total investment
            total_investment = sum(inv.investment_amount for inv in investments)

            # Calculate impact metrics
            environmental_impact = await self._calculate_environmental_impact(
                investments
            )
            social_impact = await self._calculate_social_impact(investments)
            governance_impact = await self._calculate_governance_impact(investments)

            # Calculate aggregate metrics
            carbon_avoided = sum(
                inv.impact_metrics.get("carbon_avoided", 0) for inv in investments
            )
            jobs_created = sum(
                int(inv.impact_metrics.get("jobs_created", 0)) for inv in investments
            )
            community_benefits = sum(
                inv.impact_metrics.get("community_benefits", 0) for inv in investments
            )

            # Calculate sustainability goals achieved
            sustainability_goals_achieved = (
                await self._calculate_sustainability_goals_achieved(investments)
            )

            report = ImpactReport(
                report_id=report_id,
                user_id=user_id,
                report_period=report_period,
                total_investment=total_investment,
                environmental_impact=environmental_impact,
                social_impact=social_impact,
                governance_impact=governance_impact,
                carbon_avoided=carbon_avoided,
                jobs_created=jobs_created,
                community_benefits=community_benefits,
                sustainability_goals_achieved=sustainability_goals_achieved,
                created_at=datetime.utcnow(),
            )

            self.impact_reports[user_id].append(report)

            logger.info(f"Created impact report {report_id}")
            return report

        except Exception as e:
            logger.error(f"Error creating impact report: {e}")
            raise

    async def create_esg_alert(
        self,
        company_id: str,
        alert_type: str,
        severity: str,
        description: str,
        impact_score: float,
        recommendation: str,
    ) -> ESGAlert:
        """Create ESG alert"""
        try:
            alert_id = f"ALERT_{uuid.uuid4().hex[:8]}"

            alert = ESGAlert(
                alert_id=alert_id,
                company_id=company_id,
                alert_type=alert_type,
                severity=severity,
                description=description,
                impact_score=impact_score,
                recommendation=recommendation,
                alert_date=datetime.utcnow(),
                status="active",
                created_at=datetime.utcnow(),
            )

            self.esg_alerts[company_id].append(alert)

            logger.info(f"Created ESG alert {alert_id}")
            return alert

        except Exception as e:
            logger.error(f"Error creating ESG alert: {e}")
            raise

    async def get_esg_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get ESG analytics for a user"""
        try:
            investments = self.sustainable_investments.get(user_id, [])
            portfolios = self.esg_portfolios.get(user_id, [])
            reports = self.impact_reports.get(user_id, [])

            if not investments:
                return {"error": "No investments found"}

            # Calculate portfolio metrics
            total_investment = sum(inv.investment_amount for inv in investments)
            avg_esg_score = np.mean([inv.esg_score for inv in investments])
            avg_alignment_score = np.mean([inv.alignment_score for inv in investments])
            avg_risk_score = np.mean([inv.risk_score for inv in investments])
            avg_expected_return = np.mean([inv.expected_return for inv in investments])

            # Calculate impact metrics
            total_carbon_avoided = sum(
                inv.impact_metrics.get("carbon_avoided", 0) for inv in investments
            )
            total_jobs_created = sum(
                int(inv.impact_metrics.get("jobs_created", 0)) for inv in investments
            )
            total_community_benefits = sum(
                inv.impact_metrics.get("community_benefits", 0) for inv in investments
            )

            # Calculate ESG distribution
            esg_distribution = {}
            for investment in investments:
                rating = investment.sustainability_rating
                if rating not in esg_distribution:
                    esg_distribution[rating] = 0
                esg_distribution[rating] += investment.investment_amount

            # Calculate performance vs benchmark
            benchmark_comparison = await self._calculate_benchmark_comparison(
                avg_esg_score, avg_expected_return
            )

            # Calculate sustainability goals progress
            sustainability_progress = await self._calculate_sustainability_progress(
                investments
            )

            analytics = {
                "user_id": user_id,
                "total_investment": total_investment,
                "number_of_investments": len(investments),
                "avg_esg_score": avg_esg_score,
                "avg_alignment_score": avg_alignment_score,
                "avg_risk_score": avg_risk_score,
                "avg_expected_return": avg_expected_return,
                "total_carbon_avoided": total_carbon_avoided,
                "total_jobs_created": total_jobs_created,
                "total_community_benefits": total_community_benefits,
                "esg_distribution": esg_distribution,
                "benchmark_comparison": benchmark_comparison,
                "sustainability_progress": sustainability_progress,
                "number_of_portfolios": len(portfolios),
                "number_of_reports": len(reports),
                "timestamp": datetime.utcnow(),
            }

            self.esg_analytics[user_id] = analytics

            return analytics

        except Exception as e:
            logger.error(f"Error getting ESG analytics: {e}")
            return {"error": str(e)}

    # Background tasks
    async def _update_esg_scores(self):
        """Update ESG scores"""
        while True:
            try:
                # Simulate ESG score updates
                for company_id in self.esg_scores.keys():
                    await self._update_company_esg_score(company_id)

                await asyncio.sleep(86400)  # Update daily

            except Exception as e:
                logger.error(f"Error updating ESG scores: {e}")
                await asyncio.sleep(172800)

    async def _update_company_esg_score(self, company_id: str):
        """Update company ESG score"""
        try:
            # Get latest score
            scores = self.esg_scores.get(company_id, [])
            if not scores:
                return

            latest_score = scores[-1]

            # Simulate score update
            change = np.random.normal(0, 0.05)  # 5% volatility
            new_overall_score = max(0, min(100, latest_score.overall_score + change))
            new_env_score = max(0, min(100, latest_score.environmental_score + change))
            new_social_score = max(0, min(100, latest_score.social_score + change))
            new_gov_score = max(0, min(100, latest_score.governance_score + change))

            # Create new score
            await self.create_esg_score(
                company_id,
                new_overall_score,
                new_env_score,
                new_social_score,
                new_gov_score,
                latest_score.framework,
                latest_score.data_quality,
                latest_score.coverage_percentage,
            )

        except Exception as e:
            logger.error(f"Error updating company ESG score: {e}")

    async def _monitor_esg_alerts(self):
        """Monitor ESG alerts"""
        while True:
            try:
                # Check for ESG issues
                for company_id in self.esg_scores.keys():
                    await self._check_esg_alerts(company_id)

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error monitoring ESG alerts: {e}")
                await asyncio.sleep(7200)

    async def _check_esg_alerts(self, company_id: str):
        """Check for ESG alerts"""
        try:
            scores = self.esg_scores.get(company_id, [])
            if not scores:
                return

            latest_score = scores[-1]

            # Check for low scores
            if latest_score.overall_score < 30:
                await self.create_esg_alert(
                    company_id,
                    "low_esg_score",
                    "high",
                    f"ESG score below 30: {latest_score.overall_score:.1f}",
                    latest_score.overall_score,
                    "Consider divesting or engaging with company",
                )

            # Check for environmental issues
            if latest_score.environmental_score < 25:
                await self.create_esg_alert(
                    company_id,
                    "environmental_risk",
                    "medium",
                    f"Environmental score below 25: {latest_score.environmental_score:.1f}",
                    latest_score.environmental_score,
                    "Monitor environmental performance closely",
                )

        except Exception as e:
            logger.error(f"Error checking ESG alerts: {e}")

    async def _calculate_impact_metrics(self):
        """Calculate impact metrics"""
        while True:
            try:
                for user_id, investments in self.sustainable_investments.items():
                    for investment in investments:
                        await self._update_investment_impact_metrics(investment)

                await asyncio.sleep(3600)  # Calculate every hour

            except Exception as e:
                logger.error(f"Error calculating impact metrics: {e}")
                await asyncio.sleep(7200)

    async def _update_investment_impact_metrics(
        self, investment: SustainableInvestment
    ):
        """Update investment impact metrics"""
        try:
            # Simulate impact metric updates
            investment.impact_metrics.update(
                {
                    "carbon_avoided": investment.investment_amount
                    * 0.1,  # 0.1 tons per $1000
                    "jobs_created": int(
                        investment.investment_amount / 100000
                    ),  # 1 job per $100k
                    "community_benefits": investment.investment_amount
                    * 0.05,  # 5% community benefit
                    "renewable_energy": investment.investment_amount
                    * 0.2,  # 20% renewable energy
                    "water_saved": investment.investment_amount
                    * 0.5,  # 0.5 gallons per $1000
                    "waste_reduced": investment.investment_amount
                    * 0.3,  # 0.3 tons per $1000
                }
            )

        except Exception as e:
            logger.error(f"Error updating investment impact metrics: {e}")

    async def _generate_impact_reports(self):
        """Generate impact reports"""
        while True:
            try:
                for user_id in self.sustainable_investments.keys():
                    # Generate monthly impact report
                    if datetime.utcnow().day == 1:  # First day of month
                        await self.create_impact_report(user_id, "monthly")

                await asyncio.sleep(86400)  # Check daily

            except Exception as e:
                logger.error(f"Error generating impact reports: {e}")
                await asyncio.sleep(172800)

    async def _update_portfolio_esg_metrics(self):
        """Update portfolio ESG metrics"""
        while True:
            try:
                for user_id, portfolios in self.esg_portfolios.items():
                    for portfolio in portfolios:
                        await self._update_portfolio_metrics(portfolio)

                await asyncio.sleep(3600)  # Update every hour

            except Exception as e:
                logger.error(f"Error updating portfolio ESG metrics: {e}")
                await asyncio.sleep(7200)

    async def _update_portfolio_metrics(self, portfolio: ESGPortfolio):
        """Update portfolio metrics"""
        try:
            # Get investments for this portfolio
            investments = self.sustainable_investments.get(portfolio.user_id, [])

            # Recalculate metrics
            portfolio.total_value = sum(inv.investment_amount for inv in investments)
            portfolio.esg_score = (
                np.mean([inv.esg_score for inv in investments]) if investments else 0.0
            )
            portfolio.carbon_footprint = sum(
                inv.impact_metrics.get("carbon_footprint", 0) for inv in investments
            )
            portfolio.impact_metrics = await self._calculate_portfolio_impact_metrics(
                investments
            )
            portfolio.sustainability_rating = self._calculate_sustainability_rating(
                portfolio.esg_score
            )
            portfolio.risk_score = (
                np.mean([inv.risk_score for inv in investments]) if investments else 0.0
            )
            portfolio.expected_return = (
                np.mean([inv.expected_return for inv in investments])
                if investments
                else 0.0
            )
            portfolio.benchmark_comparison = await self._calculate_benchmark_comparison(
                portfolio.esg_score, portfolio.expected_return
            )
            portfolio.last_updated = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")

    # Helper methods
    def _calculate_score_grade(self, score: float) -> str:
        """Calculate score grade"""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "average"
        elif score >= 20:
            return "poor"
        else:
            return "very_poor"

    def _calculate_sustainability_rating(self, esg_score: float) -> str:
        """Calculate sustainability rating"""
        if esg_score >= 85:
            return "AAA"
        elif esg_score >= 75:
            return "AA"
        elif esg_score >= 65:
            return "A"
        elif esg_score >= 55:
            return "BBB"
        elif esg_score >= 45:
            return "BB"
        elif esg_score >= 35:
            return "B"
        elif esg_score >= 25:
            return "CCC"
        elif esg_score >= 15:
            return "CC"
        elif esg_score >= 5:
            return "C"
        else:
            return "D"

    async def _calculate_portfolio_impact_metrics(
        self, investments: List[SustainableInvestment]
    ) -> Dict[str, float]:
        """Calculate portfolio impact metrics"""
        try:
            if not investments:
                return {}

            metrics = {}
            for investment in investments:
                for key, value in investment.impact_metrics.items():
                    if key not in metrics:
                        metrics[key] = 0
                    metrics[key] += value

            return metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio impact metrics: {e}")
            return {}

    async def _calculate_benchmark_comparison(
        self, esg_score: float, expected_return: float
    ) -> Dict[str, float]:
        """Calculate benchmark comparison"""
        try:
            # Get benchmark data
            benchmark_esg = self.sustainability_benchmarks.get("avg_esg_score", 50.0)
            benchmark_return = self.sustainability_benchmarks.get("avg_return", 0.08)

            return {
                "esg_outperformance": esg_score - benchmark_esg,
                "return_outperformance": expected_return - benchmark_return,
                "esg_percentile": (esg_score / 100) * 100,
                "return_percentile": (expected_return / 0.15)
                * 100,  # 15% max return assumption
            }

        except Exception as e:
            logger.error(f"Error calculating benchmark comparison: {e}")
            return {}

    async def _calculate_environmental_impact(
        self, investments: List[SustainableInvestment]
    ) -> Dict[str, float]:
        """Calculate environmental impact"""
        try:
            impact = {
                "carbon_avoided": 0.0,
                "renewable_energy": 0.0,
                "water_saved": 0.0,
                "waste_reduced": 0.0,
                "biodiversity_protected": 0.0,
            }

            for investment in investments:
                impact["carbon_avoided"] += investment.impact_metrics.get(
                    "carbon_avoided", 0
                )
                impact["renewable_energy"] += investment.impact_metrics.get(
                    "renewable_energy", 0
                )
                impact["water_saved"] += investment.impact_metrics.get("water_saved", 0)
                impact["waste_reduced"] += investment.impact_metrics.get(
                    "waste_reduced", 0
                )
                impact["biodiversity_protected"] += investment.impact_metrics.get(
                    "biodiversity_protected", 0
                )

            return impact

        except Exception as e:
            logger.error(f"Error calculating environmental impact: {e}")
            return {}

    async def _calculate_social_impact(
        self, investments: List[SustainableInvestment]
    ) -> Dict[str, float]:
        """Calculate social impact"""
        try:
            impact = {
                "jobs_created": 0.0,
                "community_benefits": 0.0,
                "education_supported": 0.0,
                "healthcare_improved": 0.0,
                "diversity_promoted": 0.0,
            }

            for investment in investments:
                impact["jobs_created"] += investment.impact_metrics.get(
                    "jobs_created", 0
                )
                impact["community_benefits"] += investment.impact_metrics.get(
                    "community_benefits", 0
                )
                impact["education_supported"] += investment.impact_metrics.get(
                    "education_supported", 0
                )
                impact["healthcare_improved"] += investment.impact_metrics.get(
                    "healthcare_improved", 0
                )
                impact["diversity_promoted"] += investment.impact_metrics.get(
                    "diversity_promoted", 0
                )

            return impact

        except Exception as e:
            logger.error(f"Error calculating social impact: {e}")
            return {}

    async def _calculate_governance_impact(
        self, investments: List[SustainableInvestment]
    ) -> Dict[str, float]:
        """Calculate governance impact"""
        try:
            impact = {
                "transparency_improved": 0.0,
                "ethics_promoted": 0.0,
                "stakeholder_engagement": 0.0,
                "risk_management": 0.0,
                "compliance_enhanced": 0.0,
            }

            for investment in investments:
                impact["transparency_improved"] += investment.impact_metrics.get(
                    "transparency_improved", 0
                )
                impact["ethics_promoted"] += investment.impact_metrics.get(
                    "ethics_promoted", 0
                )
                impact["stakeholder_engagement"] += investment.impact_metrics.get(
                    "stakeholder_engagement", 0
                )
                impact["risk_management"] += investment.impact_metrics.get(
                    "risk_management", 0
                )
                impact["compliance_enhanced"] += investment.impact_metrics.get(
                    "compliance_enhanced", 0
                )

            return impact

        except Exception as e:
            logger.error(f"Error calculating governance impact: {e}")
            return {}

    async def _calculate_sustainability_goals_achieved(
        self, investments: List[SustainableInvestment]
    ) -> List[str]:
        """Calculate sustainability goals achieved"""
        try:
            goals = []

            # Check UN SDGs
            total_carbon_avoided = sum(
                inv.impact_metrics.get("carbon_avoided", 0) for inv in investments
            )
            if total_carbon_avoided > 1000:  # 1000 tons
                goals.append("SDG 13: Climate Action")

            total_jobs_created = sum(
                int(inv.impact_metrics.get("jobs_created", 0)) for inv in investments
            )
            if total_jobs_created > 100:
                goals.append("SDG 8: Decent Work and Economic Growth")

            total_renewable_energy = sum(
                inv.impact_metrics.get("renewable_energy", 0) for inv in investments
            )
            if total_renewable_energy > 1000000:  # 1M kWh
                goals.append("SDG 7: Affordable and Clean Energy")

            return goals

        except Exception as e:
            logger.error(f"Error calculating sustainability goals achieved: {e}")
            return []

    async def _calculate_sustainability_progress(
        self, investments: List[SustainableInvestment]
    ) -> Dict[str, float]:
        """Calculate sustainability progress"""
        try:
            progress = {
                "carbon_reduction": 0.0,
                "renewable_energy_adoption": 0.0,
                "social_impact": 0.0,
                "governance_improvement": 0.0,
                "overall_sustainability": 0.0,
            }

            if not investments:
                return progress

            # Calculate progress metrics
            total_investment = sum(inv.investment_amount for inv in investments)
            avg_esg_score = np.mean([inv.esg_score for inv in investments])

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
            progress["social_impact"] = (
                sum(inv.impact_metrics.get("jobs_created", 0) for inv in investments)
                / total_investment
                * 100000
            )
            progress["governance_improvement"] = avg_esg_score
            progress["overall_sustainability"] = avg_esg_score

            return progress

        except Exception as e:
            logger.error(f"Error calculating sustainability progress: {e}")
            return {}

    async def _load_esg_factors(self):
        """Load ESG factors"""
        try:
            # Environmental factors
            env_factors = [
                {
                    "category": ESGCategory.ENVIRONMENTAL,
                    "factor_name": "Carbon Emissions",
                    "description": "Greenhouse gas emissions intensity",
                    "weight": 0.3,
                    "measurement_unit": "tons CO2e per $M revenue",
                    "data_source": "CDP",
                    "is_material": True,
                    "impact_type": ImpactType.NEGATIVE,
                },
                {
                    "category": ESGCategory.ENVIRONMENTAL,
                    "factor_name": "Renewable Energy",
                    "description": "Percentage of renewable energy usage",
                    "weight": 0.25,
                    "measurement_unit": "percentage",
                    "data_source": "RE100",
                    "is_material": True,
                    "impact_type": ImpactType.POSITIVE,
                },
                {
                    "category": ESGCategory.ENVIRONMENTAL,
                    "factor_name": "Water Usage",
                    "description": "Water consumption efficiency",
                    "weight": 0.2,
                    "measurement_unit": "liters per $M revenue",
                    "data_source": "GRI",
                    "is_material": True,
                    "impact_type": ImpactType.NEGATIVE,
                },
            ]

            # Social factors
            social_factors = [
                {
                    "category": ESGCategory.SOCIAL,
                    "factor_name": "Employee Safety",
                    "description": "Workplace safety record",
                    "weight": 0.3,
                    "measurement_unit": "incidents per 1000 employees",
                    "data_source": "OSHA",
                    "is_material": True,
                    "impact_type": ImpactType.NEGATIVE,
                },
                {
                    "category": ESGCategory.SOCIAL,
                    "factor_name": "Diversity & Inclusion",
                    "description": "Workforce diversity metrics",
                    "weight": 0.25,
                    "measurement_unit": "percentage",
                    "data_source": "EEOC",
                    "is_material": True,
                    "impact_type": ImpactType.POSITIVE,
                },
                {
                    "category": ESGCategory.SOCIAL,
                    "factor_name": "Community Investment",
                    "description": "Community development spending",
                    "weight": 0.2,
                    "measurement_unit": "percentage of revenue",
                    "data_source": "GRI",
                    "is_material": True,
                    "impact_type": ImpactType.POSITIVE,
                },
            ]

            # Governance factors
            gov_factors = [
                {
                    "category": ESGCategory.GOVERNANCE,
                    "factor_name": "Board Independence",
                    "description": "Percentage of independent directors",
                    "weight": 0.3,
                    "measurement_unit": "percentage",
                    "data_source": "Proxy",
                    "is_material": True,
                    "impact_type": ImpactType.POSITIVE,
                },
                {
                    "category": ESGCategory.GOVERNANCE,
                    "factor_name": "Executive Compensation",
                    "description": "CEO pay ratio",
                    "weight": 0.25,
                    "measurement_unit": "ratio",
                    "data_source": "SEC",
                    "is_material": True,
                    "impact_type": ImpactType.NEGATIVE,
                },
                {
                    "category": ESGCategory.GOVERNANCE,
                    "factor_name": "Transparency",
                    "description": "ESG reporting quality",
                    "weight": 0.2,
                    "measurement_unit": "score",
                    "data_source": "GRI",
                    "is_material": True,
                    "impact_type": ImpactType.POSITIVE,
                },
            ]

            all_factors = env_factors + social_factors + gov_factors

            for factor_data in all_factors:
                await self.create_esg_factor(**factor_data)

            logger.info("Loaded ESG factors")

        except Exception as e:
            logger.error(f"Error loading ESG factors: {e}")

    async def _initialize_sustainability_benchmarks(self):
        """Initialize sustainability benchmarks"""
        try:
            self.sustainability_benchmarks = {
                "avg_esg_score": 50.0,
                "avg_return": 0.08,
                "carbon_intensity": 0.5,  # tons CO2e per $M
                "renewable_energy_share": 0.3,  # 30%
                "diversity_score": 0.6,  # 60%
                "governance_score": 0.7,  # 70%
            }

            logger.info("Initialized sustainability benchmarks")

        except Exception as e:
            logger.error(f"Error initializing sustainability benchmarks: {e}")


# Factory function
async def get_esg_investing_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> ESGInvestingService:
    """Get ESG Investing Service instance"""
    service = ESGInvestingService(redis_client, db_session)
    await service.initialize()
    return service
