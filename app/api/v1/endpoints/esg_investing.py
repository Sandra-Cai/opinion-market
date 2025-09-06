"""
ESG Investing API Endpoints
Advanced ESG investing and sustainable finance endpoints
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Query,
    Path,
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import redis.asyncio as redis
import json
import uuid
from enum import Enum

from app.services.esg_investing import (
    ESGInvestingService,
    ESGCategory,
    ESGScore,
    SustainabilityFramework,
    ImpactType,
    get_esg_investing_service,
)
from app.services.sustainable_finance import (
    SustainableFinanceService,
    GreenBondType,
    ClimateRiskType,
    ImpactMeasurement,
    get_sustainable_finance_service,
)
from app.schemas.esg_investing import (
    ESGCreateRequest,
    ESGResponse,
    ESGScoreCreate,
    ESGScoreResponse,
    ESGFactorCreate,
    ESGFactorResponse,
    ESGImpactCreate,
    ESGImpactResponse,
    SustainableInvestmentCreate,
    SustainableInvestmentResponse,
    ESGPortfolioCreate,
    ESGPortfolioResponse,
    ImpactReportResponse,
    ESGAnalyticsResponse,
    ESGAlertResponse,
)
from app.schemas.sustainable_finance import (
    GreenBondCreate,
    GreenBondResponse,
    ClimateRiskAssessmentCreate,
    ClimateRiskAssessmentResponse,
    SustainabilityBondCreate,
    SustainabilityBondResponse,
    GreenLoanCreate,
    GreenLoanResponse,
    CarbonCreditCreate,
    CarbonCreditResponse,
    ImpactInvestmentCreate,
    ImpactInvestmentResponse,
    SustainabilityReportResponse,
    SustainableFinanceAnalyticsResponse,
)
from app.core.database import get_db
from app.core.redis_client import get_redis_client
from app.core.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket connections
websocket_connections: List[WebSocket] = []


@router.post("/esg-scores", response_model=ESGScoreResponse)
async def create_esg_score(
    esg_score_data: ESGScoreCreate,
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Create ESG score"""
    try:
        score = await esg_service.create_esg_score(
            company_id=esg_score_data.company_id,
            overall_score=esg_score_data.overall_score,
            environmental_score=esg_score_data.environmental_score,
            social_score=esg_score_data.social_score,
            governance_score=esg_score_data.governance_score,
            framework=esg_score_data.framework,
            data_quality=esg_score_data.data_quality,
            coverage_percentage=esg_score_data.coverage_percentage,
        )

        return ESGScoreResponse(
            score_id=score.score_id,
            company_id=score.company_id,
            overall_score=score.overall_score,
            environmental_score=score.environmental_score,
            social_score=score.social_score,
            governance_score=score.governance_score,
            score_grade=score.score_grade,
            framework=score.framework.value,
            data_quality=score.data_quality,
            coverage_percentage=score.coverage_percentage,
            last_updated=score.last_updated,
            created_at=score.created_at,
        )

    except Exception as e:
        logger.error(f"Error creating ESG score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/esg-scores/{company_id}", response_model=List[ESGScoreResponse])
async def get_esg_scores(
    company_id: str = Path(..., description="Company ID"),
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Get ESG scores for a company"""
    try:
        scores = esg_service.esg_scores.get(company_id, [])

        return [
            ESGScoreResponse(
                score_id=score.score_id,
                company_id=score.company_id,
                overall_score=score.overall_score,
                environmental_score=score.environmental_score,
                social_score=score.social_score,
                governance_score=score.governance_score,
                score_grade=score.score_grade,
                framework=score.framework.value,
                data_quality=score.data_quality,
                coverage_percentage=score.coverage_percentage,
                last_updated=score.last_updated,
                created_at=score.created_at,
            )
            for score in scores
        ]

    except Exception as e:
        logger.error(f"Error getting ESG scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/esg-factors", response_model=ESGFactorResponse)
async def create_esg_factor(
    factor_data: ESGFactorCreate,
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Create ESG factor"""
    try:
        factor = await esg_service.create_esg_factor(
            category=factor_data.category,
            factor_name=factor_data.factor_name,
            description=factor_data.description,
            weight=factor_data.weight,
            measurement_unit=factor_data.measurement_unit,
            data_source=factor_data.data_source,
            is_material=factor_data.is_material,
            impact_type=factor_data.impact_type,
        )

        return ESGFactorResponse(
            factor_id=factor.factor_id,
            category=factor.category.value,
            factor_name=factor.factor_name,
            description=factor.description,
            weight=factor.weight,
            measurement_unit=factor.measurement_unit,
            data_source=factor.data_source,
            is_material=factor.is_material,
            impact_type=factor.impact_type.value,
            created_at=factor.created_at,
        )

    except Exception as e:
        logger.error(f"Error creating ESG factor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/esg-factors", response_model=List[ESGFactorResponse])
async def get_esg_factors(
    category: Optional[ESGCategory] = Query(None, description="ESG category filter"),
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Get ESG factors"""
    try:
        factors = list(esg_service.esg_factors.values())

        if category:
            factors = [f for f in factors if f.category == category]

        return [
            ESGFactorResponse(
                factor_id=factor.factor_id,
                category=factor.category.value,
                factor_name=factor.factor_name,
                description=factor.description,
                weight=factor.weight,
                measurement_unit=factor.measurement_unit,
                data_source=factor.data_source,
                is_material=factor.is_material,
                impact_type=factor.impact_type.value,
                created_at=factor.created_at,
            )
            for factor in factors
        ]

    except Exception as e:
        logger.error(f"Error getting ESG factors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/esg-impacts", response_model=ESGImpactResponse)
async def create_esg_impact(
    impact_data: ESGImpactCreate,
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Create ESG impact"""
    try:
        impact = await esg_service.create_esg_impact(
            company_id=impact_data.company_id,
            factor_id=impact_data.factor_id,
            impact_value=impact_data.impact_value,
            impact_unit=impact_data.impact_unit,
            impact_type=impact_data.impact_type,
            baseline_value=impact_data.baseline_value,
            target_value=impact_data.target_value,
        )

        return ESGImpactResponse(
            impact_id=impact.impact_id,
            company_id=impact.company_id,
            factor_id=impact.factor_id,
            impact_value=impact.impact_value,
            impact_unit=impact.impact_unit,
            impact_type=impact.impact_type.value,
            measurement_date=impact.measurement_date,
            baseline_value=impact.baseline_value,
            target_value=impact.target_value,
            progress_percentage=impact.progress_percentage,
            created_at=impact.created_at,
        )

    except Exception as e:
        logger.error(f"Error creating ESG impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sustainable-investments", response_model=SustainableInvestmentResponse)
async def create_sustainable_investment(
    investment_data: SustainableInvestmentCreate,
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Create sustainable investment"""
    try:
        investment = await esg_service.create_sustainable_investment(
            user_id=current_user["id"],
            company_id=investment_data.company_id,
            investment_amount=investment_data.investment_amount,
            esg_score=investment_data.esg_score,
            sustainability_rating=investment_data.sustainability_rating,
            impact_metrics=investment_data.impact_metrics,
            alignment_score=investment_data.alignment_score,
            risk_score=investment_data.risk_score,
            expected_return=investment_data.expected_return,
        )

        return SustainableInvestmentResponse(
            investment_id=investment.investment_id,
            user_id=investment.user_id,
            company_id=investment.company_id,
            investment_amount=investment.investment_amount,
            esg_score=investment.esg_score,
            sustainability_rating=investment.sustainability_rating,
            impact_metrics=investment.impact_metrics,
            alignment_score=investment.alignment_score,
            risk_score=investment.risk_score,
            expected_return=investment.expected_return,
            investment_date=investment.investment_date,
            created_at=investment.created_at,
        )

    except Exception as e:
        logger.error(f"Error creating sustainable investment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sustainable-investments", response_model=List[SustainableInvestmentResponse]
)
async def get_sustainable_investments(
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Get user's sustainable investments"""
    try:
        investments = esg_service.sustainable_investments.get(current_user["id"], [])

        return [
            SustainableInvestmentResponse(
                investment_id=investment.investment_id,
                user_id=investment.user_id,
                company_id=investment.company_id,
                investment_amount=investment.investment_amount,
                esg_score=investment.esg_score,
                sustainability_rating=investment.sustainability_rating,
                impact_metrics=investment.impact_metrics,
                alignment_score=investment.alignment_score,
                risk_score=investment.risk_score,
                expected_return=investment.expected_return,
                investment_date=investment.investment_date,
                created_at=investment.created_at,
            )
            for investment in investments
        ]

    except Exception as e:
        logger.error(f"Error getting sustainable investments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/esg-portfolios", response_model=ESGPortfolioResponse)
async def create_esg_portfolio(
    portfolio_data: ESGPortfolioCreate,
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Create ESG portfolio"""
    try:
        portfolio = await esg_service.create_esg_portfolio(
            user_id=current_user["id"], portfolio_name=portfolio_data.portfolio_name
        )

        return ESGPortfolioResponse(
            portfolio_id=portfolio.portfolio_id,
            user_id=portfolio.user_id,
            portfolio_name=portfolio.portfolio_name,
            total_value=portfolio.total_value,
            esg_score=portfolio.esg_score,
            carbon_footprint=portfolio.carbon_footprint,
            impact_metrics=portfolio.impact_metrics,
            sustainability_rating=portfolio.sustainability_rating,
            risk_score=portfolio.risk_score,
            expected_return=portfolio.expected_return,
            benchmark_comparison=portfolio.benchmark_comparison,
            created_at=portfolio.created_at,
            last_updated=portfolio.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating ESG portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/esg-portfolios", response_model=List[ESGPortfolioResponse])
async def get_esg_portfolios(
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Get user's ESG portfolios"""
    try:
        portfolios = esg_service.esg_portfolios.get(current_user["id"], [])

        return [
            ESGPortfolioResponse(
                portfolio_id=portfolio.portfolio_id,
                user_id=portfolio.user_id,
                portfolio_name=portfolio.portfolio_name,
                total_value=portfolio.total_value,
                esg_score=portfolio.esg_score,
                carbon_footprint=portfolio.carbon_footprint,
                impact_metrics=portfolio.impact_metrics,
                sustainability_rating=portfolio.sustainability_rating,
                risk_score=portfolio.risk_score,
                expected_return=portfolio.expected_return,
                benchmark_comparison=portfolio.benchmark_comparison,
                created_at=portfolio.created_at,
                last_updated=portfolio.last_updated,
            )
            for portfolio in portfolios
        ]

    except Exception as e:
        logger.error(f"Error getting ESG portfolios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/impact-reports", response_model=List[ImpactReportResponse])
async def get_impact_reports(
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Get user's impact reports"""
    try:
        reports = esg_service.impact_reports.get(current_user["id"], [])

        return [
            ImpactReportResponse(
                report_id=report.report_id,
                user_id=report.user_id,
                report_period=report.report_period,
                total_investment=report.total_investment,
                environmental_impact=report.environmental_impact,
                social_impact=report.social_impact,
                governance_impact=report.governance_impact,
                carbon_avoided=report.carbon_avoided,
                jobs_created=report.jobs_created,
                community_benefits=report.community_benefits,
                sustainability_goals_achieved=report.sustainability_goals_achieved,
                created_at=report.created_at,
            )
            for report in reports
        ]

    except Exception as e:
        logger.error(f"Error getting impact reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/esg-analytics", response_model=ESGAnalyticsResponse)
async def get_esg_analytics(
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Get ESG analytics for user"""
    try:
        analytics = await esg_service.get_esg_analytics(current_user["id"])

        if "error" in analytics:
            raise HTTPException(status_code=404, detail=analytics["error"])

        return ESGAnalyticsResponse(**analytics)

    except Exception as e:
        logger.error(f"Error getting ESG analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/esg-alerts/{company_id}", response_model=List[ESGAlertResponse])
async def get_esg_alerts(
    company_id: str = Path(..., description="Company ID"),
    current_user: dict = Depends(get_current_user),
    esg_service: ESGInvestingService = Depends(get_esg_investing_service),
    db: Session = Depends(get_db),
):
    """Get ESG alerts for a company"""
    try:
        alerts = esg_service.esg_alerts.get(company_id, [])

        return [
            ESGAlertResponse(
                alert_id=alert.alert_id,
                company_id=alert.company_id,
                alert_type=alert.alert_type,
                severity=alert.severity,
                description=alert.description,
                impact_score=alert.impact_score,
                recommendation=alert.recommendation,
                alert_date=alert.alert_date,
                status=alert.status,
                created_at=alert.created_at,
            )
            for alert in alerts
        ]

    except Exception as e:
        logger.error(f"Error getting ESG alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Green Bonds endpoints
@router.post("/green-bonds", response_model=GreenBondResponse)
async def create_green_bond(
    bond_data: GreenBondCreate,
    current_user: dict = Depends(get_current_user),
    finance_service: SustainableFinanceService = Depends(
        get_sustainable_finance_service
    ),
    db: Session = Depends(get_db),
):
    """Create green bond"""
    try:
        bond = await finance_service.create_green_bond(
            issuer=bond_data.issuer,
            bond_type=bond_data.bond_type,
            issue_amount=bond_data.issue_amount,
            currency=bond_data.currency,
            maturity_date=bond_data.maturity_date,
            coupon_rate=bond_data.coupon_rate,
            use_of_proceeds=bond_data.use_of_proceeds,
            green_project_categories=bond_data.green_project_categories,
            third_party_verification=bond_data.third_party_verification,
            impact_reporting=bond_data.impact_reporting,
        )

        return GreenBondResponse(
            bond_id=bond.bond_id,
            issuer=bond.issuer,
            bond_type=bond.bond_type.value,
            issue_amount=bond.issue_amount,
            currency=bond.currency,
            maturity_date=bond.maturity_date,
            coupon_rate=bond.coupon_rate,
            use_of_proceeds=bond.use_of_proceeds,
            green_project_categories=bond.green_project_categories,
            third_party_verification=bond.third_party_verification,
            impact_reporting=bond.impact_reporting,
            is_active=bond.is_active,
            created_at=bond.created_at,
            last_updated=bond.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating green bond: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/green-bonds", response_model=List[GreenBondResponse])
async def get_green_bonds(
    issuer: Optional[str] = Query(None, description="Filter by issuer"),
    current_user: dict = Depends(get_current_user),
    finance_service: SustainableFinanceService = Depends(
        get_sustainable_finance_service
    ),
    db: Session = Depends(get_db),
):
    """Get green bonds"""
    try:
        all_bonds = []
        for issuer_bonds in finance_service.green_bonds.values():
            all_bonds.extend(issuer_bonds)

        if issuer:
            all_bonds = [bond for bond in all_bonds if bond.issuer == issuer]

        return [
            GreenBondResponse(
                bond_id=bond.bond_id,
                issuer=bond.issuer,
                bond_type=bond.bond_type.value,
                issue_amount=bond.issue_amount,
                currency=bond.currency,
                maturity_date=bond.maturity_date,
                coupon_rate=bond.coupon_rate,
                use_of_proceeds=bond.use_of_proceeds,
                green_project_categories=bond.green_project_categories,
                third_party_verification=bond.third_party_verification,
                impact_reporting=bond.impact_reporting,
                is_active=bond.is_active,
                created_at=bond.created_at,
                last_updated=bond.last_updated,
            )
            for bond in all_bonds
        ]

    except Exception as e:
        logger.error(f"Error getting green bonds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Carbon Credits endpoints
@router.post("/carbon-credits", response_model=CarbonCreditResponse)
async def create_carbon_credit(
    credit_data: CarbonCreditCreate,
    current_user: dict = Depends(get_current_user),
    finance_service: SustainableFinanceService = Depends(
        get_sustainable_finance_service
    ),
    db: Session = Depends(get_db),
):
    """Create carbon credit"""
    try:
        credit = await finance_service.create_carbon_credit(
            project_id=credit_data.project_id,
            project_name=credit_data.project_name,
            project_type=credit_data.project_type,
            standard=credit_data.standard,
            vintage_year=credit_data.vintage_year,
            credit_amount=credit_data.credit_amount,
            price_per_credit=credit_data.price_per_credit,
            verification_status=credit_data.verification_status,
            retirement_status=credit_data.retirement_status,
            environmental_benefits=credit_data.environmental_benefits,
        )

        return CarbonCreditResponse(
            credit_id=credit.credit_id,
            project_id=credit.project_id,
            project_name=credit.project_name,
            project_type=credit.project_type,
            standard=credit.standard,
            vintage_year=credit.vintage_year,
            credit_amount=credit.credit_amount,
            price_per_credit=credit.price_per_credit,
            total_value=credit.total_value,
            verification_status=credit.verification_status,
            retirement_status=credit.retirement_status,
            environmental_benefits=credit.environmental_benefits,
            created_at=credit.created_at,
            last_updated=credit.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating carbon credit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/carbon-credits", response_model=List[CarbonCreditResponse])
async def get_carbon_credits(
    project_type: Optional[str] = Query(None, description="Filter by project type"),
    standard: Optional[str] = Query(None, description="Filter by standard"),
    current_user: dict = Depends(get_current_user),
    finance_service: SustainableFinanceService = Depends(
        get_sustainable_finance_service
    ),
    db: Session = Depends(get_db),
):
    """Get carbon credits"""
    try:
        all_credits = []
        for project_credits in finance_service.carbon_credits.values():
            all_credits.extend(project_credits)

        if project_type:
            all_credits = [
                credit for credit in all_credits if credit.project_type == project_type
            ]

        if standard:
            all_credits = [
                credit for credit in all_credits if credit.standard == standard
            ]

        return [
            CarbonCreditResponse(
                credit_id=credit.credit_id,
                project_id=credit.project_id,
                project_name=credit.project_name,
                project_type=credit.project_type,
                standard=credit.standard,
                vintage_year=credit.vintage_year,
                credit_amount=credit.credit_amount,
                price_per_credit=credit.price_per_credit,
                total_value=credit.total_value,
                verification_status=credit.verification_status,
                retirement_status=credit.retirement_status,
                environmental_benefits=credit.environmental_benefits,
                created_at=credit.created_at,
                last_updated=credit.last_updated,
            )
            for credit in all_credits
        ]

    except Exception as e:
        logger.error(f"Error getting carbon credits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Impact Investments endpoints
@router.post("/impact-investments", response_model=ImpactInvestmentResponse)
async def create_impact_investment(
    investment_data: ImpactInvestmentCreate,
    current_user: dict = Depends(get_current_user),
    finance_service: SustainableFinanceService = Depends(
        get_sustainable_finance_service
    ),
    db: Session = Depends(get_db),
):
    """Create impact investment"""
    try:
        investment = await finance_service.create_impact_investment(
            user_id=current_user["id"],
            investment_type=investment_data.investment_type,
            investment_amount=investment_data.investment_amount,
            target_impact=investment_data.target_impact,
            impact_metrics=investment_data.impact_metrics,
            expected_return=investment_data.expected_return,
            risk_level=investment_data.risk_level,
            maturity_date=investment_data.maturity_date,
        )

        return ImpactInvestmentResponse(
            investment_id=investment.investment_id,
            user_id=investment.user_id,
            investment_type=investment.investment_type,
            investment_amount=investment.investment_amount,
            target_impact=investment.target_impact,
            impact_metrics=investment.impact_metrics,
            expected_return=investment.expected_return,
            risk_level=investment.risk_level,
            investment_date=investment.investment_date,
            maturity_date=investment.maturity_date,
            is_active=investment.is_active,
            created_at=investment.created_at,
            last_updated=investment.last_updated,
        )

    except Exception as e:
        logger.error(f"Error creating impact investment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/impact-investments", response_model=List[ImpactInvestmentResponse])
async def get_impact_investments(
    current_user: dict = Depends(get_current_user),
    finance_service: SustainableFinanceService = Depends(
        get_sustainable_finance_service
    ),
    db: Session = Depends(get_db),
):
    """Get user's impact investments"""
    try:
        investments = finance_service.impact_investments.get(current_user["id"], [])

        return [
            ImpactInvestmentResponse(
                investment_id=investment.investment_id,
                user_id=investment.user_id,
                investment_type=investment.investment_type,
                investment_amount=investment.investment_amount,
                target_impact=investment.target_impact,
                impact_metrics=investment.impact_metrics,
                expected_return=investment.expected_return,
                risk_level=investment.risk_level,
                investment_date=investment.investment_date,
                maturity_date=investment.maturity_date,
                is_active=investment.is_active,
                created_at=investment.created_at,
                last_updated=investment.last_updated,
            )
            for investment in investments
        ]

    except Exception as e:
        logger.error(f"Error getting impact investments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sustainable-finance-analytics", response_model=SustainableFinanceAnalyticsResponse
)
async def get_sustainable_finance_analytics(
    current_user: dict = Depends(get_current_user),
    finance_service: SustainableFinanceService = Depends(
        get_sustainable_finance_service
    ),
    db: Session = Depends(get_db),
):
    """Get sustainable finance analytics for user"""
    try:
        analytics = await finance_service.get_sustainable_finance_analytics(
            current_user["id"]
        )

        if "error" in analytics:
            raise HTTPException(status_code=404, detail=analytics["error"])

        return SustainableFinanceAnalyticsResponse(**analytics)

    except Exception as e:
        logger.error(f"Error getting sustainable finance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time ESG updates
@router.websocket("/ws/esg-updates")
async def websocket_esg_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time ESG updates"""
    await websocket.accept()
    websocket_connections.append(websocket)

    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(30)  # Send updates every 30 seconds

            # Get latest ESG data
            update_data = {
                "type": "esg_update",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "ESG data updated",
            }

            await websocket.send_text(json.dumps(update_data))

    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
