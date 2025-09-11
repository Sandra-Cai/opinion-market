"""
Compliance Service
Provides KYC, AML, regulatory reporting, and compliance monitoring
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
import hashlib
import re
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance status"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"
    SUSPENDED = "suspended"
    EXPIRED = "expired"


class RiskLevel(Enum):
    """Risk levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DocumentType(Enum):
    """Document types"""

    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"
    TAX_DOCUMENT = "tax_document"
    BUSINESS_LICENSE = "business_license"
    ARTICLES_OF_INCORPORATION = "articles_of_incorporation"


@dataclass
class ComplianceProfile:
    """User compliance profile"""

    profile_id: str
    user_id: int
    compliance_status: ComplianceStatus
    risk_level: RiskLevel
    kyc_status: ComplianceStatus
    aml_status: ComplianceStatus
    verification_level: int
    documents_verified: List[str]
    last_review: datetime
    next_review: datetime
    compliance_notes: List[str]
    created_at: datetime
    last_updated: datetime


@dataclass
class DocumentVerification:
    """Document verification record"""

    verification_id: str
    user_id: int
    document_type: DocumentType
    document_number: str
    document_hash: str
    verification_status: ComplianceStatus
    verification_date: Optional[datetime]
    verified_by: Optional[str]
    verification_notes: List[str]
    document_url: str
    expiry_date: Optional[datetime]
    created_at: datetime
    last_updated: datetime


@dataclass
class ComplianceRule:
    """Compliance rule definition"""

    rule_id: str
    rule_name: str
    rule_type: str  # 'kyc', 'aml', 'trading', 'reporting'
    description: str
    conditions: Dict[str, Any]
    actions: List[str]
    risk_score: int
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class ComplianceAlert:
    """Compliance alert"""

    alert_id: str
    user_id: int
    rule_id: str
    alert_type: str
    severity: RiskLevel
    description: str
    triggered_data: Dict[str, Any]
    status: str  # 'open', 'investigating', 'resolved', 'false_positive'
    assigned_to: Optional[str]
    created_at: datetime
    resolved_at: Optional[datetime]
    last_updated: datetime


@dataclass
class RegulatoryReport:
    """Regulatory report"""

    report_id: str
    report_type: str  # 'suspicious_activity', 'large_transaction', 'periodic'
    user_id: int
    report_data: Dict[str, Any]
    submission_date: datetime
    regulatory_body: str
    report_status: str  # 'draft', 'submitted', 'acknowledged', 'rejected'
    acknowledgment_reference: Optional[str]
    created_at: datetime
    last_updated: datetime


@dataclass
class ComplianceAudit:
    """Compliance audit record"""

    audit_id: str
    audit_type: str  # 'user_review', 'system_audit', 'regulatory_audit'
    target_id: str
    auditor: str
    audit_date: datetime
    findings: List[str]
    recommendations: List[str]
    compliance_score: float
    risk_assessment: RiskLevel
    created_at: datetime
    last_updated: datetime


class ComplianceService:
    """Comprehensive compliance and regulatory service"""

    def __init__(self, redis_client: redis_sync.Redis, db_session: Session):
        self.redis = redis_client
        self.db = db_session
        self.compliance_profiles: Dict[int, ComplianceProfile] = {}
        self.document_verifications: Dict[str, DocumentVerification] = {}
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.compliance_alerts: Dict[str, ComplianceAlert] = {}
        self.regulatory_reports: Dict[str, RegulatoryReport] = {}
        self.compliance_audits: Dict[str, ComplianceAudit] = {}

        # Compliance data
        self.risk_scores: Dict[int, float] = {}
        self.compliance_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.alert_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Monitoring
        self.active_monitors: Dict[str, asyncio.Task] = {}
        self.compliance_metrics: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Initialize the compliance service"""
        logger.info("Initializing Compliance Service")

        # Load existing data
        await self._load_compliance_profiles()
        await self._load_compliance_rules()
        await self._load_document_verifications()

        # Initialize default rules
        await self._initialize_default_rules()

        # Start background tasks
        asyncio.create_task(self._monitor_compliance())
        asyncio.create_task(self._update_risk_scores())
        asyncio.create_task(self._generate_regulatory_reports())

        logger.info("Compliance Service initialized successfully")

    async def create_compliance_profile(self, user_id: int) -> ComplianceProfile:
        """Create a new compliance profile"""
        try:
            profile = ComplianceProfile(
                profile_id=f"compliance_profile_{user_id}",
                user_id=user_id,
                compliance_status=ComplianceStatus.PENDING,
                risk_level=RiskLevel.MEDIUM,
                kyc_status=ComplianceStatus.PENDING,
                aml_status=ComplianceStatus.PENDING,
                verification_level=0,
                documents_verified=[],
                last_review=datetime.utcnow(),
                next_review=datetime.utcnow() + timedelta(days=30),
                compliance_notes=["Profile created"],
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.compliance_profiles[user_id] = profile
            await self._cache_compliance_profile(profile)

            logger.info(f"Created compliance profile for user {user_id}")
            return profile

        except Exception as e:
            logger.error(f"Error creating compliance profile: {e}")
            raise

    async def verify_document(
        self,
        user_id: int,
        document_type: DocumentType,
        document_number: str,
        document_url: str,
        expiry_date: Optional[datetime] = None,
    ) -> DocumentVerification:
        """Verify a user document"""
        try:
            # Generate document hash
            document_hash = hashlib.sha256(
                f"{user_id}_{document_number}_{document_type.value}".encode()
            ).hexdigest()

            verification = DocumentVerification(
                verification_id=f"verification_{user_id}_{document_type.value}_{uuid.uuid4().hex[:8]}",
                user_id=user_id,
                document_type=document_type,
                document_number=document_number,
                document_hash=document_hash,
                verification_status=ComplianceStatus.PENDING,
                verification_date=None,
                verified_by=None,
                verification_notes=["Document submitted for verification"],
                document_url=document_url,
                expiry_date=expiry_date,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.document_verifications[verification.verification_id] = verification
            await self._cache_document_verification(verification)

            # Start automated verification
            asyncio.create_task(self._automated_document_verification(verification))

            logger.info(f"Started document verification for user {user_id}")
            return verification

        except Exception as e:
            logger.error(f"Error starting document verification: {e}")
            raise

    async def approve_document(
        self, verification_id: str, verified_by: str, notes: Optional[str] = None
    ) -> bool:
        """Approve a document verification"""
        try:
            if verification_id not in self.document_verifications:
                raise ValueError(f"Verification {verification_id} not found")

            verification = self.document_verifications[verification_id]
            verification.verification_status = ComplianceStatus.APPROVED
            verification.verification_date = datetime.utcnow()
            verification.verified_by = verified_by

            if notes:
                verification.verification_notes.append(notes)

            verification.last_updated = datetime.utcnow()
            await self._cache_document_verification(verification)

            # Update user compliance profile
            await self._update_user_compliance_status(verification.user_id)

            logger.info(f"Approved document verification {verification_id}")
            return True

        except Exception as e:
            logger.error(f"Error approving document: {e}")
            raise

    async def reject_document(
        self, verification_id: str, rejected_by: str, reason: str
    ) -> bool:
        """Reject a document verification"""
        try:
            if verification_id not in self.document_verifications:
                raise ValueError(f"Verification {verification_id} not found")

            verification = self.document_verifications[verification_id]
            verification.verification_status = ComplianceStatus.REJECTED
            verification.verification_date = datetime.utcnow()
            verification.verified_by = rejected_by
            verification.verification_notes.append(f"Rejected: {reason}")

            verification.last_updated = datetime.utcnow()
            await self._cache_document_verification(verification)

            # Update user compliance status
            await self._update_user_compliance_status(verification.user_id)

            logger.info(f"Rejected document verification {verification_id}")
            return True

        except Exception as e:
            logger.error(f"Error rejecting document: {e}")
            raise

    async def create_compliance_rule(
        self,
        rule_name: str,
        rule_type: str,
        description: str,
        conditions: Dict[str, Any],
        actions: List[str],
        risk_score: int,
    ) -> ComplianceRule:
        """Create a new compliance rule"""
        try:
            rule = ComplianceRule(
                rule_id=f"rule_{rule_type}_{uuid.uuid4().hex[:8]}",
                rule_name=rule_name,
                rule_type=rule_type,
                description=description,
                conditions=conditions,
                actions=actions,
                risk_score=risk_score,
                is_active=True,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.compliance_rules[rule.rule_id] = rule
            await self._cache_compliance_rule(rule)

            logger.info(f"Created compliance rule {rule_name}")
            return rule

        except Exception as e:
            logger.error(f"Error creating compliance rule: {e}")
            raise

    async def check_compliance(
        self, user_id: int, activity_type: str, activity_data: Dict[str, Any]
    ) -> List[ComplianceAlert]:
        """Check compliance for user activity"""
        try:
            alerts = []

            # Get user compliance profile
            profile = self.compliance_profiles.get(user_id)
            if not profile:
                profile = await self.create_compliance_profile(user_id)

            # Check all applicable rules
            for rule in self.compliance_rules.values():
                if not rule.is_active:
                    continue

                if rule.rule_type in ["general", activity_type]:
                    if await self._evaluate_rule(rule, user_id, activity_data):
                        alert = await self._create_compliance_alert(
                            user_id, rule, activity_data
                        )
                        alerts.append(alert)

            # Update risk score
            await self._update_user_risk_score(user_id, alerts)

            logger.info(f"Completed compliance check for user {user_id}")
            return alerts

        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
            raise

    async def generate_regulatory_report(
        self,
        report_type: str,
        user_id: int,
        report_data: Dict[str, Any],
        regulatory_body: str,
    ) -> RegulatoryReport:
        """Generate a regulatory report"""
        try:
            report = RegulatoryReport(
                report_id=f"report_{report_type}_{user_id}_{uuid.uuid4().hex[:8]}",
                report_type=report_type,
                user_id=user_id,
                report_data=report_data,
                submission_date=datetime.utcnow(),
                regulatory_body=regulatory_body,
                report_status="draft",
                acknowledgment_reference=None,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
            )

            self.regulatory_reports[report.report_id] = report
            await self._cache_regulatory_report(report)

            # Submit report to regulatory body
            asyncio.create_task(self._submit_regulatory_report(report))

            logger.info(f"Generated regulatory report {report.report_id}")
            return report

        except Exception as e:
            logger.error(f"Error generating regulatory report: {e}")
            raise

    async def get_compliance_profile(self, user_id: int) -> ComplianceProfile:
        """Get user compliance profile"""
        try:
            profile = self.compliance_profiles.get(user_id)

            if not profile:
                profile = await self.create_compliance_profile(user_id)

            return profile

        except Exception as e:
            logger.error(f"Error getting compliance profile: {e}")
            raise

    async def get_user_risk_score(self, user_id: int) -> float:
        """Get user risk score"""
        try:
            return self.risk_scores.get(user_id, 50.0)  # Default medium risk

        except Exception as e:
            logger.error(f"Error getting user risk score: {e}")
            return 50.0

    async def _automated_document_verification(
        self, verification: DocumentVerification
    ):
        """Perform automated document verification"""
        try:
            # Simulate automated verification process
            await asyncio.sleep(5)  # Simulate processing time

            # Basic validation checks
            is_valid = await self._validate_document(verification)

            if is_valid:
                verification.verification_status = ComplianceStatus.APPROVED
                verification.verification_date = datetime.utcnow()
                verification.verified_by = "automated_system"
                verification.verification_notes.append("Automated verification passed")
            else:
                verification.verification_status = ComplianceStatus.REJECTED
                verification.verification_date = datetime.utcnow()
                verification.verified_by = "automated_system"
                verification.verification_notes.append("Automated verification failed")

            verification.last_updated = datetime.utcnow()
            await self._cache_document_verification(verification)

            # Update user compliance status
            await self._update_user_compliance_status(verification.user_id)

            logger.info(
                f"Completed automated verification for {verification.verification_id}"
            )

        except Exception as e:
            logger.error(f"Error in automated verification: {e}")

    async def _validate_document(self, verification: DocumentVerification) -> bool:
        """Validate document authenticity"""
        try:
            # Simulate document validation
            # In practice, this would use OCR, AI, and other validation techniques

            # Check document number format
            if verification.document_type == DocumentType.PASSPORT:
                # Basic passport number validation
                if not re.match(r"^[A-Z0-9]{6,9}$", verification.document_number):
                    return False

            # Check expiry date
            if (
                verification.expiry_date
                and verification.expiry_date < datetime.utcnow()
            ):
                return False

            # Simulate validation success (90% success rate)
            return np.random.random() > 0.1

        except Exception as e:
            logger.error(f"Error validating document: {e}")
            return False

    async def _evaluate_rule(
        self, rule: ComplianceRule, user_id: int, activity_data: Dict[str, Any]
    ) -> bool:
        """Evaluate if a compliance rule is triggered"""
        try:
            conditions = rule.conditions

            # Check user risk level
            if "max_risk_level" in conditions:
                profile = self.compliance_profiles.get(user_id)
                if profile and profile.risk_level.value > conditions["max_risk_level"]:
                    return True

            # Check transaction amount
            if "max_transaction_amount" in conditions:
                amount = activity_data.get("amount", 0)
                if amount > conditions["max_transaction_amount"]:
                    return True

            # Check frequency
            if "max_frequency" in conditions:
                frequency = activity_data.get("frequency", 0)
                if frequency > conditions["max_frequency"]:
                    return True

            # Check geographic restrictions
            if "restricted_countries" in conditions:
                country = activity_data.get("country", "")
                if country in conditions["restricted_countries"]:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error evaluating rule: {e}")
            return False

    async def _create_compliance_alert(
        self, user_id: int, rule: ComplianceRule, activity_data: Dict[str, Any]
    ) -> ComplianceAlert:
        """Create a compliance alert"""
        try:
            alert = ComplianceAlert(
                alert_id=f"alert_{user_id}_{rule.rule_id}_{uuid.uuid4().hex[:8]}",
                user_id=user_id,
                rule_id=rule.rule_id,
                alert_type=rule.rule_type,
                severity=RiskLevel.HIGH if rule.risk_score > 7 else RiskLevel.MEDIUM,
                description=f"Rule '{rule.rule_name}' triggered",
                triggered_data=activity_data,
                status="open",
                assigned_to=None,
                created_at=datetime.utcnow(),
                resolved_at=None,
                last_updated=datetime.utcnow(),
            )

            self.compliance_alerts[alert.alert_id] = alert
            await self._cache_compliance_alert(alert)

            # Store alert history
            self.alert_history[user_id].append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "alert_type": alert.alert_type,
                    "severity": alert.severity.value,
                    "description": alert.description,
                }
            )

            return alert

        except Exception as e:
            logger.error(f"Error creating compliance alert: {e}")
            raise

    async def _update_user_compliance_status(self, user_id: int):
        """Update user compliance status based on verifications"""
        try:
            profile = self.compliance_profiles.get(user_id)
            if not profile:
                return

            # Get user verifications
            user_verifications = [
                v for v in self.document_verifications.values() if v.user_id == user_id
            ]

            # Count approved verifications
            approved_count = len(
                [
                    v
                    for v in user_verifications
                    if v.verification_status == ComplianceStatus.APPROVED
                ]
            )

            # Update verification level
            if approved_count >= 3:
                profile.verification_level = 3
                profile.kyc_status = ComplianceStatus.APPROVED
            elif approved_count >= 2:
                profile.verification_level = 2
                profile.kyc_status = ComplianceStatus.UNDER_REVIEW
            elif approved_count >= 1:
                profile.verification_level = 1
                profile.kyc_status = ComplianceStatus.PENDING
            else:
                profile.verification_level = 0
                profile.kyc_status = ComplianceStatus.PENDING

            # Update overall compliance status
            if (
                profile.kyc_status == ComplianceStatus.APPROVED
                and profile.aml_status == ComplianceStatus.APPROVED
            ):
                profile.compliance_status = ComplianceStatus.APPROVED
            elif (
                profile.kyc_status == ComplianceStatus.REJECTED
                or profile.aml_status == ComplianceStatus.REJECTED
            ):
                profile.compliance_status = ComplianceStatus.REJECTED
            else:
                profile.compliance_status = ComplianceStatus.UNDER_REVIEW

            profile.last_updated = datetime.utcnow()
            await self._cache_compliance_profile(profile)

        except Exception as e:
            logger.error(f"Error updating user compliance status: {e}")

    async def _update_user_risk_score(
        self, user_id: int, alerts: List[ComplianceAlert]
    ):
        """Update user risk score based on compliance alerts"""
        try:
            base_score = 50.0

            # Adjust based on alert severity
            for alert in alerts:
                if alert.severity == RiskLevel.CRITICAL:
                    base_score += 20
                elif alert.severity == RiskLevel.HIGH:
                    base_score += 15
                elif alert.severity == RiskLevel.MEDIUM:
                    base_score += 10
                elif alert.severity == RiskLevel.LOW:
                    base_score += 5

            # Get profile risk level
            profile = self.compliance_profiles.get(user_id)
            if profile:
                if profile.risk_level == RiskLevel.CRITICAL:
                    base_score += 20
                elif profile.risk_level == RiskLevel.HIGH:
                    base_score += 15
                elif profile.risk_level == RiskLevel.MEDIUM:
                    base_score += 10

            # Cap risk score at 100
            final_score = min(100.0, max(0.0, base_score))

            self.risk_scores[user_id] = final_score

            # Update profile risk level
            if profile:
                if final_score >= 80:
                    profile.risk_level = RiskLevel.CRITICAL
                elif final_score >= 60:
                    profile.risk_level = RiskLevel.HIGH
                elif final_score >= 40:
                    profile.risk_level = RiskLevel.MEDIUM
                else:
                    profile.risk_level = RiskLevel.LOW

                profile.last_updated = datetime.utcnow()
                await self._cache_compliance_profile(profile)

        except Exception as e:
            logger.error(f"Error updating user risk score: {e}")

    async def _submit_regulatory_report(self, report: RegulatoryReport):
        """Submit report to regulatory body"""
        try:
            # Simulate regulatory submission
            await asyncio.sleep(2)

            # Update report status
            report.report_status = "submitted"
            report.acknowledgment_reference = f"REF_{uuid.uuid4().hex[:8].upper()}"
            report.last_updated = datetime.utcnow()

            await self._cache_regulatory_report(report)

            logger.info(f"Submitted regulatory report {report.report_id}")

        except Exception as e:
            logger.error(f"Error submitting regulatory report: {e}")

    async def _initialize_default_rules(self):
        """Initialize default compliance rules"""
        try:
            # KYC rules
            await self.create_compliance_rule(
                "Document Verification Required",
                "kyc",
                "Users must verify identity documents",
                {"min_verification_level": 2},
                ["require_document_verification"],
                5,
            )

            # AML rules
            await self.create_compliance_rule(
                "Large Transaction Monitoring",
                "aml",
                "Monitor large transactions for suspicious activity",
                {"max_transaction_amount": 10000},
                ["flag_for_review", "generate_report"],
                8,
            )

            # Trading rules
            await self.create_compliance_rule(
                "High-Frequency Trading Check",
                "trading",
                "Monitor for excessive trading activity",
                {"max_frequency": 100},
                ["flag_for_review", "limit_trading"],
                6,
            )

            # Geographic rules
            await self.create_compliance_rule(
                "Restricted Country Check",
                "trading",
                "Block transactions from restricted countries",
                {"restricted_countries": ["CU", "IR", "KP", "SD", "SY"]},
                ["block_transaction", "generate_report"],
                9,
            )

        except Exception as e:
            logger.error(f"Error initializing default rules: {e}")

    # Background tasks
    async def _monitor_compliance(self):
        """Monitor compliance across all users"""
        while True:
            try:
                # Check for compliance violations
                for user_id, profile in self.compliance_profiles.items():
                    if profile.compliance_status == ComplianceStatus.APPROVED:
                        # Periodic compliance checks
                        if datetime.utcnow() > profile.next_review:
                            await self._perform_compliance_review(user_id)

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in compliance monitoring: {e}")
                await asyncio.sleep(7200)

    async def _update_risk_scores(self):
        """Update risk scores for all users"""
        while True:
            try:
                # Update risk scores based on recent activity
                for user_id in self.compliance_profiles:
                    # Simulate risk score updates
                    current_score = self.risk_scores.get(user_id, 50.0)
                    change = np.random.normal(0, 2)  # Small random change
                    new_score = max(0, min(100, current_score + change))
                    self.risk_scores[user_id] = new_score

                await asyncio.sleep(1800)  # Update every 30 minutes

            except Exception as e:
                logger.error(f"Error updating risk scores: {e}")
                await asyncio.sleep(3600)

    async def _generate_regulatory_reports(self):
        """Generate periodic regulatory reports"""
        while True:
            try:
                # Generate periodic reports
                for user_id, profile in self.compliance_profiles.items():
                    if profile.compliance_status == ComplianceStatus.APPROVED:
                        # Check if periodic report is due
                        if datetime.utcnow() > profile.next_review:
                            await self._generate_periodic_report(user_id)

                await asyncio.sleep(86400)  # Check daily

            except Exception as e:
                logger.error(f"Error generating regulatory reports: {e}")
                await asyncio.sleep(172800)

    async def _perform_compliance_review(self, user_id: int):
        """Perform periodic compliance review"""
        try:
            profile = self.compliance_profiles[user_id]

            # Simulate compliance review
            review_score = np.random.uniform(0.7, 1.0)

            if review_score > 0.9:
                profile.compliance_status = ComplianceStatus.APPROVED
            elif review_score > 0.7:
                profile.compliance_status = ComplianceStatus.UNDER_REVIEW
            else:
                profile.compliance_status = ComplianceStatus.SUSPENDED

            profile.last_review = datetime.utcnow()
            profile.next_review = datetime.utcnow() + timedelta(days=30)
            profile.last_updated = datetime.utcnow()

            await self._cache_compliance_profile(profile)

            logger.info(f"Completed compliance review for user {user_id}")

        except Exception as e:
            logger.error(f"Error performing compliance review: {e}")

    async def _generate_periodic_report(self, user_id: int):
        """Generate periodic regulatory report"""
        try:
            profile = self.compliance_profiles[user_id]

            report_data = {
                "user_id": user_id,
                "compliance_status": profile.compliance_status.value,
                "risk_level": profile.risk_level.value,
                "verification_level": profile.verification_level,
                "last_review": profile.last_review.isoformat(),
                "risk_score": self.risk_scores.get(user_id, 50.0),
            }

            await self.generate_regulatory_report(
                "periodic", user_id, report_data, "SEC"  # Example regulatory body
            )

        except Exception as e:
            logger.error(f"Error generating periodic report: {e}")

    # Helper methods
    async def _load_compliance_profiles(self):
        """Load compliance profiles from database"""
        pass

    async def _load_compliance_rules(self):
        """Load compliance rules from database"""
        pass

    async def _load_document_verifications(self):
        """Load document verifications from database"""
        pass

    # Caching methods
    async def _cache_compliance_profile(self, profile: ComplianceProfile):
        """Cache compliance profile"""
        try:
            cache_key = f"compliance_profile:{profile.user_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "compliance_status": profile.compliance_status.value,
                        "risk_level": profile.risk_level.value,
                        "kyc_status": profile.kyc_status.value,
                        "aml_status": profile.aml_status.value,
                        "verification_level": profile.verification_level,
                        "documents_verified": profile.documents_verified,
                        "last_review": profile.last_review.isoformat(),
                        "next_review": profile.next_review.isoformat(),
                        "compliance_notes": profile.compliance_notes,
                        "created_at": profile.created_at.isoformat(),
                        "last_updated": profile.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching compliance profile: {e}")

    async def _cache_document_verification(self, verification: DocumentVerification):
        """Cache document verification"""
        try:
            cache_key = f"document_verification:{verification.verification_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "user_id": verification.user_id,
                        "document_type": verification.document_type.value,
                        "document_number": verification.document_number,
                        "verification_status": verification.verification_status.value,
                        "verification_date": (
                            verification.verification_date.isoformat()
                            if verification.verification_date
                            else None
                        ),
                        "verified_by": verification.verified_by,
                        "verification_notes": verification.verification_notes,
                        "document_url": verification.document_url,
                        "expiry_date": (
                            verification.expiry_date.isoformat()
                            if verification.expiry_date
                            else None
                        ),
                        "created_at": verification.created_at.isoformat(),
                        "last_updated": verification.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching document verification: {e}")

    async def _cache_compliance_rule(self, rule: ComplianceRule):
        """Cache compliance rule"""
        try:
            cache_key = f"compliance_rule:{rule.rule_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "rule_name": rule.rule_name,
                        "rule_type": rule.rule_type,
                        "description": rule.description,
                        "conditions": rule.conditions,
                        "actions": rule.actions,
                        "risk_score": rule.risk_score,
                        "is_active": rule.is_active,
                        "created_at": rule.created_at.isoformat(),
                        "last_updated": rule.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching compliance rule: {e}")

    async def _cache_compliance_alert(self, alert: ComplianceAlert):
        """Cache compliance alert"""
        try:
            cache_key = f"compliance_alert:{alert.alert_id}"
            await self.redis.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(
                    {
                        "user_id": alert.user_id,
                        "rule_id": alert.rule_id,
                        "alert_type": alert.alert_type,
                        "severity": alert.severity.value,
                        "description": alert.description,
                        "triggered_data": alert.triggered_data,
                        "status": alert.status,
                        "assigned_to": alert.assigned_to,
                        "created_at": alert.created_at.isoformat(),
                        "resolved_at": (
                            alert.resolved_at.isoformat() if alert.resolved_at else None
                        ),
                        "last_updated": alert.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching compliance alert: {e}")

    async def _cache_regulatory_report(self, report: RegulatoryReport):
        """Cache regulatory report"""
        try:
            cache_key = f"regulatory_report:{report.report_id}"
            await self.redis.setex(
                cache_key,
                7200,  # 2 hours TTL
                json.dumps(
                    {
                        "report_type": report.report_type,
                        "user_id": report.user_id,
                        "report_data": report.report_data,
                        "submission_date": report.submission_date.isoformat(),
                        "regulatory_body": report.regulatory_body,
                        "report_status": report.report_status,
                        "acknowledgment_reference": report.acknowledgment_reference,
                        "created_at": report.created_at.isoformat(),
                        "last_updated": report.last_updated.isoformat(),
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Error caching regulatory report: {e}")


# Factory function
async def get_compliance_service(
    redis_client: redis_sync.Redis, db_session: Session
) -> ComplianceService:
    """Get compliance service instance"""
    service = ComplianceService(redis_client, db_session)
    await service.initialize()
    return service
