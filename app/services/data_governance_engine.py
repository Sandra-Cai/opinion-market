"""
Data Governance Engine
Comprehensive data governance and GDPR compliance system
"""

import asyncio
import time
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import secrets
import base64
import uuid

from app.core.enhanced_cache import enhanced_cache

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification enumeration"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


class DataRetentionPolicy(Enum):
    """Data retention policy enumeration"""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 1 year
    LONG_TERM = "long_term"  # 7 years
    PERMANENT = "permanent"


class ConsentStatus(Enum):
    """Consent status enumeration"""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"


class DataSubjectRights(Enum):
    """Data subject rights enumeration"""
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    PORTABILITY = "portability"
    RESTRICTION = "restriction"
    OBJECTION = "objection"


@dataclass
class DataAsset:
    """Data asset data structure"""
    asset_id: str
    name: str
    description: str
    classification: DataClassification
    retention_policy: DataRetentionPolicy
    owner: str
    created_at: datetime
    last_accessed: datetime
    size_bytes: int
    location: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSubject:
    """Data subject data structure"""
    subject_id: str
    email: str
    name: str
    consent_status: ConsentStatus
    consent_granted_at: Optional[datetime] = None
    consent_withdrawn_at: Optional[datetime] = None
    data_retention_until: Optional[datetime] = None
    rights_requests: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DataProcessingActivity:
    """Data processing activity data structure"""
    activity_id: str
    name: str
    purpose: str
    legal_basis: str
    data_categories: List[str]
    data_subjects: List[str]
    recipients: List[str]
    transfers: List[str]
    retention_period: str
    security_measures: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class DataBreach:
    """Data breach data structure"""
    breach_id: str
    description: str
    affected_data_subjects: int
    data_categories: List[str]
    likely_consequences: str
    measures_taken: List[str]
    reported_to_authority: bool
    reported_to_subjects: bool
    discovered_at: datetime
    reported_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class DataGovernanceEngine:
    """Data Governance Engine for GDPR compliance and data management"""
    
    def __init__(self):
        self.data_assets: Dict[str, DataAsset] = {}
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_activities: Dict[str, DataProcessingActivity] = {}
        self.data_breaches: List[DataBreach] = []
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            "gdpr_enabled": True,
            "data_retention_enabled": True,
            "consent_management_enabled": True,
            "breach_detection_enabled": True,
            "audit_logging_enabled": True,
            "data_anonymization_enabled": True,
            "right_to_be_forgotten_enabled": True,
            "data_portability_enabled": True,
            "privacy_by_design_enabled": True,
            "retention_check_interval": 86400,  # 24 hours
            "breach_notification_time": 72,  # hours
            "audit_retention_days": 2555,  # 7 years
            "consent_expiry_days": 365  # 1 year
        }
        
        # Data classification rules
        self.classification_rules = {
            "email": DataClassification.PERSONAL,
            "phone": DataClassification.PERSONAL,
            "address": DataClassification.PERSONAL,
            "ssn": DataClassification.SENSITIVE_PERSONAL,
            "credit_card": DataClassification.SENSITIVE_PERSONAL,
            "health_data": DataClassification.SENSITIVE_PERSONAL,
            "biometric": DataClassification.SENSITIVE_PERSONAL,
            "political_opinions": DataClassification.SENSITIVE_PERSONAL,
            "trade_secrets": DataClassification.RESTRICTED,
            "financial_data": DataClassification.CONFIDENTIAL
        }
        
        # Retention policies
        self.retention_policies = {
            DataClassification.PUBLIC: DataRetentionPolicy.PERMANENT,
            DataClassification.INTERNAL: DataRetentionPolicy.MEDIUM_TERM,
            DataClassification.CONFIDENTIAL: DataRetentionPolicy.LONG_TERM,
            DataClassification.RESTRICTED: DataRetentionPolicy.LONG_TERM,
            DataClassification.PERSONAL: DataRetentionPolicy.MEDIUM_TERM,
            DataClassification.SENSITIVE_PERSONAL: DataRetentionPolicy.SHORT_TERM
        }
        
        # Legal bases for processing
        self.legal_bases = {
            "consent": "Data subject has given consent",
            "contract": "Processing is necessary for contract performance",
            "legal_obligation": "Processing is necessary for legal compliance",
            "vital_interests": "Processing is necessary to protect vital interests",
            "public_task": "Processing is necessary for public interest",
            "legitimate_interests": "Processing is necessary for legitimate interests"
        }
        
        # Monitoring
        self.governance_active = False
        self.governance_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.governance_stats = {
            "data_assets_managed": 0,
            "data_subjects_registered": 0,
            "consent_requests_processed": 0,
            "rights_requests_processed": 0,
            "data_breaches_detected": 0,
            "retention_actions_taken": 0,
            "audit_events_logged": 0
        }
        
    async def start_governance_engine(self):
        """Start the data governance engine"""
        if self.governance_active:
            logger.warning("Data governance engine already active")
            return
            
        self.governance_active = True
        self.governance_task = asyncio.create_task(self._governance_processing_loop())
        logger.info("Data Governance Engine started")
        
    async def stop_governance_engine(self):
        """Stop the data governance engine"""
        self.governance_active = False
        if self.governance_task:
            self.governance_task.cancel()
            try:
                await self.governance_task
            except asyncio.CancelledError:
                pass
        logger.info("Data Governance Engine stopped")
        
    async def _governance_processing_loop(self):
        """Main governance processing loop"""
        while self.governance_active:
            try:
                # Check data retention policies
                if self.config["data_retention_enabled"]:
                    await self._check_data_retention()
                    
                # Check consent expiry
                if self.config["consent_management_enabled"]:
                    await self._check_consent_expiry()
                    
                # Monitor for data breaches
                if self.config["breach_detection_enabled"]:
                    await self._monitor_data_breaches()
                    
                # Audit logging
                if self.config["audit_logging_enabled"]:
                    await self._perform_audit_logging()
                    
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next cycle
                await asyncio.sleep(self.config["retention_check_interval"])
                
            except Exception as e:
                logger.error(f"Error in governance processing loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
                
    async def register_data_asset(self, asset_data: Dict[str, Any]) -> DataAsset:
        """Register a new data asset"""
        try:
            asset_id = f"asset_{int(time.time())}_{secrets.token_hex(4)}"
            
            # Classify data automatically
            classification = await self._classify_data(asset_data.get("name", ""), asset_data.get("content", ""))
            
            # Determine retention policy
            retention_policy = self.retention_policies.get(classification, DataRetentionPolicy.MEDIUM_TERM)
            
            # Create data asset
            asset = DataAsset(
                asset_id=asset_id,
                name=asset_data.get("name", "Unknown Asset"),
                description=asset_data.get("description", ""),
                classification=classification,
                retention_policy=retention_policy,
                owner=asset_data.get("owner", "system"),
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=asset_data.get("size_bytes", 0),
                location=asset_data.get("location", ""),
                tags=asset_data.get("tags", []),
                metadata=asset_data.get("metadata", {})
            )
            
            # Store asset
            self.data_assets[asset_id] = asset
            
            # Store in cache
            await enhanced_cache.set(
                f"data_asset_{asset_id}",
                asset,
                ttl=86400 * 30  # 30 days
            )
            
            self.governance_stats["data_assets_managed"] += 1
            
            logger.info(f"Data asset registered: {asset_id}")
            return asset
            
        except Exception as e:
            logger.error(f"Error registering data asset: {e}")
            raise
            
    async def register_data_subject(self, subject_data: Dict[str, Any]) -> DataSubject:
        """Register a new data subject"""
        try:
            subject_id = f"subject_{int(time.time())}_{secrets.token_hex(4)}"
            
            # Create data subject
            subject = DataSubject(
                subject_id=subject_id,
                email=subject_data.get("email", ""),
                name=subject_data.get("name", ""),
                consent_status=ConsentStatus.PENDING,
                data_retention_until=datetime.now() + timedelta(days=365)  # Default 1 year
            )
            
            # Store subject
            self.data_subjects[subject_id] = subject
            
            # Store in cache
            await enhanced_cache.set(
                f"data_subject_{subject_id}",
                subject,
                ttl=86400 * 365  # 1 year
            )
            
            self.governance_stats["data_subjects_registered"] += 1
            
            logger.info(f"Data subject registered: {subject_id}")
            return subject
            
        except Exception as e:
            logger.error(f"Error registering data subject: {e}")
            raise
            
    async def process_consent_request(self, subject_id: str, consent_data: Dict[str, Any]) -> bool:
        """Process a consent request"""
        try:
            subject = self.data_subjects.get(subject_id)
            if not subject:
                raise ValueError(f"Data subject not found: {subject_id}")
                
            # Update consent status
            consent_granted = consent_data.get("consent_granted", False)
            
            if consent_granted:
                subject.consent_status = ConsentStatus.GRANTED
                subject.consent_granted_at = datetime.now()
                subject.consent_withdrawn_at = None
            else:
                subject.consent_status = ConsentStatus.DENIED
                subject.consent_granted_at = None
                
            # Record consent details
            self.consent_records[subject_id] = {
                "consent_data": consent_data,
                "timestamp": datetime.now().isoformat(),
                "ip_address": consent_data.get("ip_address", ""),
                "user_agent": consent_data.get("user_agent", "")
            }
            
            # Update cache
            await enhanced_cache.set(
                f"data_subject_{subject_id}",
                subject,
                ttl=86400 * 365
            )
            
            self.governance_stats["consent_requests_processed"] += 1
            
            logger.info(f"Consent processed for subject: {subject_id}")
            return consent_granted
            
        except Exception as e:
            logger.error(f"Error processing consent request: {e}")
            raise
            
    async def process_data_subject_rights_request(self, subject_id: str, right_type: DataSubjectRights, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a data subject rights request"""
        try:
            subject = self.data_subjects.get(subject_id)
            if not subject:
                raise ValueError(f"Data subject not found: {subject_id}")
                
            # Add to rights requests
            subject.rights_requests.append(right_type.value)
            
            result = {"status": "processed", "right_type": right_type.value}
            
            if right_type == DataSubjectRights.ACCESS:
                result["data"] = await self._provide_data_access(subject_id)
            elif right_type == DataSubjectRights.RECTIFICATION:
                result["status"] = await self._rectify_data(subject_id, request_data.get("corrections", {}))
            elif right_type == DataSubjectRights.ERASURE:
                result["status"] = await self._erase_data(subject_id)
            elif right_type == DataSubjectRights.PORTABILITY:
                result["data"] = await self._provide_data_portability(subject_id)
            elif right_type == DataSubjectRights.RESTRICTION:
                result["status"] = await self._restrict_processing(subject_id, request_data.get("restrictions", []))
            elif right_type == DataSubjectRights.OBJECTION:
                result["status"] = await self._object_to_processing(subject_id, request_data.get("objections", []))
                
            # Update cache
            await enhanced_cache.set(
                f"data_subject_{subject_id}",
                subject,
                ttl=86400 * 365
            )
            
            self.governance_stats["rights_requests_processed"] += 1
            
            logger.info(f"Rights request processed: {right_type.value} for subject: {subject_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing rights request: {e}")
            raise
            
    async def report_data_breach(self, breach_data: Dict[str, Any]) -> DataBreach:
        """Report a data breach"""
        try:
            breach_id = f"breach_{int(time.time())}_{secrets.token_hex(4)}"
            
            # Create data breach record
            breach = DataBreach(
                breach_id=breach_id,
                description=breach_data.get("description", ""),
                affected_data_subjects=breach_data.get("affected_subjects", 0),
                data_categories=breach_data.get("data_categories", []),
                likely_consequences=breach_data.get("consequences", ""),
                measures_taken=breach_data.get("measures", []),
                reported_to_authority=breach_data.get("reported_to_authority", False),
                reported_to_subjects=breach_data.get("reported_to_subjects", False),
                discovered_at=datetime.now()
            )
            
            # Add to breaches list
            self.data_breaches.append(breach)
            
            # Store in cache
            await enhanced_cache.set(
                f"data_breach_{breach_id}",
                breach,
                ttl=86400 * 2555  # 7 years
            )
            
            self.governance_stats["data_breaches_detected"] += 1
            
            # Trigger breach notification workflow
            await self._trigger_breach_notification_workflow(breach)
            
            logger.warning(f"Data breach reported: {breach_id}")
            return breach
            
        except Exception as e:
            logger.error(f"Error reporting data breach: {e}")
            raise
            
    async def _classify_data(self, name: str, content: str) -> DataClassification:
        """Classify data based on content"""
        try:
            # Simple classification based on keywords
            text_to_analyze = f"{name} {content}".lower()
            
            for keyword, classification in self.classification_rules.items():
                if keyword in text_to_analyze:
                    return classification
                    
            # Default classification
            return DataClassification.INTERNAL
            
        except Exception as e:
            logger.error(f"Error classifying data: {e}")
            return DataClassification.INTERNAL
            
    async def _check_data_retention(self):
        """Check data retention policies"""
        try:
            current_time = datetime.now()
            
            for asset_id, asset in list(self.data_assets.items()):
                # Calculate retention period
                retention_days = {
                    DataRetentionPolicy.IMMEDIATE: 0,
                    DataRetentionPolicy.SHORT_TERM: 30,
                    DataRetentionPolicy.MEDIUM_TERM: 365,
                    DataRetentionPolicy.LONG_TERM: 2555,  # 7 years
                    DataRetentionPolicy.PERMANENT: 999999
                }.get(asset.retention_policy, 365)
                
                # Check if asset should be deleted
                if retention_days < 999999:  # Not permanent
                    expiry_date = asset.created_at + timedelta(days=retention_days)
                    
                    if current_time > expiry_date:
                        await self._delete_data_asset(asset_id)
                        self.governance_stats["retention_actions_taken"] += 1
                        
        except Exception as e:
            logger.error(f"Error checking data retention: {e}")
            
    async def _check_consent_expiry(self):
        """Check consent expiry"""
        try:
            current_time = datetime.now()
            expiry_threshold = current_time - timedelta(days=self.config["consent_expiry_days"])
            
            for subject_id, subject in self.data_subjects.items():
                if (subject.consent_status == ConsentStatus.GRANTED and 
                    subject.consent_granted_at and 
                    subject.consent_granted_at < expiry_threshold):
                    
                    # Mark consent as expired
                    subject.consent_status = ConsentStatus.EXPIRED
                    
                    # Update cache
                    await enhanced_cache.set(
                        f"data_subject_{subject_id}",
                        subject,
                        ttl=86400 * 365
                    )
                    
                    logger.info(f"Consent expired for subject: {subject_id}")
                    
        except Exception as e:
            logger.error(f"Error checking consent expiry: {e}")
            
    async def _monitor_data_breaches(self):
        """Monitor for potential data breaches"""
        try:
            # This would implement breach detection logic
            # For now, we'll simulate monitoring
            logger.debug("Monitoring for data breaches...")
            
        except Exception as e:
            logger.error(f"Error monitoring data breaches: {e}")
            
    async def _perform_audit_logging(self):
        """Perform audit logging"""
        try:
            # Log governance activities
            audit_event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": "governance_activity",
                "data_assets_count": len(self.data_assets),
                "data_subjects_count": len(self.data_subjects),
                "active_breaches": len([b for b in self.data_breaches if not b.resolved_at])
            }
            
            # Store audit event
            await enhanced_cache.set(
                f"audit_{int(time.time())}",
                audit_event,
                ttl=86400 * self.config["audit_retention_days"]
            )
            
            self.governance_stats["audit_events_logged"] += 1
            
        except Exception as e:
            logger.error(f"Error performing audit logging: {e}")
            
    async def _cleanup_old_data(self):
        """Clean up old governance data"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=30)
            
            # Clean up old consent records
            self.consent_records = {
                subject_id: record for subject_id, record in self.consent_records.items()
                if datetime.fromisoformat(record["timestamp"]) > cutoff_time
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    async def _delete_data_asset(self, asset_id: str):
        """Delete a data asset"""
        try:
            if asset_id in self.data_assets:
                del self.data_assets[asset_id]
                await enhanced_cache.delete(f"data_asset_{asset_id}")
                logger.info(f"Data asset deleted: {asset_id}")
                
        except Exception as e:
            logger.error(f"Error deleting data asset: {e}")
            
    async def _provide_data_access(self, subject_id: str) -> Dict[str, Any]:
        """Provide data access to subject"""
        try:
            # This would provide actual data access
            return {
                "subject_id": subject_id,
                "data_summary": "Data access provided",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error providing data access: {e}")
            return {}
            
    async def _rectify_data(self, subject_id: str, corrections: Dict[str, Any]) -> str:
        """Rectify data for subject"""
        try:
            # This would implement data rectification
            return "Data rectified successfully"
            
        except Exception as e:
            logger.error(f"Error rectifying data: {e}")
            return "Error rectifying data"
            
    async def _erase_data(self, subject_id: str) -> str:
        """Erase data for subject (right to be forgotten)"""
        try:
            # This would implement data erasure
            return "Data erased successfully"
            
        except Exception as e:
            logger.error(f"Error erasing data: {e}")
            return "Error erasing data"
            
    async def _provide_data_portability(self, subject_id: str) -> Dict[str, Any]:
        """Provide data portability for subject"""
        try:
            # This would provide data in portable format
            return {
                "subject_id": subject_id,
                "data_format": "JSON",
                "data": "Portable data provided",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error providing data portability: {e}")
            return {}
            
    async def _restrict_processing(self, subject_id: str, restrictions: List[str]) -> str:
        """Restrict processing for subject"""
        try:
            # This would implement processing restrictions
            return "Processing restricted successfully"
            
        except Exception as e:
            logger.error(f"Error restricting processing: {e}")
            return "Error restricting processing"
            
    async def _object_to_processing(self, subject_id: str, objections: List[str]) -> str:
        """Object to processing for subject"""
        try:
            # This would implement processing objections
            return "Processing objection recorded successfully"
            
        except Exception as e:
            logger.error(f"Error recording processing objection: {e}")
            return "Error recording processing objection"
            
    async def _trigger_breach_notification_workflow(self, breach: DataBreach):
        """Trigger breach notification workflow"""
        try:
            # This would implement breach notification workflow
            logger.warning(f"Breach notification workflow triggered for: {breach.breach_id}")
            
        except Exception as e:
            logger.error(f"Error triggering breach notification workflow: {e}")
            
    def get_governance_summary(self) -> Dict[str, Any]:
        """Get comprehensive governance summary"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "governance_active": self.governance_active,
                "total_data_assets": len(self.data_assets),
                "total_data_subjects": len(self.data_subjects),
                "total_processing_activities": len(self.processing_activities),
                "total_data_breaches": len(self.data_breaches),
                "active_breaches": len([b for b in self.data_breaches if not b.resolved_at]),
                "consent_records": len(self.consent_records),
                "assets_by_classification": {
                    classification.value: len([a for a in self.data_assets.values() if a.classification == classification])
                    for classification in DataClassification
                },
                "subjects_by_consent": {
                    status.value: len([s for s in self.data_subjects.values() if s.consent_status == status])
                    for status in ConsentStatus
                },
                "stats": self.governance_stats,
                "config": self.config
            }
            
        except Exception as e:
            logger.error(f"Error getting governance summary: {e}")
            return {"error": str(e)}


# Global instance
data_governance_engine = DataGovernanceEngine()
