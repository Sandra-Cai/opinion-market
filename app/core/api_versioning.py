"""
Advanced API Versioning System
Provides sophisticated API versioning with multiple strategies and backward compatibility
"""

import re
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path
import json

from fastapi import Request, HTTPException, status
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class VersioningStrategy(Enum):
    """API versioning strategies"""
    URL_PATH = "url_path"  # /api/v1/endpoint
    HEADER = "header"      # Accept: application/vnd.api+json;version=1
    QUERY_PARAM = "query_param"  # ?version=1
    SUBDOMAIN = "subdomain"  # v1.api.example.com
    CUSTOM_HEADER = "custom_header"  # X-API-Version: 1

class VersionStatus(Enum):
    """Version status types"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    RETIRED = "retired"

@dataclass
class APIVersion:
    """Represents an API version"""
    version: str
    status: VersionStatus
    release_date: datetime
    deprecation_date: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    retirement_date: Optional[datetime] = None
    description: str = ""
    changelog: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    migration_guide: Optional[str] = None
    supported_until: Optional[datetime] = None

@dataclass
class VersioningConfig:
    """Configuration for API versioning"""
    strategy: VersioningStrategy
    default_version: str
    supported_versions: List[str]
    versions: Dict[str, APIVersion]
    header_name: str = "X-API-Version"
    query_param_name: str = "version"
    accept_header_pattern: str = r"application/vnd\.api\+json;version=(\d+)"
    subdomain_pattern: str = r"v(\d+)\.api\."
    path_pattern: str = r"/api/v(\d+)/"

class APIVersionManager:
    """Advanced API version manager"""
    
    def __init__(self, config: VersioningConfig):
        self.config = config
        self.version_routes: Dict[str, List[APIRoute]] = {}
        self.deprecation_warnings: Dict[str, List[str]] = {}
        self.version_analytics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize version analytics
        for version in config.supported_versions:
            self.version_analytics[version] = {
                "requests": 0,
                "errors": 0,
                "last_accessed": None,
                "unique_users": set(),
                "endpoints_used": set()
            }
    
    def extract_version(self, request: Request) -> Optional[str]:
        """Extract API version from request"""
        if self.config.strategy == VersioningStrategy.URL_PATH:
            return self._extract_from_path(request)
        elif self.config.strategy == VersioningStrategy.HEADER:
            return self._extract_from_accept_header(request)
        elif self.config.strategy == VersioningStrategy.QUERY_PARAM:
            return self._extract_from_query_param(request)
        elif self.config.strategy == VersioningStrategy.SUBDOMAIN:
            return self._extract_from_subdomain(request)
        elif self.config.strategy == VersioningStrategy.CUSTOM_HEADER:
            return self._extract_from_custom_header(request)
        
        return None
    
    def _extract_from_path(self, request: Request) -> Optional[str]:
        """Extract version from URL path"""
        path = request.url.path
        match = re.search(self.config.path_pattern, path)
        return match.group(1) if match else None
    
    def _extract_from_accept_header(self, request: Request) -> Optional[str]:
        """Extract version from Accept header"""
        accept_header = request.headers.get("Accept", "")
        match = re.search(self.config.accept_header_pattern, accept_header)
        return match.group(1) if match else None
    
    def _extract_from_query_param(self, request: Request) -> Optional[str]:
        """Extract version from query parameter"""
        return request.query_params.get(self.config.query_param_name)
    
    def _extract_from_subdomain(self, request: Request) -> Optional[str]:
        """Extract version from subdomain"""
        host = request.headers.get("Host", "")
        match = re.search(self.config.subdomain_pattern, host)
        return match.group(1) if match else None
    
    def _extract_from_custom_header(self, request: Request) -> Optional[str]:
        """Extract version from custom header"""
        return request.headers.get(self.config.header_name)
    
    def validate_version(self, version: str) -> bool:
        """Validate if version is supported"""
        return version in self.config.supported_versions
    
    def get_version_info(self, version: str) -> Optional[APIVersion]:
        """Get version information"""
        return self.config.versions.get(version)
    
    def check_version_status(self, version: str) -> VersionStatus:
        """Check the status of a version"""
        version_info = self.get_version_info(version)
        if not version_info:
            return VersionStatus.RETIRED
        
        now = datetime.utcnow()
        
        if version_info.retirement_date and now > version_info.retirement_date:
            return VersionStatus.RETIRED
        elif version_info.sunset_date and now > version_info.sunset_date:
            return VersionStatus.SUNSET
        elif version_info.deprecation_date and now > version_info.deprecation_date:
            return VersionStatus.DEPRECATED
        else:
            return VersionStatus.ACTIVE
    
    def get_deprecation_warnings(self, version: str) -> List[str]:
        """Get deprecation warnings for a version"""
        version_info = self.get_version_info(version)
        if not version_info:
            return ["Version not found"]
        
        warnings = []
        now = datetime.utcnow()
        
        if version_info.deprecation_date and now > version_info.deprecation_date:
            warnings.append(f"API version {version} is deprecated")
            
            if version_info.sunset_date:
                days_until_sunset = (version_info.sunset_date - now).days
                if days_until_sunset > 0:
                    warnings.append(f"API version {version} will be sunset in {days_until_sunset} days")
                else:
                    warnings.append(f"API version {version} has been sunset")
            
            if version_info.retirement_date:
                days_until_retirement = (version_info.retirement_date - now).days
                if days_until_retirement > 0:
                    warnings.append(f"API version {version} will be retired in {days_until_retirement} days")
                else:
                    warnings.append(f"API version {version} has been retired")
        
        return warnings
    
    def track_version_usage(self, version: str, request: Request, endpoint: str, user_id: Optional[str] = None):
        """Track version usage for analytics"""
        if version not in self.version_analytics:
            return
        
        analytics = self.version_analytics[version]
        analytics["requests"] += 1
        analytics["last_accessed"] = datetime.utcnow()
        analytics["endpoints_used"].add(endpoint)
        
        if user_id:
            analytics["unique_users"].add(user_id)
    
    def get_version_analytics(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Get version usage analytics"""
        if version:
            if version not in self.version_analytics:
                return {}
            
            analytics = self.version_analytics[version].copy()
            analytics["unique_users"] = len(analytics["unique_users"])
            analytics["endpoints_used"] = list(analytics["endpoints_used"])
            return analytics
        
        # Return all analytics
        result = {}
        for v, analytics in self.version_analytics.items():
            result[v] = analytics.copy()
            result[v]["unique_users"] = len(analytics["unique_users"])
            result[v]["endpoints_used"] = list(analytics["endpoints_used"])
        
        return result
    
    def get_supported_versions(self) -> List[Dict[str, Any]]:
        """Get list of supported versions with their status"""
        versions = []
        
        for version in self.config.supported_versions:
            version_info = self.get_version_info(version)
            status = self.check_version_status(version)
            analytics = self.get_version_analytics(version)
            
            versions.append({
                "version": version,
                "status": status.value,
                "release_date": version_info.release_date.isoformat() if version_info else None,
                "deprecation_date": version_info.deprecation_date.isoformat() if version_info and version_info.deprecation_date else None,
                "sunset_date": version_info.sunset_date.isoformat() if version_info and version_info.sunset_date else None,
                "retirement_date": version_info.retirement_date.isoformat() if version_info and version_info.retirement_date else None,
                "description": version_info.description if version_info else "",
                "analytics": analytics
            })
        
        return versions
    
    def get_migration_guide(self, from_version: str, to_version: str) -> Optional[str]:
        """Get migration guide between versions"""
        from_info = self.get_version_info(from_version)
        to_info = self.get_version_info(to_version)
        
        if not from_info or not to_info:
            return None
        
        # In a real implementation, you'd have detailed migration guides
        # For now, return a basic guide
        return f"Migration guide from v{from_version} to v{to_version}: {to_info.migration_guide or 'No migration guide available'}"
    
    def get_breaking_changes(self, from_version: str, to_version: str) -> List[str]:
        """Get breaking changes between versions"""
        from_info = self.get_version_info(from_version)
        to_info = self.get_version_info(to_version)
        
        if not from_info or not to_info:
            return []
        
        # In a real implementation, you'd compare the versions
        # For now, return the breaking changes from the target version
        return to_info.breaking_changes
    
    def create_version_response(self, version: str, data: Any, request: Request) -> JSONResponse:
        """Create versioned response with appropriate headers"""
        response = JSONResponse(content=data)
        
        # Add version headers
        response.headers["X-API-Version"] = version
        response.headers["X-API-Version-Status"] = self.check_version_status(version).value
        
        # Add deprecation warnings
        warnings = self.get_deprecation_warnings(version)
        if warnings:
            response.headers["X-API-Deprecation-Warning"] = "; ".join(warnings)
        
        # Add sunset information
        version_info = self.get_version_info(version)
        if version_info and version_info.sunset_date:
            response.headers["X-API-Sunset-Date"] = version_info.sunset_date.isoformat()
        
        # Add retirement information
        if version_info and version_info.retirement_date:
            response.headers["X-API-Retirement-Date"] = version_info.retirement_date.isoformat()
        
        return response
    
    def handle_version_error(self, version: str, request: Request) -> JSONResponse:
        """Handle version-related errors"""
        if not self.validate_version(version):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Unsupported API version",
                    "message": f"API version {version} is not supported",
                    "supported_versions": self.config.supported_versions,
                    "default_version": self.config.default_version
                },
                headers={"X-API-Supported-Versions": ", ".join(self.config.supported_versions)}
            )
        
        status_type = self.check_version_status(version)
        
        if status_type == VersionStatus.RETIRED:
            return JSONResponse(
                status_code=status.HTTP_410_GONE,
                content={
                    "error": "API version retired",
                    "message": f"API version {version} has been retired",
                    "supported_versions": self.config.supported_versions
                }
            )
        elif status_type == VersionStatus.SUNSET:
            return JSONResponse(
                status_code=status.HTTP_410_GONE,
                content={
                    "error": "API version sunset",
                    "message": f"API version {version} has been sunset",
                    "supported_versions": self.config.supported_versions
                }
            )
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "API version error",
                "message": f"API version {version} is not available",
                "supported_versions": self.config.supported_versions
            }
        )

# Default versioning configuration
def create_default_versioning_config() -> VersioningConfig:
    """Create default versioning configuration"""
    now = datetime.utcnow()
    
    versions = {
        "1": APIVersion(
            version="1",
            status=VersionStatus.ACTIVE,
            release_date=now - timedelta(days=365),
            description="Initial API version with core functionality",
            changelog=[
                "Initial release with user management",
                "Market creation and trading",
                "Basic analytics and reporting"
            ]
        ),
        "2": APIVersion(
            version="2",
            status=VersionStatus.ACTIVE,
            release_date=now - timedelta(days=180),
            description="Enhanced API with advanced features",
            changelog=[
                "Added advanced order types",
                "Enhanced analytics and reporting",
                "Improved error handling",
                "Added rate limiting"
            ],
            breaking_changes=[
                "Changed response format for market data",
                "Updated authentication flow",
                "Modified error response structure"
            ],
            migration_guide="See migration guide for details on breaking changes"
        ),
        "3": APIVersion(
            version="3",
            status=VersionStatus.DEPRECATED,
            release_date=now - timedelta(days=90),
            deprecation_date=now - timedelta(days=30),
            sunset_date=now + timedelta(days=60),
            retirement_date=now + timedelta(days=90),
            description="Latest API version with modern features",
            changelog=[
                "Added real-time WebSocket support",
                "Enhanced security features",
                "Improved performance monitoring",
                "Added advanced caching"
            ],
            breaking_changes=[
                "Changed WebSocket message format",
                "Updated authentication headers",
                "Modified rate limiting response format"
            ],
            migration_guide="See migration guide for details on breaking changes"
        )
    }
    
    return VersioningConfig(
        strategy=VersioningStrategy.URL_PATH,
        default_version="2",
        supported_versions=["1", "2", "3"],
        versions=versions
    )

# Global version manager
api_version_manager = APIVersionManager(create_default_versioning_config())
