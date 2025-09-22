"""
API Versioning Management Endpoints
Provides management and monitoring of API versioning through API
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from typing import Dict, List, Any, Optional
import logging

from app.core.api_versioning import (
    api_version_manager,
    VersioningStrategy,
    VersionStatus,
    APIVersion
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/versions")
async def get_supported_versions():
    """Get list of supported API versions with their status"""
    try:
        versions = api_version_manager.get_supported_versions()
        
        return {
            "success": True,
            "data": {
                "versions": versions,
                "default_version": api_version_manager.config.default_version,
                "strategy": api_version_manager.config.strategy.value
            }
        }
    except Exception as e:
        logger.error(f"Error getting supported versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/versions/{version}")
async def get_version_info(version: str):
    """Get detailed information about a specific version"""
    try:
        version_info = api_version_manager.get_version_info(version)
        if not version_info:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
        
        status = api_version_manager.check_version_status(version)
        analytics = api_version_manager.get_version_analytics(version)
        warnings = api_version_manager.get_deprecation_warnings(version)
        
        return {
            "success": True,
            "data": {
                "version": version,
                "status": status.value,
                "release_date": version_info.release_date.isoformat(),
                "deprecation_date": version_info.deprecation_date.isoformat() if version_info.deprecation_date else None,
                "sunset_date": version_info.sunset_date.isoformat() if version_info.sunset_date else None,
                "retirement_date": version_info.retirement_date.isoformat() if version_info.retirement_date else None,
                "description": version_info.description,
                "changelog": version_info.changelog,
                "breaking_changes": version_info.breaking_changes,
                "migration_guide": version_info.migration_guide,
                "warnings": warnings,
                "analytics": analytics
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting version info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics")
async def get_version_analytics(version: Optional[str] = None):
    """Get version usage analytics"""
    try:
        analytics = api_version_manager.get_version_analytics(version)
        
        return {
            "success": True,
            "data": {
                "analytics": analytics,
                "version": version if version else "all"
            }
        }
    except Exception as e:
        logger.error(f"Error getting version analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/migration/{from_version}/{to_version}")
async def get_migration_guide(from_version: str, to_version: str):
    """Get migration guide between versions"""
    try:
        # Validate versions
        if not api_version_manager.validate_version(from_version):
            raise HTTPException(status_code=400, detail=f"Source version {from_version} not found")
        
        if not api_version_manager.validate_version(to_version):
            raise HTTPException(status_code=400, detail=f"Target version {to_version} not found")
        
        migration_guide = api_version_manager.get_migration_guide(from_version, to_version)
        breaking_changes = api_version_manager.get_breaking_changes(from_version, to_version)
        
        return {
            "success": True,
            "data": {
                "from_version": from_version,
                "to_version": to_version,
                "migration_guide": migration_guide,
                "breaking_changes": breaking_changes
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting migration guide: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_versioning_status():
    """Get overall API versioning status"""
    try:
        config = api_version_manager.config
        
        return {
            "success": True,
            "data": {
                "strategy": config.strategy.value,
                "default_version": config.default_version,
                "supported_versions": config.supported_versions,
                "total_versions": len(config.supported_versions),
                "active_versions": len([v for v in config.supported_versions 
                                     if api_version_manager.check_version_status(v) == VersionStatus.ACTIVE]),
                "deprecated_versions": len([v for v in config.supported_versions 
                                          if api_version_manager.check_version_status(v) == VersionStatus.DEPRECATED]),
                "sunset_versions": len([v for v in config.supported_versions 
                                      if api_version_manager.check_version_status(v) == VersionStatus.SUNSET]),
                "retired_versions": len([v for v in config.supported_versions 
                                       if api_version_manager.check_version_status(v) == VersionStatus.RETIRED])
            }
        }
    except Exception as e:
        logger.error(f"Error getting versioning status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies")
async def get_versioning_strategies():
    """Get available versioning strategies"""
    strategies = [
        {
            "name": strategy.value,
            "description": _get_strategy_description(strategy)
        }
        for strategy in VersioningStrategy
    ]
    
    return {
        "success": True,
        "data": {
            "strategies": strategies,
            "current_strategy": api_version_manager.config.strategy.value
        }
    }

@router.get("/deprecation-warnings")
async def get_deprecation_warnings():
    """Get all deprecation warnings for active versions"""
    try:
        warnings = {}
        
        for version in api_version_manager.config.supported_versions:
            version_warnings = api_version_manager.get_deprecation_warnings(version)
            if version_warnings:
                warnings[version] = version_warnings
        
        return {
            "success": True,
            "data": {
                "warnings": warnings,
                "total_warnings": sum(len(w) for w in warnings.values())
            }
        }
    except Exception as e:
        logger.error(f"Error getting deprecation warnings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/endpoints/{version}")
async def get_version_endpoints(version: str):
    """Get available endpoints for a specific version"""
    try:
        if not api_version_manager.validate_version(version):
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
        
        # In a real implementation, you'd get the actual endpoints for the version
        # For now, return a mock list
        endpoints = [
            {
                "path": f"/api/v{version}/users",
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "description": "User management endpoints"
            },
            {
                "path": f"/api/v{version}/markets",
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "description": "Market management endpoints"
            },
            {
                "path": f"/api/v{version}/trades",
                "methods": ["GET", "POST"],
                "description": "Trading endpoints"
            },
            {
                "path": f"/api/v{version}/analytics",
                "methods": ["GET"],
                "description": "Analytics endpoints"
            }
        ]
        
        return {
            "success": True,
            "data": {
                "version": version,
                "endpoints": endpoints,
                "total_endpoints": len(endpoints)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting version endpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/changelog/{version}")
async def get_version_changelog(version: str):
    """Get changelog for a specific version"""
    try:
        version_info = api_version_manager.get_version_info(version)
        if not version_info:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
        
        return {
            "success": True,
            "data": {
                "version": version,
                "changelog": version_info.changelog,
                "release_date": version_info.release_date.isoformat(),
                "description": version_info.description
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting version changelog: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/breaking-changes/{version}")
async def get_version_breaking_changes(version: str):
    """Get breaking changes for a specific version"""
    try:
        version_info = api_version_manager.get_version_info(version)
        if not version_info:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
        
        return {
            "success": True,
            "data": {
                "version": version,
                "breaking_changes": version_info.breaking_changes,
                "migration_guide": version_info.migration_guide
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting version breaking changes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/current")
async def get_current_version(request: Request):
    """Get current API version from request"""
    try:
        version = getattr(request.state, 'api_version', None)
        if not version:
            version = api_version_manager.config.default_version
        
        version_info = api_version_manager.get_version_info(version)
        status = api_version_manager.check_version_status(version)
        
        return {
            "success": True,
            "data": {
                "current_version": version,
                "status": status.value,
                "description": version_info.description if version_info else "",
                "release_date": version_info.release_date.isoformat() if version_info else None
            }
        }
    except Exception as e:
        logger.error(f"Error getting current version: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _get_strategy_description(strategy: VersioningStrategy) -> str:
    """Get description for versioning strategy"""
    descriptions = {
        VersioningStrategy.URL_PATH: "Version in URL path (e.g., /api/v1/endpoint)",
        VersioningStrategy.HEADER: "Version in Accept header (e.g., Accept: application/vnd.api+json;version=1)",
        VersioningStrategy.QUERY_PARAM: "Version in query parameter (e.g., ?version=1)",
        VersioningStrategy.SUBDOMAIN: "Version in subdomain (e.g., v1.api.example.com)",
        VersioningStrategy.CUSTOM_HEADER: "Version in custom header (e.g., X-API-Version: 1)"
    }
    return descriptions.get(strategy, "Unknown strategy")
