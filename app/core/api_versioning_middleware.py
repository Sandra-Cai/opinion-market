"""
API Versioning Middleware for FastAPI
Integrates API versioning with FastAPI applications
"""

import time
from typing import Callable, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

from app.core.api_versioning import api_version_manager, VersioningStrategy

logger = logging.getLogger(__name__)

class APIVersioningMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for API versioning"""
    
    def __init__(
        self,
        app: ASGIApp,
        enabled: bool = True,
        skip_paths: Optional[list] = None,
        track_analytics: bool = True,
        add_version_headers: bool = True
    ):
        super().__init__(app)
        self.enabled = enabled
        self.skip_paths = skip_paths or ["/health", "/docs", "/openapi.json", "/redoc", "/metrics"]
        self.track_analytics = track_analytics
        self.add_version_headers = add_version_headers
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through API versioning middleware"""
        
        # Skip versioning if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Skip versioning for certain paths
        if self._should_skip_path(request.url.path):
            return await call_next(request)
        
        # Extract API version from request
        version = api_version_manager.extract_version(request)
        
        # If no version specified, use default
        if not version:
            version = api_version_manager.config.default_version
        
        # Validate version
        if not api_version_manager.validate_version(version):
            return api_version_manager.handle_version_error(version, request)
        
        # Check version status
        status = api_version_manager.check_version_status(version)
        
        # Handle retired or sunset versions
        if status.value in ["retired", "sunset"]:
            return api_version_manager.handle_version_error(version, request)
        
        # Add version to request state for use in endpoints
        request.state.api_version = version
        request.state.api_version_status = status
        
        # Track analytics if enabled
        if self.track_analytics:
            self._track_version_usage(version, request)
        
        # Process request
        response = await call_next(request)
        
        # Add version headers if enabled
        if self.add_version_headers:
            self._add_version_headers(response, version)
        
        return response
    
    def _should_skip_path(self, path: str) -> bool:
        """Check if path should be skipped for versioning"""
        return any(path.startswith(skip_path) for skip_path in self.skip_paths)
    
    def _track_version_usage(self, version: str, request: Request):
        """Track version usage for analytics"""
        try:
            # Extract endpoint from path
            endpoint = self._extract_endpoint(request.url.path)
            
            # Extract user ID if available
            user_id = self._extract_user_id(request)
            
            # Track usage
            api_version_manager.track_version_usage(version, request, endpoint, user_id)
            
        except Exception as e:
            logger.error(f"Error tracking version usage: {e}")
    
    def _extract_endpoint(self, path: str) -> str:
        """Extract endpoint from path"""
        # Remove version from path for endpoint tracking
        import re
        pattern = api_version_manager.config.path_pattern
        endpoint = re.sub(pattern, "/api/", path)
        return endpoint
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        # Try to get user ID from JWT token or session
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            # In a real implementation, you'd decode the JWT token here
            # For now, we'll extract a simple user ID from the token
            token = authorization.split(" ")[1]
            return f"user_{hash(token) % 10000}"
        
        return None
    
    def _add_version_headers(self, response: Response, version: str):
        """Add version headers to response"""
        try:
            # Add basic version headers
            response.headers["X-API-Version"] = version
            response.headers["X-API-Version-Status"] = api_version_manager.check_version_status(version).value
            
            # Add deprecation warnings
            warnings = api_version_manager.get_deprecation_warnings(version)
            if warnings:
                response.headers["X-API-Deprecation-Warning"] = "; ".join(warnings)
            
            # Add sunset information
            version_info = api_version_manager.get_version_info(version)
            if version_info:
                if version_info.sunset_date:
                    response.headers["X-API-Sunset-Date"] = version_info.sunset_date.isoformat()
                
                if version_info.retirement_date:
                    response.headers["X-API-Retirement-Date"] = version_info.retirement_date.isoformat()
                
                # Add supported versions
                response.headers["X-API-Supported-Versions"] = ", ".join(api_version_manager.config.supported_versions)
                
                # Add default version
                response.headers["X-API-Default-Version"] = api_version_manager.config.default_version
            
        except Exception as e:
            logger.error(f"Error adding version headers: {e}")

class APIVersioningConfig:
    """Configuration for API versioning middleware"""
    
    def __init__(
        self,
        enabled: bool = True,
        skip_paths: Optional[list] = None,
        track_analytics: bool = True,
        add_version_headers: bool = True
    ):
        self.enabled = enabled
        self.skip_paths = skip_paths or []
        self.track_analytics = track_analytics
        self.add_version_headers = add_version_headers

def create_api_versioning_middleware(config: APIVersioningConfig) -> APIVersioningMiddleware:
    """Create API versioning middleware with configuration"""
    return APIVersioningMiddleware(
        app=None,  # Will be set by FastAPI
        enabled=config.enabled,
        skip_paths=config.skip_paths,
        track_analytics=config.track_analytics,
        add_version_headers=config.add_version_headers
    )

# Default configuration
default_api_versioning_config = APIVersioningConfig(
    enabled=True,
    skip_paths=["/health", "/docs", "/openapi.json", "/redoc", "/metrics", "/favicon.ico"],
    track_analytics=True,
    add_version_headers=True
)
