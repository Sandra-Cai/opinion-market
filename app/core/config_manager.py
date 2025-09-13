"""
Advanced configuration management system
"""

import os
import yaml
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "opinion_market"
    user: str = "postgres"
    password: str = "password"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 100
    socket_timeout: int = 5
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    password_min_length: int = 8
    require_special_chars: bool = True
    session_timeout_minutes: int = 60


@dataclass
class APIConfig:
    """API configuration"""
    title: str = "Opinion Market API"
    description: str = "Advanced prediction market platform"
    version: str = "2.0.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_window: int = 60


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_metrics: bool = True
    enable_health_checks: bool = True
    enable_performance_monitoring: bool = True
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"
    log_level: str = "INFO"
    enable_request_logging: bool = True
    enable_database_logging: bool = False


@dataclass
class CacheConfig:
    """Cache configuration"""
    enable_memory_cache: bool = True
    memory_cache_size: int = 10000
    memory_cache_ttl: int = 300
    enable_redis_cache: bool = True
    redis_cache_ttl: int = 3600
    cache_key_prefix: str = "opinion_market"


@dataclass
class ApplicationConfig:
    """Main application configuration"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


class ConfigManager:
    """Advanced configuration management"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config: Optional[ApplicationConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables"""
        # Start with defaults
        self.config = ApplicationConfig()
        
        # Load from file if provided
        if self.config_file and Path(self.config_file).exists():
            self._load_from_file()
        
        # Override with environment variables
        self._load_from_environment()
        
        logger.info(f"Configuration loaded for environment: {self.config.environment.value}")
    
    def _load_from_file(self):
        """Load configuration from YAML or JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
            
            # Update configuration with file values
            self._update_config_from_dict(file_config)
            logger.info(f"Configuration loaded from file: {self.config_file}")
        
        except Exception as e:
            logger.error(f"Failed to load configuration file {self.config_file}: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Application
            'APP_ENVIRONMENT': ('environment', lambda x: Environment(x.lower())),
            'APP_DEBUG': ('debug', lambda x: x.lower() in ('true', '1', 'yes')),
            'APP_HOST': ('host', str),
            'APP_PORT': ('port', int),
            'APP_WORKERS': ('workers', int),
            
            # Database
            'DB_HOST': ('database.host', str),
            'DB_PORT': ('database.port', int),
            'DB_NAME': ('database.name', str),
            'DB_USER': ('database.user', str),
            'DB_PASSWORD': ('database.password', str),
            'DB_POOL_SIZE': ('database.pool_size', int),
            'DB_MAX_OVERFLOW': ('database.max_overflow', int),
            
            # Redis
            'REDIS_HOST': ('redis.host', str),
            'REDIS_PORT': ('redis.port', int),
            'REDIS_PASSWORD': ('redis.password', str),
            'REDIS_DB': ('redis.db', int),
            
            # Security
            'SECRET_KEY': ('security.secret_key', str),
            'ACCESS_TOKEN_EXPIRE_MINUTES': ('security.access_token_expire_minutes', int),
            'MAX_LOGIN_ATTEMPTS': ('security.max_login_attempts', int),
            
            # Monitoring
            'LOG_LEVEL': ('monitoring.log_level', str),
            'ENABLE_METRICS': ('monitoring.enable_metrics', lambda x: x.lower() in ('true', '1', 'yes')),
        }
        
        for env_var, (config_path, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    self._set_nested_config(config_path, converted_value)
                except Exception as e:
                    logger.warning(f"Failed to convert environment variable {env_var}={value}: {e}")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                if isinstance(value, dict):
                    # Handle nested configuration
                    nested_config = getattr(self.config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self.config, key, value)
    
    def _set_nested_config(self, path: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if hasattr(config, key):
                config = getattr(config, key)
            else:
                return
        
        if hasattr(config, keys[-1]):
            setattr(config, keys[-1], value)
    
    def get_config(self) -> ApplicationConfig:
        """Get current configuration"""
        return self.config
    
    def reload_config(self):
        """Reload configuration from file and environment"""
        self._load_config()
        logger.info("Configuration reloaded")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate required fields
        if not self.config.security.secret_key or self.config.security.secret_key == "your-secret-key-change-in-production":
            validation_results["warnings"].append("Using default secret key in production is not secure")
        
        if self.config.environment == Environment.PRODUCTION and self.config.debug:
            validation_results["errors"].append("Debug mode should not be enabled in production")
            validation_results["valid"] = False
        
        if self.config.database.password == "password":
            validation_results["warnings"].append("Using default database password")
        
        # Validate numeric ranges
        if self.config.database.pool_size < 1 or self.config.database.pool_size > 100:
            validation_results["errors"].append("Database pool size should be between 1 and 100")
            validation_results["valid"] = False
        
        if self.config.port < 1 or self.config.port > 65535:
            validation_results["errors"].append("Port should be between 1 and 65535")
            validation_results["valid"] = False
        
        return validation_results
    
    def export_config(self, format: str = "yaml") -> str:
        """Export configuration to string"""
        config_dict = self._config_to_dict(self.config)
        
        if format.lower() == "yaml":
            return yaml.dump(config_dict, default_flow_style=False)
        elif format.lower() == "json":
            return json.dumps(config_dict, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _config_to_dict(self, config: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        if hasattr(config, '__dataclass_fields__'):
            result = {}
            for field_name, field_info in config.__dataclass_fields__.items():
                value = getattr(config, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    result[field_name] = self._config_to_dict(value)
                elif isinstance(value, Enum):
                    result[field_name] = value.value
                else:
                    result[field_name] = value
            return result
        return config


# Global configuration manager
config_manager = ConfigManager()
