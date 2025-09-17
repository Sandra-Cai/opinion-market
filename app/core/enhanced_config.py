"""
Enhanced Configuration Management System
Advanced configuration with validation, hot reloading, and environment-specific settings
"""

import os
import json
import yaml
import functools
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Type, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigSource(Enum):
    """Configuration source types"""
    ENVIRONMENT = "environment"
    FILE = "file"
    DATABASE = "database"
    REMOTE = "remote"


@dataclass
class ConfigValidationRule:
    """Configuration validation rule"""
    field_name: str
    field_type: Type
    required: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    custom_validator: Optional[callable] = None


@dataclass
class ConfigChange:
    """Configuration change record"""
    field_name: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    source: ConfigSource
    user_id: Optional[str] = None


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration files"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
            logger.info(f"Configuration file changed: {event.src_path}")
            asyncio.create_task(self.config_manager.reload_config())


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "prefer"


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str
    max_connections: int = 100
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    refresh_token_expire_days: int = 30
    password_min_length: int = 8
    password_require_special: bool = True
    password_require_numbers: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    rate_limit_requests: int = 100
    rate_limit_window: int = 60


@dataclass
class APIConfig:
    """API configuration"""
    title: str = "Opinion Market API"
    version: str = "2.0.0"
    description: str = "Advanced prediction market platform"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    file_path: str = "logs/app.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True
    enable_remote: bool = False
    remote_endpoint: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_tracing: bool = True
    tracing_endpoint: Optional[str] = None
    enable_profiling: bool = False
    profiling_port: int = 6060


@dataclass
class CacheConfig:
    """Cache configuration"""
    default_ttl: int = 3600  # 1 hour
    max_size: int = 1000
    enable_compression: bool = True
    compression_level: int = 6
    enable_persistence: bool = False
    persistence_path: str = "cache/data"


@dataclass
class AIConfig:
    """AI/ML configuration"""
    enable_predictions: bool = True
    model_update_interval: int = 3600  # 1 hour
    prediction_cache_ttl: int = 1800  # 30 minutes
    max_prediction_history: int = 10000
    enable_sentiment_analysis: bool = True
    sentiment_update_interval: int = 300  # 5 minutes


@dataclass
class BlockchainConfig:
    """Blockchain configuration"""
    enable_integration: bool = False
    network: str = "ethereum"
    rpc_url: Optional[str] = None
    private_key: Optional[str] = None
    contract_address: Optional[str] = None
    gas_limit: int = 200000
    gas_price: int = 20


class EnhancedConfigManager:
    """Enhanced configuration management system"""
    
    def __init__(self, config_path: str = "config", environment: Environment = None):
        self.config_path = Path(config_path)
        self.environment = environment or self._detect_environment()
        self.config_data: Dict[str, Any] = {}
        self.config_changes: List[ConfigChange] = []
        self.validation_rules: List[ConfigValidationRule] = []
        self.observers: List[callable] = []
        self.file_observer: Optional[Observer] = None
        self._lock = threading.Lock()
        
        # Initialize configuration sections
        self.database = DatabaseConfig(url="")
        self.redis = RedisConfig(url="")
        self.security = SecurityConfig(secret_key="")
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.monitoring = MonitoringConfig()
        self.cache = CacheConfig()
        self.ai = AIConfig()
        self.blockchain = BlockchainConfig()
        
        # Initialize validation rules
        self._initialize_validation_rules()
        
        # Load initial configuration
        self.load_config()

    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        try:
            return Environment(env_name)
        except ValueError:
            logger.warning(f"Unknown environment: {env_name}, defaulting to development")
            return Environment.DEVELOPMENT

    def _initialize_validation_rules(self):
        """Initialize configuration validation rules"""
        self.validation_rules = [
            ConfigValidationRule("database.url", str, required=True),
            ConfigValidationRule("database.pool_size", int, min_value=1, max_value=100),
            ConfigValidationRule("redis.url", str, required=True),
            ConfigValidationRule("security.secret_key", str, required=True, min_value=32),
            ConfigValidationRule("security.access_token_expire_minutes", int, min_value=1, max_value=10080),
            ConfigValidationRule("api.cors_origins", list, required=True),
            ConfigValidationRule("logging.level", str, allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
            ConfigValidationRule("monitoring.metrics_port", int, min_value=1024, max_value=65535),
        ]

    def load_config(self):
        """Load configuration from all sources"""
        with self._lock:
            try:
                # Load from environment variables
                self._load_from_environment()
                
                # Load from configuration files
                self._load_from_files()
                
                # Validate configuration
                self._validate_config()
                
                # Update configuration objects
                self._update_config_objects()
                
                logger.info(f"Configuration loaded successfully for {self.environment.value} environment")
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise

    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            "DATABASE_URL": "database.url",
            "REDIS_URL": "redis.url",
            "SECRET_KEY": "security.secret_key",
            "ENVIRONMENT": "environment",
            "LOG_LEVEL": "logging.level",
            "API_TITLE": "api.title",
            "API_VERSION": "api.version",
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, value)

    def _load_from_files(self):
        """Load configuration from files"""
        config_files = [
            self.config_path / f"config.{self.environment.value}.yaml",
            self.config_path / f"config.{self.environment.value}.yml",
            self.config_path / f"config.{self.environment.value}.json",
            self.config_path / "config.yaml",
            self.config_path / "config.yml",
            self.config_path / "config.json",
        ]
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        if config_file.suffix in ['.yaml', '.yml']:
                            file_config = yaml.safe_load(f)
                        else:
                            file_config = json.load(f)
                    
                    self._merge_config(file_config)
                    logger.info(f"Loaded configuration from {config_file}")
                    break
                    
                except Exception as e:
                    logger.error(f"Failed to load configuration from {config_file}: {e}")

    def _merge_config(self, new_config: Dict[str, Any]):
        """Merge new configuration with existing"""
        def merge_dicts(base: Dict[str, Any], update: Dict[str, Any]):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dicts(base[key], value)
                else:
                    base[key] = value
        
        merge_dicts(self.config_data, new_config)

    def _set_nested_value(self, path: str, value: Any):
        """Set nested configuration value"""
        keys = path.split('.')
        current = self.config_data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value

    def _get_nested_value(self, path: str, default: Any = None) -> Any:
        """Get nested configuration value"""
        keys = path.split('.')
        current = self.config_data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def _validate_config(self):
        """Validate configuration against rules"""
        for rule in self.validation_rules:
            value = self._get_nested_value(rule.field_name)
            
            if rule.required and value is None:
                raise ValueError(f"Required configuration field missing: {rule.field_name}")
            
            if value is not None:
                # Type validation
                if not isinstance(value, rule.field_type):
                    try:
                        value = rule.field_type(value)
                        self._set_nested_value(rule.field_name, value)
                    except (ValueError, TypeError):
                        raise ValueError(f"Invalid type for {rule.field_name}: expected {rule.field_type.__name__}")
                
                # Range validation (only for numeric types)
                if rule.min_value is not None and isinstance(value, (int, float)) and value < rule.min_value:
                    raise ValueError(f"Value too small for {rule.field_name}: {value} < {rule.min_value}")
                
                if rule.max_value is not None and isinstance(value, (int, float)) and value > rule.max_value:
                    raise ValueError(f"Value too large for {rule.field_name}: {value} > {rule.max_value}")
                
                # Allowed values validation
                if rule.allowed_values is not None and value not in rule.allowed_values:
                    raise ValueError(f"Invalid value for {rule.field_name}: {value} not in {rule.allowed_values}")
                
                # Custom validation
                if rule.custom_validator and not rule.custom_validator(value):
                    raise ValueError(f"Custom validation failed for {rule.field_name}")

    def _update_config_objects(self):
        """Update configuration objects with loaded data"""
        # Update database config
        db_config = self._get_nested_value("database", {})
        if db_config:
            self.database = DatabaseConfig(**db_config)
        
        # Update redis config
        redis_config = self._get_nested_value("redis", {})
        if redis_config:
            self.redis = RedisConfig(**redis_config)
        
        # Update security config
        security_config = self._get_nested_value("security", {})
        if security_config:
            self.security = SecurityConfig(**security_config)
        
        # Update API config
        api_config = self._get_nested_value("api", {})
        if api_config:
            self.api = APIConfig(**api_config)
        
        # Update logging config
        logging_config = self._get_nested_value("logging", {})
        if logging_config:
            self.logging = LoggingConfig(**logging_config)
        
        # Update monitoring config
        monitoring_config = self._get_nested_value("monitoring", {})
        if monitoring_config:
            self.monitoring = MonitoringConfig(**monitoring_config)
        
        # Update cache config
        cache_config = self._get_nested_value("cache", {})
        if cache_config:
            self.cache = CacheConfig(**cache_config)
        
        # Update AI config
        ai_config = self._get_nested_value("ai", {})
        if ai_config:
            self.ai = AIConfig(**ai_config)
        
        # Update blockchain config
        blockchain_config = self._get_nested_value("blockchain", {})
        if blockchain_config:
            self.blockchain = BlockchainConfig(**blockchain_config)

    async def reload_config(self):
        """Reload configuration from sources"""
        logger.info("Reloading configuration...")
        
        old_config = self.config_data.copy()
        self.load_config()
        
        # Detect changes
        changes = self._detect_changes(old_config, self.config_data)
        
        # Record changes
        for change in changes:
            self.config_changes.append(change)
        
        # Notify observers
        for observer in self.observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(changes)
                else:
                    observer(changes)
            except Exception as e:
                logger.error(f"Error notifying configuration observer: {e}")
        
        logger.info(f"Configuration reloaded with {len(changes)} changes")

    def _detect_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> List[ConfigChange]:
        """Detect configuration changes"""
        changes = []
        
        def compare_dicts(old: Dict[str, Any], new: Dict[str, Any], path: str = ""):
            for key, new_value in new.items():
                current_path = f"{path}.{key}" if path else key
                old_value = old.get(key)
                
                if old_value != new_value:
                    changes.append(ConfigChange(
                        field_name=current_path,
                        old_value=old_value,
                        new_value=new_value,
                        timestamp=datetime.utcnow(),
                        source=ConfigSource.FILE
                    ))
                
                if isinstance(new_value, dict) and isinstance(old_value, dict):
                    compare_dicts(old_value, new_value, current_path)
        
        compare_dicts(old_config, new_config)
        return changes

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._get_nested_value(key, default)

    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.ENVIRONMENT):
        """Set configuration value"""
        old_value = self._get_nested_value(key)
        self._set_nested_value(key, value)
        
        # Record change
        change = ConfigChange(
            field_name=key,
            old_value=old_value,
            new_value=value,
            timestamp=datetime.utcnow(),
            source=source
        )
        self.config_changes.append(change)
        
        # Notify observers
        for observer in self.observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    asyncio.create_task(observer([change]))
                else:
                    observer([change])
            except Exception as e:
                logger.error(f"Error notifying configuration observer: {e}")

    def add_observer(self, observer: callable):
        """Add configuration change observer"""
        self.observers.append(observer)

    def remove_observer(self, observer: callable):
        """Remove configuration change observer"""
        if observer in self.observers:
            self.observers.remove(observer)

    def start_file_watching(self):
        """Start watching configuration files for changes"""
        if self.file_observer is None:
            self.file_observer = Observer()
            handler = ConfigFileHandler(self)
            self.file_observer.schedule(handler, str(self.config_path), recursive=True)
            self.file_observer.start()
            logger.info("Started watching configuration files for changes")

    def stop_file_watching(self):
        """Stop watching configuration files"""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            self.file_observer = None
            logger.info("Stopped watching configuration files")

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "environment": self.environment.value,
            "config_path": str(self.config_path),
            "sections": {
                "database": {
                    "url": self.database.url[:20] + "..." if len(self.database.url) > 20 else self.database.url,
                    "pool_size": self.database.pool_size
                },
                "redis": {
                    "url": self.redis.url[:20] + "..." if len(self.redis.url) > 20 else self.redis.url,
                    "max_connections": self.redis.max_connections
                },
                "security": {
                    "algorithm": self.security.algorithm,
                    "token_expire_minutes": self.security.access_token_expire_minutes
                },
                "api": {
                    "title": self.api.title,
                    "version": self.api.version
                },
                "logging": {
                    "level": self.logging.level,
                    "format": self.logging.format
                },
                "monitoring": {
                    "enable_metrics": self.monitoring.enable_metrics,
                    "metrics_port": self.monitoring.metrics_port
                }
            },
            "total_changes": len(self.config_changes),
            "recent_changes": [
                {
                    "field": change.field_name,
                    "timestamp": change.timestamp.isoformat(),
                    "source": change.source.value
                }
                for change in self.config_changes[-10:]  # Last 10 changes
            ]
        }

    def export_config(self, format: str = "yaml") -> str:
        """Export configuration to string"""
        if format.lower() == "yaml":
            return yaml.dump(self.config_data, default_flow_style=False)
        elif format.lower() == "json":
            return json.dumps(self.config_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def cleanup_old_changes(self, days: int = 30):
        """Clean up old configuration changes"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        self.config_changes = [
            change for change in self.config_changes 
            if change.timestamp >= cutoff_time
        ]


# Global configuration manager instance
enhanced_config_manager = EnhancedConfigManager()


# Configuration decorators
def config_value(key: str, default: Any = None):
    """Decorator to inject configuration values"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            value = enhanced_config_manager.get(key, default)
            return func(*args, **kwargs, config_value=value)
        return wrapper
    return decorator


def require_config(key: str):
    """Decorator to require configuration value"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            value = enhanced_config_manager.get(key)
            if value is None:
                raise ValueError(f"Required configuration missing: {key}")
            return func(*args, **kwargs, config_value=value)
        return wrapper
    return decorator
