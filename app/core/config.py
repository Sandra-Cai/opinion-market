from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional, Dict, Any
import os
import secrets
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = Field(default=True, description="Enable debug mode")
    LOG_LEVEL: LogLevel = LogLevel.INFO
    
    # Application
    APP_NAME: str = "Opinion Market"
    APP_VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Opinion Market Platform"
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=1, le=1440)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1, le=30)
    PASSWORD_MIN_LENGTH: int = Field(default=8, ge=6, le=128)
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://user:password@localhost/opinion_market",
        description="Database connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, ge=1, le=100)
    DATABASE_MAX_OVERFLOW: int = Field(default=30, ge=0, le=200)
    DATABASE_POOL_TIMEOUT: int = Field(default=30, ge=1, le=300)
    DATABASE_POOL_RECYCLE: int = Field(default=3600, ge=300, le=7200)
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = Field(default=0, ge=0, le=15)
    REDIS_POOL_SIZE: int = Field(default=10, ge=1, le=50)
    
    # CORS
    ALLOWED_HOSTS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = Field(default=100, ge=1, le=10000)
    RATE_LIMIT_WINDOW: int = Field(default=60, ge=1, le=3600)  # seconds
    
    # Email Configuration
    SMTP_TLS: bool = True
    SMTP_PORT: int = Field(default=587, ge=1, le=65535)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = None
    
    # Trading Configuration
    DEFAULT_TRADING_FEE: float = Field(default=0.02, ge=0.0, le=0.1)
    MIN_TRADE_AMOUNT: float = Field(default=1.0, ge=0.01, le=1000.0)
    MAX_TRADE_AMOUNT: float = Field(default=10000.0, ge=100.0, le=1000000.0)
    DEFAULT_LIQUIDITY: float = Field(default=1000.0, ge=100.0, le=100000.0)
    
    # Market Configuration
    MARKET_VERIFICATION_REQUIRED: bool = True
    MARKET_DISPUTE_WINDOW_DAYS: int = Field(default=7, ge=1, le=30)
    MARKET_TRENDING_THRESHOLD: float = Field(default=50.0, ge=0.0, le=100.0)
    
    # Monitoring and Logging
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = Field(default=9090, ge=1000, le=65535)
    LOG_FORMAT: str = "json"  # json or text
    LOG_FILE: Optional[str] = None
    LOG_ROTATION: str = "daily"
    LOG_RETENTION_DAYS: int = Field(default=30, ge=1, le=365)
    
    # Performance
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = Field(default=300, ge=1, le=3600)  # seconds
    ENABLE_COMPRESSION: bool = True
    MAX_REQUEST_SIZE: int = Field(default=10485760, ge=1024, le=104857600)  # 10MB
    
    # WebSocket Configuration
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, ge=5, le=300)
    WS_MAX_CONNECTIONS: int = Field(default=1000, ge=10, le=10000)
    
    # Machine Learning
    ML_ENABLED: bool = True
    ML_MODEL_PATH: str = "models/"
    ML_PREDICTION_CACHE_TTL: int = Field(default=600, ge=60, le=3600)
    
    # Blockchain (Optional)
    BLOCKCHAIN_ENABLED: bool = False
    ETHEREUM_RPC_URL: Optional[str] = None
    POLYGON_RPC_URL: Optional[str] = None
    ARBITRUM_RPC_URL: Optional[str] = None
    
    # External APIs
    NEWS_API_KEY: Optional[str] = None
    SENTIMENT_API_KEY: Optional[str] = None
    
    # File Storage
    UPLOAD_DIR: str = "uploads/"
    MAX_FILE_SIZE: int = Field(default=5242880, ge=1024, le=52428800)  # 5MB
    ALLOWED_FILE_TYPES: List[str] = ["jpg", "jpeg", "png", "gif", "pdf", "txt"]
    
    # Backup Configuration
    BACKUP_ENABLED: bool = True
    BACKUP_SCHEDULE: str = "0 2 * * *"  # Daily at 2 AM
    BACKUP_RETENTION_DAYS: int = Field(default=30, ge=1, le=365)
    
    # Health Check
    HEALTH_CHECK_INTERVAL: int = Field(default=30, ge=5, le=300)
    HEALTH_CHECK_TIMEOUT: int = Field(default=10, ge=1, le=60)
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v.startswith(("postgresql://", "sqlite:///", "mysql://")):
            raise ValueError("DATABASE_URL must be a valid database URL")
        return v
    
    @validator("REDIS_URL")
    def validate_redis_url(cls, v):
        if not v.startswith("redis://"):
            raise ValueError("REDIS_URL must be a valid Redis URL")
        return v
    
    @validator("ALLOWED_HOSTS")
    def validate_allowed_hosts(cls, v):
        if not v:
            raise ValueError("ALLOWED_HOSTS cannot be empty")
        return v
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT == Environment.TESTING
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration for SQLAlchemy"""
        return {
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "pool_timeout": self.DATABASE_POOL_TIMEOUT,
            "pool_recycle": self.DATABASE_POOL_RECYCLE,
            "pool_pre_ping": True,
        }
    
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "password": self.REDIS_PASSWORD,
            "db": self.REDIS_DB,
            "max_connections": self.REDIS_POOL_SIZE,
            "retry_on_timeout": True,
            "socket_keepalive": True,
            "socket_keepalive_options": {},
        }
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allow_origins": self.ALLOWED_HOSTS,
            "allow_credentials": self.CORS_ALLOW_CREDENTIALS,
            "allow_methods": self.CORS_ALLOW_METHODS,
            "allow_headers": self.CORS_ALLOW_HEADERS,
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        validate_assignment = True


# Create settings instance
settings = Settings()

# Environment-specific overrides
if settings.is_production:
    settings.DEBUG = False
    settings.LOG_LEVEL = LogLevel.WARNING
elif settings.is_testing:
    settings.DEBUG = True
    settings.LOG_LEVEL = LogLevel.DEBUG
