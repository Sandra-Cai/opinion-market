from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/opinion_market"

    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Email
    SMTP_TLS: bool = True
    SMTP_PORT: int = 587
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""

    # App
    APP_NAME: str = "Opinion Market"
    DEBUG: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
