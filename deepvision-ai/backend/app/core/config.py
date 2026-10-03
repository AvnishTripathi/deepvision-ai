"""
DeepVision AI — Core Configuration
Manages environment variables, database settings, and system parameters.
"""

from typing import List, Optional, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Project Identity
    PROJECT_NAME: str = "DeepVision AI"
    PROJECT_VERSION: str = "1.0.0-dev"
    API_V1_STR: str = "/api/v1"
    APP_ENV: str = "development"
    DEBUG: bool = True

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "t", "yes", "y", "debug", "dev", "development")
        return bool(v)

    # Database & Caching
    DATABASE_URL: str = "sqlite:///./deepvision.db"
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"

    # Security & Tokens
    SECRET_KEY: str = "deepvision-ai-super-secure-production-secret-key-2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Networking & CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Storage Paths
    STORAGE_DIR: str = "./storage"
    MODELS_DIR: str = "./models"
    MAX_IMAGE_SIZE_MB: int = 50
    MAX_VIDEO_SIZE_MB: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
