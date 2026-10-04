"""
DeepVision AI — Core Configuration
Provides strongly typed, validated settings powered by Pydantic Settings.
Configuration is loaded from environment variables and .env files.
"""

import os
from typing import List, Optional, Any, Union
import json
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application configuration."""

    # Application Metadata
    APP_NAME: str = "DeepVision AI"
    SERVICE_NAME: str = "deepvision-api"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    API_V1_STR: str = "/api/v1"

    # Runtime Debugging
    DEBUG: bool = True

    # Database & Caching
    DATABASE_URL: Optional[str] = "sqlite:///./deepvision.db"
    REDIS_URL: Optional[str] = "redis://localhost:6379/0"

    # Cross-Origin Resource Sharing
    CORS_ORIGINS: Union[List[str], str] = ["*"]

    # Logging
    LOG_LEVEL: str = "INFO"

    # AI Engine & Hardware Inference Settings
    AI_DEVICE: str = "auto"
    AI_MODEL_NAME: str = "scrfd_500m"
    AI_MODEL_VERSION: str = "1.0.0"
    AI_CONFIDENCE_THRESHOLD: float = 0.50
    AI_MAX_IMAGE_SIZE_MB: int = 10
    AI_MAX_IMAGE_DIMENSION: int = 4096
    AI_ENABLE_GPU: bool = False
    AI_MODEL_PATH: Optional[str] = None

    # Storage Paths & Limits
    STORAGE_DIR: str = "./storage"
    MODELS_DIR: str = "./models"
    MAX_IMAGE_SIZE_MB: int = 50
    MAX_VIDEO_SIZE_MB: int = 500

    # Security
    SECRET_KEY: str = "deepvision-ai-development-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    @field_validator("CORS_ORIGINS")
    @classmethod
    def parse_cors_origins(cls, v: Union[List[str], str]) -> List[str]:
        """Supports comma-separated strings, JSON arrays, or native Python lists."""
        if isinstance(v, list):
            return [str(item).strip() for item in v]
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return ["*"]
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed]
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        return ["*"]

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level(cls, v: Any) -> str:
        """Normalizes and validates logging levels."""
        val = str(v).upper().strip()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        return val if val in allowed else "INFO"

    @field_validator("AI_DEVICE", mode="before")
    @classmethod
    def validate_ai_device(cls, v: Any) -> str:
        """Validates AI inference hardware target."""
        val = str(v).lower().strip()
        allowed = {"auto", "cpu", "gpu", "cuda", "mps", "directml"}
        return val if val in allowed else "auto"

    @field_validator("AI_CONFIDENCE_THRESHOLD")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        """Validates confidence threshold is within valid probability bounds [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"AI_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {v}")
        return float(v)

    @field_validator("AI_MAX_IMAGE_SIZE_MB")
    @classmethod
    def validate_max_image_size(cls, v: int) -> int:
        """Ensures max image payload size is positive."""
        if int(v) <= 0:
            raise ValueError(f"AI_MAX_IMAGE_SIZE_MB must be positive, got {v}")
        return int(v)

    @field_validator("AI_MAX_IMAGE_DIMENSION")
    @classmethod
    def validate_max_dimension(cls, v: int) -> int:
        """Ensures max image dimension is a reasonable non-zero dimension."""
        if int(v) < 64:
            raise ValueError(f"AI_MAX_IMAGE_DIMENSION must be at least 64px, got {v}")
        return int(v)

    @field_validator("AI_ENABLE_GPU", mode="before")
    @classmethod
    def parse_enable_gpu(cls, v: Any) -> bool:
        """Parses boolean strings for GPU enablement flag."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "t", "yes", "y", "enable", "enabled")
        return bool(v)

    @property
    def ai_max_image_bytes(self) -> int:
        """Returns max image upload size converted to bytes."""
        return self.AI_MAX_IMAGE_SIZE_MB * 1024 * 1024

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v: Any) -> bool:
        """Tolerates booleans, numeric representations, or environment release flags."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "t", "yes", "y", "debug", "dev", "development")
        return bool(v)

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT.lower() in ("test", "testing")

    def resolve_model_path(self, model_name: Optional[str] = None) -> Optional[str]:
        """
        Locates physical ONNX model weights on disk across standard repository search paths.
        Returns the resolved absolute path or None if no weights are found.
        """
        if self.AI_MODEL_PATH and os.path.exists(self.AI_MODEL_PATH):
            return os.path.abspath(self.AI_MODEL_PATH)

        name = model_name or self.AI_MODEL_NAME
        candidate_filenames = [
            f"{name}.onnx",
            "face_detection_yunet_2023mar.onnx",
            "scrfd_500m_bnkps_shape640x640.onnx",
            "scrfd_500m.onnx",
        ]

        base_dir = os.path.dirname(os.path.abspath(__file__))
        search_dirs = [
            os.path.abspath(os.path.join(self.MODELS_DIR, "face_detection")),
            os.path.abspath(os.path.join(base_dir, "../../../models/face_detection")),
            os.path.abspath(os.path.join(base_dir, "../../../../models/face_detection")),
            os.path.abspath("./models/face_detection"),
            os.path.abspath("../models/face_detection"),
            os.path.abspath("./deepvision-ai/models/face_detection"),
        ]

        for s_dir in search_dirs:
            if os.path.isdir(s_dir):
                for fname in candidate_filenames:
                    cand = os.path.join(s_dir, fname)
                    if os.path.exists(cand):
                        return cand
                for f in os.listdir(s_dir):
                    if f.endswith(".onnx"):
                        return os.path.join(s_dir, f)

        return None

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global singleton settings instance
settings = Settings()
