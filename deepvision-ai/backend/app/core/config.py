"""
DeepVision AI — Core Configuration
Provides strongly typed, validated settings powered by Pydantic Settings.
Configuration is loaded from environment variables and .env files.
"""

import os
from typing import List, Optional, Any, Union
import json
from pydantic import field_validator, model_validator
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
    AI_NMS_THRESHOLD: float = 0.30
    AI_MAX_IMAGE_SIZE_MB: int = 10
    AI_MAX_IMAGE_DIMENSION: int = 4096
    AI_ENABLE_GPU: bool = False
    AI_MODEL_PATH: Optional[str] = None
    AI_REQUIRE_MODEL: bool = False


    # Pipeline Modular Feature Toggles (Step 19)
    AI_ENABLE_LANDMARKS: bool = True
    AI_ENABLE_QUALITY_ANALYSIS: bool = True
    AI_ENABLE_POSE_ANALYSIS: bool = True

    # Face Quality & Optics Configurable Thresholds (Step 19)
    FACE_BLUR_THRESHOLD: float = 50.0
    FACE_BRIGHTNESS_MIN: float = 60.0
    FACE_BRIGHTNESS_MAX: float = 200.0
    FACE_QUALITY_THRESHOLD: float = 45.0

    # Face Blur & Sharpness Thresholds (Step 7)
    AI_BLUR_SHARP_THRESHOLD: float = 100.0
    AI_BLUR_SLIGHTLY_BLURRED_THRESHOLD: float = 50.0

    # Face Brightness & Illumination Thresholds (Step 8)
    AI_BRIGHTNESS_MIN_ACCEPTABLE: float = 60.0
    AI_BRIGHTNESS_MAX_ACCEPTABLE: float = 200.0

    # Face Visibility Assessment Thresholds (Step 10)
    AI_VISIBILITY_EXCELLENT_THRESHOLD: float = 0.82
    AI_VISIBILITY_GOOD_THRESHOLD: float = 0.65
    AI_VISIBILITY_FAIR_THRESHOLD: float = 0.45

    # Face Quality Classification Thresholds (Step 13)
    AI_QUALITY_EXCELLENT_THRESHOLD: float = 80.0
    AI_QUALITY_GOOD_THRESHOLD: float = 65.0
    AI_QUALITY_ACCEPTABLE_THRESHOLD: float = 45.0

    # Storage Paths & Limits
    STORAGE_DIR: str = "./storage"
    MODELS_DIR: str = "./models"
    MAX_IMAGE_SIZE_MB: int = 50
    MAX_VIDEO_SIZE_MB: int = 500

    # Video Face Detection & Multi-Face Tracking Settings (Step 5.11)
    VIDEO_DETECTION_INTERVAL: int = 1
    VIDEO_ANALYSIS_INTERVAL: int = 5
    VIDEO_MAX_FPS: Optional[float] = None
    VIDEO_TRACKER_HIGH_THRESHOLD: float = 0.50
    VIDEO_TRACKER_LOW_THRESHOLD: float = 0.15
    VIDEO_TRACKER_MAX_TIME_LOST: int = 30
    VIDEO_TRACKER_MIN_HITS: int = 2

    # Video Security & Resource Limits (Step 5.24)
    VIDEO_MAX_FILE_SIZE_MB: int = 500
    VIDEO_MAX_DURATION_SECONDS: float = 600.0
    VIDEO_MAX_WIDTH: int = 4096
    VIDEO_MAX_HEIGHT: int = 4096
    VIDEO_MAX_FRAME_COUNT: int = 18000

    # Security & Networking Hardening
    SECRET_KEY: str = "deepvision-ai-development-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALLOWED_HOSTS: Union[List[str], str] = ["*"]

    # Rate Limiting (Step 7.4)
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_IMAGE_ANALYSIS: str = "60/minute"
    RATE_LIMIT_VIDEO_ANALYSIS: str = "10/minute"

    # Concurrency Resource Protection (Step 7.6 / 7.8)
    VIDEO_MAX_CONCURRENT_JOBS: int = 4
    MAX_CONCURRENT_VIDEO_JOBS: Optional[int] = None

    @model_validator(mode="after")
    def sync_concurrency_settings(self):
        if self.MAX_CONCURRENT_VIDEO_JOBS is not None:
            self.VIDEO_MAX_CONCURRENT_JOBS = self.MAX_CONCURRENT_VIDEO_JOBS
        return self

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

    @field_validator("ALLOWED_HOSTS")
    @classmethod
    def parse_allowed_hosts(cls, v: Union[List[str], str]) -> List[str]:
        """Supports comma-separated strings or list of allowed HTTP hosts."""
        if isinstance(v, list):
            return [str(item).strip() for item in v]
        if isinstance(v, str):
            v = v.strip()
            if not v or v == "*":
                return ["*"]
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

    @field_validator("AI_NMS_THRESHOLD")
    @classmethod
    def validate_nms_threshold(cls, v: float) -> float:
        """Validates Non-Maximum Suppression IoU threshold is within [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"AI_NMS_THRESHOLD must be between 0.0 and 1.0, got {v}")
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

    @field_validator(
        "AI_QUALITY_EXCELLENT_THRESHOLD",
        "AI_QUALITY_GOOD_THRESHOLD",
        "AI_QUALITY_ACCEPTABLE_THRESHOLD",
    )
    @classmethod
    def validate_quality_threshold(cls, v: float) -> float:
        """Ensures quality thresholds are within the 0 to 100 range."""
        if not (0.0 <= v <= 100.0):
            raise ValueError(f"Quality threshold must be between 0.0 and 100.0, got {v}")
        return float(v)

    @field_validator(
        "AI_ENABLE_LANDMARKS",
        "AI_ENABLE_QUALITY_ANALYSIS",
        "AI_ENABLE_POSE_ANALYSIS",
        "AI_ENABLE_GPU",
        mode="before",
    )
    @classmethod
    def parse_boolean_flags(cls, v: Any) -> bool:
        """Parses boolean strings, integers, or literals for feature flags."""
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.lower().strip() in ("true", "1", "t", "yes", "y", "enable", "enabled", "on")
        return bool(v)

    @field_validator("FACE_BLUR_THRESHOLD")
    @classmethod
    def validate_face_blur_threshold(cls, v: float) -> float:
        """Ensures FACE_BLUR_THRESHOLD is non-negative."""
        if float(v) < 0.0:
            raise ValueError(f"FACE_BLUR_THRESHOLD must be non-negative, got {v}")
        return float(v)

    @field_validator("FACE_BRIGHTNESS_MIN", "FACE_BRIGHTNESS_MAX")
    @classmethod
    def validate_face_brightness_bounds(cls, v: float) -> float:
        """Ensures brightness thresholds fall within 8-bit luminance bounds [0.0, 255.0]."""
        if not (0.0 <= float(v) <= 255.0):
            raise ValueError(f"Brightness threshold must be in range [0.0, 255.0], got {v}")
        return float(v)

    @field_validator("FACE_QUALITY_THRESHOLD")
    @classmethod
    def validate_face_quality_threshold(cls, v: float) -> float:
        """Ensures FACE_QUALITY_THRESHOLD falls within score bounds [0.0, 100.0]."""
        if not (0.0 <= float(v) <= 100.0):
            raise ValueError(f"FACE_QUALITY_THRESHOLD must be between 0.0 and 100.0, got {v}")
        return float(v)

    @model_validator(mode="after")
    def validate_brightness_order(self) -> "Settings":
        """Ensures FACE_BRIGHTNESS_MIN is strictly less than FACE_BRIGHTNESS_MAX."""
        if self.FACE_BRIGHTNESS_MIN >= self.FACE_BRIGHTNESS_MAX:
            raise ValueError(
                f"FACE_BRIGHTNESS_MIN ({self.FACE_BRIGHTNESS_MIN}) must be strictly less than "
                f"FACE_BRIGHTNESS_MAX ({self.FACE_BRIGHTNESS_MAX})"
            )
        return self

    def get_public_ai_config(self) -> dict:
        """
        Returns safe, non-sensitive AI pipeline configuration.
        Guarantees zero exposure of secrets (SECRET_KEY, DB passwords, tokens).
        """
        return {
            "model_name": self.AI_MODEL_NAME,
            "model_version": self.AI_MODEL_VERSION,
            "device": self.AI_DEVICE,
            "confidence_threshold": self.AI_CONFIDENCE_THRESHOLD,
            "nms_threshold": self.AI_NMS_THRESHOLD,
            "enable_landmarks": self.AI_ENABLE_LANDMARKS,
            "enable_quality_analysis": self.AI_ENABLE_QUALITY_ANALYSIS,
            "enable_pose_analysis": self.AI_ENABLE_POSE_ANALYSIS,
            "face_blur_threshold": self.FACE_BLUR_THRESHOLD,
            "face_brightness_min": self.FACE_BRIGHTNESS_MIN,
            "face_brightness_max": self.FACE_BRIGHTNESS_MAX,
            "face_quality_threshold": self.FACE_QUALITY_THRESHOLD,
        }

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
