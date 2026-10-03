"""
DeepVision AI — Database Models Package
Architecture prepared for upcoming domain models:
- users
- analysis_jobs
- media_assets
- analysis_results
- authentication_logs
- model_versions
"""

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

__all__ = [
    "Base",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    # Planned future entity exports:
    # "User",
    # "AnalysisJob",
    # "MediaAsset",
    # "AnalysisResult",
    # "AuthenticationLog",
    # "ModelVersion",
]
