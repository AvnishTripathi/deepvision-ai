"""
DeepVision AI — Database Models Package
Exports enterprise PostgreSQL / SQLAlchemy 2.0 domain entities:
- User (users)
- MediaAsset (media_assets)
- ModelVersion (model_versions)
- AnalysisJob (analysis_jobs)
- AnalysisResult (analysis_results)
- SystemLog (system_logs)

CRITICAL PRIVACY MANDATE:
Zero biometric identity, 1:1 / 1:N face matching, or face image storage tables.
"""

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.user import User
from app.models.media_asset import MediaAsset
from app.models.model_version import ModelVersion
from app.models.analysis_job import AnalysisJob
from app.models.analysis_result import AnalysisResult
from app.models.system_log import SystemLog

__all__ = [
    "Base",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "User",
    "MediaAsset",
    "ModelVersion",
    "AnalysisJob",
    "AnalysisResult",
    "SystemLog",
]

