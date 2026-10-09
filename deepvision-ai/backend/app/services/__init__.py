"""
DeepVision AI — Application Services Package
"""

from app.services.ai_service import AIService, get_ai_service
from app.services.face_analysis_service import (
    FaceAnalysisService,
    get_face_analysis_service,
)

__all__ = [
    "AIService",
    "get_ai_service",
    "FaceAnalysisService",
    "get_face_analysis_service",
]
