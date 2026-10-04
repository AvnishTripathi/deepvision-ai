"""
Re-export AI schemas under app.ai.schemas namespace for internal AI module convenience.
"""

from app.schemas.ai import (
    BoundingBox,
    LandmarkPoint,
    FaceQualityScore,
    FaceDetectionResult,
    FrameInferenceResult,
    PerformanceMetrics,
)

__all__ = [
    "BoundingBox",
    "LandmarkPoint",
    "FaceQualityScore",
    "FaceDetectionResult",
    "FrameInferenceResult",
    "PerformanceMetrics",
]
