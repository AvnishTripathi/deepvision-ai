"""
DeepVision AI — Schemas Package Initialization
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

