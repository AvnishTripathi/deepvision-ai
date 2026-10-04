"""
Re-export AI schemas under app.ai.schemas namespace for internal AI module convenience.
"""

from app.schemas.ai import (
    BoundingBox,
    NormalizedBoundingBox,
    LandmarkPoint,
    FaceQualityScore,
    FaceDetection,
    FaceDetectionResult,
    DetectionResponse,
    FrameInferenceResult,
    PerformanceMetrics,
)

__all__ = [
    "BoundingBox",
    "NormalizedBoundingBox",
    "LandmarkPoint",
    "FaceQualityScore",
    "FaceDetection",
    "FaceDetectionResult",
    "DetectionResponse",
    "FrameInferenceResult",
    "PerformanceMetrics",
]
