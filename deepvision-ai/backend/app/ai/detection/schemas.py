"""
DeepVision AI — Detection Schemas
Standardized schemas for face bounding boxes, facial landmarks, face detections,
and API detection responses.
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
