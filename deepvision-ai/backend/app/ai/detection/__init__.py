"""
DeepVision AI — Face Detection Package
Contains modular face detection interfaces, implementations, preprocessing, and result schemas.
"""

from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.detector import FaceDetector
from app.ai.detection.schemas import (
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
from app.ai.detection.preprocessing import (
    load_image,
    letterbox_resize,
    unletterbox_coordinates,
    prepare_input_tensor,
)

__all__ = [
    "BaseFaceDetector",
    "FaceDetector",
    "BoundingBox",
    "NormalizedBoundingBox",
    "LandmarkPoint",
    "FaceQualityScore",
    "FaceDetection",
    "FaceDetectionResult",
    "DetectionResponse",
    "FrameInferenceResult",
    "PerformanceMetrics",
    "load_image",
    "letterbox_resize",
    "unletterbox_coordinates",
    "prepare_input_tensor",
]
