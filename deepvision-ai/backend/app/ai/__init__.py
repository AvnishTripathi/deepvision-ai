"""
DeepVision AI — AI & Computer Vision Architecture Package
Exposes core inference abstractions, preprocessing utilities, detection engines,
and analytics modules.
"""

from app.ai.preprocessing import (
    load_image,
    letterbox_resize,
    unletterbox_coordinates,
    prepare_input_tensor,
)
from app.ai.inference.base import BaseInferenceEngine
from app.ai.inference.onnx_engine import ONNXInferenceEngine
from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.detector import FaceDetector
from app.ai.analysis.quality import analyze_face_quality
from app.schemas.ai import (
    BoundingBox,
    LandmarkPoint,
    FaceQualityScore,
    FaceDetectionResult,
    FrameInferenceResult,
    PerformanceMetrics,
)

__all__ = [
    "load_image",
    "letterbox_resize",
    "unletterbox_coordinates",
    "prepare_input_tensor",
    "BaseInferenceEngine",
    "ONNXInferenceEngine",
    "BaseFaceDetector",
    "FaceDetector",
    "analyze_face_quality",
    "BoundingBox",
    "LandmarkPoint",
    "FaceQualityScore",
    "FaceDetectionResult",
    "FrameInferenceResult",
    "PerformanceMetrics",
]
