"""
DeepVision AI — AI & Computer Vision Architecture Package
Exposes core inference abstractions, preprocessing utilities, detection engines,
device configuration, and analytics modules.
"""

from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.detector import FaceDetector
from app.ai.detection.preprocessing import (
    load_image,
    letterbox_resize,
    unletterbox_coordinates,
    prepare_input_tensor,
)
from app.ai.detection.schemas import (
    BoundingBox,
    LandmarkPoint,
    FaceQualityScore,
    FaceDetectionResult,
    FrameInferenceResult,
    PerformanceMetrics,
)
from app.ai.inference.runtime import InferenceRuntime
from app.ai.inference.device import (
    resolve_device,
    is_gpu_available,
    is_cuda_available,
    get_available_gpu_providers,
    get_optimal_providers,
    get_recommended_thread_count,
    get_device_telemetry,
)
from app.ai.analysis.quality import analyze_face_quality
from app.ai.common.types import (
    ImageSource,
    BoundingBoxCoords,
    Point2D,
    DeviceType,
    DeviceTarget,
    ModelArchitecture,
    FACIAL_LANDMARK_NAMES,
)

__all__ = [
    # Detection
    "BaseFaceDetector",
    "FaceDetector",
    "load_image",
    "letterbox_resize",
    "unletterbox_coordinates",
    "prepare_input_tensor",
    # Schemas
    "BoundingBox",
    "LandmarkPoint",
    "FaceQualityScore",
    "FaceDetectionResult",
    "FrameInferenceResult",
    "PerformanceMetrics",
    # Inference & Hardware Abstraction
    "DeviceType",
    "InferenceRuntime",
    "resolve_device",
    "is_gpu_available",
    "is_cuda_available",
    "get_available_gpu_providers",
    "get_optimal_providers",
    "get_recommended_thread_count",
    "get_device_telemetry",
    # Analysis
    "analyze_face_quality",
    # Common Types
    "ImageSource",
    "BoundingBoxCoords",
    "Point2D",
    "DeviceTarget",
    "ModelArchitecture",
    "FACIAL_LANDMARK_NAMES",
]
