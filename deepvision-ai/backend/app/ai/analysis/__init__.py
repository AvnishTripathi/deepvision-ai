"""
DeepVision AI — Comprehensive Face Analysis Package
Exports schemas, landmark processing, 3D head pose estimation, metrics, and quality analysis.
"""

from app.ai.analysis.schemas import (
    PoseCategory,
    BlurCategory,
    ExposureCategory,
    SizeCategory,
    QualityGrade,
    FacialLandmarkMetrics,
    HeadPoseEstimation,
    FaceSizeMetrics,
    IlluminationMetrics,
    ClarityMetrics,
    ComprehensiveFaceQuality,
    StructuredFaceAnalysis,
    AIAnalysisResponse,
)
from app.ai.analysis.landmarks import (
    LandmarkProcessor,
    compute_landmark_metrics,
    STANDARD_LANDMARK_NAMES,
    LANDMARK_NAME_ALIASES,
    normalize_landmark_name,
    landmarks_to_dict,
    format_structured_landmarks,
    landmarks_to_list,
)
from app.ai.analysis.pose import (
    CANONICAL_3D_MODEL_POINTS,
    HeadPoseEstimator,
    estimate_head_pose,
)
from app.ai.analysis.metrics import (
    compute_face_size_metrics,
    compute_clarity_metrics,
    compute_illumination_metrics,
)
from app.ai.analysis.quality import (
    compute_sharpness,
    compute_brightness_and_contrast,
    analyze_face_quality,
    evaluate_quality_grade,
    analyze_face_comprehensive,
    FaceAnalyzer,
)

__all__ = [
    # Schemas & Enums
    "PoseCategory",
    "BlurCategory",
    "ExposureCategory",
    "SizeCategory",
    "QualityGrade",
    "FacialLandmarkMetrics",
    "HeadPoseEstimation",
    "FaceSizeMetrics",
    "IlluminationMetrics",
    "ClarityMetrics",
    "ComprehensiveFaceQuality",
    "StructuredFaceAnalysis",
    "AIAnalysisResponse",
    # Landmark Processing
    "LandmarkProcessor",
    "compute_landmark_metrics",
    "STANDARD_LANDMARK_NAMES",
    "LANDMARK_NAME_ALIASES",
    "normalize_landmark_name",
    "landmarks_to_dict",
    "format_structured_landmarks",
    "landmarks_to_list",
    # 3D Head Pose
    "CANONICAL_3D_MODEL_POINTS",
    "HeadPoseEstimator",
    "estimate_head_pose",
    # Metrics
    "compute_face_size_metrics",
    "compute_clarity_metrics",
    "compute_illumination_metrics",
    # Quality Analysis & Legacy
    "compute_sharpness",
    "compute_brightness_and_contrast",
    "analyze_face_quality",
    "evaluate_quality_grade",
    "analyze_face_comprehensive",
    "FaceAnalyzer",
]
