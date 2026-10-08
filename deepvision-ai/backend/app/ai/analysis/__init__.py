"""
DeepVision AI — Comprehensive Face Analysis Package
Exports schemas, landmark processing, 3D head pose estimation, metrics, and quality analysis.
"""

from app.ai.analysis.schemas import (
    PoseCategory,
    BlurCategory,
    BlurStatus,
    ExposureCategory,
    BrightnessStatus,
    FaceVisibility,
    VisibilityStatus,
    FaceVisibilityAssessment,
    FaceQualityFactors,
    SizeCategory,
    QualityGrade,
    QualityClassification,
    FaceQualityClassification,
    QualityCategory,
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
    validate_and_clip_landmarks,
    validate_and_clip_landmarks_list,
)
from app.ai.analysis.pose import (
    CANONICAL_3D_MODEL_POINTS,
    HeadPoseEstimator,
    estimate_head_pose,
    get_pose_estimation_status,
)
from app.ai.analysis.metrics import (
    compute_face_size_metrics,
    calculate_face_size,
    estimate_face_blur,
    compute_clarity_metrics,
    calculate_face_brightness,
    compute_illumination_metrics,
    calculate_face_visibility,
)
from app.ai.analysis.quality import (
    compute_sharpness,
    compute_brightness_and_contrast,
    analyze_face_quality,
    evaluate_quality_grade,
    classify_face_quality,
    classify_quality,
    calculate_face_quality_score,
    analyze_face_comprehensive,
    FaceAnalyzer,
    FaceQualityAnalyzer,
)

__all__ = [
    # Schemas & Enums
    "PoseCategory",
    "BlurCategory",
    "BlurStatus",
    "ExposureCategory",
    "BrightnessStatus",
    "FaceVisibility",
    "VisibilityStatus",
    "FaceVisibilityAssessment",
    "FaceQualityFactors",
    "SizeCategory",
    "QualityGrade",
    "QualityClassification",
    "FaceQualityClassification",
    "QualityCategory",
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
    "validate_and_clip_landmarks",
    "validate_and_clip_landmarks_list",
    # 3D Head Pose
    "CANONICAL_3D_MODEL_POINTS",
    "HeadPoseEstimator",
    "estimate_head_pose",
    "get_pose_estimation_status",
    # Metrics
    "compute_face_size_metrics",
    "calculate_face_size",
    "estimate_face_blur",
    "compute_clarity_metrics",
    "calculate_face_brightness",
    "compute_illumination_metrics",
    "calculate_face_visibility",
    # Quality Analysis & Legacy
    "compute_sharpness",
    "compute_brightness_and_contrast",
    "analyze_face_quality",
    "evaluate_quality_grade",
    "classify_face_quality",
    "classify_quality",
    "calculate_face_quality_score",
    "analyze_face_comprehensive",
    "FaceAnalyzer",
    "FaceQualityAnalyzer",
]


