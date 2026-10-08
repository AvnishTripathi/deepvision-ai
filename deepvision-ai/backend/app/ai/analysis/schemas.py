"""
DeepVision AI — Face Analysis Schemas
Strongly typed Pydantic v2 schemas for facial landmarks, head pose estimation,
size metrics, illumination, clarity, and comprehensive face quality scoring.
"""

from enum import Enum
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from app.schemas.ai import (
    BoundingBox,
    FaceBoundingBox,
    NormalizedBoundingBox,
    LandmarkPoint,
)


class PoseCategory(str, Enum):
    """Categorical classification of head pose orientation."""
    FRONTAL = "FRONTAL"
    SLIGHT_TURN = "SLIGHT_TURN"
    PROFILE = "PROFILE"
    UPWARD = "UPWARD"
    DOWNWARD = "DOWNWARD"


class BlurCategory(str, Enum):
    """Categorical classification of facial sharpness/blur."""
    SHARP = "SHARP"
    SLIGHTLY_BLURRED = "SLIGHTLY_BLURRED"
    BLURRED = "BLURRED"
    VERY_BLURRED = "VERY_BLURRED"


class ExposureCategory(str, Enum):
    """Categorical classification of facial illumination and exposure."""
    OPTIMAL = "OPTIMAL"
    UNDEREXPOSED = "UNDEREXPOSED"
    OVEREXPOSED = "OVEREXPOSED"
    HARSH_CONTRAST = "HARSH_CONTRAST"
    LOW_CONTRAST = "LOW_CONTRAST"


class SizeCategory(str, Enum):
    """Categorical classification of face scale in image pixels."""
    TINY = "TINY"          # < 32px
    SMALL = "SMALL"        # 32px - 80px
    MEDIUM = "MEDIUM"      # 80px - 200px
    LARGE = "LARGE"        # > 200px


class QualityGrade(str, Enum):
    """Overall qualitative grade for biometric/visual face utility."""
    EXCELLENT = "EXCELLENT"  # >= 0.85
    GOOD = "GOOD"            # >= 0.70
    FAIR = "FAIR"            # >= 0.50
    POOR = "POOR"            # >= 0.30
    UNUSABLE = "UNUSABLE"    # < 0.30


class FacialLandmarkMetrics(BaseModel):
    """Geometric and spatial relationship metrics computed from facial landmarks."""
    interpupillary_distance_px: float = Field(
        ..., ge=0.0, description="Inter-pupillary distance (IPD) between eyes in pixels"
    )
    eye_tilt_angle_deg: float = Field(
        ..., description="Tilt angle of the eye-line relative to horizontal (degrees)"
    )
    eye_to_nose_distance_px: float = Field(
        ..., ge=0.0, description="Mean Euclidean distance from eyes to nose tip in pixels"
    )
    nose_to_mouth_distance_px: float = Field(
        ..., ge=0.0, description="Mean Euclidean distance from nose tip to mouth corners in pixels"
    )
    symmetry_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Facial bilateral symmetry score (1.0 = perfectly symmetric)"
    )

    model_config = ConfigDict(populate_by_name=True)


class HeadPoseEstimation(BaseModel):
    """3D Head pose orientation estimated via Perspective-n-Point (solvePnP)."""
    yaw: float = Field(
        ..., description="Yaw rotation angle in degrees (negative = turned left, positive = turned right)"
    )
    pitch: float = Field(
        ..., description="Pitch rotation angle in degrees (negative = tilted down, positive = tilted up)"
    )
    roll: float = Field(
        ..., description="Roll rotation angle in degrees (tilt sideways, ear towards shoulder)"
    )
    frontality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized frontality score (1.0 = directly facing camera)"
    )
    pose_category: PoseCategory = Field(
        ..., description="Qualitative category of facial orientation"
    )

    model_config = ConfigDict(populate_by_name=True)


class FaceSizeMetrics(BaseModel):
    """Scale, dimensions, and resolution diagnostics for detected face ROI."""
    width_px: float = Field(..., ge=0.0, description="Face bounding box width in pixels")
    height_px: float = Field(..., ge=0.0, description="Face bounding box height in pixels")
    aspect_ratio: float = Field(..., ge=0.0, description="Width-to-height ratio (W / H)")
    area_px: float = Field(..., ge=0.0, description="Face bounding box area in square pixels")
    relative_area: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of face area relative to entire image canvas"
    )
    size_category: SizeCategory = Field(..., description="Qualitative size category")
    is_sufficient_size: bool = Field(
        True, description="Whether face resolution meets minimum biometric processing threshold"
    )

    model_config = ConfigDict(populate_by_name=True)


class IlluminationMetrics(BaseModel):
    """Illumination, exposure, and contrast evaluation for face ROI."""
    brightness: float = Field(
        ..., ge=0.0, le=255.0, description="Mean luminance brightness (0 = pitch black, 255 = white)"
    )
    contrast: float = Field(
        ..., ge=0.0, description="Standard deviation luminance contrast"
    )
    exposure_category: ExposureCategory = Field(
        ..., description="Qualitative illumination and exposure classification"
    )
    illumination_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized illumination quality score (0.0 to 1.0)"
    )

    model_config = ConfigDict(populate_by_name=True)


class ClarityMetrics(BaseModel):
    """Sharpness, blur detection, and frequency clarity measurements."""
    sharpness: float = Field(
        ..., ge=0.0, description="Laplacian variance sharpness measurement"
    )
    blur_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized clarity score (1.0 = sharp, 0.0 = heavily blurred)"
    )
    blur_category: BlurCategory = Field(
        ..., description="Qualitative blur and sharpness classification"
    )

    model_config = ConfigDict(populate_by_name=True)


class ComprehensiveFaceQuality(BaseModel):
    """
    Unified multi-metric face quality evaluation.
    Preserves full backward compatibility with legacy FaceQualityScore.
    """
    overall_quality: float = Field(
        ..., ge=0.0, le=1.0, description="Composite weighted quality score (0.0 to 1.0)"
    )
    sharpness: float = Field(..., ge=0.0, description="Laplacian variance sharpness measurement")
    brightness: float = Field(..., ge=0.0, le=255.0, description="Mean luminance brightness (0-255)")
    contrast: float = Field(..., ge=0.0, description="Standard deviation luminance contrast")
    is_usable: bool = Field(True, description="Whether face quality meets minimum processing threshold")

    # Extended granular diagnostic breakdowns
    grade: QualityGrade = Field(
        default=QualityGrade.GOOD, description="Qualitative face quality grade"
    )
    clarity: Optional[ClarityMetrics] = Field(default=None, description="Detailed clarity & blur metrics")
    illumination: Optional[IlluminationMetrics] = Field(default=None, description="Detailed illumination metrics")
    size: Optional[FaceSizeMetrics] = Field(default=None, description="Detailed scale and dimension metrics")
    pose: Optional[HeadPoseEstimation] = Field(default=None, description="Estimated 3D head pose orientation")

    model_config = ConfigDict(populate_by_name=True)


class StructuredFaceAnalysis(BaseModel):
    """
    Complete structured face analysis output combining detection, geometry, pose, and quality.
    """
    face_id: Optional[Union[str, int]] = Field(default=None, description="Face identifier")
    bbox: BoundingBox = Field(..., description="Face bounding box coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    landmarks: List[LandmarkPoint] = Field(default_factory=list, description="5-point facial landmarks")
    landmark_metrics: Optional[FacialLandmarkMetrics] = Field(
        default=None, description="Inter-pupillary distance and bilateral symmetry metrics"
    )
    pose: Optional[HeadPoseEstimation] = Field(
        default=None, description="3D Head pose angles (yaw, pitch, roll) and frontality"
    )
    size: FaceSizeMetrics = Field(..., description="Face resolution and scale metrics")
    illumination: IlluminationMetrics = Field(..., description="Face exposure and illumination metrics")
    clarity: ClarityMetrics = Field(..., description="Face sharpness and blur metrics")
    quality: ComprehensiveFaceQuality = Field(..., description="Composite quality assessment and grade")

    model_config = ConfigDict(populate_by_name=True)


class AIAnalysisResponse(BaseModel):
    """
    Standard top-level face analysis response schema.
    Returns detection, landmark geometry, 3D pose, and comprehensive quality metrics.
    """
    success: bool = Field(True, description="Analysis execution status")
    model: str = Field(..., description="Active detection model name")
    device: str = Field(..., description="Inference device used (e.g. CPU, GPU)")
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")
    processing_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    face_count: int = Field(0, description="Total number of analyzed faces")
    faces: List[StructuredFaceAnalysis] = Field(default_factory=list, description="List of structured face analyses")
    request_id: Optional[str] = Field(default=None, description="Unique request trace identifier")

    model_config = ConfigDict(populate_by_name=True)

