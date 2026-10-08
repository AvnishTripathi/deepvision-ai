"""
DeepVision AI — Face Analysis Schemas
Strongly typed Pydantic v2 schemas for facial landmarks, head pose estimation,
size metrics, illumination, clarity, and comprehensive face quality scoring.
"""

from enum import Enum
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator

from app.schemas.ai import (
    BoundingBox,
    FaceBoundingBox,
    NormalizedBoundingBox,
    LandmarkPoint,
    LandmarkCoordinate,
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


class BlurStatus(str, Enum):
    """Qualitative classification of face sharpness/blur status (Step 7)."""
    SHARP = "sharp"
    SLIGHTLY_BLURRED = "slightly_blurred"
    BLURRY = "blurry"


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
    face_width: float = Field(..., ge=0.0, description="Face bounding box width in pixels")
    face_height: float = Field(..., ge=0.0, description="Face bounding box height in pixels")
    face_area: float = Field(..., ge=0.0, description="Face bounding box area in square pixels")
    face_area_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of face area relative to entire image canvas area (face_area / image_area)"
    )
    width_px: Optional[float] = Field(default=None, ge=0.0, description="Alias for face_width")
    height_px: Optional[float] = Field(default=None, ge=0.0, description="Alias for face_height")
    aspect_ratio: float = Field(..., ge=0.0, description="Width-to-height ratio (W / H)")
    area_px: Optional[float] = Field(default=None, ge=0.0, description="Alias for face_area")
    relative_area: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Alias for face_area_ratio"
    )
    size_category: SizeCategory = Field(..., description="Qualitative size category")
    is_sufficient_size: bool = Field(
        True, description="Whether face resolution meets minimum biometric processing threshold"
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def populate_size_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Map width
            if "face_width" not in data and "width_px" in data:
                data["face_width"] = data["width_px"]
            elif "width_px" not in data and "face_width" in data:
                data["width_px"] = data["face_width"]

            # Map height
            if "face_height" not in data and "height_px" in data:
                data["face_height"] = data["height_px"]
            elif "height_px" not in data and "face_height" in data:
                data["height_px"] = data["face_height"]

            # Map area
            if "face_area" not in data and "area_px" in data:
                data["face_area"] = data["area_px"]
            elif "area_px" not in data and "face_area" in data:
                data["area_px"] = data["face_area"]

            # Map relative area / ratio
            if "face_area_ratio" not in data and "relative_area" in data:
                data["face_area_ratio"] = data["relative_area"]
            elif "relative_area" not in data and "face_area_ratio" in data:
                data["relative_area"] = data["face_area_ratio"]
        return data


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
    """
    Sharpness and blur estimation diagnostics based on Laplacian variance.
    Note: Blur score reflects high-frequency spatial variation and focus;
    it is not a universal quality measurement (factors such as smooth skin,
    diffuse lighting, or cosmetic makeup can lower variance without loss of fidelity).
    """
    sharpness: float = Field(
        ..., ge=0.0, description="Laplacian variance sharpness measurement"
    )
    blur_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized clarity score (1.0 = sharp, 0.0 = heavily blurred)"
    )
    blur_status: BlurStatus = Field(
        default=BlurStatus.SHARP,
        description="Qualitative blur status classification ('sharp', 'slightly_blurred', or 'blurry')",
    )
    blur_category: BlurCategory = Field(
        default=BlurCategory.SHARP,
        description="Qualitative blur and sharpness classification (SHARP, SLIGHTLY_BLURRED, BLURRED, VERY_BLURRED)",
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def sync_blur_status_and_category(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync blur_status -> blur_category if blur_category omitted
            if "blur_status" in data and "blur_category" not in data:
                raw_st = str(data["blur_status"]).lower()
                if raw_st == "sharp":
                    data["blur_category"] = BlurCategory.SHARP
                elif raw_st == "slightly_blurred":
                    data["blur_category"] = BlurCategory.SLIGHTLY_BLURRED
                elif raw_st in ("blurry", "blurred"):
                    data["blur_category"] = BlurCategory.BLURRED
            # Sync blur_category -> blur_status if blur_status omitted
            elif "blur_category" in data and "blur_status" not in data:
                raw_cat = str(data["blur_category"]).upper()
                if raw_cat == "SHARP":
                    data["blur_status"] = BlurStatus.SHARP
                elif raw_cat == "SLIGHTLY_BLURRED":
                    data["blur_status"] = BlurStatus.SLIGHTLY_BLURRED
                else:
                    data["blur_status"] = BlurStatus.BLURRY
        return data


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
    landmarks: Dict[str, LandmarkCoordinate] = Field(
        default_factory=dict,
        description="Structured facial landmarks map {name: {x, y}}",
    )
    landmarks_list: Optional[List[LandmarkPoint]] = Field(
        default=None,
        description="Optional list representation of landmark points",
    )
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

    @model_validator(mode="before")
    @classmethod
    def normalize_landmarks_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw_lm = data.get("landmarks")
            if isinstance(raw_lm, list):
                from app.ai.analysis.landmarks import landmarks_to_dict
                data["landmarks_list"] = raw_lm
                data["landmarks"] = landmarks_to_dict(raw_lm)
            elif isinstance(raw_lm, dict):
                from app.ai.analysis.landmarks import landmarks_to_dict
                data["landmarks"] = landmarks_to_dict(raw_lm)
        return data


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

