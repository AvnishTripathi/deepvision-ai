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


class BrightnessStatus(str, Enum):
    """Qualitative classification of face illumination status (Step 8)."""
    TOO_DARK = "too_dark"
    ACCEPTABLE = "acceptable"
    TOO_BRIGHT = "too_bright"


class FaceVisibility(str, Enum):
    """
    Qualitative face visibility classification (Step 10).
    Engineering heuristic indicating how clearly a face is observable.
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


VisibilityStatus = FaceVisibility


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


class QualityClassification(str, Enum):
    """
    Overall qualitative face quality classification (Step 13).
    Categorizes the measurable quality score (0 - 100) into standardized tiers:
    - excellent: High-definition, sharp, optimally exposed, strong symmetry & landmarks (>= 80.0)
    - good: Minor imperfections, fully suitable for recognition and detection (>= 65.0)
    - acceptable: Marginal lighting or mild blur, but retains discernible facial geometry (>= 45.0)
    - poor: Severely degraded, blurry, extreme exposure, or critical defects (< 45.0)
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


FaceQualityClassification = QualityClassification
QualityCategory = QualityClassification


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
    """
    3D Head pose orientation estimated via Perspective-n-Point (solvePnP).
    Supports actual yaw, pitch, and roll in degrees (Step 12).

    Disclaimer:
        This is an engineering approximation for face orientation triage;
        it does not claim clinical, surgical, or sub-degree biometric accuracy.
    """
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
        default=1.0, ge=0.0, le=1.0, description="Normalized frontality score (1.0 = directly facing camera)"
    )
    pose_category: PoseCategory = Field(
        default=PoseCategory.FRONTAL, description="Qualitative category of facial orientation"
    )

    model_config = ConfigDict(populate_by_name=True)

    def to_angles_dict(self) -> Dict[str, float]:
        """Returns {yaw, pitch, roll} dictionary conforming strictly to Step 12 specification."""
        return {
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
        }



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
    """
    Illumination, exposure, and contrast evaluation for face ROI.
    Evaluates mean luminance brightness, contrast, and highlight/shadow clipping.
    """
    brightness: float = Field(
        ..., ge=0.0, le=255.0, description="Mean luminance brightness (0 = pitch black, 255 = white)"
    )
    contrast: float = Field(
        ..., ge=0.0, description="Standard deviation luminance contrast"
    )
    brightness_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Normalized brightness suitability score (0.0 to 1.0)"
    )
    brightness_status: BrightnessStatus = Field(
        default=BrightnessStatus.ACCEPTABLE,
        description="Qualitative brightness classification: 'too_dark', 'acceptable', or 'too_bright'",
    )
    exposure_category: ExposureCategory = Field(
        default=ExposureCategory.OPTIMAL,
        description="Qualitative illumination and exposure classification",
    )
    illumination_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Normalized composite illumination quality score (0.0 to 1.0)"
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def sync_brightness_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync illumination_score <-> brightness_score
            if "brightness_score" not in data and "illumination_score" in data:
                data["brightness_score"] = data["illumination_score"]
            elif "illumination_score" not in data and "brightness_score" in data:
                data["illumination_score"] = data["brightness_score"]

            # Sync brightness_status -> exposure_category if missing
            if "brightness_status" in data and "exposure_category" not in data:
                status = str(data["brightness_status"]).lower()
                if status == "too_dark":
                    data["exposure_category"] = ExposureCategory.UNDEREXPOSED
                elif status == "too_bright":
                    data["exposure_category"] = ExposureCategory.OVEREXPOSED
                else:
                    data["exposure_category"] = ExposureCategory.OPTIMAL
            # Sync exposure_category -> brightness_status if missing
            elif "exposure_category" in data and "brightness_status" not in data:
                exp = str(data["exposure_category"]).upper()
                if exp == "UNDEREXPOSED":
                    data["brightness_status"] = BrightnessStatus.TOO_DARK
                elif exp == "OVEREXPOSED":
                    data["brightness_status"] = BrightnessStatus.TOO_BRIGHT
                else:
                    data["brightness_status"] = BrightnessStatus.ACCEPTABLE
        return data


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


class FaceVisibilityAssessment(BaseModel):
    """
    Detailed explainable heuristic visibility assessment breakdown (Step 10).
    Synthesizes face size, blur, brightness, landmark availability, and landmark confidence.
    """
    visibility: FaceVisibility = Field(
        ..., description="Qualitative visibility rating: excellent, good, fair, or poor"
    )
    visibility_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized composite visibility score (0.0 to 1.0)"
    )
    size_factor: float = Field(
        ..., ge=0.0, le=1.0, description="Resolution and face scale factor (0.0 to 1.0)"
    )
    blur_factor: float = Field(
        ..., ge=0.0, le=1.0, description="Sharpness and clarity factor (0.0 to 1.0)"
    )
    brightness_factor: float = Field(
        ..., ge=0.0, le=1.0, description="Luminance and exposure factor (0.0 to 1.0)"
    )
    landmark_availability_factor: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of standard facial landmarks detected (0.0 to 1.0)"
    )
    landmark_confidence_factor: float = Field(
        ..., ge=0.0, le=1.0, description="Mean confidence of detected landmarks or fallback (0.0 to 1.0)"
    )
    details: Optional[str] = Field(
        default=None, description="Diagnostic notes highlighting primary limiting visibility factors"
    )

    model_config = ConfigDict(populate_by_name=True)


class FaceQualityFactors(BaseModel):
    """
    Granular breakdown of transparent, measurable factors for the 0-100 Face Quality Score (Step 11).
    Combines Sharpness, Brightness, Contrast, Face Size, and Landmark Quality.
    """
    sharpness: float = Field(..., ge=0.0, le=100.0, description="Laplacian variance edge definition (0-100)")
    brightness: float = Field(..., ge=0.0, le=100.0, description="Photopic luminance exposure suitability (0-100)")
    contrast: float = Field(..., ge=0.0, le=100.0, description="Luminance dynamic range / contrast (0-100)")
    size: float = Field(..., ge=0.0, le=100.0, description="Pixel resolution and biometric scale (0-100)")
    landmark: float = Field(..., ge=0.0, le=100.0, description="Landmark geometry, symmetry, and confidence (0-100)")

    model_config = ConfigDict(populate_by_name=True)


class ComprehensiveFaceQuality(BaseModel):
    """
    Unified multi-metric face quality evaluation.
    Preserves full backward compatibility with legacy FaceQualityScore.
    """
    quality_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall face quality score on a transparent 0-100 scale"
    )
    overall_quality: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Composite weighted quality score normalized to (0.0 to 1.0)"
    )
    sharpness: float = Field(..., ge=0.0, description="Laplacian variance sharpness measurement")
    brightness: float = Field(..., ge=0.0, le=255.0, description="Mean luminance brightness (0-255)")
    contrast: float = Field(..., ge=0.0, description="Standard deviation luminance contrast")
    is_usable: bool = Field(True, description="Whether face quality meets minimum processing threshold")

    # Extended granular diagnostic breakdowns
    classification: QualityClassification = Field(
        default=QualityClassification.GOOD,
        description="Overall qualitative face quality classification (excellent, good, acceptable, poor)",
    )
    grade: QualityGrade = Field(
        default=QualityGrade.GOOD, description="Qualitative face quality grade"
    )
    factors: Optional[FaceQualityFactors] = Field(
        default=None, description="Transparent 0-100 breakdown across measurable visual factors"
    )
    visibility: Optional[FaceVisibility] = Field(
        default=None, description="Qualitative face visibility rating (excellent, good, fair, poor)"
    )
    visibility_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Normalized composite visibility score (0.0 to 1.0)"
    )
    clarity: Optional[ClarityMetrics] = Field(default=None, description="Detailed clarity & blur metrics")
    illumination: Optional[IlluminationMetrics] = Field(default=None, description="Detailed illumination metrics")
    size: Optional[FaceSizeMetrics] = Field(default=None, description="Detailed scale and dimension metrics")
    pose: Optional[HeadPoseEstimation] = Field(default=None, description="Estimated 3D head pose orientation")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def sync_quality_scores(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "quality_score" not in data and "overall_quality" in data and data["overall_quality"] is not None:
                data["quality_score"] = round(float(data["overall_quality"]) * 100.0, 1)
            elif "overall_quality" not in data and "quality_score" in data and data["quality_score"] is not None:
                data["overall_quality"] = round(float(data["quality_score"]) / 100.0, 3)

            # Sync classification from quality_score if omitted
            if "classification" not in data or data["classification"] is None:
                score = data.get("quality_score")
                if score is None and data.get("overall_quality") is not None:
                    score = float(data["overall_quality"]) * 100.0
                if score is not None:
                    from app.ai.analysis.quality import classify_face_quality
                    data["classification"] = classify_face_quality(float(score))
        return data


class StructuredFaceAnalysis(BaseModel):
    """
    Complete structured face analysis output combining detection, geometry, pose, quality, and visibility.
    """
    face_id: Optional[Union[str, int]] = Field(default=None, description="Face identifier")
    bbox: BoundingBox = Field(..., description="Face bounding box coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    quality_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall face quality score on a 0-100 scale"
    )
    classification: QualityClassification = Field(
        default=QualityClassification.GOOD,
        description="Overall qualitative face quality classification (excellent, good, acceptable, poor)",
    )
    visibility: FaceVisibility = Field(
        default=FaceVisibility.GOOD,
        description="Qualitative face visibility rating (excellent, good, fair, poor)",
    )
    visibility_assessment: Optional[FaceVisibilityAssessment] = Field(
        default=None,
        description="Detailed breakdown of visibility assessment factors",
    )
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

            # Sync visibility from visibility_assessment if present
            if "visibility" not in data and "visibility_assessment" in data and data["visibility_assessment"]:
                va = data["visibility_assessment"]
                if isinstance(va, dict) and "visibility" in va:
                    data["visibility"] = va["visibility"]
                elif hasattr(va, "visibility"):
                    data["visibility"] = va.visibility

            # Sync quality_score from quality if present
            if "quality_score" not in data and "quality" in data and data["quality"]:
                q = data["quality"]
                if isinstance(q, dict) and "quality_score" in q:
                    data["quality_score"] = q["quality_score"]
                elif hasattr(q, "quality_score"):
                    data["quality_score"] = q.quality_score
                elif isinstance(q, dict) and "overall_quality" in q:
                    data["quality_score"] = round(float(q["overall_quality"]) * 100.0, 1)
                elif hasattr(q, "overall_quality"):
                    data["quality_score"] = round(float(q.overall_quality) * 100.0, 1)

            # Sync classification from quality if present
            if "classification" not in data or data["classification"] is None:
                q = data.get("quality")
                if isinstance(q, dict) and "classification" in q:
                    data["classification"] = q["classification"]
                elif hasattr(q, "classification"):
                    data["classification"] = q.classification
                elif "quality_score" in data and data["quality_score"] is not None:
                    from app.ai.analysis.quality import classify_face_quality
                    data["classification"] = classify_face_quality(float(data["quality_score"]))
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

