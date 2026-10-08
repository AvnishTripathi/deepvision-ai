"""
DeepVision AI — Comprehensive Face Quality Analytics & Scoring Module
Combines sharpness, illumination, resolution, and 3D head pose into explainable quality metrics.
Maintains full backward compatibility with legacy FaceQualityScore callers.
"""

from typing import Tuple, Optional, List, Union, Any
import numpy as np
import cv2

from app.schemas.ai import (
    BoundingBox,
    LandmarkPoint,
    FaceQualityScore,
)
from app.ai.analysis.schemas import (
    ComprehensiveFaceQuality,
    QualityGrade,
    StructuredFaceAnalysis,
    FacialLandmarkMetrics,
    HeadPoseEstimation,
    FaceSizeMetrics,
    IlluminationMetrics,
    ClarityMetrics,
    FaceVisibility,
    FaceVisibilityAssessment,
)
from app.ai.analysis.landmarks import (
    compute_landmark_metrics,
    validate_and_clip_landmarks_list,
)
from app.ai.analysis.pose import estimate_head_pose
from app.ai.analysis.metrics import (
    compute_face_size_metrics,
    compute_clarity_metrics,
    compute_illumination_metrics,
    calculate_face_visibility,
)

import logging
logger = logging.getLogger("deepvision.ai.analysis")


# -----------------------------------------------------------------------------
# 1. Legacy Compatibility Functions
# -----------------------------------------------------------------------------

def compute_sharpness(crop_rgb: np.ndarray) -> float:
    """
    Computes sharpness using the variance of the Laplacian filter.
    Higher values indicate sharp, in-focus faces; low values indicate motion blur.
    """
    if crop_rgb.size == 0 or crop_rgb.shape[0] < 4 or crop_rgb.shape[1] < 4:
        return 0.0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return round(float(laplacian.var()), 2)


def compute_brightness_and_contrast(crop_rgb: np.ndarray) -> Tuple[float, float]:
    """
    Computes mean luminance (brightness) and standard deviation (contrast).
    Brightness: 0 (pitch black) to 255 (overexposed).
    """
    if crop_rgb.size == 0 or crop_rgb.shape[0] < 4 or crop_rgb.shape[1] < 4:
        return 0.0, 0.0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return round(brightness, 2), round(contrast, 2)


def analyze_face_quality(
    crop_rgb: np.ndarray,
    min_sharpness_threshold: float = 50.0,
    min_brightness_threshold: float = 40.0,
    max_brightness_threshold: float = 215.0,
) -> FaceQualityScore:
    """
    Legacy entrypoint returning standard FaceQualityScore.
    Preserves exact backward compatibility with all Step 2/3 callers and unit tests.
    """
    if crop_rgb.size == 0 or crop_rgb.shape[0] < 10 or crop_rgb.shape[1] < 10:
        return FaceQualityScore(
            overall_quality=0.0,
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            is_usable=False,
        )

    clarity = compute_clarity_metrics(crop_rgb, min_sharpness_threshold)
    illum = compute_illumination_metrics(crop_rgb)

    # Composite weighted quality score
    composite = (clarity.blur_score * 0.55) + (illum.illumination_score * 0.45)
    overall_quality = round(min(1.0, max(0.0, composite)), 3)

    is_usable = (
        clarity.sharpness >= min_sharpness_threshold and
        min_brightness_threshold <= illum.brightness <= max_brightness_threshold and
        illum.contrast >= 15.0
    )

    return FaceQualityScore(
        overall_quality=overall_quality,
        sharpness=clarity.sharpness,
        brightness=illum.brightness,
        contrast=illum.contrast,
        is_usable=is_usable,
    )


# -----------------------------------------------------------------------------
# 2. Comprehensive Multi-Metric Quality & Analysis Engine (Step 4)
# -----------------------------------------------------------------------------

def evaluate_quality_grade(overall_score: float) -> QualityGrade:
    """Maps composite quality score to qualitative grade."""
    if overall_score >= 0.85:
        return QualityGrade.EXCELLENT
    elif overall_score >= 0.70:
        return QualityGrade.GOOD
    elif overall_score >= 0.50:
        return QualityGrade.FAIR
    elif overall_score >= 0.30:
        return QualityGrade.POOR
    else:
        return QualityGrade.UNUSABLE


def analyze_face_comprehensive(
    face_crop_rgb: np.ndarray,
    bbox: BoundingBox,
    image_shape: Tuple[int, int],
    confidence: float = 0.95,
    landmarks: Optional[List[LandmarkPoint]] = None,
    face_id: Optional[Union[str, int]] = None,
    min_sharpness: float = 50.0,
) -> StructuredFaceAnalysis:
    """
    Executes deep multi-dimensional face analytics:
    1. Landmark metrics & symmetry
    2. 3D Head pose estimation (Yaw, Pitch, Roll, Frontality)
    3. Physical scale & resolution metrics
    4. Clarity & blur measurement
    5. Illumination & contrast diagnostics
    6. Composite explainable quality scoring & grading
    """
    img_h, img_w = image_shape

    # Validate and clip input landmarks against image boundaries [0, width] and [0, height]
    validated_landmarks: Optional[List[LandmarkPoint]] = None
    if landmarks:
        try:
            validated_landmarks = validate_and_clip_landmarks_list(
                landmarks=landmarks,
                image_width=img_w,
                image_height=img_h,
            )
        except Exception as lm_val_exc:
            logger.warning("Failed validating landmark boundaries for face %s: %s", face_id, lm_val_exc)
            validated_landmarks = None

    # 1. Landmark & Symmetry Metrics
    lm_metrics: Optional[FacialLandmarkMetrics] = None
    if validated_landmarks and len(validated_landmarks) >= 5:
        try:
            lm_metrics = compute_landmark_metrics(validated_landmarks)
        except Exception as lm_exc:
            logger.warning("Failed computing landmark metrics for face %s: %s", face_id, lm_exc)
            lm_metrics = None

    ipd = lm_metrics.interpupillary_distance_px if lm_metrics else None

    # 2. 3D Head Pose & Orientation
    pose: Optional[HeadPoseEstimation] = None
    if validated_landmarks and len(validated_landmarks) >= 5:
        try:
            pose = estimate_head_pose(validated_landmarks, image_shape)
        except Exception as pose_exc:
            logger.warning("Failed estimating head pose for face %s: %s", face_id, pose_exc)
            pose = None

    # 3. Size & Scale Metrics
    size = compute_face_size_metrics(bbox, image_shape, ipd=ipd)

    # 4. Clarity & Blur Metrics
    clarity = compute_clarity_metrics(face_crop_rgb, min_sharpness_threshold=min_sharpness)

    # 5. Illumination & Exposure Metrics
    illum = compute_illumination_metrics(face_crop_rgb)

    # 6. Composite Weighted Quality Score
    # Weights: Clarity (40%), Illumination (30%), Scale/Size (15%), Pose Frontality (15%)
    w_clarity = 0.40
    w_illum = 0.30
    w_size = 0.15
    w_pose = 0.15

    score_clarity = clarity.blur_score
    score_illum = illum.illumination_score
    score_size = 1.0 if size.is_sufficient_size else 0.40
    score_pose = pose.frontality_score if pose else 0.80

    composite_score = (
        w_clarity * score_clarity
        + w_illum * score_illum
        + w_size * score_size
        + w_pose * score_pose
    )
    overall_quality = round(min(1.0, max(0.0, composite_score)), 3)
    grade = evaluate_quality_grade(overall_quality)

    # Usability determination
    is_usable = (
        clarity.sharpness >= min_sharpness and
        40.0 <= illum.brightness <= 215.0 and
        illum.contrast >= 15.0 and
        size.is_sufficient_size and
        (pose.frontality_score >= 0.40 if pose else True)
    )

    # 7. Face Visibility Assessment (Step 10)
    visibility_assessment = calculate_face_visibility(
        face_crop_rgb=face_crop_rgb,
        bbox=bbox,
        image_shape=image_shape,
        size_metrics=size,
        clarity_metrics=clarity,
        illumination_metrics=illum,
        landmarks=validated_landmarks,
        detection_confidence=confidence,
    )

    quality_score = ComprehensiveFaceQuality(
        overall_quality=overall_quality,
        sharpness=clarity.sharpness,
        brightness=illum.brightness,
        contrast=illum.contrast,
        is_usable=is_usable,
        grade=grade,
        visibility=visibility_assessment.visibility,
        visibility_score=visibility_assessment.visibility_score,
        clarity=clarity,
        illumination=illum,
        size=size,
        pose=pose,
    )

    return StructuredFaceAnalysis(
        face_id=face_id,
        bbox=bbox,
        confidence=round(confidence, 4),
        visibility=visibility_assessment.visibility,
        visibility_assessment=visibility_assessment,
        landmarks=validated_landmarks or [],
        landmark_metrics=lm_metrics,
        pose=pose,
        size=size,
        illumination=illum,
        clarity=clarity,
        quality=quality_score,
    )



class FaceAnalyzer:
    """
    High-level orchestrator that applies comprehensive face analysis
    to detected faces across an image.
    Never crashes the request if one face has corrupted or missing landmarks/geometry.
    """

    def __init__(self, min_sharpness: float = 50.0):
        self.min_sharpness = min_sharpness

    def analyze_detections(
        self,
        image_rgb: np.ndarray,
        detections: List[Any],
    ) -> List[StructuredFaceAnalysis]:
        """
        Analyzes a list of FaceDetection items on the input RGB image.
        Applies per-face fault isolation so that an anomaly in one face never crashes the entire batch.
        """
        img_h, img_w = image_rgb.shape[:2]
        analyses: List[StructuredFaceAnalysis] = []

        for d in detections:
            try:
                x1 = max(0, min(img_w, int(d.bbox.x1)))
                y1 = max(0, min(img_h, int(d.bbox.y1)))
                x2 = max(0, min(img_w, int(d.bbox.x2)))
                y2 = max(0, min(img_h, int(d.bbox.y2)))

                if x2 > x1 and y2 > y1:
                    crop = image_rgb[y1:y2, x1:x2]
                else:
                    crop = np.empty((0, 0, 3), dtype=np.uint8)

                analysis = analyze_face_comprehensive(
                    face_crop_rgb=crop,
                    bbox=d.bbox,
                    image_shape=(img_h, img_w),
                    confidence=d.confidence,
                    landmarks=d.landmarks,
                    face_id=d.face_id,
                    min_sharpness=self.min_sharpness,
                )
                analyses.append(analysis)
            except Exception as exc:
                logger.error("Error analyzing face detection %s: %s", getattr(d, 'face_id', 'unknown'), exc, exc_info=True)
                try:
                    safe_crop = np.empty((0, 0, 3), dtype=np.uint8)
                    fallback_analysis = StructuredFaceAnalysis(
                        face_id=getattr(d, "face_id", None),
                        bbox=d.bbox,
                        confidence=round(float(getattr(d, "confidence", 0.0)), 4),
                        landmarks=[],
                        landmark_metrics=None,
                        pose=None,
                        size=compute_face_size_metrics(d.bbox, (img_h, img_w)),
                        illumination=compute_illumination_metrics(safe_crop),
                        clarity=compute_clarity_metrics(safe_crop, min_sharpness_threshold=self.min_sharpness),
                        quality=ComprehensiveFaceQuality(
                            overall_quality=0.0,
                            sharpness=0.0,
                            brightness=0.0,
                            contrast=0.0,
                            is_usable=False,
                            grade=QualityGrade.UNUSABLE,
                        ),
                    )
                    analyses.append(fallback_analysis)
                except Exception:
                    pass

        return analyses


# Alias for explicit clarity and specification alignment
FaceQualityAnalyzer = FaceAnalyzer
