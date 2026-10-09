"""
DeepVision AI — Comprehensive Face Quality Analytics & Scoring Module
Combines sharpness, illumination, resolution, and 3D head pose into explainable quality metrics.
Maintains full backward compatibility with legacy FaceQualityScore callers.
"""

import time
from typing import Tuple, Optional, List, Union, Dict, Any
import numpy as np
import cv2


from app.core.config import settings
from app.schemas.ai import (
    BoundingBox,
    LandmarkPoint,
    FaceQualityScore,
)
from app.ai.analysis.schemas import (
    ComprehensiveFaceQuality,
    QualityGrade,
    QualityClassification,
    StructuredFaceAnalysis,
    FacialLandmarkMetrics,
    HeadPoseEstimation,
    FaceSizeMetrics,
    IlluminationMetrics,
    ClarityMetrics,
    BlurStatus,
    BrightnessStatus,
    FaceVisibility,
    FaceVisibilityAssessment,
    FaceQualityFactors,
)

from app.ai.analysis.landmarks import (
    compute_landmark_metrics,
    validate_and_clip_landmarks_list,
)
from app.ai.analysis.pose import estimate_head_pose, get_pose_estimation_status
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
    """Maps composite quality score (0.0 to 1.0 or 0 to 100) to qualitative grade."""
    # Support both normalized (0.0 - 1.0) and 0 - 100 scales
    score = overall_score if overall_score <= 1.0 else (overall_score / 100.0)
    if score >= 0.85:
        return QualityGrade.EXCELLENT
    elif score >= 0.70:
        return QualityGrade.GOOD
    elif score >= 0.50:
        return QualityGrade.FAIR
    elif score >= 0.30:
        return QualityGrade.POOR
    else:
        return QualityGrade.UNUSABLE


def classify_face_quality(
    quality_score: float,
    excellent_threshold: Optional[float] = None,
    good_threshold: Optional[float] = None,
    acceptable_threshold: Optional[float] = None,
) -> QualityClassification:
    """
    Overall qualitative face quality classification (Step 13).
    Categorizes the measurable face quality score into:
        - 'excellent'
        - 'good'
        - 'acceptable'
        - 'poor'

    Threshold assumptions (configurable):
        - excellent: score >= excellent_threshold (default 80.0)
        - good: score >= good_threshold (default 65.0)
        - acceptable: score >= acceptable_threshold (default 45.0)
        - poor: score < acceptable_threshold (< 45.0)

    Thresholds are fully configurable via function arguments or application settings:
        - settings.AI_QUALITY_EXCELLENT_THRESHOLD
        - settings.AI_QUALITY_GOOD_THRESHOLD
        - settings.AI_QUALITY_ACCEPTABLE_THRESHOLD

    Supports both 0-100 scale (default) and normalized 0.0-1.0 scale transparently.
    """
    # Scale normalized scores (0.0 < score <= 1.0) to 0 - 100
    score_100 = float(quality_score) * 100.0 if 0.0 < quality_score <= 1.0 else float(quality_score)
    score_100 = max(0.0, min(100.0, score_100))

    th_excellent = (
        excellent_threshold
        if excellent_threshold is not None
        else getattr(settings, "AI_QUALITY_EXCELLENT_THRESHOLD", 80.0)
    )
    th_good = (
        good_threshold
        if good_threshold is not None
        else getattr(settings, "AI_QUALITY_GOOD_THRESHOLD", 65.0)
    )
    th_acceptable = (
        acceptable_threshold
        if acceptable_threshold is not None
        else getattr(settings, "FACE_QUALITY_THRESHOLD", getattr(settings, "AI_QUALITY_ACCEPTABLE_THRESHOLD", 45.0))
    )

    # Scale thresholds if provided in normalized 0.0 - 1.0 format
    if 0.0 < th_excellent <= 1.0:
        th_excellent *= 100.0
    if 0.0 < th_good <= 1.0:
        th_good *= 100.0
    if 0.0 < th_acceptable <= 1.0:
        th_acceptable *= 100.0

    if score_100 >= th_excellent:
        return QualityClassification.EXCELLENT
    elif score_100 >= th_good:
        return QualityClassification.GOOD
    elif score_100 >= th_acceptable:
        return QualityClassification.ACCEPTABLE
    else:
        return QualityClassification.POOR


classify_quality = classify_face_quality


def calculate_face_quality_score(
    face_crop_rgb: Optional[np.ndarray] = None,
    bbox: Optional[BoundingBox] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    clarity_metrics: Optional[ClarityMetrics] = None,
    illumination_metrics: Optional[IlluminationMetrics] = None,
    size_metrics: Optional[FaceSizeMetrics] = None,
    landmark_metrics: Optional[FacialLandmarkMetrics] = None,
    landmarks: Optional[Union[List[Any], Dict[str, Any]]] = None,
    detection_confidence: float = 0.95,
) -> Tuple[float, FaceQualityFactors, QualityGrade]:
    """
    Transparent Face Quality Scoring Engine (Step 11).
    Calculates an explainable 0 - 100 quality score based on measurable physical/geometric optics.

    Factors combined:
        1. Sharpness (25%): Laplacian variance edge-frequency definition.
        2. Brightness (20%): ITU-R BT.601 photopic luminance exposure against sensor dynamic limits.
        3. Contrast (15%): Standard deviation of luminance dynamic range.
        4. Face Size (20%): Pixel resolution & scale sufficiency relative to biometric minimums.
        5. Landmark Quality (20%): Structural availability, bilateral facial symmetry, and localization certainty.

    Formula:
        Score_base = 0.25 * Sharpness + 0.20 * Brightness + 0.15 * Contrast + 0.20 * Size + 0.20 * Landmark

    Bottleneck Gating:
        Critical defects (e.g. pitch-black darkness, severe motion blur, or sub-32px scale)
        cap the maximum composite score to avoid artificially elevating unusable faces.

    Returns:
        Tuple of (quality_score_0_to_100, FaceQualityFactors, QualityGrade).
    """
    # 1. Sharpness Factor (0 - 100)
    if clarity_metrics is None and face_crop_rgb is not None and face_crop_rgb.size > 0:
        try:
            clarity_metrics = compute_clarity_metrics(face_crop_rgb)
        except Exception:
            clarity_metrics = None

    if clarity_metrics is not None:
        f_sharpness = float(clarity_metrics.blur_score) * 100.0
        if clarity_metrics.blur_status == BlurStatus.BLURRY:
            f_sharpness = min(f_sharpness, 35.0)
        elif clarity_metrics.blur_status == BlurStatus.SLIGHTLY_BLURRED:
            f_sharpness = min(74.0, max(40.0, f_sharpness))
    else:
        f_sharpness = 70.0
    f_sharpness = round(min(100.0, max(0.0, f_sharpness)), 1)

    # 2. Brightness Factor (0 - 100)
    if illumination_metrics is None and face_crop_rgb is not None and face_crop_rgb.size > 0:
        try:
            illumination_metrics = compute_illumination_metrics(face_crop_rgb)
        except Exception:
            illumination_metrics = None

    if illumination_metrics is not None:
        f_brightness = float(illumination_metrics.brightness_score) * 100.0
        if illumination_metrics.brightness_status in (BrightnessStatus.TOO_DARK, BrightnessStatus.TOO_BRIGHT):
            f_brightness = min(f_brightness, 50.0)
    else:
        f_brightness = 70.0
    f_brightness = round(min(100.0, max(0.0, f_brightness)), 1)

    # 3. Contrast Factor (0 - 100)
    if illumination_metrics is not None:
        std_c = float(illumination_metrics.contrast)
    elif face_crop_rgb is not None and face_crop_rgb.size > 0:
        std_c = float(compute_brightness_and_contrast(face_crop_rgb)[1])
    else:
        std_c = 35.0

    if std_c < 15.0:
        f_contrast = max(0.0, (std_c / 15.0) * 50.0)
    elif std_c < 40.0:
        f_contrast = 50.0 + ((std_c - 15.0) / 25.0) * 50.0
    elif std_c <= 65.0:
        f_contrast = 100.0
    else:
        f_contrast = max(40.0, 100.0 - ((std_c - 65.0) / 35.0) * 30.0)
    f_contrast = round(min(100.0, max(0.0, f_contrast)), 1)

    # 4. Face Size Factor (0 - 100)
    if size_metrics is None and bbox is not None and image_shape is not None:
        try:
            size_metrics = compute_face_size_metrics(bbox, image_shape)
        except Exception:
            size_metrics = None

    if size_metrics is not None:
        min_dim = min(size_metrics.face_width, size_metrics.face_height)
        if min_dim <= 0:
            f_size = 0.0
        elif min_dim < 32.0:
            f_size = max(5.0, (min_dim / 32.0) * 40.0)
        elif min_dim < 80.0:
            f_size = 40.0 + ((min_dim - 32.0) / 48.0) * 35.0
        elif min_dim < 200.0:
            f_size = 75.0 + ((min_dim - 80.0) / 120.0) * 20.0
        else:
            f_size = min(100.0, 95.0 + min(5.0, (min_dim - 200.0) / 400.0 * 5.0))
    else:
        f_size = 70.0
    f_size = round(min(100.0, max(0.0, f_size)), 1)

    # 5. Landmark Quality Factor (0 - 100)
    if landmarks is None or len(landmarks) == 0:
        a_lm = 0.20
    else:
        n_lm = len(landmarks)
        a_lm = 1.0 if n_lm >= 5 else max(0.20, n_lm / 5.0)

    if landmark_metrics and landmark_metrics.symmetry_ratio is not None:
        s_sym = float(landmark_metrics.symmetry_ratio)
    elif landmarks and len(landmarks) >= 5:
        s_sym = 0.85
    else:
        s_sym = 0.60

    confidences: List[float] = []
    if landmarks:
        items = landmarks.values() if isinstance(landmarks, dict) else landmarks
        for pt in items:
            if isinstance(pt, dict) and pt.get("confidence") is not None:
                confidences.append(float(pt["confidence"]))
            elif hasattr(pt, "confidence") and pt.confidence is not None:
                confidences.append(float(pt.confidence))

    if confidences:
        c_lm = float(np.mean(confidences))
    elif landmarks and len(landmarks) > 0:
        c_lm = min(1.0, max(0.0, float(detection_confidence)))
    else:
        c_lm = min(0.50, max(0.0, float(detection_confidence) * 0.50))

    f_landmark = round(100.0 * (0.40 * a_lm + 0.35 * s_sym + 0.25 * c_lm), 1)
    f_landmark = round(min(100.0, max(0.0, f_landmark)), 1)

    # Weighted baseline combination (weights sum to 1.0)
    baseline = (
        0.25 * f_sharpness
        + 0.20 * f_brightness
        + 0.15 * f_contrast
        + 0.20 * f_size
        + 0.20 * f_landmark
    )

    # Critical bottleneck gating
    critical_min = min(f_sharpness, f_brightness, f_size)
    if critical_min < 25.0:
        composite = min(baseline, critical_min * 1.80)
    else:
        composite = baseline

    quality_score = round(min(100.0, max(0.0, composite)), 1)
    overall_normalized = round(quality_score / 100.0, 3)
    grade = evaluate_quality_grade(overall_normalized)

    factors = FaceQualityFactors(
        sharpness=f_sharpness,
        brightness=f_brightness,
        contrast=f_contrast,
        size=f_size,
        landmark=f_landmark,
    )

    return quality_score, factors, grade


def analyze_face_comprehensive(
    face_crop_rgb: np.ndarray,
    bbox: BoundingBox,
    image_shape: Tuple[int, int],
    confidence: float = 0.95,
    landmarks: Optional[List[LandmarkPoint]] = None,
    face_id: Optional[Union[str, int]] = None,
    min_sharpness: Optional[float] = None,
    stage_timing: Optional[Dict[str, float]] = None,
    enable_landmarks: Optional[bool] = None,
    enable_quality_analysis: Optional[bool] = None,
    enable_pose_analysis: Optional[bool] = None,
    brightness_min: Optional[float] = None,
    brightness_max: Optional[float] = None,
    quality_threshold: Optional[float] = None,
) -> StructuredFaceAnalysis:
    """
    Executes deep multi-dimensional face analytics:
    1. Landmark metrics & symmetry
    2. 3D Head pose estimation (Yaw, Pitch, Roll, Frontality)
    3. Physical scale & resolution metrics
    4. Clarity & blur measurement
    5. Illumination & contrast diagnostics
    6. Composite transparent quality scoring (0-100) & grading
    7. Face visibility assessment
    """
    img_h, img_w = image_shape

    eff_min_sharpness = min_sharpness if min_sharpness is not None else getattr(settings, "FACE_BLUR_THRESHOLD", 50.0)
    eff_enable_landmarks = enable_landmarks if enable_landmarks is not None else getattr(settings, "AI_ENABLE_LANDMARKS", True)
    eff_enable_quality = enable_quality_analysis if enable_quality_analysis is not None else getattr(settings, "AI_ENABLE_QUALITY_ANALYSIS", True)
    eff_enable_pose = enable_pose_analysis if enable_pose_analysis is not None else getattr(settings, "AI_ENABLE_POSE_ANALYSIS", True)
    eff_b_min = brightness_min if brightness_min is not None else getattr(settings, "FACE_BRIGHTNESS_MIN", 60.0)
    eff_b_max = brightness_max if brightness_max is not None else getattr(settings, "FACE_BRIGHTNESS_MAX", 200.0)
    eff_q_thresh = quality_threshold if quality_threshold is not None else getattr(settings, "FACE_QUALITY_THRESHOLD", 45.0)

    # 1. Landmark & Symmetry Metrics
    t_lm = time.perf_counter() if stage_timing is not None else 0.0
    validated_landmarks: Optional[List[LandmarkPoint]] = None
    if eff_enable_landmarks and landmarks:
        try:
            validated_landmarks = validate_and_clip_landmarks_list(
                landmarks=landmarks,
                image_width=img_w,
                image_height=img_h,
            )
        except Exception as lm_val_exc:
            logger.warning("Failed validating landmark boundaries for face %s: %s", face_id, lm_val_exc)
            validated_landmarks = None

    lm_metrics: Optional[FacialLandmarkMetrics] = None
    if eff_enable_landmarks and validated_landmarks and len(validated_landmarks) >= 5:
        try:
            lm_metrics = compute_landmark_metrics(validated_landmarks)
        except Exception as lm_exc:
            logger.warning("Failed computing landmark metrics for face %s: %s", face_id, lm_exc)
            lm_metrics = None

    if stage_timing is not None and eff_enable_landmarks:
        stage_timing["landmark"] += (time.perf_counter() - t_lm)

    ipd = lm_metrics.interpupillary_distance_px if lm_metrics else None

    # 2. 3D Head Pose & Orientation
    t_pose = time.perf_counter() if stage_timing is not None else 0.0
    pose: Optional[HeadPoseEstimation] = None
    if eff_enable_pose and eff_enable_landmarks and validated_landmarks and len(validated_landmarks) >= 5:
        try:
            pose, failure_reason = get_pose_estimation_status(validated_landmarks, image_shape)
            if pose is None and failure_reason:
                logger.debug("Head pose estimation unavailable for face %s: %s", face_id, failure_reason)
        except Exception as pose_exc:
            logger.warning("Failed estimating head pose for face %s: %s", face_id, pose_exc)
            pose = None

    if stage_timing is not None and eff_enable_pose and eff_enable_landmarks:
        stage_timing["pose"] += (time.perf_counter() - t_pose)

    # 3. Quality, Clarity & Illumination Analytics
    t_qual = time.perf_counter() if stage_timing is not None else 0.0
    try:
        size = compute_face_size_metrics(bbox, image_shape, ipd=ipd)
    except Exception as sz_exc:
        logger.warning("Failed computing size metrics for face %s: %s", face_id, sz_exc)
        size = compute_face_size_metrics(bbox, (max(1, img_h), max(1, img_w)))

    try:
        clarity = compute_clarity_metrics(face_crop_rgb, min_sharpness_threshold=eff_min_sharpness)
    except Exception as cl_exc:
        logger.warning("Failed computing clarity metrics for face %s: %s", face_id, cl_exc)
        clarity = ClarityMetrics(
            sharpness=0.0,
            blur_score=0.0,
            blur_status=BlurStatus.BLURRY,
        )

    try:
        illum = compute_illumination_metrics(face_crop_rgb, min_acceptable=eff_b_min, max_acceptable=eff_b_max)
    except Exception as il_exc:
        logger.warning("Failed computing illumination metrics for face %s: %s", face_id, il_exc)
        illum = IlluminationMetrics(
            brightness=0.0,
            contrast=0.0,
            brightness_score=0.0,
            brightness_status=BrightnessStatus.TOO_DARK,
        )

    # 4. Composite Transparent Quality Scoring (0 - 100) (Step 11)
    try:
        quality_score_100, quality_factors, grade = calculate_face_quality_score(
            face_crop_rgb=face_crop_rgb,
            bbox=bbox,
            image_shape=image_shape,
            clarity_metrics=clarity,
            illumination_metrics=illum,
            size_metrics=size,
            landmark_metrics=lm_metrics,
            landmarks=validated_landmarks,
            detection_confidence=confidence,
        )
    except Exception as q_exc:
        logger.warning("Failed calculating face quality score for face %s: %s", face_id, q_exc)
        quality_score_100 = 0.0
        quality_factors = FaceQualityFactors(sharpness=0.0, brightness=0.0, contrast=0.0, size=0.0, landmark=0.0)
        grade = QualityGrade.UNUSABLE

    overall_quality = round(quality_score_100 / 100.0, 3)

    # Usability determination
    is_usable = (
        clarity.sharpness >= eff_min_sharpness and
        eff_b_min <= illum.brightness <= eff_b_max and
        illum.contrast >= 15.0 and
        size.is_sufficient_size and
        (pose.frontality_score >= 0.40 if pose else True) and
        quality_score_100 >= eff_q_thresh
    )

    # 5. Face Visibility Assessment (Step 10)
    try:
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
    except Exception as vis_exc:
        logger.warning("Failed calculating visibility assessment for face %s: %s", face_id, vis_exc)
        visibility_assessment = FaceVisibilityAssessment(
            visibility=FaceVisibility.POOR,
            visibility_score=0.0,
            size_factor=0.0,
            blur_factor=0.0,
            brightness_factor=0.0,
            landmark_availability_factor=0.0,
            landmark_confidence_factor=0.0,
            details="Visibility evaluation failed",
        )

    # 6. Face Quality Classification (Step 13)
    try:
        classification = classify_face_quality(quality_score_100, acceptable_threshold=eff_q_thresh)
    except Exception:
        classification = QualityClassification.POOR

    quality_score = ComprehensiveFaceQuality(
        quality_score=quality_score_100,
        overall_quality=overall_quality,
        classification=classification,
        factors=quality_factors,
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
    if stage_timing is not None:
        stage_timing["quality"] += (time.perf_counter() - t_qual)

    return StructuredFaceAnalysis(
        face_id=face_id,
        bbox=bbox,
        confidence=round(confidence, 4),
        quality_score=quality_score_100,
        classification=classification,
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

    def __init__(
        self,
        min_sharpness: Optional[float] = None,
        enable_landmarks: Optional[bool] = None,
        enable_quality_analysis: Optional[bool] = None,
        enable_pose_analysis: Optional[bool] = None,
        brightness_min: Optional[float] = None,
        brightness_max: Optional[float] = None,
        quality_threshold: Optional[float] = None,
    ):
        self.min_sharpness = min_sharpness if min_sharpness is not None else getattr(settings, "FACE_BLUR_THRESHOLD", 50.0)
        self.enable_landmarks = enable_landmarks if enable_landmarks is not None else getattr(settings, "AI_ENABLE_LANDMARKS", True)
        self.enable_quality_analysis = enable_quality_analysis if enable_quality_analysis is not None else getattr(settings, "AI_ENABLE_QUALITY_ANALYSIS", True)
        self.enable_pose_analysis = enable_pose_analysis if enable_pose_analysis is not None else getattr(settings, "AI_ENABLE_POSE_ANALYSIS", True)
        self.brightness_min = brightness_min if brightness_min is not None else getattr(settings, "FACE_BRIGHTNESS_MIN", 60.0)
        self.brightness_max = brightness_max if brightness_max is not None else getattr(settings, "FACE_BRIGHTNESS_MAX", 200.0)
        self.quality_threshold = quality_threshold if quality_threshold is not None else getattr(settings, "FACE_QUALITY_THRESHOLD", 45.0)

    def analyze_detections(
        self,
        image_rgb: np.ndarray,
        detections: List[Any],
        stage_timing: Optional[Dict[str, float]] = None,
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
                    stage_timing=stage_timing,
                    enable_landmarks=self.enable_landmarks,
                    enable_quality_analysis=self.enable_quality_analysis,
                    enable_pose_analysis=self.enable_pose_analysis,
                    brightness_min=self.brightness_min,
                    brightness_max=self.brightness_max,
                    quality_threshold=self.quality_threshold,
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
                        classification=QualityClassification.POOR,
                        quality=ComprehensiveFaceQuality(
                            quality_score=0.0,
                            overall_quality=0.0,
                            classification=QualityClassification.POOR,
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
