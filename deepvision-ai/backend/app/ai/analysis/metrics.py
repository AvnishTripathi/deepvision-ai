"""
DeepVision AI — Face Quality Metrics Module
Calculates physical scale, blur/clarity measurements, and illumination/exposure diagnostics.
"""

from typing import Tuple, Optional, List, Dict, Union, Any
import numpy as np
import cv2

from app.core.config import settings
from app.schemas.ai import BoundingBox, LandmarkPoint, LandmarkCoordinate
from app.ai.analysis.schemas import (
    FaceSizeMetrics,
    SizeCategory,
    ClarityMetrics,
    BlurCategory,
    BlurStatus,
    IlluminationMetrics,
    ExposureCategory,
    BrightnessStatus,
    FaceVisibility,
    FaceVisibilityAssessment,
)



# -----------------------------------------------------------------------------
# 1. Face Size & Scale Analysis
# -----------------------------------------------------------------------------

def compute_face_size_metrics(
    bbox: BoundingBox,
    image_shape: Tuple[int, int],
    ipd: Optional[float] = None,
    min_face_dimension: float = 32.0,
) -> FaceSizeMetrics:
    """
    Evaluates scale, resolution, and pixel density of the face bounding box.

    Args:
        bbox: Absolute bounding box in native image pixels.
        image_shape: (height, width) of the original image canvas.
        ipd: Optional inter-pupillary distance in pixels.
        min_face_dimension: Minimum recommended pixel width/height for biometric usability.
    """
    img_h, img_w = image_shape
    w = max(0.0, float(bbox.x2 - bbox.x1))
    h = max(0.0, float(bbox.y2 - bbox.y1))
    area = w * h
    canvas_area = float(max(1, img_h * img_w))
    relative_area = min(1.0, area / canvas_area)
    aspect_ratio = round(w / h, 3) if h > 0.0 else 1.0

    # Scale classification
    max_dim = max(w, h)
    if max_dim < 32.0:
        category = SizeCategory.TINY
    elif max_dim < 80.0:
        category = SizeCategory.SMALL
    elif max_dim < 200.0:
        category = SizeCategory.MEDIUM
    else:
        category = SizeCategory.LARGE

    # Sufficiency check (minimum width & height >= 32px or IPD >= 20px)
    is_sufficient = (w >= min_face_dimension and h >= min_face_dimension)
    if ipd is not None and ipd < 20.0:
        is_sufficient = False

    return FaceSizeMetrics(
        face_width=round(w, 2),
        face_height=round(h, 2),
        face_area=round(area, 2),
        face_area_ratio=round(relative_area, 4),
        width_px=round(w, 2),
        height_px=round(h, 2),
        aspect_ratio=aspect_ratio,
        area_px=round(area, 2),
        relative_area=round(relative_area, 4),
        size_category=category,
        is_sufficient_size=is_sufficient,
    )


# Explicit task alias function
calculate_face_size = compute_face_size_metrics


# -----------------------------------------------------------------------------
# 2. Clarity & Blur Detection
# -----------------------------------------------------------------------------

def estimate_face_blur(
    face_crop_rgb: np.ndarray,
    sharp_threshold: Optional[float] = None,
    slightly_blurred_threshold: Optional[float] = None,
) -> ClarityMetrics:
    """
    Measures facial sharpness and estimates motion/defocus blur using Laplacian variance.

    Mathematical formulation:
        L(x, y) = ∇² I(x, y) = ∂²I/∂x² + ∂²I/∂y²
        sharpness = Var(L) = (1/N) * Σ (L(x, y) - μ_L)²

    Threshold assumptions:
        - sharpness >= sharp_threshold (default 100.0): In-focus high-frequency edge definition ('sharp').
        - slightly_blurred_threshold <= sharpness < sharp_threshold (default 50.0 to 100.0): Minor defocus/motion ('slightly_blurred').
        - sharpness < slightly_blurred_threshold: Lacks critical edge definition ('blurry').

    CRITICAL NOTE ON INTERPRETATION:
        Laplacian variance is an edge-frequency measurement, NOT a universal quality measurement.
        Smooth makeup, flat lighting, or naturally soft youth skin reduce Laplacian variance without
        loss of biometric fidelity, whereas ISO noise or sensor grain can artificially inflate sharpness.

    Args:
        face_crop_rgb: Cropped face ROI in RGB numpy array format.
        sharp_threshold: Optional threshold for 'sharp' status (defaults to settings.AI_BLUR_SHARP_THRESHOLD or 100.0).
        slightly_blurred_threshold: Optional threshold for 'slightly_blurred' status (defaults to settings.AI_BLUR_SLIGHTLY_BLURRED_THRESHOLD or 50.0).

    Returns:
        ClarityMetrics with blur_score, blur_status ('sharp', 'slightly_blurred', 'blurry'), and raw sharpness.
    """
    effective_sharp_thresh = (
        sharp_threshold
        if sharp_threshold is not None
        else getattr(settings, "AI_BLUR_SHARP_THRESHOLD", 100.0)
    )
    effective_slight_thresh = (
        slightly_blurred_threshold
        if slightly_blurred_threshold is not None
        else getattr(settings, "FACE_BLUR_THRESHOLD", getattr(settings, "AI_BLUR_SLIGHTLY_BLURRED_THRESHOLD", 50.0))
    )

    if face_crop_rgb.size == 0 or face_crop_rgb.shape[0] < 4 or face_crop_rgb.shape[1] < 4:
        return ClarityMetrics(
            sharpness=0.0,
            blur_score=0.0,
            blur_status=BlurStatus.BLURRY,
            blur_category=BlurCategory.VERY_BLURRED,
        )

    gray = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    sharpness = round(variance, 2)

    # Normalize sharpness to a continuous score [0.0, 1.0]
    blur_score = min(1.0, max(0.0, (sharpness - 15.0) / 185.0))
    blur_score = round(blur_score, 3)

    if sharpness >= effective_sharp_thresh:
        status = BlurStatus.SHARP
        category = BlurCategory.SHARP
    elif sharpness >= effective_slight_thresh:
        status = BlurStatus.SLIGHTLY_BLURRED
        category = BlurCategory.SLIGHTLY_BLURRED
    else:
        status = BlurStatus.BLURRY
        category = BlurCategory.BLURRED if sharpness >= 20.0 else BlurCategory.VERY_BLURRED

    return ClarityMetrics(
        sharpness=sharpness,
        blur_score=blur_score,
        blur_status=status,
        blur_category=category,
    )


def compute_clarity_metrics(
    face_crop_rgb: np.ndarray,
    min_sharpness_threshold: float = 50.0,
    sharp_threshold: float = 100.0,
) -> ClarityMetrics:
    """Backward-compatible wrapper for estimate_face_blur."""
    return estimate_face_blur(
        face_crop_rgb=face_crop_rgb,
        sharp_threshold=sharp_threshold,
        slightly_blurred_threshold=min_sharpness_threshold,
    )


# -----------------------------------------------------------------------------
# 3. Illumination & Exposure Analysis
# -----------------------------------------------------------------------------

def calculate_face_brightness(
    face_crop_rgb: np.ndarray,
    min_acceptable: Optional[float] = None,
    max_acceptable: Optional[float] = None,
) -> IlluminationMetrics:
    """
    Calculates brightness and exposure information for each face crop.

    Mathematical formulation (ITU-R BT.601 standard):
        Y(x, y) = 0.299 * R + 0.587 * G + 0.114 * B
        mean brightness = (1/N) * Σ Y(x, y)
        contrast = sqrt((1/N) * Σ (Y(x, y) - brightness)²)

    Threshold assumptions (configurable):
        - brightness < min_acceptable (default 60.0): 'too_dark' (underexposed shadow clipping).
        - min_acceptable <= brightness <= max_acceptable (default 60.0 to 200.0): 'acceptable' (balanced facial luminance).
        - brightness > max_acceptable (default 200.0): 'too_bright' (overexposed highlight clipping).

    Score calculation (brightness_score in [0.0, 1.0]):
        - When too dark: scaled linearly from 0.0 at brightness=0 to 0.70 at min_acceptable.
        - When acceptable: scaled continuously between [0.70, 1.0], centered near optimal midpoint (~130.0).
        - When too bright: scaled linearly from 0.70 at max_acceptable to 0.0 at brightness=255.

    Args:
        face_crop_rgb: Cropped face ROI in RGB format.
        min_acceptable: Optional cutoff below which face is 'too_dark' (defaults to settings.AI_BRIGHTNESS_MIN_ACCEPTABLE or 60.0).
        max_acceptable: Optional cutoff above which face is 'too_bright' (defaults to settings.AI_BRIGHTNESS_MAX_ACCEPTABLE or 200.0).

    Returns:
        IlluminationMetrics containing brightness, contrast, brightness_score, and brightness_status.
    """
    effective_min = (
        min_acceptable
        if min_acceptable is not None
        else getattr(settings, "FACE_BRIGHTNESS_MIN", getattr(settings, "AI_BRIGHTNESS_MIN_ACCEPTABLE", 60.0))
    )
    effective_max = (
        max_acceptable
        if max_acceptable is not None
        else getattr(settings, "FACE_BRIGHTNESS_MAX", getattr(settings, "AI_BRIGHTNESS_MAX_ACCEPTABLE", 200.0))
    )

    if face_crop_rgb.size == 0 or face_crop_rgb.shape[0] < 4 or face_crop_rgb.shape[1] < 4:
        return IlluminationMetrics(
            brightness=0.0,
            contrast=0.0,
            brightness_score=0.0,
            brightness_status=BrightnessStatus.TOO_DARK,
            exposure_category=ExposureCategory.UNDEREXPOSED,
            illumination_score=0.0,
        )

    gray = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    brightness = round(mean_val, 2)
    contrast = round(std_val, 2)

    # Status classification against configurable thresholds
    if brightness < effective_min:
        status = BrightnessStatus.TOO_DARK
        exp_category = ExposureCategory.UNDEREXPOSED
        score_brightness = max(0.0, (brightness / effective_min) * 0.70) if effective_min > 0 else 0.0
    elif brightness > effective_max:
        status = BrightnessStatus.TOO_BRIGHT
        exp_category = ExposureCategory.OVEREXPOSED
        denom = 255.0 - effective_max
        score_brightness = max(0.0, ((255.0 - brightness) / denom) * 0.70) if denom > 0 else 0.0
    else:
        status = BrightnessStatus.ACCEPTABLE
        exp_category = ExposureCategory.OPTIMAL
        # Smooth trapezoid/midpoint centered at (min + max) / 2
        midpoint = (effective_min + effective_max) / 2.0
        half_range = max(1.0, (effective_max - effective_min) / 2.0)
        deviation = abs(brightness - midpoint) / half_range
        score_brightness = max(0.70, 1.0 - 0.30 * deviation)

    score_brightness = round(min(1.0, max(0.0, score_brightness)), 3)

    # Composite illumination score incorporates contrast
    score_contrast = min(1.0, max(0.0, contrast / 50.0))
    illumination_score = round(score_brightness * 0.75 + score_contrast * 0.25, 3)

    # Contrast fine-tuning on exposure category
    if status == BrightnessStatus.ACCEPTABLE:
        if contrast < 16.0:
            exp_category = ExposureCategory.LOW_CONTRAST
        elif contrast > 75.0:
            exp_category = ExposureCategory.HARSH_CONTRAST

    return IlluminationMetrics(
        brightness=brightness,
        contrast=contrast,
        brightness_score=score_brightness,
        brightness_status=status,
        exposure_category=exp_category,
        illumination_score=illumination_score,
    )


def compute_illumination_metrics(
    face_crop_rgb: np.ndarray,
    min_acceptable: Optional[float] = None,
    max_acceptable: Optional[float] = None,
) -> IlluminationMetrics:
    """Backward-compatible wrapper for calculate_face_brightness."""
    return calculate_face_brightness(
        face_crop_rgb=face_crop_rgb,
        min_acceptable=min_acceptable,
        max_acceptable=max_acceptable,
    )


# -----------------------------------------------------------------------------
# 4. Face Visibility Assessment (Step 10)
# -----------------------------------------------------------------------------

def calculate_face_visibility(
    face_crop_rgb: Optional[np.ndarray] = None,
    bbox: Optional[BoundingBox] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    size_metrics: Optional[FaceSizeMetrics] = None,
    clarity_metrics: Optional[ClarityMetrics] = None,
    illumination_metrics: Optional[IlluminationMetrics] = None,
    landmarks: Optional[Union[List[Any], Dict[str, Any]]] = None,
    detection_confidence: float = 0.95,
    excellent_threshold: Optional[float] = None,
    good_threshold: Optional[float] = None,
    fair_threshold: Optional[float] = None,
) -> FaceVisibilityAssessment:
    """
    Evaluates overall face visibility using an explainable engineering heuristic (Step 10).

    Synthesizes five distinct visual dimensions:
        1. Face Size: Pixel scale and resolution sufficiency relative to biometric minimums.
        2. Blur: High-frequency spatial clarity based on Laplacian variance.
        3. Brightness: Photopic luminance exposure based on ITU-R BT.601 standards.
        4. Landmark Availability: Ratio of detectable key facial landmarks (5 canonical points).
        5. Landmark Confidence: Explicit landmark confidence or fallback detection certainty.

    Threshold mapping (configurable):
        - visibility_score >= excellent_threshold (default 0.82) -> 'excellent'
        - visibility_score >= good_threshold (default 0.65) -> 'good'
        - visibility_score >= fair_threshold (default 0.45) -> 'fair'
        - otherwise -> 'poor'

    A critical bottleneck rule prevents faces with severely crippled individual factors
    (e.g., pitch-black illumination, complete motion blur, or microscopic resolution)
    from receiving high ratings.

    Note:
        This is an engineering heuristic designed for production AI processing pipelines,
        not an absolute or universal ground-truth visibility measurement.

    Args:
        face_crop_rgb: Cropped face ROI in RGB format (used to compute blur/brightness if not provided).
        bbox: Absolute bounding box in native image pixels (used to compute size if not provided).
        image_shape: (height, width) of the original image canvas.
        size_metrics: Optional precomputed FaceSizeMetrics.
        clarity_metrics: Optional precomputed ClarityMetrics.
        illumination_metrics: Optional precomputed IlluminationMetrics.
        landmarks: Optional list or dictionary of detected facial landmarks.
        detection_confidence: Face detector confidence score in [0.0, 1.0].
        excellent_threshold: Cutoff score for 'excellent' visibility (defaults to 0.82).
        good_threshold: Cutoff score for 'good' visibility (defaults to 0.65).
        fair_threshold: Cutoff score for 'fair' visibility (defaults to 0.45).

    Returns:
        FaceVisibilityAssessment containing visibility rating, score, breakdown factors, and diagnostics.
    """
    # 1. Size Factor (S_size in [0.0, 1.0])
    if size_metrics is None and bbox is not None and image_shape is not None:
        try:
            size_metrics = compute_face_size_metrics(bbox, image_shape)
        except Exception:
            size_metrics = None

    if size_metrics is not None:
        min_dim = min(size_metrics.face_width, size_metrics.face_height)
        if min_dim <= 0:
            s_size = 0.0
        elif min_dim < 32.0:
            s_size = max(0.10, (min_dim / 32.0) * 0.40)
        elif min_dim < 80.0:
            s_size = 0.40 + ((min_dim - 32.0) / 48.0) * 0.35
        elif min_dim < 200.0:
            s_size = 0.75 + ((min_dim - 80.0) / 120.0) * 0.20
        else:
            s_size = min(1.0, 0.95 + min(0.05, (min_dim - 200.0) / 400.0 * 0.05))
    else:
        s_size = 0.70

    # 2. Blur / Sharpness Factor (S_blur in [0.0, 1.0])
    if clarity_metrics is None and face_crop_rgb is not None and face_crop_rgb.size > 0:
        try:
            clarity_metrics = compute_clarity_metrics(face_crop_rgb)
        except Exception:
            clarity_metrics = None

    if clarity_metrics is not None:
        s_blur = float(clarity_metrics.blur_score)
        if clarity_metrics.blur_status == BlurStatus.BLURRY:
            s_blur = min(s_blur, 0.35)
        elif clarity_metrics.blur_status == BlurStatus.SLIGHTLY_BLURRED:
            s_blur = min(0.74, max(0.40, s_blur))
    else:
        s_blur = 0.70

    # 3. Brightness / Illumination Factor (S_brightness in [0.0, 1.0])
    if illumination_metrics is None and face_crop_rgb is not None and face_crop_rgb.size > 0:
        try:
            illumination_metrics = calculate_face_brightness(face_crop_rgb)
        except Exception:
            illumination_metrics = None

    if illumination_metrics is not None:
        s_brightness = float(illumination_metrics.brightness_score)
        if illumination_metrics.brightness_status in (BrightnessStatus.TOO_DARK, BrightnessStatus.TOO_BRIGHT):
            s_brightness = min(s_brightness, 0.50)
    else:
        s_brightness = 0.70

    # 4. Landmark Availability Factor (S_lm_avail in [0.0, 1.0])
    if landmarks is None or len(landmarks) == 0:
        s_lm_avail = 0.20
    else:
        lm_count = len(landmarks)
        if lm_count >= 5:
            s_lm_avail = 1.0
        else:
            s_lm_avail = max(0.20, min(1.0, lm_count / 5.0))

    # 5. Landmark Confidence Factor (S_lm_conf in [0.0, 1.0])
    confidences: List[float] = []
    if landmarks:
        items = landmarks.values() if isinstance(landmarks, dict) else landmarks
        for pt in items:
            if isinstance(pt, dict) and pt.get("confidence") is not None:
                confidences.append(float(pt["confidence"]))
            elif hasattr(pt, "confidence") and pt.confidence is not None:
                confidences.append(float(pt.confidence))

    if confidences:
        s_lm_conf = float(np.mean(confidences))
    elif landmarks and len(landmarks) > 0:
        # Fallback to detection confidence when landmarks exist without individual scores
        s_lm_conf = min(1.0, max(0.0, float(detection_confidence)))
    else:
        # No landmarks detected at all
        s_lm_conf = min(0.50, max(0.0, float(detection_confidence) * 0.50))

    # Clamp all factors to [0.0, 1.0]
    s_size = round(min(1.0, max(0.0, s_size)), 3)
    s_blur = round(min(1.0, max(0.0, s_blur)), 3)
    s_brightness = round(min(1.0, max(0.0, s_brightness)), 3)
    s_lm_avail = round(min(1.0, max(0.0, s_lm_avail)), 3)
    s_lm_conf = round(min(1.0, max(0.0, s_lm_conf)), 3)

    # Weighted baseline score
    weighted_base = (
        0.25 * s_size
        + 0.25 * s_blur
        + 0.25 * s_brightness
        + 0.15 * s_lm_avail
        + 0.10 * s_lm_conf
    )

    # Critical bottleneck gating: severe defects cap maximum visibility
    critical_min = min(s_size, s_blur, s_brightness)
    if critical_min < 0.35:
        composite = min(weighted_base, critical_min * 1.60)
    else:
        composite = weighted_base

    visibility_score = round(min(1.0, max(0.0, composite)), 3)

    # Configurable threshold mapping
    th_excellent = (
        excellent_threshold
        if excellent_threshold is not None
        else getattr(settings, "AI_VISIBILITY_EXCELLENT_THRESHOLD", 0.82)
    )
    th_good = (
        good_threshold
        if good_threshold is not None
        else getattr(settings, "AI_VISIBILITY_GOOD_THRESHOLD", 0.65)
    )
    th_fair = (
        fair_threshold
        if fair_threshold is not None
        else getattr(settings, "AI_VISIBILITY_FAIR_THRESHOLD", 0.45)
    )

    if visibility_score >= th_excellent and critical_min >= 0.60 and s_lm_avail >= 0.80:
        visibility = FaceVisibility.EXCELLENT
    elif visibility_score >= th_good and critical_min >= 0.38:
        visibility = FaceVisibility.GOOD
    elif visibility_score >= th_fair:
        visibility = FaceVisibility.FAIR
    else:
        visibility = FaceVisibility.POOR

    # Explanatory diagnostics
    details_parts: List[str] = []
    if s_blur < 0.40:
        details_parts.append("Significant blur detected")
    if s_brightness < 0.50:
        details_parts.append("Suboptimal illumination (underexposed or overexposed)")
    if s_size < 0.40:
        details_parts.append("Low face resolution / tiny scale")
    if s_lm_avail < 0.60:
        details_parts.append("Incomplete facial landmarks")
    if not details_parts:
        details_str = "Optimal facial visibility across all metrics"
    else:
        details_str = "; ".join(details_parts)

    return FaceVisibilityAssessment(
        visibility=visibility,
        visibility_score=visibility_score,
        size_factor=s_size,
        blur_factor=s_blur,
        brightness_factor=s_brightness,
        landmark_availability_factor=s_lm_avail,
        landmark_confidence_factor=s_lm_conf,
        details=details_str,
    )

