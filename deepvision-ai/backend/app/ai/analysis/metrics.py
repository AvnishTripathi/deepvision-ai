"""
DeepVision AI — Face Quality Metrics Module
Calculates physical scale, blur/clarity measurements, and illumination/exposure diagnostics.
"""

from typing import Tuple, Optional
import numpy as np
import cv2

from app.core.config import settings
from app.schemas.ai import BoundingBox
from app.ai.analysis.schemas import (
    FaceSizeMetrics,
    SizeCategory,
    ClarityMetrics,
    BlurCategory,
    BlurStatus,
    IlluminationMetrics,
    ExposureCategory,
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
        else getattr(settings, "AI_BLUR_SLIGHTLY_BLURRED_THRESHOLD", 50.0)
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

def compute_illumination_metrics(
    face_crop_rgb: np.ndarray,
) -> IlluminationMetrics:
    """
    Evaluates mean luminance brightness, contrast, and highlights/shadows.
    """
    if face_crop_rgb.size == 0 or face_crop_rgb.shape[0] < 4 or face_crop_rgb.shape[1] < 4:
        return IlluminationMetrics(
            brightness=0.0,
            contrast=0.0,
            exposure_category=ExposureCategory.UNDEREXPOSED,
            illumination_score=0.0,
        )

    gray = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    brightness = round(mean_val, 2)
    contrast = round(std_val, 2)

    # Exposure classification
    if brightness < 55.0:
        category = ExposureCategory.UNDEREXPOSED
    elif brightness > 210.0:
        category = ExposureCategory.OVEREXPOSED
    elif contrast < 16.0:
        category = ExposureCategory.LOW_CONTRAST
    elif contrast > 75.0:
        category = ExposureCategory.HARSH_CONTRAST
    else:
        category = ExposureCategory.OPTIMAL

    # Continuous illumination quality score [0.0, 1.0]
    # Optimal brightness is centered around 100-160 with contrast between 25-60
    if brightness < 60.0:
        score_brightness = max(0.0, brightness / 60.0)
    elif brightness > 200.0:
        score_brightness = max(0.0, (255.0 - brightness) / 55.0)
    else:
        score_brightness = 1.0

    score_contrast = min(1.0, max(0.0, contrast / 50.0))
    illumination_score = round(score_brightness * 0.70 + score_contrast * 0.30, 3)

    return IlluminationMetrics(
        brightness=brightness,
        contrast=contrast,
        exposure_category=category,
        illumination_score=illumination_score,
    )
