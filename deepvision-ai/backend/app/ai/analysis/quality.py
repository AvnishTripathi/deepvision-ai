"""
DeepVision AI — Face Quality Analytics Module
Evaluates image clarity, blurriness, exposure, and contrast to provide
explainable quality scores for detected faces.
"""

from typing import Tuple
import cv2
import numpy as np
from app.schemas.ai import FaceQualityScore


def compute_sharpness(crop_rgb: np.ndarray) -> float:
    """
    Computes sharpness using the variance of the Laplacian filter.
    Higher values indicate sharp, in-focus faces; low values indicate motion blur or out-of-focus capture.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return round(variance, 2)


def compute_brightness_and_contrast(crop_rgb: np.ndarray) -> Tuple[float, float]:
    """
    Computes mean luminance (brightness) and standard deviation (contrast).
    Brightness: 0 (pitch black) to 255 (overexposed).
    """
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
    Executes comprehensive multi-metric quality assessment on a cropped face image.
    
    Returns:
        FaceQualityScore with composite quality score [0.0, 1.0] and usability flag.
    """
    if crop_rgb.size == 0 or crop_rgb.shape[0] < 10 or crop_rgb.shape[1] < 10:
        return FaceQualityScore(
            overall_quality=0.0,
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            is_usable=False,
        )

    sharpness = compute_sharpness(crop_rgb)
    brightness, contrast = compute_brightness_and_contrast(crop_rgb)

    # Normalize sharpness score (50-300+ typical range mapped to 0.0 - 1.0)
    norm_sharpness = min(1.0, max(0.0, (sharpness - 20.0) / 280.0))

    # Normalize brightness (ideal range ~90-160)
    if brightness < 60.0:
        norm_brightness = max(0.0, brightness / 60.0)
    elif brightness > 200.0:
        norm_brightness = max(0.0, (255.0 - brightness) / 55.0)
    else:
        norm_brightness = 1.0

    # Normalize contrast (typical std 20-80)
    norm_contrast = min(1.0, max(0.0, contrast / 60.0))

    # Composite weighted quality score
    composite_quality = (norm_sharpness * 0.50) + (norm_brightness * 0.30) + (norm_contrast * 0.20)
    overall_quality = round(min(1.0, max(0.0, composite_quality)), 3)

    # Determine usability
    is_usable = (
        sharpness >= min_sharpness_threshold and
        min_brightness_threshold <= brightness <= max_brightness_threshold and
        contrast >= 15.0
    )

    return FaceQualityScore(
        overall_quality=overall_quality,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        is_usable=is_usable,
    )

