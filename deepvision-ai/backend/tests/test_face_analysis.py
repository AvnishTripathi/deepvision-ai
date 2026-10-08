"""
DeepVision AI — Comprehensive Face Analysis Unit & Integration Tests (Step 4)
Tests 5-point facial landmarks, geometric metrics, 3D head pose estimation,
clarity/blur detection, illumination diagnostics, face size analysis,
composite quality scoring, and the POST /api/v1/ai/analyze API endpoint.
"""

import os
import io
import math
import numpy as np
import cv2
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas.ai import BoundingBox, LandmarkPoint, LandmarkCoordinate, FaceQualityScore
from app.ai.detection.schemas import FaceDetection
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
    landmarks_to_dict,
    landmarks_to_list,
    normalize_landmark_name,
    STANDARD_LANDMARK_NAMES,
    validate_and_clip_landmarks,
    validate_and_clip_landmarks_list,
)
from app.ai.analysis.pose import (
    HeadPoseEstimator,
    estimate_head_pose,
    CANONICAL_3D_MODEL_POINTS,
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
    calculate_face_quality_score,
    analyze_face_comprehensive,
    FaceAnalyzer,
    FaceQualityAnalyzer,
)

from app.services.ai_service import get_ai_service


# -----------------------------------------------------------------------------
# Fixtures & Helper Generators
# -----------------------------------------------------------------------------

def create_canonical_landmarks(
    offset_x: float = 100.0,
    offset_y: float = 100.0,
    scale: float = 1.0,
) -> list[LandmarkPoint]:
    """Generates standard canonical frontal 5-point facial landmarks."""
    return [
        LandmarkPoint(name="right_eye", x=offset_x + 35.0 * scale, y=offset_y + 40.0 * scale),
        LandmarkPoint(name="left_eye", x=offset_x + 85.0 * scale, y=offset_y + 40.0 * scale),
        LandmarkPoint(name="nose_tip", x=offset_x + 60.0 * scale, y=offset_y + 65.0 * scale),
        LandmarkPoint(name="mouth_right", x=offset_x + 42.0 * scale, y=offset_y + 90.0 * scale),
        LandmarkPoint(name="mouth_left", x=offset_x + 78.0 * scale, y=offset_y + 90.0 * scale),
    ]


def create_synthetic_face_crop(width: int = 120, height: int = 120, blur: bool = False) -> np.ndarray:
    """Creates a synthetic textured face-like RGB crop for sharpness and illumination tests."""
    crop = np.full((height, width, 3), 160, dtype=np.uint8)
    # Add contrast features (eyes, nose, mouth regions)
    crop[30:50, 25:45] = [40, 40, 40]    # Right eye
    crop[30:50, 75:95] = [40, 40, 40]    # Left eye
    crop[55:75, 50:70] = [80, 60, 60]    # Nose
    crop[85:98, 35:85] = [180, 60, 70]   # Mouth
    # Add fine texture noise
    rng = np.random.RandomState(42)
    noise = rng.randint(-20, 20, (height, width, 3))
    crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if blur:
        crop = cv2.GaussianBlur(crop, (25, 25), 9.0)
    return crop


# -----------------------------------------------------------------------------
# 1. Landmark Extraction & Geometric Metrics Tests
# -----------------------------------------------------------------------------

class TestLandmarkMetrics:
    def test_compute_landmark_metrics_standard(self):
        landmarks = create_canonical_landmarks(offset_x=50.0, offset_y=50.0, scale=1.0)
        metrics = compute_landmark_metrics(landmarks)

        assert metrics is not None
        assert isinstance(metrics, FacialLandmarkMetrics)
        # Eyes are at x=85 and x=135, so IPD = 50.0 px
        assert metrics.interpupillary_distance_px == 50.0
        # Horizontal eye-line tilt angle should be 0.0 deg
        assert abs(metrics.eye_tilt_angle_deg) < 0.1
        # Bilateral symmetry should be virtually 1.0 (perfectly centered nose)
        assert metrics.symmetry_ratio >= 0.95
        assert metrics.eye_to_nose_distance_px > 0.0
        assert metrics.nose_to_mouth_distance_px > 0.0

    def test_compute_landmark_metrics_tilted_eyes(self):
        landmarks = [
            LandmarkPoint(name="right_eye", x=50.0, y=50.0),
            LandmarkPoint(name="left_eye", x=100.0, y=80.0),  # left eye tilted downward
            LandmarkPoint(name="nose_tip", x=75.0, y=75.0),
            LandmarkPoint(name="mouth_right", x=60.0, y=100.0),
            LandmarkPoint(name="mouth_left", x=90.0, y=115.0),
        ]
        metrics = compute_landmark_metrics(landmarks)
        assert metrics is not None
        # Tilt angle atan2(30, 50) ~= 30.96 degrees
        assert 25.0 < metrics.eye_tilt_angle_deg < 35.0

    def test_compute_landmark_metrics_asymmetric(self):
        landmarks = [
            LandmarkPoint(name="right_eye", x=50.0, y=50.0),
            LandmarkPoint(name="left_eye", x=150.0, y=50.0),
            LandmarkPoint(name="nose_tip", x=60.0, y=80.0),  # nose severely shifted rightwards (near right eye)
            LandmarkPoint(name="mouth_right", x=55.0, y=110.0),
            LandmarkPoint(name="mouth_left", x=145.0, y=110.0),
        ]
        metrics = compute_landmark_metrics(landmarks)
        assert metrics is not None
        # Nose is much closer to right eye than left eye, symmetry ratio should be low
        assert metrics.symmetry_ratio < 0.50

    def test_compute_landmark_metrics_missing_or_empty(self):
        assert compute_landmark_metrics([]) is None
        # Incomplete landmark list (missing mouth_left)
        incomplete = [
            LandmarkPoint(name="right_eye", x=50.0, y=50.0),
            LandmarkPoint(name="left_eye", x=100.0, y=50.0),
            LandmarkPoint(name="nose_tip", x=75.0, y=75.0),
            LandmarkPoint(name="mouth_right", x=60.0, y=100.0),
        ]
        assert compute_landmark_metrics(incomplete) is None

    def test_crop_relative_landmarks(self):
        processor = LandmarkProcessor()
        landmarks = [
            LandmarkPoint(name="right_eye", x=150.0, y=120.0),
            LandmarkPoint(name="left_eye", x=200.0, y=120.0),
        ]
        rel = processor.extract_crop_relative_landmarks(landmarks, crop_origin=(100.0, 100.0))
        assert rel[0].x == 50.0
        assert rel[0].y == 20.0
        assert rel[1].x == 100.0
        assert rel[1].y == 20.0

    def test_landmarks_to_dict_canonical_5_points(self):
        landmarks = [
            LandmarkPoint(name="left_eye", x=150.0, y=180.0),
            LandmarkPoint(name="right_eye", x=220.0, y=180.0),
            LandmarkPoint(name="nose", x=185.0, y=220.0),
            LandmarkPoint(name="left_mouth", x=160.0, y=260.0),
            LandmarkPoint(name="right_mouth", x=210.0, y=260.0),
        ]
        structured = landmarks_to_dict(landmarks)
        assert isinstance(structured, dict)
        assert len(structured) == 5
        assert "left_eye" in structured
        assert "right_eye" in structured
        assert "nose" in structured
        assert "left_mouth" in structured
        assert "right_mouth" in structured
        assert structured["left_eye"].x == 150.0
        assert structured["left_eye"].y == 180.0
        assert structured["right_eye"].x == 220.0
        assert structured["right_eye"].y == 180.0
        assert structured["nose"].x == 185.0
        assert structured["nose"].y == 220.0

    def test_landmarks_to_dict_alias_normalization(self):
        # Model returns nose_tip, mouth_left, mouth_right
        landmarks = [
            LandmarkPoint(name="right_eye", x=220.0, y=180.0),
            LandmarkPoint(name="left_eye", x=150.0, y=180.0),
            LandmarkPoint(name="nose_tip", x=185.0, y=220.0),
            LandmarkPoint(name="mouth_left", x=160.0, y=260.0),
            LandmarkPoint(name="mouth_right", x=210.0, y=260.0),
        ]
        structured = landmarks_to_dict(landmarks)
        # Should be accessible via standard canonical names
        assert "nose" in structured
        assert "left_mouth" in structured
        assert "right_mouth" in structured
        assert structured["nose"].x == 185.0
        assert structured["left_mouth"].x == 160.0
        assert structured["right_mouth"].x == 210.0

    def test_landmarks_to_dict_subset_never_invents_missing(self):
        # Model only returns 3 landmarks (like in user prompt example)
        landmarks = [
            LandmarkPoint(name="left_eye", x=150.0, y=180.0),
            LandmarkPoint(name="right_eye", x=220.0, y=180.0),
            LandmarkPoint(name="nose", x=185.0, y=220.0),
        ]
        structured = landmarks_to_dict(landmarks)
        assert len(structured) == 3
        assert "left_mouth" not in structured
        assert "right_mouth" not in structured
        assert structured["left_eye"].x == 150.0
        assert structured["right_eye"].x == 220.0
        assert structured["nose"].x == 185.0

    def test_landmarks_to_dict_preserves_extended_landmarks(self):
        # Model returns additional landmarks (e.g. 68-point model points)
        landmarks = [
            LandmarkPoint(name="left_eye", x=150.0, y=180.0),
            LandmarkPoint(name="right_eye", x=220.0, y=180.0),
            LandmarkPoint(name="nose", x=185.0, y=220.0),
            LandmarkPoint(name="chin", x=185.0, y=290.0),
            LandmarkPoint(name="left_eyebrow", x=145.0, y=160.0),
            LandmarkPoint(name="right_eyebrow", x=225.0, y=160.0),
        ]
        structured = landmarks_to_dict(landmarks)
        assert len(structured) == 6
        assert "chin" in structured
        assert "left_eyebrow" in structured
        assert "right_eyebrow" in structured
        assert structured["chin"].x == 185.0
        assert structured["chin"].y == 290.0

    def test_structured_face_analysis_landmarks_dict_format(self):
        crop = create_synthetic_face_crop(width=120, height=120)
        bbox = BoundingBox(x1=100, y1=100, x2=220, y2=220)
        landmarks = [
            LandmarkPoint(name="left_eye", x=150.0, y=180.0),
            LandmarkPoint(name="right_eye", x=220.0, y=180.0),
            LandmarkPoint(name="nose_tip", x=185.0, y=220.0),
            LandmarkPoint(name="mouth_left", x=160.0, y=260.0),
            LandmarkPoint(name="mouth_right", x=210.0, y=260.0),
        ]
        analysis = analyze_face_comprehensive(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(640, 640),
            confidence=0.98,
            landmarks=landmarks,
            face_id="face_test",
        )
        assert isinstance(analysis.landmarks, dict)
        assert "left_eye" in analysis.landmarks
        assert "right_eye" in analysis.landmarks
        assert "nose" in analysis.landmarks
        assert "left_mouth" in analysis.landmarks
        assert "right_mouth" in analysis.landmarks
        assert analysis.landmarks["left_eye"].x == 150.0
        assert analysis.landmarks["left_eye"].y == 180.0
        # Check dict model_dump conforms to {landmarks: {left_eye: {x: 150, y: 180}, ...}}
        dumped = analysis.model_dump()
        assert dumped["landmarks"]["left_eye"]["x"] == 150.0
        assert dumped["landmarks"]["left_eye"]["y"] == 180.0
        assert dumped["landmarks"]["nose"]["x"] == 185.0


# -----------------------------------------------------------------------------
# 2. 3D Head Pose & Orientation Estimation Tests
# -----------------------------------------------------------------------------

class TestHeadPoseEstimation:
    def test_frontal_head_pose(self):
        landmarks = create_canonical_landmarks(offset_x=250.0, offset_y=200.0, scale=1.5)
        pose = estimate_head_pose(landmarks, image_shape=(640, 640))

        assert pose is not None
        assert isinstance(pose, HeadPoseEstimation)
        # Canonical frontal points should yield frontality >= 0.70
        assert pose.frontality_score >= 0.70
        assert pose.pose_category in (PoseCategory.FRONTAL, PoseCategory.SLIGHT_TURN)
        assert -45.0 <= pose.yaw <= 45.0
        assert -45.0 <= pose.pitch <= 45.0
        assert -45.0 <= pose.roll <= 45.0

    def test_turned_head_pose_yaw_detection(self):
        # Shift nose tip towards left eye to simulate a head turn
        landmarks = [
            LandmarkPoint(name="right_eye", x=200.0, y=200.0),
            LandmarkPoint(name="left_eye", x=300.0, y=200.0),
            LandmarkPoint(name="nose_tip", x=285.0, y=240.0),  # heavily shifted towards left eye
            LandmarkPoint(name="mouth_right", x=220.0, y=290.0),
            LandmarkPoint(name="mouth_left", x=290.0, y=290.0),
        ]
        pose = estimate_head_pose(landmarks, image_shape=(640, 640))
        assert pose is not None
        assert pose.pose_category in (PoseCategory.SLIGHT_TURN, PoseCategory.PROFILE)
        assert pose.frontality_score < 0.90

    def test_tilted_head_pose_pitch_categories(self):
        # Downward pitch (nose much lower than normal)
        landmarks_down = [
            LandmarkPoint(name="right_eye", x=200.0, y=200.0),
            LandmarkPoint(name="left_eye", x=300.0, y=200.0),
            LandmarkPoint(name="nose_tip", x=250.0, y=180.0),  # nose high relative to eyes (head tilted downward)
            LandmarkPoint(name="mouth_right", x=220.0, y=250.0),
            LandmarkPoint(name="mouth_left", x=280.0, y=250.0),
        ]
        pose = estimate_head_pose(landmarks_down, image_shape=(640, 640))
        assert pose is not None
        assert isinstance(pose.pitch, float)

    def test_invalid_landmarks_and_degenerate_canvas(self):
        assert estimate_head_pose([], image_shape=(640, 640)) is None
        landmarks = create_canonical_landmarks()
        assert estimate_head_pose(landmarks, image_shape=(0, 0)) is None

    def test_solve_pnp_fallback_heuristic(self):
        estimator = HeadPoseEstimator()
        lm_map = {
            "right_eye": (100.0, 100.0),
            "left_eye": (200.0, 100.0),
            "nose_tip": (150.0, 140.0),
            "mouth_right": (120.0, 180.0),
            "mouth_left": (180.0, 180.0),
        }
        fallback_pose = estimator._geometric_heuristic_pose_fallback(lm_map)
        assert isinstance(fallback_pose, HeadPoseEstimation)
        assert fallback_pose.pose_category == PoseCategory.FRONTAL
        assert fallback_pose.frontality_score >= 0.80


# -----------------------------------------------------------------------------
# 3. Clarity & Blur Detection Tests
# -----------------------------------------------------------------------------

class TestClarityMetrics:
    def test_sharp_crop_metrics(self):
        crop = create_synthetic_face_crop(width=100, height=100, blur=False)
        clarity = estimate_face_blur(crop)

        assert isinstance(clarity, ClarityMetrics)
        assert clarity.sharpness > 50.0
        assert clarity.blur_score > 0.15
        assert clarity.blur_status in (BlurStatus.SHARP, BlurStatus.SLIGHTLY_BLURRED)
        assert clarity.blur_category in (BlurCategory.SHARP, BlurCategory.SLIGHTLY_BLURRED)

    def test_blurred_crop_metrics(self):
        crop = create_synthetic_face_crop(width=100, height=100, blur=True)
        clarity = estimate_face_blur(crop)

        assert clarity.sharpness < 40.0
        assert clarity.blur_score < 0.20
        assert clarity.blur_status == BlurStatus.BLURRY
        assert clarity.blur_category in (BlurCategory.BLURRED, BlurCategory.VERY_BLURRED)

    def test_empty_or_degenerate_crop(self):
        empty_crop = np.empty((0, 0, 3), dtype=np.uint8)
        clarity = estimate_face_blur(empty_crop)
        assert clarity.sharpness == 0.0
        assert clarity.blur_score == 0.0
        assert clarity.blur_status == BlurStatus.BLURRY
        assert clarity.blur_category == BlurCategory.VERY_BLURRED

    def test_configurable_blur_thresholds(self):
        """Verifies that configurable sharp and slightly_blurred thresholds dynamically alter status."""
        crop = create_synthetic_face_crop(width=100, height=100, blur=False)
        sharpness_val = estimate_face_blur(crop).sharpness

        # Case 1: Ultra-high sharp_threshold (e.g. 5000) forces it to slightly_blurred or blurry
        high_thresh_clarity = estimate_face_blur(
            crop,
            sharp_threshold=sharpness_val + 500.0,
            slightly_blurred_threshold=sharpness_val - 10.0,
        )
        assert high_thresh_clarity.blur_status == BlurStatus.SLIGHTLY_BLURRED

        # Case 2: Very high slightly_blurred_threshold forces it to blurry
        blurry_thresh_clarity = estimate_face_blur(
            crop,
            sharp_threshold=sharpness_val + 1000.0,
            slightly_blurred_threshold=sharpness_val + 500.0,
        )
        assert blurry_thresh_clarity.blur_status == BlurStatus.BLURRY

        # Case 3: Very low sharp_threshold (e.g. 10.0) forces even slightly blurry crop to sharp
        low_thresh_clarity = estimate_face_blur(
            crop,
            sharp_threshold=10.0,
            slightly_blurred_threshold=5.0,
        )
        assert low_thresh_clarity.blur_status == BlurStatus.SHARP


# -----------------------------------------------------------------------------
# 4. Illumination & Exposure Diagnostics Tests
# -----------------------------------------------------------------------------

class TestIlluminationMetrics:
    def test_optimal_exposure(self):
        crop = create_synthetic_face_crop(width=100, height=100)
        illum = calculate_face_brightness(crop)

        assert isinstance(illum, IlluminationMetrics)
        assert 60.0 <= illum.brightness <= 200.0
        assert illum.contrast >= 15.0
        assert illum.brightness_status == BrightnessStatus.ACCEPTABLE
        assert illum.brightness_score >= 0.70
        # Backward compatibility checks
        assert illum.exposure_category == ExposureCategory.OPTIMAL
        assert illum.illumination_score >= 0.70

    def test_underexposed_dark_face(self):
        dark_crop = np.full((80, 80, 3), 25, dtype=np.uint8)
        illum = calculate_face_brightness(dark_crop)

        assert illum.brightness < 55.0
        assert illum.brightness_status == BrightnessStatus.TOO_DARK
        assert illum.brightness_score < 0.60
        assert illum.exposure_category == ExposureCategory.UNDEREXPOSED

    def test_overexposed_bright_face(self):
        bright_crop = np.full((80, 80, 3), 245, dtype=np.uint8)
        illum = calculate_face_brightness(bright_crop)

        assert illum.brightness > 210.0
        assert illum.brightness_status == BrightnessStatus.TOO_BRIGHT
        assert illum.brightness_score < 0.70
        assert illum.exposure_category == ExposureCategory.OVEREXPOSED

    def test_low_contrast_flat_face(self):
        flat_crop = np.full((80, 80, 3), 128, dtype=np.uint8)
        illum = compute_illumination_metrics(flat_crop)

        assert illum.contrast < 5.0
        assert illum.exposure_category == ExposureCategory.LOW_CONTRAST
        assert illum.brightness_status == BrightnessStatus.ACCEPTABLE

    def test_configurable_brightness_thresholds(self):
        """Verifies that configurable min_acceptable and max_acceptable thresholds dynamically alter status."""
        crop = np.full((80, 80, 3), 100, dtype=np.uint8)

        # Baseline: 100 is acceptable under default (60, 200)
        baseline = calculate_face_brightness(crop)
        assert baseline.brightness_status == BrightnessStatus.ACCEPTABLE

        # High min_acceptable (e.g. 120.0) -> 100 becomes too_dark
        strict_dark = calculate_face_brightness(crop, min_acceptable=120.0, max_acceptable=220.0)
        assert strict_dark.brightness_status == BrightnessStatus.TOO_DARK

        # Low max_acceptable (e.g. 90.0) -> 100 becomes too_bright
        strict_bright = calculate_face_brightness(crop, min_acceptable=40.0, max_acceptable=90.0)
        assert strict_bright.brightness_status == BrightnessStatus.TOO_BRIGHT

    def test_empty_or_degenerate_crop_brightness(self):
        empty_crop = np.empty((0, 0, 3), dtype=np.uint8)
        illum = calculate_face_brightness(empty_crop)

        assert illum.brightness == 0.0
        assert illum.brightness_score == 0.0
        assert illum.brightness_status == BrightnessStatus.TOO_DARK
        assert illum.illumination_score == 0.0



# -----------------------------------------------------------------------------
# 5. Face Size & Scale Analysis Tests
# -----------------------------------------------------------------------------

class TestFaceSizeMetrics:
    def test_tiny_face_detection(self):
        bbox = BoundingBox(x1=10, y1=10, x2=30, y2=30)  # 20x20 px
        size = compute_face_size_metrics(bbox, image_shape=(1080, 1920))

        assert size.size_category == SizeCategory.TINY
        assert not size.is_sufficient_size
        assert size.width_px == 20.0
        assert size.height_px == 20.0
        assert size.aspect_ratio == 1.0

    def test_small_face_detection(self):
        bbox = BoundingBox(x1=100, y1=100, x2=160, y2=160)  # 60x60 px
        size = compute_face_size_metrics(bbox, image_shape=(1080, 1920))

        assert size.size_category == SizeCategory.SMALL
        assert size.is_sufficient_size

    def test_medium_face_detection(self):
        bbox = BoundingBox(x1=100, y1=100, x2=250, y2=250)  # 150x150 px
        size = compute_face_size_metrics(bbox, image_shape=(1080, 1920))

        assert size.size_category == SizeCategory.MEDIUM
        assert size.is_sufficient_size

    def test_large_face_detection(self):
        bbox = BoundingBox(x1=100, y1=100, x2=450, y2=450)  # 350x350 px
        size = compute_face_size_metrics(bbox, image_shape=(1080, 1920))

        assert size.size_category == SizeCategory.LARGE
        assert size.is_sufficient_size

    def test_interpupillary_distance_sufficiency_cutoff(self):
        bbox = BoundingBox(x1=100, y1=100, x2=150, y2=150)
        # IPD < 20px marks face as insufficient for biometric matching
        size = compute_face_size_metrics(bbox, image_shape=(1080, 1920), ipd=15.0)
        assert not size.is_sufficient_size

    def test_face_size_measurable_properties(self):
        """Verifies calculation of face_width, face_height, face_area, and face_area_ratio."""
        img_w, img_h = 1000, 1000
        bbox = BoundingBox(x1=200, y1=300, x2=400, y2=600)  # w=200, h=300

        size = calculate_face_size(bbox, image_shape=(img_h, img_w))

        # Check explicit measurable properties
        assert size.face_width == 200.0
        assert size.face_height == 300.0
        assert size.face_area == 60000.0  # 200 * 300
        # face_area_ratio = face_area / image_area = 60000 / (1000 * 1000) = 0.06
        expected_ratio = round(60000.0 / (1000 * 1000), 4)
        assert size.face_area_ratio == expected_ratio
        assert size.aspect_ratio == round(200.0 / 300.0, 3)

        # Check backward compatibility aliases
        assert size.width_px == 200.0
        assert size.height_px == 300.0
        assert size.area_px == 60000.0
        assert size.relative_area == expected_ratio

    def test_face_quality_analyzer_alias(self):
        """Verifies FaceQualityAnalyzer can be instantiated and behaves identically to FaceAnalyzer."""
        analyzer = FaceQualityAnalyzer(min_sharpness=50.0)
        assert isinstance(analyzer, FaceAnalyzer)
        assert analyzer.min_sharpness == 50.0


# -----------------------------------------------------------------------------
# 6. Composite Face Quality & Grading Tests
# -----------------------------------------------------------------------------

class TestComprehensiveQualityScoring:
    def test_grade_boundaries(self):
        # Normalized 0.0 - 1.0 scale
        assert evaluate_quality_grade(0.92) == QualityGrade.EXCELLENT
        assert evaluate_quality_grade(0.78) == QualityGrade.GOOD
        assert evaluate_quality_grade(0.60) == QualityGrade.FAIR
        assert evaluate_quality_grade(0.40) == QualityGrade.POOR
        assert evaluate_quality_grade(0.15) == QualityGrade.UNUSABLE
        # 0 - 100 scale
        assert evaluate_quality_grade(92.0) == QualityGrade.EXCELLENT
        assert evaluate_quality_grade(78.0) == QualityGrade.GOOD
        assert evaluate_quality_grade(60.0) == QualityGrade.FAIR
        assert evaluate_quality_grade(40.0) == QualityGrade.POOR
        assert evaluate_quality_grade(15.0) == QualityGrade.UNUSABLE

    def test_calculate_face_quality_score_factors(self):
        crop = create_synthetic_face_crop(width=140, height=140, blur=False)
        bbox = BoundingBox(x1=80, y1=80, x2=220, y2=220)
        landmarks = create_canonical_landmarks(offset_x=80.0, offset_y=80.0)

        score, factors, grade = calculate_face_quality_score(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(720, 1280),
            landmarks=landmarks,
            detection_confidence=0.98,
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0
        assert score >= 70.0  # Sharp, medium-sized, canonical landmarks
        assert isinstance(factors, FaceQualityFactors)
        assert 0.0 <= factors.sharpness <= 100.0
        assert 0.0 <= factors.brightness <= 100.0
        assert 0.0 <= factors.contrast <= 100.0
        assert 0.0 <= factors.size <= 100.0
        assert 0.0 <= factors.landmark <= 100.0
        assert grade in (QualityGrade.EXCELLENT, QualityGrade.GOOD)

    def test_quality_score_severe_defect_gating(self):
        # Case 1: Pitch black face (luminance 10)
        dark_crop = np.full((100, 100, 3), 10, dtype=np.uint8)
        bbox = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        landmarks = create_canonical_landmarks(offset_x=50.0, offset_y=50.0)

        dark_score, dark_factors, dark_grade = calculate_face_quality_score(
            face_crop_rgb=dark_crop,
            bbox=bbox,
            image_shape=(720, 1280),
            landmarks=landmarks,
        )
        assert dark_score < 40.0
        assert dark_factors.brightness <= 30.0
        assert dark_grade in (QualityGrade.POOR, QualityGrade.UNUSABLE)

        # Case 2: Tiny face (16x16 px)
        tiny_bbox = BoundingBox(x1=10, y1=10, x2=26, y2=26)
        tiny_crop = np.full((16, 16, 3), 128, dtype=np.uint8)

        tiny_score, tiny_factors, tiny_grade = calculate_face_quality_score(
            face_crop_rgb=tiny_crop,
            bbox=tiny_bbox,
            image_shape=(1080, 1920),
            landmarks=None,
        )
        assert tiny_score < 40.0
        assert tiny_factors.size <= 25.0
        assert tiny_grade in (QualityGrade.POOR, QualityGrade.UNUSABLE)

    def test_analyze_face_comprehensive_sharp_face(self):
        crop = create_synthetic_face_crop(width=120, height=120, blur=False)
        bbox = BoundingBox(x1=100, y1=100, x2=220, y2=220)
        landmarks = create_canonical_landmarks(offset_x=100.0, offset_y=100.0)

        analysis = analyze_face_comprehensive(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(640, 640),
            confidence=0.98,
            landmarks=landmarks,
            face_id="face_0",
        )

        assert isinstance(analysis, StructuredFaceAnalysis)
        assert analysis.face_id == "face_0"
        assert analysis.confidence == 0.98
        assert 0.0 <= analysis.quality_score <= 100.0
        assert analysis.quality_score == analysis.quality.quality_score
        assert analysis.quality.overall_quality == round(analysis.quality_score / 100.0, 3)
        assert analysis.quality.factors is not None
        assert analysis.landmark_metrics is not None
        assert analysis.pose is not None
        assert analysis.size.is_sufficient_size
        assert analysis.quality.is_usable
        assert analysis.quality.grade in (QualityGrade.EXCELLENT, QualityGrade.GOOD, QualityGrade.FAIR)

        # Check serialization contains quality_score and factors
        dumped = analysis.model_dump()
        assert "quality_score" in dumped
        assert 0.0 <= dumped["quality_score"] <= 100.0
        assert "factors" in dumped["quality"]
        assert "sharpness" in dumped["quality"]["factors"]
        assert "brightness" in dumped["quality"]["factors"]
        assert "contrast" in dumped["quality"]["factors"]
        assert "size" in dumped["quality"]["factors"]
        assert "landmark" in dumped["quality"]["factors"]

    def test_analyze_face_comprehensive_blurred_face_unusable(self):
        crop = create_synthetic_face_crop(width=120, height=120, blur=True)
        bbox = BoundingBox(x1=100, y1=100, x2=220, y2=220)
        landmarks = create_canonical_landmarks(offset_x=100.0, offset_y=100.0)

        analysis = analyze_face_comprehensive(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(640, 640),
            confidence=0.95,
            landmarks=landmarks,
            min_sharpness=50.0,
        )

        assert not analysis.quality.is_usable
        assert analysis.clarity.blur_category in (BlurCategory.BLURRED, BlurCategory.VERY_BLURRED)
        assert analysis.quality_score < 60.0



# -----------------------------------------------------------------------------
# 7. Backward Compatibility Tests
# -----------------------------------------------------------------------------

class TestLegacyCompatibility:
    def test_analyze_face_quality_returns_legacy_schema(self):
        crop = create_synthetic_face_crop(width=100, height=100)
        quality = analyze_face_quality(crop)

        assert isinstance(quality, FaceQualityScore)
        assert 0.0 <= quality.overall_quality <= 1.0
        assert quality.sharpness >= 0.0
        assert 0.0 <= quality.brightness <= 255.0
        assert quality.contrast >= 0.0
        assert isinstance(quality.is_usable, bool)

    def test_compute_sharpness_and_contrast_helpers(self):
        crop = create_synthetic_face_crop(width=100, height=100)
        sharpness = compute_sharpness(crop)
        brightness, contrast = compute_brightness_and_contrast(crop)

        assert isinstance(sharpness, float)
        assert isinstance(brightness, float)
        assert isinstance(contrast, float)
        assert sharpness > 0.0
        assert 0.0 <= brightness <= 255.0
        assert contrast > 0.0


# -----------------------------------------------------------------------------
# 8. FaceAnalyzer Orchestration Tests
# -----------------------------------------------------------------------------

class TestFaceAnalyzerOrchestrator:
    def test_analyze_detections_batch(self):
        canvas = np.full((400, 400, 3), 128, dtype=np.uint8)
        # Place synthetic face on canvas
        face_patch = create_synthetic_face_crop(width=100, height=100)
        canvas[50:150, 50:150] = face_patch

        detections = [
            FaceDetection(
                face_id="det_1",
                bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150),
                confidence=0.97,
                landmarks=create_canonical_landmarks(offset_x=50.0, offset_y=50.0),
            )
        ]

        analyzer = FaceAnalyzer(min_sharpness=50.0)
        analyses = analyzer.analyze_detections(canvas, detections)

        assert len(analyses) == 1
        res = analyses[0]
        assert res.face_id == "det_1"
        assert res.confidence == 0.97
        assert res.quality.overall_quality > 0.0
        assert res.landmark_metrics is not None
        assert res.pose is not None


# -----------------------------------------------------------------------------
# 9. API Endpoint Integration Tests (POST /api/v1/ai/analyze)
# -----------------------------------------------------------------------------

@pytest.mark.anyio
class TestAnalysisAPIEndpoint:
    async def test_api_analyze_endpoint_success(self):
        # Generate valid JPEG with synthetic face
        img = create_synthetic_face_crop(width=200, height=200)
        success, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        assert success
        img_bytes = enc.tobytes()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ai/analyze",
                files={"file": ("face.jpg", io.BytesIO(img_bytes), "image/jpeg")},
                params={"confidence_threshold": 0.5},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "faces" in data
        assert "face_count" in data
        assert "processing_time_ms" in data
        assert data["image_width"] == 200
        assert data["image_height"] == 200

    async def test_api_analyze_endpoint_invalid_file_rejected(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/ai/analyze",
                files={"file": ("corrupt.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
            )

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "error" in data


# -----------------------------------------------------------------------------
# 10. Landmark Coordinate Validation & Boundary Enforcement Tests (Step 5)
# -----------------------------------------------------------------------------

class TestLandmarkValidationAndRobustness:
    """
    Tests landmark coordinate boundary enforcement (0 <= x <= w, 0 <= y <= h),
    low-confidence filtering, non-finite value rejection, partial landmark safety,
    and multi-face batch fault isolation.
    """

    def test_landmark_boundary_clipping(self):
        """Verifies that out-of-bounds landmark coordinates are clamped strictly to [0, w] and [0, h]."""
        w, h = 640, 480
        raw_landmarks = [
            LandmarkPoint(name="left_eye", x=-50.0, y=100.0, confidence=0.9),
            LandmarkPoint(name="right_eye", x=700.0, y=-20.0, confidence=0.85),
            LandmarkPoint(name="nose_tip", x=320.0, y=550.0, confidence=0.95),
            LandmarkPoint(name="mouth_left", x=280.0, y=350.0, confidence=0.9),
            LandmarkPoint(name="mouth_right", x=360.0, y=350.0, confidence=0.9),
        ]

        clipped_dict = validate_and_clip_landmarks(raw_landmarks, image_width=w, image_height=h)

        # left_eye was -50.0 -> clamped to 0.0
        assert clipped_dict["left_eye"].x == 0.0
        assert clipped_dict["left_eye"].y == 100.0

        # right_eye was x=700.0, y=-20.0 -> clamped to x=640.0, y=0.0
        assert clipped_dict["right_eye"].x == 640.0
        assert clipped_dict["right_eye"].y == 0.0

        # nose was y=550.0 -> clamped to y=480.0 (normalized from nose_tip to nose)
        assert clipped_dict["nose"].x == 320.0
        assert clipped_dict["nose"].y == 480.0

        # All points strictly satisfy 0 <= x <= w and 0 <= y <= h
        for name, coord in clipped_dict.items():
            assert 0.0 <= coord.x <= float(w)
            assert 0.0 <= coord.y <= float(h)

        # Verify list variant as well
        clipped_list = validate_and_clip_landmarks_list(raw_landmarks, image_width=w, image_height=h)
        assert len(clipped_list) == 5
        for lm in clipped_list:
            assert 0.0 <= lm.x <= float(w)
            assert 0.0 <= lm.y <= float(h)

    def test_landmark_is_within_bounds_checker(self):
        """Tests the is_within_bounds method on LandmarkPoint and LandmarkCoordinate."""
        w, h = 800, 600

        valid_pt = LandmarkPoint(name="nose", x=400.0, y=300.0)
        assert valid_pt.is_within_bounds(w, h) is True

        neg_x = LandmarkPoint(name="nose", x=-5.0, y=300.0)
        assert neg_x.is_within_bounds(w, h) is False

        over_w = LandmarkPoint(name="nose", x=850.0, y=300.0)
        assert over_w.is_within_bounds(w, h) is False

        over_h = LandmarkPoint(name="nose", x=400.0, y=650.0)
        assert over_h.is_within_bounds(w, h) is False

        # LandmarkCoordinate
        valid_coord = LandmarkCoordinate(x=100.0, y=100.0)
        assert valid_coord.is_within_bounds(w, h) is True

        neg_coord = LandmarkCoordinate(x=100.0, y=-1.0)
        assert neg_coord.is_within_bounds(w, h) is False

    def test_nan_and_inf_landmarks_discarded(self):
        """Verifies that non-finite landmark values (NaN / Inf) are safely discarded without error."""
        w, h = 640, 480
        corrupt_landmarks = [
            LandmarkPoint(name="left_eye", x=float("nan"), y=150.0),
            LandmarkPoint(name="right_eye", x=250.0, y=float("inf")),
            LandmarkPoint(name="nose_tip", x=200.0, y=200.0),
        ]

        cleaned = validate_and_clip_landmarks(corrupt_landmarks, image_width=w, image_height=h)
        assert "left_eye" not in cleaned
        assert "right_eye" not in cleaned
        assert "nose" in cleaned
        assert cleaned["nose"].x == 200.0

    def test_low_confidence_filtering(self):
        """Verifies that landmarks with confidence below threshold are filtered out when requested."""
        w, h = 640, 480
        landmarks = [
            LandmarkPoint(name="left_eye", x=150.0, y=150.0, confidence=0.92),
            LandmarkPoint(name="right_eye", x=250.0, y=150.0, confidence=0.35),  # Low confidence
            LandmarkPoint(name="nose_tip", x=200.0, y=200.0, confidence=0.88),
        ]

        filtered = validate_and_clip_landmarks(landmarks, image_width=w, image_height=h, min_confidence=0.50)
        assert "left_eye" in filtered
        assert "nose" in filtered
        assert "right_eye" not in filtered

    def test_missing_and_partial_landmarks_safety(self):
        """Verifies that incomplete landmarks (e.g. 1 or 2 points) return None safely without crashing."""
        partial_landmarks = [
            LandmarkPoint(name="left_eye", x=150.0, y=150.0),
            LandmarkPoint(name="right_eye", x=250.0, y=150.0),
        ]

        # Landmark metrics requires all 5 regions
        metrics = compute_landmark_metrics(partial_landmarks)
        assert metrics is None

        # Head pose requires all 5 regions
        pose = estimate_head_pose(partial_landmarks, image_shape=(480, 640))
        assert pose is None

    def test_collinear_or_identical_landmarks_safety(self):
        """Verifies that degenerate coincident landmarks do not cause ZeroDivisionError or crash."""
        coincident_landmarks = [
            LandmarkPoint(name="left_eye", x=100.0, y=100.0),
            LandmarkPoint(name="right_eye", x=100.0, y=100.0),
            LandmarkPoint(name="nose_tip", x=100.0, y=100.0),
            LandmarkPoint(name="mouth_left", x=100.0, y=100.0),
            LandmarkPoint(name="mouth_right", x=100.0, y=100.0),
        ]

        # Should handle 0 IPD and 0 distances cleanly without raising
        metrics = compute_landmark_metrics(coincident_landmarks)
        assert metrics is not None
        assert metrics.interpupillary_distance_px == 0.0

        # Pose estimation should fallback safely without raising
        pose = estimate_head_pose(coincident_landmarks, image_shape=(480, 640))
        assert pose is not None

    def test_analyze_face_comprehensive_with_no_landmarks(self):
        """Ensures comprehensive face analysis succeeds with zero landmarks (landmarks=None)."""
        crop = create_synthetic_face_crop(width=100, height=100)
        bbox = BoundingBox(x1=50, y1=50, x2=150, y2=150)

        analysis = analyze_face_comprehensive(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(300, 300),
            landmarks=None,
        )

        assert analysis is not None
        assert analysis.landmark_metrics is None
        assert analysis.pose is None
        assert analysis.landmarks == {}
        assert analysis.quality.overall_quality > 0.0
        assert analysis.size.width_px == 100.0

    def test_analyze_face_comprehensive_with_partial_landmarks(self):
        """Ensures comprehensive face analysis succeeds with partial landmarks (e.g. 2 points)."""
        crop = create_synthetic_face_crop(width=100, height=100)
        bbox = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        partial_lms = [
            LandmarkPoint(name="left_eye", x=70.0, y=70.0),
            LandmarkPoint(name="right_eye", x=130.0, y=70.0),
        ]

        analysis = analyze_face_comprehensive(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(300, 300),
            landmarks=partial_lms,
        )

        assert analysis is not None
        assert analysis.landmark_metrics is None
        assert analysis.pose is None
        assert "left_eye" in analysis.landmarks
        assert "right_eye" in analysis.landmarks
        assert analysis.quality.overall_quality > 0.0

    def test_multi_face_fault_isolation(self):
        """
        CRITICAL TEST: Verifies that a batch containing faces with corrupt, missing,
        and valid landmarks NEVER crashes the complete request.
        All faces in the batch must be processed.
        """
        canvas = np.zeros((400, 600, 3), dtype=np.uint8)

        # Face 1: Fully valid canonical landmarks
        face_1 = FaceDetection(
            face_id="valid_face",
            bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150),
            confidence=0.96,
            landmarks=create_canonical_landmarks(offset_x=50.0, offset_y=50.0),
        )

        # Face 2: Corrupted landmarks with NaN and out-of-bounds coordinates
        face_2 = FaceDetection(
            face_id="corrupt_face",
            bbox=BoundingBox(x1=200, y1=50, x2=300, y2=150),
            confidence=0.90,
            landmarks=[
                LandmarkPoint(name="left_eye", x=float("nan"), y=80.0),
                LandmarkPoint(name="right_eye", x=9999.0, y=-500.0),
            ],
        )

        # Face 3: Incomplete / zero landmarks
        face_3 = FaceDetection(
            face_id="no_landmarks_face",
            bbox=BoundingBox(x1=350, y1=50, x2=450, y2=150),
            confidence=0.92,
            landmarks=None,
        )

        analyzer = FaceAnalyzer(min_sharpness=50.0)
        analyses = analyzer.analyze_detections(canvas, [face_1, face_2, face_3])

        # Batch MUST contain results for all 3 faces without crashing
        assert len(analyses) == 3

        # Face 1 has full analysis
        assert analyses[0].face_id == "valid_face"
        assert analyses[0].landmark_metrics is not None
        assert analyses[0].pose is not None

        # Face 2 has clipped/cleaned landmarks, did not crash
        assert analyses[1].face_id == "corrupt_face"
        assert analyses[1].landmark_metrics is None  # Handled safely

        # Face 3 has no landmarks, did not crash
        assert analyses[2].face_id == "no_landmarks_face"
        assert analyses[2].landmark_metrics is None
        assert analyses[2].pose is None


# -----------------------------------------------------------------------------
# 10. Face Visibility Assessment Tests (Step 10)
# -----------------------------------------------------------------------------

class TestFaceVisibilityAssessment:
    def test_excellent_visibility_face(self):
        crop = create_synthetic_face_crop(width=160, height=160, blur=False)
        bbox = BoundingBox(x1=100, y1=100, x2=260, y2=260)
        landmarks = create_canonical_landmarks(offset_x=100.0, offset_y=100.0)
        for lm in landmarks:
            lm.confidence = 0.98

        assessment = calculate_face_visibility(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(1080, 1920),
            landmarks=landmarks,
            detection_confidence=0.99,
        )

        assert isinstance(assessment, FaceVisibilityAssessment)
        assert assessment.visibility == FaceVisibility.EXCELLENT
        assert assessment.visibility_score >= 0.82
        assert assessment.size_factor >= 0.80
        assert assessment.blur_factor >= 0.70
        assert assessment.brightness_factor >= 0.70
        assert assessment.landmark_availability_factor == 1.0
        assert assessment.landmark_confidence_factor >= 0.95
        assert "optimal" in assessment.details.lower()

    def test_good_visibility_face(self):
        # Medium size face (75x75 px)
        crop = create_synthetic_face_crop(width=75, height=75, blur=False)
        bbox = BoundingBox(x1=100, y1=100, x2=175, y2=175)
        landmarks = create_canonical_landmarks(offset_x=100.0, offset_y=100.0)

        assessment = calculate_face_visibility(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(720, 1280),
            landmarks=landmarks,
            detection_confidence=0.90,
        )

        assert assessment.visibility in (FaceVisibility.GOOD, FaceVisibility.EXCELLENT)
        assert assessment.visibility_score >= 0.65

    def test_fair_visibility_face(self):
        # Small face with moderate blur and partial landmarks (3 landmarks)
        clarity = ClarityMetrics(
            sharpness=45.0,
            blur_score=0.48,
            blur_status=BlurStatus.SLIGHTLY_BLURRED,
        )
        illum = IlluminationMetrics(
            brightness=75.0,
            contrast=20.0,
            brightness_score=0.68,
            brightness_status=BrightnessStatus.ACCEPTABLE,
        )
        bbox = BoundingBox(x1=50, y1=50, x2=95, y2=95)  # 45x45 px
        partial_landmarks = [
            LandmarkPoint(name="left_eye", x=60.0, y=60.0),
            LandmarkPoint(name="right_eye", x=85.0, y=60.0),
            LandmarkPoint(name="nose", x=72.0, y=75.0),
        ]

        assessment = calculate_face_visibility(
            bbox=bbox,
            image_shape=(1080, 1920),
            clarity_metrics=clarity,
            illumination_metrics=illum,
            landmarks=partial_landmarks,
            detection_confidence=0.75,
        )

        assert assessment.visibility == FaceVisibility.FAIR
        assert 0.45 <= assessment.visibility_score < 0.65
        assert assessment.landmark_availability_factor == 0.60  # 3/5




    def test_poor_visibility_pitch_black(self):
        # Completely dark crop (luminance = 10)
        dark_crop = np.full((120, 120, 3), 10, dtype=np.uint8)
        bbox = BoundingBox(x1=50, y1=50, x2=170, y2=170)
        landmarks = create_canonical_landmarks(offset_x=50.0, offset_y=50.0)

        assessment = calculate_face_visibility(
            face_crop_rgb=dark_crop,
            bbox=bbox,
            image_shape=(720, 1280),
            landmarks=landmarks,
            detection_confidence=0.85,
        )

        assert assessment.visibility == FaceVisibility.POOR
        assert assessment.visibility_score < 0.45
        assert assessment.brightness_factor <= 0.50
        assert "illumination" in assessment.details.lower()

    def test_poor_visibility_heavily_blurred(self):
        # Heavily blurred crop
        crop = create_synthetic_face_crop(width=100, height=100, blur=True)
        # Apply additional strong blur
        blurred_crop = cv2.GaussianBlur(crop, (25, 25), 10.0)
        bbox = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        landmarks = create_canonical_landmarks(offset_x=50.0, offset_y=50.0)

        assessment = calculate_face_visibility(
            face_crop_rgb=blurred_crop,
            bbox=bbox,
            image_shape=(720, 1280),
            landmarks=landmarks,
            detection_confidence=0.85,
        )

        assert assessment.visibility == FaceVisibility.POOR
        assert assessment.visibility_score < 0.45
        assert assessment.blur_factor <= 0.35
        assert "blur" in assessment.details.lower()

    def test_poor_visibility_tiny_scale(self):
        # Tiny 16x16 pixel face
        crop = np.full((16, 16, 3), 128, dtype=np.uint8)
        bbox = BoundingBox(x1=10, y1=10, x2=26, y2=26)

        assessment = calculate_face_visibility(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(1080, 1920),
            landmarks=None,
            detection_confidence=0.75,
        )

        assert assessment.visibility == FaceVisibility.POOR
        assert assessment.size_factor <= 0.30
        assert "resolution" in assessment.details.lower() or "tiny" in assessment.details.lower()

    def test_landmark_confidence_explicit_vs_fallback(self):
        crop = create_synthetic_face_crop(width=120, height=120)
        bbox = BoundingBox(x1=50, y1=50, x2=170, y2=170)

        # Case 1: Explicit landmark confidences
        lms_explicit = [
            LandmarkPoint(name="left_eye", x=80.0, y=80.0, confidence=0.90),
            LandmarkPoint(name="right_eye", x=140.0, y=80.0, confidence=0.80),
        ]
        ass_explicit = calculate_face_visibility(
            face_crop_rgb=crop, bbox=bbox, image_shape=(640, 640), landmarks=lms_explicit
        )
        assert ass_explicit.landmark_confidence_factor == 0.85

        # Case 2: Landmarks present without confidence scores -> falls back to detection_confidence
        lms_no_conf = [
            LandmarkPoint(name="left_eye", x=80.0, y=80.0, confidence=None),
            LandmarkPoint(name="right_eye", x=140.0, y=80.0, confidence=None),
        ]
        ass_fallback = calculate_face_visibility(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(640, 640),
            landmarks=lms_no_conf,
            detection_confidence=0.92,
        )
        assert ass_fallback.landmark_confidence_factor == 0.92

        # Case 3: No landmarks at all -> falls back to halved detection confidence
        ass_none = calculate_face_visibility(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(640, 640),
            landmarks=[],
            detection_confidence=0.80,
        )
        assert ass_none.landmark_confidence_factor == 0.40

    def test_configurable_visibility_thresholds(self):
        crop = create_synthetic_face_crop(width=120, height=120)
        bbox = BoundingBox(x1=50, y1=50, x2=170, y2=170)
        landmarks = create_canonical_landmarks(offset_x=50.0, offset_y=50.0)

        # Baseline
        base = calculate_face_visibility(
            face_crop_rgb=crop, bbox=bbox, image_shape=(640, 640), landmarks=landmarks
        )

        # Overly strict thresholds demote it
        strict = calculate_face_visibility(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(640, 640),
            landmarks=landmarks,
            excellent_threshold=base.visibility_score + 0.05,
            good_threshold=base.visibility_score + 0.04,
            fair_threshold=base.visibility_score + 0.02,
        )
        assert strict.visibility == FaceVisibility.POOR


        # Lenient thresholds elevate lower scores
        lenient = calculate_face_visibility(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(640, 640),
            landmarks=landmarks,
            excellent_threshold=0.50,
            good_threshold=0.40,
            fair_threshold=0.30,
        )
        assert lenient.visibility == FaceVisibility.EXCELLENT

    def test_integration_in_structured_face_analysis(self):
        crop = create_synthetic_face_crop(width=140, height=140)
        bbox = BoundingBox(x1=80, y1=80, x2=220, y2=220)
        landmarks = create_canonical_landmarks(offset_x=80.0, offset_y=80.0)

        analysis = analyze_face_comprehensive(
            face_crop_rgb=crop,
            bbox=bbox,
            image_shape=(720, 1280),
            confidence=0.97,
            landmarks=landmarks,
            face_id="vis_test_1",
        )

        assert isinstance(analysis.visibility, FaceVisibility)
        assert analysis.visibility in (FaceVisibility.EXCELLENT, FaceVisibility.GOOD)
        assert analysis.visibility_assessment is not None
        assert analysis.visibility_assessment.visibility == analysis.visibility
        assert analysis.quality.visibility == analysis.visibility
        assert analysis.quality.visibility_score == analysis.visibility_assessment.visibility_score

        # Verify JSON dump includes "visibility" and "visibility_assessment"
        dumped = analysis.model_dump()
        assert "visibility" in dumped
        assert dumped["visibility"] in ("excellent", "good", "fair", "poor")
        assert "visibility_assessment" in dumped
        assert "visibility_score" in dumped["visibility_assessment"]


