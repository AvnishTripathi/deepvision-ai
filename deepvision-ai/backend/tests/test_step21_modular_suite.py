"""
DeepVision AI — Comprehensive Feature Requirements Test Suite (Step 21)

Validates all required verification modules without any GPU dependency (CPU/Mock only):
1. Landmarks:
   - Valid landmarks
   - Coordinate validation (clipping and boundaries)
   - Missing landmarks
2. Blur:
   - Sharp image
   - Blurred image
3. Brightness:
   - Dark image
   - Normal image
   - Bright image
4. Contrast:
   - Low contrast
   - Normal contrast
5. Face Size:
   - Small face
   - Large face
6. Quality:
   - Quality score range 0–100
   - Classification thresholds
7. Pose:
   - Valid pose
   - Unavailable pose
8. API:
   - One face
   - Multiple faces
   - No face
   - Invalid image
   - Partial analysis failure
"""

import io
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import pytest
from PIL import Image
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas.ai import (
    BoundingBox,
    FaceBoundingBox,
    LandmarkPoint,
    LandmarkCoordinate,
    DetectedFaceItem,
    FaceDetectionQuality,
)
from app.ai.detection.schemas import FaceDetection, FrameInferenceResult
from app.ai.analysis.schemas import (
    BlurStatus,
    BrightnessStatus,
    ExposureCategory,
    SizeCategory,
    QualityClassification,
    QualityGrade,
    PoseCategory,
)
from app.ai.analysis.landmarks import (
    compute_landmark_metrics,
    validate_and_clip_landmarks,
    validate_and_clip_landmarks_list,
    landmarks_to_dict,
)
from app.ai.analysis.metrics import (
    compute_clarity_metrics,
    compute_illumination_metrics,
    compute_face_size_metrics,
)
from app.ai.analysis.quality import (
    calculate_face_quality_score,
    classify_face_quality,
    analyze_face_comprehensive,
)
from app.ai.analysis.pose import (
    estimate_head_pose,
    get_pose_estimation_status,
)
from app.services.face_analysis_service import FaceAnalysisService


# =============================================================================
# Helper Utilities & Fixtures (CPU Synthetic Generation)
# =============================================================================

def make_canonical_5_landmarks(offset_x: float = 100.0, offset_y: float = 100.0) -> list[LandmarkPoint]:
    """Generates 5 realistic facial landmarks for testing."""
    return [
        LandmarkPoint(name="right_eye", x=offset_x + 35.0, y=offset_y + 40.0, confidence=0.98),
        LandmarkPoint(name="left_eye", x=offset_x + 85.0, y=offset_y + 40.0, confidence=0.98),
        LandmarkPoint(name="nose_tip", x=offset_x + 60.0, y=offset_y + 65.0, confidence=0.95),
        LandmarkPoint(name="mouth_right", x=offset_x + 42.0, y=offset_y + 90.0, confidence=0.92),
        LandmarkPoint(name="mouth_left", x=offset_x + 78.0, y=offset_y + 90.0, confidence=0.92),
    ]


def make_synthetic_face_crop(width: int = 120, height: int = 120, blur_sigma: float = 0.0) -> np.ndarray:
    """Generates synthetic textured face crop on CPU."""
    img = np.full((height, width, 3), 150, dtype=np.uint8)
    # Add contrast features for face-like texture
    img[25:45, 20:45] = [30, 30, 30]     # Right eye
    img[25:45, 75:100] = [30, 30, 30]    # Left eye
    img[50:70, 50:70] = [80, 50, 50]     # Nose
    img[80:95, 35:85] = [170, 40, 50]    # Mouth
    # Add high-frequency noise
    noise = np.random.RandomState(42).randint(-25, 25, (height, width, 3))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if blur_sigma > 0.0:
        k = int(round(blur_sigma * 4)) | 1
        img = cv2.GaussianBlur(img, (k, k), blur_sigma)
    return img


def make_jpeg_bytes(width: int = 640, height: int = 480, color: tuple = (120, 130, 140)) -> bytes:
    """Generates a valid JPEG byte stream in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# =============================================================================
# 1. Landmarks Tests
# =============================================================================

class TestLandmarks:
    """Validates landmark parsing, bounds clipping, and missing landmark handling."""

    def test_valid_landmarks(self):
        """Verifies parsing and geometric metric calculation for valid landmarks."""
        lms = make_canonical_5_landmarks(100.0, 100.0)
        metrics = compute_landmark_metrics(lms)

        assert metrics is not None
        assert metrics.interpupillary_distance_px == 50.0  # 185 - 135
        assert 0.0 <= metrics.symmetry_ratio <= 1.0
        assert abs(metrics.eye_tilt_angle_deg) < 1.0  # Level eyes

        # Dictionary format
        lms_dict = landmarks_to_dict(lms)
        assert len(lms_dict) == 5
        assert "right_eye" in lms_dict
        assert "left_eye" in lms_dict
        assert lms_dict["right_eye"].x == 135.0
        assert lms_dict["right_eye"].y == 140.0

    def test_coordinate_validation(self):
        """Verifies that out-of-bounds coordinates are safely clipped to canvas limits."""
        out_of_bounds = [
            LandmarkPoint(name="right_eye", x=-50.0, y=100.0),      # Negative x
            LandmarkPoint(name="left_eye", x=700.0, y=100.0),       # x > width (640)
            LandmarkPoint(name="nose_tip", x=300.0, y=-20.0),       # Negative y
            LandmarkPoint(name="mouth_right", x=250.0, y=550.0),    # y > height (480)
            LandmarkPoint(name="mouth_left", x=350.0, y=400.0),
        ]
        clipped_list = validate_and_clip_landmarks_list(out_of_bounds, image_width=640, image_height=480)
        clipped_dict = validate_and_clip_landmarks(out_of_bounds, image_width=640, image_height=480)

        assert len(clipped_list) == 5
        assert clipped_list[0].x == 0.0
        assert clipped_list[1].x == 640.0
        assert clipped_list[2].y == 0.0
        assert clipped_list[3].y == 480.0
        assert clipped_list[4].x == 350.0

        # Dict representation
        assert clipped_dict["right_eye"].x == 0.0
        assert clipped_dict["left_eye"].x == 640.0
        assert "nose" in clipped_dict or "nose_tip" in clipped_dict
        nose_key = "nose" if "nose" in clipped_dict else "nose_tip"
        assert clipped_dict[nose_key].y == 0.0

    def test_missing_landmarks(self):
        """Verifies graceful degradation when landmarks are None or empty."""
        service = FaceAnalysisService()

        # None landmarks
        d_none, l_none = service.extract_and_validate_landmarks(None, (480, 640))
        assert d_none == {}
        assert l_none == []

        # Empty landmarks list
        d_empty, l_empty = service.extract_and_validate_landmarks([], (480, 640))
        assert d_empty == {}
        assert l_empty == []

        # Metric calculation on empty list safely returns None
        assert compute_landmark_metrics([]) is None


# =============================================================================
# 2. Blur Tests
# =============================================================================

class TestBlur:
    """Validates sharpness and blur detection via Laplacian variance."""

    def test_sharp_image(self):
        """Verifies sharp image detection with high Laplacian variance."""
        sharp_crop = make_synthetic_face_crop(width=160, height=160, blur_sigma=0.0)
        clarity = compute_clarity_metrics(sharp_crop, min_sharpness_threshold=50.0)

        assert clarity.sharpness >= 50.0
        assert clarity.blur_status in (BlurStatus.SHARP, BlurStatus.SLIGHTLY_BLURRED)
        assert clarity.blur_score > 0.40

    def test_blurred_image(self):
        """Verifies blurred image detection with low Laplacian variance."""
        blurred_crop = make_synthetic_face_crop(width=160, height=160, blur_sigma=8.0)
        clarity = compute_clarity_metrics(blurred_crop, min_sharpness_threshold=50.0)

        assert clarity.sharpness < 50.0
        assert clarity.blur_status == BlurStatus.BLURRY
        assert clarity.blur_score < 0.35


# =============================================================================
# 3. Brightness Tests
# =============================================================================

class TestBrightness:
    """Validates photopic illumination and exposure classification."""

    def test_dark_image(self):
        """Verifies underexposed dark image detection (luminance < min_acceptable)."""
        dark_crop = np.full((100, 100, 3), 25, dtype=np.uint8)  # Mean luminance = 25
        illum = compute_illumination_metrics(dark_crop, min_acceptable=60.0, max_acceptable=200.0)

        assert illum.brightness == 25.0
        assert illum.brightness_status == BrightnessStatus.TOO_DARK
        assert illum.exposure_category == ExposureCategory.UNDEREXPOSED
        assert illum.brightness_score < 0.50

    def test_normal_image(self):
        """Verifies properly exposed normal image detection (60.0 <= luminance <= 200.0)."""
        normal_crop = np.full((100, 100, 3), 130, dtype=np.uint8)
        # Add slight natural facial contrast variations
        normal_crop[20:80, 20:80] = 160
        normal_crop[40:60, 40:60] = 95
        illum = compute_illumination_metrics(normal_crop, min_acceptable=60.0, max_acceptable=200.0)

        assert 60.0 <= illum.brightness <= 200.0
        assert illum.brightness_status == BrightnessStatus.ACCEPTABLE
        assert illum.exposure_category == ExposureCategory.OPTIMAL
        assert illum.brightness_score >= 0.85

    def test_bright_image(self):
        """Verifies overexposed bright image detection (luminance > max_acceptable)."""
        bright_crop = np.full((100, 100, 3), 235, dtype=np.uint8)  # Mean luminance = 235
        illum = compute_illumination_metrics(bright_crop, min_acceptable=60.0, max_acceptable=200.0)

        assert illum.brightness == 235.0
        assert illum.brightness_status == BrightnessStatus.TOO_BRIGHT
        assert illum.exposure_category == ExposureCategory.OVEREXPOSED
        assert illum.brightness_score < 0.50


# =============================================================================
# 4. Contrast Tests
# =============================================================================

class TestContrast:
    """Validates luminance standard deviation dynamic range diagnostics."""

    def test_low_contrast(self):
        """Verifies low contrast detection on washed-out / flat luminance scenes."""
        # Flat image with minimal standard deviation (< 5.0)
        low_c_crop = np.full((100, 100, 3), 128, dtype=np.uint8)
        illum = compute_illumination_metrics(low_c_crop)
        service = FaceAnalysisService()
        score, status = service.analyze_contrast(illum)

        assert score < 15.0
        assert status == "low_contrast"

    def test_normal_contrast(self):
        """Verifies normal contrast detection on balanced scenes with rich dynamic range."""
        textured_crop = make_synthetic_face_crop(width=120, height=120)
        illum = compute_illumination_metrics(textured_crop)
        service = FaceAnalysisService()
        score, status = service.analyze_contrast(illum)

        assert 16.0 <= score <= 75.0
        assert status == "acceptable"


# =============================================================================
# 5. Face Size Tests
# =============================================================================

class TestFaceSize:
    """Validates scale, resolution, and biometric sufficiency diagnostics."""

    def test_small_face(self):
        """Verifies small face detection (< 32px or < 80px)."""
        tiny_bbox = BoundingBox(x1=100.0, y1=100.0, x2=124.0, y2=124.0)  # 24x24 px
        size_metrics = compute_face_size_metrics(tiny_bbox, image_shape=(480, 640))

        assert size_metrics.face_width == 24.0
        assert size_metrics.face_height == 24.0
        assert size_metrics.size_category == SizeCategory.TINY
        assert size_metrics.is_sufficient_size is False

        # Small category (50x50 px)
        small_bbox = BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=150.0)  # 50x50 px
        size_small = compute_face_size_metrics(small_bbox, image_shape=(480, 640))
        assert size_small.size_category == SizeCategory.SMALL
        assert size_small.is_sufficient_size is True

    def test_large_face(self):
        """Verifies large high-resolution face detection (> 200px)."""
        large_bbox = BoundingBox(x1=50.0, y1=50.0, x2=300.0, y2=300.0)  # 250x250 px
        size_metrics = compute_face_size_metrics(large_bbox, image_shape=(480, 640))

        assert size_metrics.face_width == 250.0
        assert size_metrics.face_height == 250.0
        assert size_metrics.size_category == SizeCategory.LARGE
        assert size_metrics.is_sufficient_size is True
        assert size_metrics.face_area_ratio > 0.15


# =============================================================================
# 6. Quality Tests
# =============================================================================

class TestQuality:
    """Validates transparent 0-100 quality scoring and configurable classification."""

    def test_quality_score_range_0_to_100(self):
        """Verifies that the transparent quality score is strictly bounded in [0.0, 100.0]."""
        test_crops = [
            make_synthetic_face_crop(width=160, height=160, blur_sigma=0.0),  # Sharp
            make_synthetic_face_crop(width=160, height=160, blur_sigma=12.0), # Blurred
            np.full((80, 80, 3), 15, dtype=np.uint8),                          # Dark
            np.full((80, 80, 3), 245, dtype=np.uint8),                         # Bright
            np.empty((0, 0, 3), dtype=np.uint8),                               # Empty
        ]

        bbox = BoundingBox(x1=50.0, y1=50.0, x2=200.0, y2=200.0)
        lms = make_canonical_5_landmarks(50.0, 50.0)

        for crop in test_crops:
            score, factors, grade = calculate_face_quality_score(
                face_crop_rgb=crop,
                bbox=bbox,
                image_shape=(480, 640),
                landmarks=lms,
            )
            assert 0.0 <= score <= 100.0
            assert 0.0 <= factors.sharpness <= 100.0
            assert 0.0 <= factors.brightness <= 100.0
            assert 0.0 <= factors.contrast <= 100.0
            assert 0.0 <= factors.size <= 100.0
            assert 0.0 <= factors.landmark <= 100.0
            assert isinstance(grade, QualityGrade)

    def test_classification_thresholds(self):
        """Verifies standardized quality classification tiers against score thresholds."""
        assert classify_face_quality(88.5) == QualityClassification.EXCELLENT
        assert classify_face_quality(80.0) == QualityClassification.EXCELLENT
        assert classify_face_quality(75.0) == QualityClassification.GOOD
        assert classify_face_quality(65.0) == QualityClassification.GOOD
        assert classify_face_quality(55.0) == QualityClassification.ACCEPTABLE
        assert classify_face_quality(45.0) == QualityClassification.ACCEPTABLE
        assert classify_face_quality(40.0) == QualityClassification.POOR
        assert classify_face_quality(0.0) == QualityClassification.POOR

        # Configurable threshold customization
        assert classify_face_quality(70.0, excellent_threshold=70.0) == QualityClassification.EXCELLENT
        assert classify_face_quality(50.0, acceptable_threshold=55.0) == QualityClassification.POOR


# =============================================================================
# 7. Pose Tests
# =============================================================================

class TestPose:
    """Validates 3D Head Pose estimation (Yaw, Pitch, Roll) and unavailable state handling."""

    def test_valid_pose(self):
        """Verifies Euler angle orientation calculation on realistic landmarks."""
        lms = make_canonical_5_landmarks(200.0, 150.0)
        pose = estimate_head_pose(lms, image_shape=(480, 640))

        assert pose is not None
        assert isinstance(pose.yaw, float)
        assert isinstance(pose.pitch, float)
        assert isinstance(pose.roll, float)
        # Bounded physiological limits
        assert -90.0 <= pose.yaw <= 90.0
        assert -90.0 <= pose.pitch <= 90.0
        assert -90.0 <= pose.roll <= 90.0
        assert 0.0 <= pose.frontality_score <= 1.0
        assert pose.pose_category in list(PoseCategory)

        # Dictionary format
        angles = pose.to_angles_dict()
        assert "yaw" in angles and "pitch" in angles and "roll" in angles

    def test_unavailable_pose(self):
        """Verifies that degenerate geometry cleanly returns None rather than fake numbers."""
        service = FaceAnalysisService()

        # Missing landmarks
        pose_none, _ = service.estimate_pose([], (480, 640))
        assert pose_none is None

        # Less than 5 points
        partial_lms = [
            LandmarkPoint(name="right_eye", x=100.0, y=100.0),
            LandmarkPoint(name="left_eye", x=150.0, y=100.0),
        ]
        pose_partial, _ = service.estimate_pose(partial_lms, (480, 640))
        assert pose_partial is None

        # Degenerate collinear / collapsed points
        collapsed_lms = [
            LandmarkPoint(name="right_eye", x=100.0, y=100.0),
            LandmarkPoint(name="left_eye", x=100.0, y=100.0),
            LandmarkPoint(name="nose_tip", x=100.0, y=100.0),
            LandmarkPoint(name="mouth_right", x=100.0, y=100.0),
            LandmarkPoint(name="mouth_left", x=100.0, y=100.0),
        ]
        pose_collapsed, _ = service.estimate_pose(collapsed_lms, (480, 640))
        assert pose_collapsed is None

        # Documented failure reason
        pose_status, reason = get_pose_estimation_status(collapsed_lms, (480, 640))
        assert pose_status is None
        assert reason is not None
        assert "degenerate_geometry" in reason


# =============================================================================
# 8. API Tests
# =============================================================================

@pytest.mark.anyio
class TestAPI:
    """Validates HTTP API endpoints under single, multiple, empty, invalid, and partial scenarios."""

    async def test_api_one_face(self):
        """Verifies API detection response for a single detected face."""
        test_jpeg = make_jpeg_bytes()
        lms = make_canonical_5_landmarks(100.0, 100.0)
        mock_detection = FaceDetection(
            face_id=0,
            bbox=BoundingBox(x1=100.0, y1=100.0, x2=250.0, y2=250.0),
            confidence=0.98,
            landmarks=lms,
        )
        mock_frame = FrameInferenceResult(
            image_width=640,
            image_height=480,
            detections=[mock_detection],
            inference_time_ms=14.0,
            processing_time_ms=14.0,
            device="CPU",
            model_name="test-detector",
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch("app.services.ai_service.AIService.detect_faces", return_value=mock_frame):
                files = {"file": ("face.jpg", test_jpeg, "image/jpeg")}
                response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["face_count"] == 1
        assert len(data["faces"]) == 1
        face = data["faces"][0]
        assert face["confidence"] == 0.98
        assert face["bbox"]["x1"] == 100.0
        assert face["landmarks"] is not None
        assert face["quality"] is not None
        assert face["pose"] is not None

    async def test_api_multiple_faces(self):
        """Verifies API detection response for multiple detected faces."""
        test_jpeg = make_jpeg_bytes()
        detections = [
            FaceDetection(face_id=0, bbox=BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0), confidence=0.97, landmarks=make_canonical_5_landmarks(50.0, 50.0)),
            FaceDetection(face_id=1, bbox=BoundingBox(x1=200.0, y1=50.0, x2=300.0, y2=150.0), confidence=0.95, landmarks=make_canonical_5_landmarks(200.0, 50.0)),
            FaceDetection(face_id=2, bbox=BoundingBox(x1=350.0, y1=50.0, x2=450.0, y2=150.0), confidence=0.92, landmarks=make_canonical_5_landmarks(350.0, 50.0)),
        ]
        mock_frame = FrameInferenceResult(
            image_width=640,
            image_height=480,
            detections=detections,
            inference_time_ms=22.0,
            processing_time_ms=22.0,
            device="CPU",
            model_name="test-detector",
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch("app.services.ai_service.AIService.detect_faces", return_value=mock_frame):
                files = {"file": ("multi.jpg", test_jpeg, "image/jpeg")}
                response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["face_count"] == 3
        assert len(data["faces"]) == 3

    async def test_api_no_face(self):
        """Verifies API detection response when no faces are detected in the image."""
        test_jpeg = make_jpeg_bytes()
        mock_frame = FrameInferenceResult(
            image_width=640,
            image_height=480,
            detections=[],
            inference_time_ms=8.0,
            processing_time_ms=8.0,
            device="CPU",
            model_name="test-detector",
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch("app.services.ai_service.AIService.detect_faces", return_value=mock_frame):
                files = {"file": ("blank.jpg", test_jpeg, "image/jpeg")}
                response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["face_count"] == 0
        assert data["faces"] == []

    async def test_api_invalid_image(self):
        """Verifies that invalid or corrupted image payloads return appropriate client errors."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # 1. Non-image text payload
            files = {"file": ("corrupt.txt", b"this is not an image", "text/plain")}
            response = await client.post("/api/v1/ai/detect", files=files)
            assert response.status_code in (400, 422)

            # 2. Corrupted pseudo-image bytes
            files_bad = {"file": ("corrupt.jpg", b"\xff\xd8\xff\x00truncated_garbage", "image/jpeg")}
            resp_bad = await client.post("/api/v1/ai/detect", files=files_bad)
            assert resp_bad.status_code in (400, 422)

    async def test_api_partial_analysis_failure(self):
        """
        Verifies that partial analysis failure (e.g. pose estimation unavailable)
        returns HTTP 200 with partial results rather than failing the request.
        """
        test_jpeg = make_jpeg_bytes()
        # Degenerate collapsed landmarks where pose estimation is unavailable
        collapsed_lms = [
            LandmarkPoint(name="right_eye", x=100.0, y=100.0),
            LandmarkPoint(name="left_eye", x=100.0, y=100.0),
            LandmarkPoint(name="nose_tip", x=100.0, y=100.0),
            LandmarkPoint(name="mouth_right", x=100.0, y=100.0),
            LandmarkPoint(name="mouth_left", x=100.0, y=100.0),
        ]
        mock_detection = FaceDetection(
            face_id=0,
            bbox=BoundingBox(x1=100.0, y1=100.0, x2=250.0, y2=250.0),
            confidence=0.96,
            landmarks=collapsed_lms,
        )
        mock_frame = FrameInferenceResult(
            image_width=640,
            image_height=480,
            detections=[mock_detection],
            inference_time_ms=12.0,
            processing_time_ms=12.0,
            device="CPU",
            model_name="test-detector",
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch("app.services.ai_service.AIService.detect_faces", return_value=mock_frame):
                files = {"file": ("partial.jpg", test_jpeg, "image/jpeg")}
                response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["face_count"] == 1
        face = data["faces"][0]
        # Partial result: detection, landmarks, and quality succeed; pose is null
        assert face["confidence"] == 0.96
        assert face["landmarks"] is not None
        assert face["quality"] is not None
        assert face["pose"] is None
