"""
DeepVision AI — Error Handling & Fault Isolation Resilience Test Suite (Step 20)
Verifies that individual sub-component failures NEVER fail the entire face detection request:
1. Landmark model unavailable / disabled -> Detection + Quality SUCCESS, Landmarks None, Pose UNAVAILABLE.
2. Landmark inference / extraction failure -> Detection + Quality SUCCESS, Landmarks None, Pose UNAVAILABLE.
3. Invalid face crop (e.g. 0-pixel width/height, out of bounds) -> Safe empty crop handling, no crash.
4. Missing landmarks -> Detection + Quality SUCCESS, Landmarks None, Pose UNAVAILABLE.
5. Pose estimation failure (solvePnP failure / collinear points) -> Detection + Landmarks + Quality SUCCESS, Pose UNAVAILABLE.
6. Quality analysis failure -> Detection + Landmarks + Pose SUCCESS, Quality fallback.
7. Multi-face batch fault isolation -> Anomaly in one face never crashes the batch.
8. API endpoints (/api/v1/ai/detect and /api/v1/ai/analyze) return HTTP 200 with partial results.
"""

import io
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from PIL import Image
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas.ai import (
    BoundingBox,
    FaceBoundingBox,
    LandmarkPoint,
    DetectedFaceItem,
    FaceDetectionQuality,
)
from app.ai.detection.schemas import FaceDetection, FrameInferenceResult
from app.services.face_analysis_service import FaceAnalysisService, get_face_analysis_service
from app.services.ai_service import AIService
from app.ai.analysis.pose import get_pose_estimation_status, estimate_head_pose
from app.ai.analysis.quality import analyze_face_comprehensive, FaceAnalyzer


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def synthetic_rgb_image() -> np.ndarray:
    """Creates a 640x480 RGB image canvas."""
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Add a mock face pattern
    cv2_rect = img[100:300, 150:350]
    cv2_rect[:] = np.random.randint(80, 180, size=cv2_rect.shape, dtype=np.uint8)
    return img


@pytest.fixture
def valid_5_landmarks() -> list[LandmarkPoint]:
    """Standard 5 canonical facial landmark points."""
    return [
        LandmarkPoint(name="right_eye", x=200.0, y=160.0, confidence=0.98),
        LandmarkPoint(name="left_eye", x=280.0, y=160.0, confidence=0.98),
        LandmarkPoint(name="nose_tip", x=240.0, y=210.0, confidence=0.95),
        LandmarkPoint(name="mouth_right", x=210.0, y=250.0, confidence=0.92),
        LandmarkPoint(name="mouth_left", x=270.0, y=250.0, confidence=0.92),
    ]


@pytest.fixture
def test_jpeg_bytes() -> bytes:
    """Generates a valid JPEG byte payload."""
    img = Image.new("RGB", (640, 480), color=(120, 140, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Test 1: Landmark Model Unavailable or Disabled
# -----------------------------------------------------------------------------

def test_landmark_model_unavailable_or_disabled(synthetic_rgb_image):
    """
    When landmarks are disabled or unavailable:
    - Detection: SUCCESS
    - Landmarks: UNAVAILABLE (None)
    - Blur: SUCCESS
    - Brightness: SUCCESS
    - Pose: UNAVAILABLE (None)
    """
    service = FaceAnalysisService(enable_landmarks=False)
    detection = FaceDetection(
        face_id=0,
        bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
        confidence=0.97,
        landmarks=None,
    )

    item = service.analyze_face_detection_item(
        image_rgb=synthetic_rgb_image,
        detection=detection,
        image_shape=(480, 640),
    )

    assert item.confidence == 0.97
    assert item.landmarks is None
    assert item.pose is None
    assert item.quality is not None
    assert item.quality.blur_score >= 0.0
    assert item.quality.brightness_score >= 0.0
    assert item.quality.contrast_score >= 0.0
    assert item.quality.classification in ("excellent", "good", "acceptable", "poor")


# -----------------------------------------------------------------------------
# Test 2: Landmark Inference Failure
# -----------------------------------------------------------------------------

def test_landmark_inference_failure_gracefully_degraded(synthetic_rgb_image):
    """
    If landmark inference/parsing raises an unexpected exception:
    - Detection: SUCCESS
    - Landmarks: UNAVAILABLE (None)
    - Quality: SUCCESS
    - Pose: UNAVAILABLE (None)
    """
    service = FaceAnalysisService(enable_landmarks=True)
    detection = FaceDetection(
        face_id=1,
        bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
        confidence=0.95,
        landmarks=[LandmarkPoint(name="corrupted", x=float("nan"), y=float("inf"))],
    )

    with patch.object(service, "extract_and_validate_landmarks", side_effect=ValueError("Simulated landmark parsing failure")):
        item = service.analyze_face_detection_item(
            image_rgb=synthetic_rgb_image,
            detection=detection,
            image_shape=(480, 640),
        )

    assert item.confidence == 0.95
    assert item.landmarks is None
    assert item.pose is None
    assert item.quality is not None
    assert item.quality.blur_score >= 0.0


# -----------------------------------------------------------------------------
# Test 3: Invalid Face Crop Handling
# -----------------------------------------------------------------------------

def test_invalid_face_crop_does_not_crash(synthetic_rgb_image, valid_5_landmarks):
    """
    Invalid face crop (e.g., x2 <= x1, y2 <= y1, or completely outside canvas):
    - Handled via safe empty array.
    - Does not crash the detection or landmarks or pose.
    - Quality marks face as unusable without crashing.
    """
    service = FaceAnalysisService()
    # Degenerate inverted bbox
    detection = FaceDetection(
        face_id=2,
        bbox=BoundingBox(x1=350.0, y1=300.0, x2=150.0, y2=100.0),  # x2 < x1, y2 < y1
        confidence=0.90,
        landmarks=valid_5_landmarks,
    )

    item = service.analyze_face_detection_item(
        image_rgb=synthetic_rgb_image,
        detection=detection,
        image_shape=(480, 640),
    )

    assert item is not None
    assert item.confidence == 0.90
    assert item.landmarks is not None  # Landmarks are preserved!
    assert item.quality is not None
    assert item.quality.is_usable is False
    assert item.quality.score == 0.0 or item.quality.classification == "poor"


# -----------------------------------------------------------------------------
# Test 4: Missing Landmarks
# -----------------------------------------------------------------------------

def test_missing_landmarks_returns_partial_results(synthetic_rgb_image):
    """
    When landmarks are None or empty:
    - Detection: SUCCESS
    - Blur: SUCCESS
    - Brightness: SUCCESS
    - Pose: UNAVAILABLE (None)
    """
    service = FaceAnalysisService()
    detection = FaceDetection(
        face_id=3,
        bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
        confidence=0.96,
        landmarks=None,
    )

    item = service.analyze_face_detection_item(
        image_rgb=synthetic_rgb_image,
        detection=detection,
        image_shape=(480, 640),
    )

    assert item.landmarks is None
    assert item.pose is None
    assert item.quality is not None
    assert item.quality.blur_status in ("sharp", "slightly_blurred", "blurry")
    assert item.quality.brightness_status in ("acceptable", "too_dark", "too_bright")


# -----------------------------------------------------------------------------
# Test 5: Pose Estimation Failure Returns Partial Results
# -----------------------------------------------------------------------------

def test_pose_estimation_failure_returns_partial_results(synthetic_rgb_image):
    """
    When solvePnP or pose estimation fails (e.g. collinear points or exception):
    - Detection: SUCCESS
    - Landmarks: SUCCESS
    - Blur: SUCCESS
    - Brightness: SUCCESS
    - Pose: UNAVAILABLE (None)
    """
    service = FaceAnalysisService()
    # Degenerate collinear points where solvePnP fails
    collinear_landmarks = [
        LandmarkPoint(name="right_eye", x=200.0, y=200.0, confidence=0.9),
        LandmarkPoint(name="left_eye", x=200.0, y=200.0, confidence=0.9),
        LandmarkPoint(name="nose_tip", x=200.0, y=200.0, confidence=0.9),
        LandmarkPoint(name="mouth_right", x=200.0, y=200.0, confidence=0.9),
        LandmarkPoint(name="mouth_left", x=200.0, y=200.0, confidence=0.9),
    ]
    detection = FaceDetection(
        face_id=4,
        bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
        confidence=0.94,
        landmarks=collinear_landmarks,
    )

    item = service.analyze_face_detection_item(
        image_rgb=synthetic_rgb_image,
        detection=detection,
        image_shape=(480, 640),
    )

    assert item.confidence == 0.94
    assert item.landmarks is not None
    assert item.pose is None  # Pose is UNAVAILABLE
    assert item.quality is not None
    assert item.quality.blur_score >= 0.0
    assert item.quality.brightness_score >= 0.0

    # Also test get_pose_estimation_status documentation
    pose_obj, failure_reason = get_pose_estimation_status(collinear_landmarks, (480, 640))
    assert pose_obj is None
    assert failure_reason is not None
    assert "degenerate_geometry" in failure_reason


def test_pose_exception_caught_cleanly(synthetic_rgb_image, valid_5_landmarks):
    """Even if estimate_pose raises an unexpected exception, request succeeds."""
    service = FaceAnalysisService()
    detection = FaceDetection(
        face_id=5,
        bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
        confidence=0.99,
        landmarks=valid_5_landmarks,
    )

    with patch.object(service, "estimate_pose", side_effect=RuntimeError("Simulated OpenCV solvePnP crash")):
        item = service.analyze_face_detection_item(
            image_rgb=synthetic_rgb_image,
            detection=detection,
            image_shape=(480, 640),
        )

    assert item.confidence == 0.99
    assert item.landmarks is not None
    assert item.pose is None  # Pose cleanly None
    assert item.quality is not None


# -----------------------------------------------------------------------------
# Test 6: Quality Analysis Failure Isolation
# -----------------------------------------------------------------------------

def test_quality_analysis_failure_isolation(synthetic_rgb_image, valid_5_landmarks):
    """
    If calculate_quality raises an unexpected exception:
    - Detection: SUCCESS
    - Landmarks: SUCCESS
    - Pose: SUCCESS
    - Quality: Fallback / safe defaults
    """
    service = FaceAnalysisService()
    detection = FaceDetection(
        face_id=6,
        bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
        confidence=0.95,
        landmarks=valid_5_landmarks,
    )

    with patch.object(service, "calculate_quality", side_effect=Exception("Simulated scoring error")):
        item = service.analyze_face_detection_item(
            image_rgb=synthetic_rgb_image,
            detection=detection,
            image_shape=(480, 640),
        )

    assert item.confidence == 0.95
    assert item.landmarks is not None
    assert item.pose is not None
    assert item.quality is not None
    assert item.quality.score == 0.0
    assert item.quality.classification == "poor"


# -----------------------------------------------------------------------------
# Test 7: Multi-Face Batch Fault Isolation
# -----------------------------------------------------------------------------

def test_multi_face_batch_fault_isolation(synthetic_rgb_image, valid_5_landmarks):
    """
    A batch containing 3 faces:
    - Face 1: Standard valid face.
    - Face 2: Corrupted face with invalid crop and collinear landmarks.
    - Face 3: Missing landmarks.
    Ensures that an anomaly in Face 2 does NOT crash Face 1 or Face 3!
    """
    service = FaceAnalysisService()

    collinear_lms = [
        LandmarkPoint(name="right_eye", x=10.0, y=10.0),
        LandmarkPoint(name="left_eye", x=10.0, y=10.0),
        LandmarkPoint(name="nose_tip", x=10.0, y=10.0),
        LandmarkPoint(name="mouth_right", x=10.0, y=10.0),
        LandmarkPoint(name="mouth_left", x=10.0, y=10.0),
    ]

    detections = [
        FaceDetection(face_id=0, bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0), confidence=0.98, landmarks=valid_5_landmarks),
        FaceDetection(face_id=1, bbox=BoundingBox(x1=300.0, y1=300.0, x2=200.0, y2=200.0), confidence=0.85, landmarks=collinear_lms),
        FaceDetection(face_id=2, bbox=BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=150.0), confidence=0.92, landmarks=None),
    ]

    results = service.analyze_detected_faces(
        image_rgb=synthetic_rgb_image,
        detections=detections,
        image_shape=(480, 640),
    )

    assert len(results) == 3

    # Face 0: Full results
    assert results[0].landmarks is not None
    assert results[0].pose is not None
    assert results[0].quality is not None

    # Face 1: Degenerate -> landmarks present, pose None, quality poor
    assert results[1].landmarks is not None
    assert results[1].pose is None
    assert results[1].quality is not None

    # Face 2: Missing landmarks -> landmarks None, pose None, quality valid
    assert results[2].landmarks is None
    assert results[2].pose is None
    assert results[2].quality is not None


# -----------------------------------------------------------------------------
# Test 8: End-to-End API /api/v1/ai/detect Partial Results
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_api_detect_returns_partial_results_when_pose_unavailable(test_jpeg_bytes):
    """
    POST /api/v1/ai/detect returns HTTP 200 with partial results
    even when pose estimation cannot resolve on a detected face.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Mock detector to return a face with degenerate landmarks (collinear)
        collinear_lms = [
            LandmarkPoint(name="right_eye", x=200.0, y=200.0, confidence=0.9),
            LandmarkPoint(name="left_eye", x=200.0, y=200.0, confidence=0.9),
            LandmarkPoint(name="nose_tip", x=200.0, y=200.0, confidence=0.9),
            LandmarkPoint(name="mouth_right", x=200.0, y=200.0, confidence=0.9),
            LandmarkPoint(name="mouth_left", x=200.0, y=200.0, confidence=0.9),
        ]
        mock_detection = FaceDetection(
            face_id=0,
            bbox=BoundingBox(x1=100.0, y1=100.0, x2=300.0, y2=300.0),
            confidence=0.97,
            landmarks=collinear_lms,
        )
        mock_frame = FrameInferenceResult(
            image_width=640,
            image_height=480,
            detections=[mock_detection],
            inference_time_ms=12.5,
            processing_time_ms=12.5,
            device="CPU",
            model_name="test-detector",
        )

        with patch("app.services.ai_service.AIService.detect_faces", return_value=mock_frame):
            files = {"file": ("test.jpg", test_jpeg_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["face_count"] == 1
        face = data["faces"][0]
        assert face["confidence"] == 0.97
        assert face["landmarks"] is not None
        assert face["quality"] is not None
        # Crucial requirement: Pose is None (null) and request succeeded with partial results!
        assert face["pose"] is None


@pytest.mark.anyio
async def test_api_detect_returns_partial_results_when_landmarks_missing(test_jpeg_bytes):
    """
    POST /api/v1/ai/detect returns HTTP 200 with partial results
    when landmarks are completely missing from the detection model.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        mock_detection = FaceDetection(
            face_id=0,
            bbox=BoundingBox(x1=100.0, y1=100.0, x2=300.0, y2=300.0),
            confidence=0.95,
            landmarks=None,
        )
        mock_frame = FrameInferenceResult(
            image_width=640,
            image_height=480,
            detections=[mock_detection],
            inference_time_ms=10.0,
            processing_time_ms=10.0,
            device="CPU",
            model_name="test-detector",
        )

        with patch("app.services.ai_service.AIService.detect_faces", return_value=mock_frame):
            files = {"file": ("test.jpg", test_jpeg_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["face_count"] == 1
        face = data["faces"][0]
        assert face["confidence"] == 0.95
        assert face["landmarks"] is None
        assert face["pose"] is None
        assert face["quality"] is not None
        assert face["quality"]["blur_score"] >= 0.0
        assert face["quality"]["brightness_score"] >= 0.0


# -----------------------------------------------------------------------------
# Test 9: End-to-End API /api/v1/ai/analyze Partial Results
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_api_analyze_returns_partial_results_on_pose_failure(test_jpeg_bytes):
    """
    POST /api/v1/ai/analyze returns HTTP 200 with partial results
    when pose estimation cannot resolve.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        collinear_lms = [
            LandmarkPoint(name="right_eye", x=200.0, y=200.0, confidence=0.9),
            LandmarkPoint(name="left_eye", x=200.0, y=200.0, confidence=0.9),
            LandmarkPoint(name="nose_tip", x=200.0, y=200.0, confidence=0.9),
            LandmarkPoint(name="mouth_right", x=200.0, y=200.0, confidence=0.9),
            LandmarkPoint(name="mouth_left", x=200.0, y=200.0, confidence=0.9),
        ]
        mock_detection = FaceDetection(
            face_id=0,
            bbox=BoundingBox(x1=100.0, y1=100.0, x2=300.0, y2=300.0),
            confidence=0.96,
            landmarks=collinear_lms,
        )
        mock_frame = FrameInferenceResult(
            image_width=640,
            image_height=480,
            detections=[mock_detection],
            inference_time_ms=15.0,
            processing_time_ms=15.0,
            device="CPU",
            model_name="test-detector",
        )

        with patch("app.services.ai_service.AIService.detect_faces", return_value=mock_frame):
            files = {"file": ("test.jpg", test_jpeg_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/analyze", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["face_count"] == 1
        face = data["faces"][0]
        assert face["pose"] is None
        assert face["quality"] is not None
        assert face["quality"]["quality_score"] >= 0.0
