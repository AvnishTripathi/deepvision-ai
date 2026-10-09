"""
DeepVision AI — Performance Latency Stage Measurement Test Suite (Step 18)
Validates granular, non-fabricated telemetry for separate pipeline execution stages:
  - detection_time_ms
  - landmark_time_ms
  - quality_analysis_time_ms
  - pose_time_ms
  - total_analysis_time_ms
"""

import io
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services.face_analysis_service import FaceAnalysisService, get_face_analysis_service
from app.services.ai_service import AIService, get_ai_service
from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.schemas import FaceDetection, FrameInferenceResult
from app.schemas.ai import (
    BoundingBox,
    LandmarkPoint,
    AnalysisPerformanceBreakdown,
)


@pytest.fixture
def synthetic_face_frame():
    """Generates a realistic 640x480 RGB frame containing a textured face region."""
    arr = np.ones((480, 640, 3), dtype=np.uint8) * 120
    np.random.seed(42)
    noise = np.random.randint(-20, 20, (200, 200, 3), dtype=np.int16)
    arr[100:300, 150:350] = np.clip(130 + noise, 0, 255).astype(np.uint8)
    return arr


@pytest.fixture
def canonical_landmarks():
    """5 standard facial landmark points."""
    return [
        LandmarkPoint(name="right_eye", x=200.0, y=170.0),
        LandmarkPoint(name="left_eye", x=300.0, y=170.0),
        LandmarkPoint(name="nose_tip", x=250.0, y=210.0),
        LandmarkPoint(name="mouth_right", x=210.0, y=260.0),
        LandmarkPoint(name="mouth_left", x=290.0, y=260.0),
    ]


@pytest.fixture
def valid_test_image_bytes():
    """Valid in-memory JPEG bytes."""
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# 1. FaceAnalysisService Granular Latency Unit Tests
# -----------------------------------------------------------------------------

def test_face_analysis_service_timing_breakdown_with_faces(synthetic_face_frame, canonical_landmarks):
    """
    Verifies that FaceAnalysisService accurately records all 5 separate stage metrics
    when analyzing faces.
    """
    service = FaceAnalysisService(min_sharpness=50.0)
    detection = FaceDetection(
        face_id=1,
        bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
        confidence=0.98,
        landmarks=canonical_landmarks,
    )

    items, timing = service.analyze_detected_faces_with_timing(
        image_rgb=synthetic_face_frame,
        detections=[detection],
        image_shape=(480, 640),
        detection_time_ms=15.4,
    )

    assert len(items) == 1
    assert isinstance(timing, AnalysisPerformanceBreakdown)
    assert timing.detection_time_ms == 15.4
    assert isinstance(timing.landmark_time_ms, float)
    assert timing.landmark_time_ms >= 0.0
    assert isinstance(timing.quality_analysis_time_ms, float)
    assert timing.quality_analysis_time_ms >= 0.0
    assert isinstance(timing.pose_time_ms, float)
    assert timing.pose_time_ms >= 0.0
    assert isinstance(timing.total_analysis_time_ms, float)
    assert timing.total_analysis_time_ms >= 0.0

    # Ensure self.last_timing matches returned breakdown
    assert service.last_timing is not None
    assert service.last_timing.landmark_time_ms == timing.landmark_time_ms
    assert service.last_timing.pose_time_ms == timing.pose_time_ms


def test_face_analysis_service_timing_breakdown_zero_faces(synthetic_face_frame):
    """
    Verifies that FaceAnalysisService handles 0 faces safely, reporting 0.0 for
    face-dependent stages while total analysis measures actual empty loop time.
    """
    service = FaceAnalysisService(min_sharpness=50.0)

    items, timing = service.analyze_detected_faces_with_timing(
        image_rgb=synthetic_face_frame,
        detections=[],
        image_shape=(480, 640),
        detection_time_ms=10.2,
    )

    assert len(items) == 0
    assert timing.detection_time_ms == 10.2
    assert timing.landmark_time_ms == 0.0
    assert timing.quality_analysis_time_ms == 0.0
    assert timing.pose_time_ms == 0.0
    assert timing.total_analysis_time_ms >= 0.0


def test_face_analysis_service_structured_detections_timing(synthetic_face_frame, canonical_landmarks):
    """
    Verifies analyze_structured_detections_with_timing measures all stage latencies
    for comprehensive analysis (/ai/analyze pipeline).
    """
    service = FaceAnalysisService(min_sharpness=50.0)
    detection = FaceDetection(
        face_id=1,
        bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
        confidence=0.98,
        landmarks=canonical_landmarks,
    )

    analyses, timing = service.analyze_structured_detections_with_timing(
        image_rgb=synthetic_face_frame,
        detections=[detection],
        image_shape=(480, 640),
        detection_time_ms=12.5,
    )

    assert len(analyses) == 1
    assert timing.detection_time_ms == 12.5
    assert timing.landmark_time_ms >= 0.0
    assert timing.quality_analysis_time_ms >= 0.0
    assert timing.pose_time_ms >= 0.0
    assert timing.total_analysis_time_ms >= 0.0


# -----------------------------------------------------------------------------
# 2. End-to-End API Route Tests (/ai/detect & /ai/analyze)
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_detect_endpoint_returns_separate_stage_latencies(valid_test_image_bytes, canonical_landmarks):
    """
    Step 18: Verifies POST /api/v1/ai/detect returns actual measured stage latencies:
      - detection_time_ms
      - landmark_time_ms
      - quality_analysis_time_ms
      - pose_time_ms
      - total_analysis_time_ms
    Both in response.timing and at top-level.
    """
    mock_detector = MagicMock(spec=BaseFaceDetector)
    mock_detector.is_loaded = True

    mock_detector.detect.return_value = FrameInferenceResult(
        image_width=640,
        image_height=480,
        preprocessing_time_ms=1.5,
        inference_time_ms=18.5,
        postprocessing_time_ms=2.0,
        total_time_ms=22.0,
        processing_time_ms=22.0,
        device="cpu",
        model_name="yunet",
        detections=[
            FaceDetection(
                bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
                confidence=0.97,
                landmarks=canonical_landmarks,
            )
        ],
    )

    operational_service = AIService(detector=mock_detector)
    app.dependency_overrides[get_ai_service] = lambda: operational_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test.jpg", valid_test_image_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 200
            data = response.json()

            # 1. Top-Level Stage Metrics
            assert "detection_time_ms" in data
            assert "landmark_time_ms" in data
            assert "quality_analysis_time_ms" in data
            assert "pose_time_ms" in data
            assert "total_analysis_time_ms" in data

            assert data["detection_time_ms"] == 18.5
            assert isinstance(data["landmark_time_ms"], float)
            assert data["landmark_time_ms"] >= 0.0
            assert isinstance(data["quality_analysis_time_ms"], float)
            assert data["quality_analysis_time_ms"] >= 0.0
            assert isinstance(data["pose_time_ms"], float)
            assert data["pose_time_ms"] >= 0.0
            assert isinstance(data["total_analysis_time_ms"], float)
            assert data["total_analysis_time_ms"] >= 0.0

            # 2. Timing Breakdown Nested Object (Backward Compatibility)
            timing = data["timing"]
            assert timing["preprocessing_time_ms"] == 1.5
            assert timing["inference_time_ms"] == 18.5
            assert timing["postprocessing_time_ms"] == 2.0
            assert timing["detection_time_ms"] == 18.5
            assert timing["landmark_time_ms"] == data["landmark_time_ms"]
            assert timing["quality_analysis_time_ms"] == data["quality_analysis_time_ms"]
            assert timing["pose_time_ms"] == data["pose_time_ms"]
            assert timing["total_analysis_time_ms"] == data["total_analysis_time_ms"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_analyze_endpoint_returns_separate_stage_latencies(valid_test_image_bytes, canonical_landmarks):
    """
    Step 18: Verifies POST /api/v1/ai/analyze returns actual measured stage latencies:
      - detection_time_ms
      - landmark_time_ms
      - quality_analysis_time_ms
      - pose_time_ms
      - total_analysis_time_ms
    Both in response.timing and at top-level.
    """
    mock_detector = MagicMock(spec=BaseFaceDetector)
    mock_detector.is_loaded = True

    mock_detector.detect.return_value = FrameInferenceResult(
        image_width=640,
        image_height=480,
        preprocessing_time_ms=1.2,
        inference_time_ms=20.0,
        postprocessing_time_ms=1.8,
        total_time_ms=23.0,
        processing_time_ms=23.0,
        device="cpu",
        model_name="yunet",
        detections=[
            FaceDetection(
                bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
                confidence=0.98,
                landmarks=canonical_landmarks,
            )
        ],
    )

    operational_service = AIService(detector=mock_detector)
    app.dependency_overrides[get_ai_service] = lambda: operational_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test.jpg", valid_test_image_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/analyze", files=files)

            assert response.status_code == 200
            data = response.json()

            # 1. Top-Level Stage Metrics
            assert "detection_time_ms" in data
            assert "landmark_time_ms" in data
            assert "quality_analysis_time_ms" in data
            assert "pose_time_ms" in data
            assert "total_analysis_time_ms" in data

            assert data["detection_time_ms"] == 20.0
            assert isinstance(data["landmark_time_ms"], float)
            assert data["landmark_time_ms"] >= 0.0
            assert isinstance(data["quality_analysis_time_ms"], float)
            assert data["quality_analysis_time_ms"] >= 0.0
            assert isinstance(data["pose_time_ms"], float)
            assert data["pose_time_ms"] >= 0.0
            assert isinstance(data["total_analysis_time_ms"], float)
            assert data["total_analysis_time_ms"] >= 0.0

            # 2. Timing Breakdown Object
            timing = data["timing"]
            assert timing is not None
            assert timing["detection_time_ms"] == 20.0
            assert timing["landmark_time_ms"] == data["landmark_time_ms"]
            assert timing["quality_analysis_time_ms"] == data["quality_analysis_time_ms"]
            assert timing["pose_time_ms"] == data["pose_time_ms"]
            assert timing["total_analysis_time_ms"] == data["total_analysis_time_ms"]
    finally:
        app.dependency_overrides.clear()
