"""
DeepVision AI — Detector & API Verification Test Suite (Step 3, Task 16)

Comprehensive automated tests covering:
1. Detector:
   - Model loading (valid weights, missing file error handling, CPU runtime)
   - Valid image inference (facial localization, landmarks, confidence)
   - No-face image (zero false positives on blank/landscape scenes)
   - Multiple-face image (multi-face localization)
   - Bounding box validation (bounds checking, clipping, rejection of degenerate boxes)
   - Confidence filtering (threshold gating)
2. API (POST /api/v1/ai/detect):
   - Valid image upload (HTTP 200 with structured response)
   - Invalid file (corrupt binary stream -> HTTP 400)
   - Unsupported file (text/PDF/unsupported MIME -> HTTP 400)
   - Oversized file (>10MB -> HTTP 413)
   - Empty upload (0-byte payload -> HTTP 400)
   - Detector failure (unconfigured -> HTTP 503; runtime error -> HTTP 500)
3. Performance:
   - Processing time is a real measured value (strictly positive float, stage breakdown)
   - Zero GPU dependency (runs 100% reliably on standard CPU runners)
   - Mocking unavailable hardware (GPU -> CPU fallback)
"""

import os
import io
import time
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
import cv2
from PIL import Image
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import settings, Settings
from app.services.ai_service import AIService, get_ai_service
from app.ai.detection.detector import FaceDetector
from app.ai.detection.schemas import (
    BoundingBox,
    FaceBoundingBox,
    FaceDetection,
    FrameInferenceResult,
)
from app.ai.inference.device import resolve_device, is_gpu_available, is_cuda_available
from app.core.exceptions import DetectorNotConfiguredException, InferenceError


# Path to test fixtures
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
ONE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")
MULTIPLE_FACES_PATH = os.path.join(FIXTURES_DIR, "multiple_faces.jpg")
NO_FACE_PATH = os.path.join(FIXTURES_DIR, "no_face.jpg")


def _read_fixture_bytes(path: str) -> bytes:
    assert os.path.exists(path), f"Fixture not found at {path}"
    with open(path, "rb") as f:
        return f.read()


def _read_fixture_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    assert bgr is not None, f"Could not read fixture at {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# =============================================================================
# 1. DETECTOR TESTS
# =============================================================================

class TestDetectorSuite:
    """Verifies core AI detector model loading, inference, filtering, and geometry."""

    def test_detector_model_loading_success(self):
        """Detector: model loading - verifies model initializes on CPU without GPU dependency."""
        model_path = settings.resolve_model_path()
        assert model_path is not None, "Model path could not be resolved"
        assert os.path.exists(model_path), f"Physical model file not found at {model_path}"

        detector = FaceDetector(model_path=model_path, device="cpu")
        assert detector.is_loaded is True
        assert detector.device == "cpu"
        assert detector.model_path == os.path.abspath(model_path)

    def test_detector_model_loading_missing_file_handled_safely(self):
        """Detector: model loading - verifies non-existent model path leaves detector safely unloaded without unhandled crash."""
        detector = FaceDetector(model_path="non_existent/path/to/model.onnx", device="cpu")
        assert detector.is_loaded is False

    def test_detector_valid_image_inference(self):
        """Detector: valid image inference - verifies face localization and landmarks on real portrait."""
        rgb_img = _read_fixture_rgb(ONE_FACE_PATH)
        h, w = rgb_img.shape[:2]

        model_path = settings.resolve_model_path()
        detector = FaceDetector(model_path=model_path, device="cpu", confidence_threshold=0.5)

        result = detector.detect(rgb_img)
        assert isinstance(result, FrameInferenceResult)
        assert result.face_count >= 1
        assert len(result.detections) >= 1

        first_detection = result.detections[0]
        assert first_detection.confidence >= 0.5
        assert 0.0 <= first_detection.bbox.x1 < first_detection.bbox.x2 <= float(w)
        assert 0.0 <= first_detection.bbox.y1 < first_detection.bbox.y2 <= float(h)
        assert first_detection.landmarks is not None
        assert len(first_detection.landmarks) == 5

    def test_detector_no_face_image_zero_false_positives(self):
        """Detector: no-face image - verifies zero false positive detections on genuine non-face image."""
        rgb_img = _read_fixture_rgb(NO_FACE_PATH)

        model_path = settings.resolve_model_path()
        detector = FaceDetector(model_path=model_path, device="cpu", confidence_threshold=0.5)

        result = detector.detect(rgb_img)
        assert result.face_count == 0
        assert len(result.detections) == 0

    def test_detector_multiple_face_image_localization(self):
        """Detector: multiple-face image - detects multiple distinct faces on multi-face scene."""
        rgb_img = _read_fixture_rgb(MULTIPLE_FACES_PATH)
        h, w = rgb_img.shape[:2]

        model_path = settings.resolve_model_path()
        detector = FaceDetector(model_path=model_path, device="cpu", confidence_threshold=0.45)

        result = detector.detect(rgb_img)
        assert result.face_count >= 2, f"Expected >= 2 faces, got {result.face_count}"

        # Ensure detected boxes are non-degenerate and distinct
        boxes = [d.bbox for d in result.detections]
        for box in boxes:
            assert box.x1 >= 0.0 and box.y1 >= 0.0
            assert box.x2 <= float(w) and box.y2 <= float(h)
            assert box.x2 > box.x1
            assert box.y2 > box.y1

    def test_detector_bounding_box_validation_rules(self):
        """Detector: bounding box validation - verifies bounds checking, clipping, and invalid box rejection."""
        # Valid box inside canvas 1000x800
        valid_box = BoundingBox(x1=50.0, y1=60.0, x2=200.0, y2=250.0)
        assert valid_box.is_valid(image_width=1000, image_height=800) is True

        # Inverted coordinates (x2 <= x1)
        inverted_box = BoundingBox(x1=200.0, y1=60.0, x2=50.0, y2=250.0)
        assert inverted_box.is_valid(image_width=1000, image_height=800) is False

        # Negative coordinates
        negative_box = BoundingBox(x1=-10.0, y1=20.0, x2=100.0, y2=150.0)
        assert negative_box.is_valid(image_width=1000, image_height=800) is False

        # Exceeding canvas bounds
        overflow_box = BoundingBox(x1=100.0, y1=100.0, x2=1200.0, y2=600.0)
        assert overflow_box.is_valid(image_width=1000, image_height=800) is False

        # Clipping partially overflowing box to boundaries
        clipped = overflow_box.clip_to_bounds(image_width=1000, image_height=800)
        assert clipped is not None
        assert clipped.x2 == 1000.0
        assert clipped.is_valid(image_width=1000, image_height=800) is True

        # Degenerate box completely out-of-bounds clipped returns None
        out_of_bounds = BoundingBox(x1=1100.0, y1=100.0, x2=1300.0, y2=200.0)
        assert out_of_bounds.clip_to_bounds(image_width=1000, image_height=800) is None

    def test_detector_confidence_filtering(self):
        """Detector: confidence filtering - verifies setting higher threshold filters out lower confidence detections."""
        rgb_img = _read_fixture_rgb(ONE_FACE_PATH)
        model_path = settings.resolve_model_path()

        detector_low = FaceDetector(model_path=model_path, device="cpu", confidence_threshold=0.30)
        result_low = detector_low.detect(rgb_img)

        # Extremely high threshold should yield fewer or equal detections
        detector_high = FaceDetector(model_path=model_path, device="cpu", confidence_threshold=0.999)
        result_high = detector_high.detect(rgb_img)

        assert result_low.face_count >= result_high.face_count
        for d in result_high.detections:
            assert d.confidence >= 0.999


# =============================================================================
# 2. API ENDPOINT TESTS (POST /api/v1/ai/detect)
# =============================================================================

class TestAPISuite:
    """Verifies HTTP API layer validation, upload processing, and error handling."""

    @pytest.mark.anyio
    async def test_api_valid_image_upload(self):
        """API: valid image upload - verifies HTTP 200 and schema adherence for JPEG/PNG."""
        image_bytes = _read_fixture_bytes(ONE_FACE_PATH)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("portrait.jpg", image_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["model"] is not None
            assert data["device"] == "CPU"
            assert data["image"]["width"] > 0
            assert data["image"]["height"] > 0
            assert data["face_count"] >= 1
            assert isinstance(data["faces"], list)
            assert len(data["faces"]) == data["face_count"]
            assert "timing" in data
            assert data["timing"]["total_processing_time_ms"] > 0

    @pytest.mark.anyio
    async def test_api_invalid_file_corrupt_stream(self):
        """API: invalid file - verifies HTTP 400 when corrupt/truncated bytes are uploaded."""
        corrupt_bytes = b"\xff\xd8\xff\xe0CORRUPTED_BINARY_STREAM_DATA_TRUNCATED"
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 400
            data = response.json()
            assert data["success"] is False
            assert "error" in data
            assert "message" in data

    @pytest.mark.anyio
    async def test_api_unsupported_file_format(self):
        """API: unsupported file - verifies HTTP 400 for non-image or unsupported formats."""
        text_bytes = b"This is a text file payload, not a valid image container."
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("document.txt", text_bytes, "text/plain")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 400
            data = response.json()
            assert data["success"] is False

    @pytest.mark.anyio
    async def test_api_oversized_file_rejected_with_413(self):
        """API: oversized file - verifies HTTP 413 when payload exceeds configured size limit."""
        # Create an in-memory byte stream larger than 10MB (11MB)
        large_bytes = b"\x00" * (11 * 1024 * 1024)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("huge_image.jpg", large_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 413
            data = response.json()
            assert data["success"] is False
            assert data["error"] == "payload_too_large"

    @pytest.mark.anyio
    async def test_api_empty_upload_rejected_with_400(self):
        """API: empty upload - verifies HTTP 400 when a 0-byte file is submitted."""
        empty_bytes = b""
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("empty.jpg", empty_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 400
            data = response.json()
            assert data["success"] is False
            assert "empty" in data["message"].lower() or data["error"] == "empty_file"

    @pytest.mark.anyio
    async def test_api_detector_failure_unconfigured_model(self):
        """API: detector failure - verifies HTTP 503 when model is not operational, without fabricating detections."""
        image_bytes = _read_fixture_bytes(ONE_FACE_PATH)
        transport = ASGITransport(app=app)

        # Mock AIService as unconfigured/non-operational
        mock_unconfigured_service = MagicMock(spec=AIService)
        mock_unconfigured_service.is_operational = False

        app.dependency_overrides[get_ai_service] = lambda: mock_unconfigured_service
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                files = {"file": ("sample.jpg", image_bytes, "image/jpeg")}
                response = await client.post("/api/v1/ai/detect", files=files)

                assert response.status_code == 503
                data = response.json()
                assert data["success"] is False
                assert data["error"] == "detector_not_configured"
                assert "faces" not in data or data.get("faces") is None
        finally:
            app.dependency_overrides.pop(get_ai_service, None)

    @pytest.mark.anyio
    async def test_api_detector_failure_internal_inference_error_sanitized(self):
        """API: detector failure - verifies HTTP 500 masks internal tracebacks on unexpected inference crash."""
        image_bytes = _read_fixture_bytes(ONE_FACE_PATH)
        transport = ASGITransport(app=app)

        # Mock AIService raising an internal crash
        mock_crashing_service = MagicMock(spec=AIService)
        mock_crashing_service.is_operational = True
        mock_crashing_service.detect_faces.side_effect = RuntimeError("Internal CUDA memory fault 0xDEADBEEF")

        app.dependency_overrides[get_ai_service] = lambda: mock_crashing_service
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                files = {"file": ("sample.jpg", image_bytes, "image/jpeg")}
                response = await client.post("/api/v1/ai/detect", files=files)

                assert response.status_code == 500
                data = response.json()
                assert data["success"] is False
                assert data["error"] == "inference_error"
                # Internal stack trace / sensitive details should not be exposed in user-facing message
                assert "0xDEADBEEF" not in data["message"]
        finally:
            app.dependency_overrides.pop(get_ai_service, None)


# =============================================================================
# 3. PERFORMANCE & HARDWARE INDEPENDENCE TESTS
# =============================================================================

class TestPerformanceSuite:
    """Verifies execution telemetry, real processing times, and GPU independence."""

    def test_processing_time_is_real_measured_value(self):
        """Performance: verifies processing times are strictly real measured non-zero values with breakdown."""
        rgb_img = _read_fixture_rgb(ONE_FACE_PATH)
        model_path = settings.resolve_model_path()
        detector = FaceDetector(model_path=model_path, device="cpu")

        # Run inference multiple times
        t0 = time.perf_counter()
        result1 = detector.detect(rgb_img)
        elapsed1 = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        result2 = detector.detect(rgb_img)
        elapsed2 = (time.perf_counter() - t0) * 1000

        # 1. Total latency must be a positive non-zero float
        assert result1.total_processing_time_ms > 0.0
        assert result2.total_processing_time_ms > 0.0
        assert isinstance(result1.total_processing_time_ms, float)

        # 2. Total latency must be in a realistic physical range (e.g. <= actual wall-clock elapsed)
        assert result1.total_processing_time_ms <= elapsed1 + 5.0
        assert result2.total_processing_time_ms <= elapsed2 + 5.0

        # 3. Sub-stage timing breakdown must be real measured values
        assert result1.preprocessing_time_ms > 0.0
        assert result1.inference_time_ms > 0.0
        assert result1.postprocessing_time_ms >= 0.0
        assert result1.total_processing_time_ms >= result1.inference_time_ms

    def test_tests_must_not_depend_on_gpu(self):
        """Performance: verifies that detector functions completely without GPU hardware."""
        # Explicitly simulate an environment with NO GPU available
        with patch("app.ai.inference.device.is_gpu_available", return_value=False), \
             patch("app.ai.inference.device.is_cuda_available", return_value=False):

            # Hardware resolution must select CPU
            device = resolve_device(requested_device="auto")
            assert device == "cpu"

            # Model loads and infers on CPU without issue
            model_path = settings.resolve_model_path()
            detector = FaceDetector(model_path=model_path, device="cpu")
            assert detector.is_loaded is True
            assert detector.device == "cpu"

            rgb_img = _read_fixture_rgb(ONE_FACE_PATH)
            result = detector.detect(rgb_img)
            assert result.face_count >= 1

    def test_mock_unavailable_hardware_graceful_fallback(self):
        """Performance: verifies requesting unavailable GPU hardware safely falls back to CPU without crashing."""
        # Simulate GPU requested by user/config, but hardware runtime is missing
        with patch("app.ai.inference.device.is_gpu_available", return_value=False):
            # Requesting "gpu" or "cuda" on host without GPU
            resolved = resolve_device(requested_device="cuda")
            assert resolved == "cpu"

            resolved_gpu = resolve_device(requested_device="gpu")
            assert resolved_gpu == "cpu"
