"""
DeepVision AI — AI Detection API Endpoint Test Suite
Validates the development endpoint POST /api/v1/ai/detect and GET /api/v1/ai/status:
- File type validation (rejects invalid extensions & MIME types)
- File size validation (rejects empty 0-byte and oversized files)
- Corrupted file validation (rejects corrupted byte streams)
- Image dimensions validation (rejects images exceeding max dimension limits)
- Operational state checking (returns proper development error when detector is not configured)
- Zero-fabrication enforcement (never returns fake detections when unconfigured or on blank scenes)
- Operational inference execution (returns standardized structured JSON when model is loaded)
"""

import io
from unittest.mock import MagicMock
import numpy as np
import pytest
from PIL import Image
import cv2
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services.ai_service import AIService, get_ai_service
from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.schemas import FrameInferenceResult, FaceDetection, BoundingBox
from app.core.config import settings


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Creates a valid 640x480 RGB JPEG byte stream."""
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Creates a valid 320x240 RGB PNG byte stream."""
    img = Image.new("RGB", (320, 240), color=(50, 80, 120))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def oversized_dimension_png_bytes() -> bytes:
    """Creates an image exceeding AI_MAX_IMAGE_DIMENSION (4096px)."""
    # Create 4100x10 image (small memory footprint, exceeds 4096px limit)
    img = Image.new("RGB", (4100, 10), color=(20, 20, 20))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# -------------------------------------------------------------------------
# Test 1: Model Not Operational / Development Error Handling
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_detect_endpoint_returns_development_error_when_model_not_configured(valid_jpeg_bytes):
    """
    When no model weights are loaded (detector not operational),
    endpoint returns HTTP 503 and structured development error explaining
    that the detector is not configured. NEVER returns fabricated detections.
    """
    mock_unconfigured = MagicMock(spec=BaseFaceDetector)
    mock_unconfigured.is_loaded = False
    unconfigured_service = AIService(detector=mock_unconfigured)
    app.dependency_overrides[get_ai_service] = lambda: unconfigured_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test_face.jpg", valid_jpeg_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 503
            data = response.json()
            assert data["success"] is False
            assert data["error"] == "detector_not_configured"
            assert "not operational" in data["message"]
            assert "models/face_detection" in data["detail"]
            assert data["model"] == settings.AI_MODEL_NAME
            assert data["device"] == settings.AI_DEVICE.upper()
            # Verify NO fabricated detections were returned
            assert "faces" not in data or data.get("faces") is None
    finally:
        app.dependency_overrides.clear()


# -------------------------------------------------------------------------
# Test 2: File Type Validation
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_detect_endpoint_rejects_text_file():
    """Verify endpoint rejects text files with HTTP 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("notes.txt", b"This is plain text, not an image.", "text/plain")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "unsupported_media_type"
        assert "Unsupported file type" in data["message"]


@pytest.mark.anyio
async def test_detect_endpoint_rejects_pdf_file():
    """Verify endpoint rejects PDF files with HTTP 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("document.pdf", b"%PDF-1.4 dummy pdf bytes", "application/pdf")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "unsupported_media_type"


# -------------------------------------------------------------------------
# Test 3: File Size Validation
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_detect_endpoint_rejects_empty_file():
    """Verify endpoint rejects 0-byte uploads with HTTP 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "empty_file"


@pytest.mark.anyio
async def test_detect_endpoint_rejects_oversized_payload():
    """Verify endpoint rejects files exceeding AI_MAX_IMAGE_SIZE_MB limit with HTTP 413."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create oversized payload exceeding 10MB
        oversized_bytes = b"x" * (settings.ai_max_image_bytes + 1024)
        files = {"file": ("huge.jpg", oversized_bytes, "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 413
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "payload_too_large"


# -------------------------------------------------------------------------
# Test 4: Corrupted Files Validation
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_detect_endpoint_rejects_corrupted_image():
    """Verify endpoint rejects corrupt bytes disguised as JPEG with HTTP 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        corrupted_bytes = b"\xFF\xD8\xFF\xE0corrupted_garbage_bytes_without_valid_image_stream"
        files = {"file": ("corrupted.jpg", corrupted_bytes, "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] in ("corrupted_image", "decoding_error")


# -------------------------------------------------------------------------
# Test 5: Image Dimensions Validation
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_detect_endpoint_rejects_oversized_dimensions(oversized_dimension_png_bytes):
    """Verify endpoint rejects image exceeding 4096px dimension limit with HTTP 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("wide.png", oversized_dimension_png_bytes, "image/png")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] in ("dimensions_too_large", "invalid_dimensions")


# -------------------------------------------------------------------------
# Test 6: Operational Model Detection — Structured Success Output
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_detect_endpoint_success_with_operational_detector(valid_jpeg_bytes):
    """
    When an operational detector session is active, endpoint returns HTTP 200
    with exact requested schema:
    {
      "success": true,
      "model": "selected-model",
      "device": "CPU",
      "image": { "width": 640, "height": 480 },
      "processing_time_ms": 42.5,
      "faces": [{ "bbox": { "x1": 100, "y1": 120, "x2": 300, "y2": 350 }, "confidence": 0.96 }]
    }
    """
    # Create mock operational detector
    mock_detector = MagicMock(spec=BaseFaceDetector)
    mock_detector.is_loaded = True  # Model is operational!

    mock_detector.detect.return_value = FrameInferenceResult(
        image_width=640,
        image_height=480,
        processing_time_ms=42.5,
        device="cpu",
        model_name="scrfd_500m",
        detections=[
            FaceDetection(
                bbox=BoundingBox(x1=100.0, y1=120.0, x2=300.0, y2=350.0),
                confidence=0.96,
            )
        ],
    )

    operational_service = AIService(detector=mock_detector)
    app.dependency_overrides[get_ai_service] = lambda: operational_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("sample.jpg", valid_jpeg_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 200
            data = response.json()

            # Verify schema keys
            assert data["success"] is True
            assert data["model"] == "scrfd_500m"
            assert data["device"] == "CPU"
            assert data["image"] == {"width": 640, "height": 480}
            assert data["processing_time_ms"] == 42.5
            assert data["face_count"] == 1

            # Verify faces array
            assert len(data["faces"]) == 1
            face = data["faces"][0]
            assert face["confidence"] == 0.96
            assert face["bbox"] == {
                "x1": 100.0,
                "y1": 120.0,
                "x2": 300.0,
                "y2": 350.0,
            }
    finally:
        app.dependency_overrides.clear()


# -------------------------------------------------------------------------
# Test 7: Operational Model with 0 Faces (No False Positives / Fabrication)
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_detect_endpoint_zero_faces_never_fabricated(valid_png_bytes):
    """
    When an image contains no faces, endpoint returns faces: [] (empty list).
    Validates that fabricated detections are NEVER returned.
    """
    mock_detector = MagicMock(spec=BaseFaceDetector)
    mock_detector.is_loaded = True

    mock_detector.detect.return_value = FrameInferenceResult(
        image_width=320,
        image_height=240,
        processing_time_ms=18.2,
        device="cpu",
        model_name="scrfd_500m",
        detections=[],  # No faces found
    )

    operational_service = AIService(detector=mock_detector)
    app.dependency_overrides[get_ai_service] = lambda: operational_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("nature_landscape.png", valid_png_bytes, "image/png")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["face_count"] == 0
            assert data["faces"] == []  # Completely empty, no fake faces
            assert data["image"] == {"width": 320, "height": 240}
    finally:
        app.dependency_overrides.clear()


# -------------------------------------------------------------------------
# Test 8: AI Status Telemetry Endpoint
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_ai_status_endpoint():
    """Verify GET /api/v1/ai/status returns readiness status and telemetry."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/ai/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "is_operational" in data
        assert data["model_name"] == settings.AI_MODEL_NAME
        assert data["max_image_size_mb"] == settings.AI_MAX_IMAGE_SIZE_MB
        assert "telemetry" in data
