"""
DeepVision AI — Structured Logging & Performance Telemetry Test Suite
Validates that:
- AI inference forward-passes emit structured logs capturing:
  model, device, image_width, image_height, number_of_faces, processing_time_ms, request_id.
- Sensitive image payloads (raw bytes, pixel matrices, base64 strings) are NEVER logged.
- Uploaded images are processed strictly in-memory and NEVER saved permanently to disk.
- Request ID tracing propagates seamlessly across headers, telemetry, and response payloads.
"""

import os
import io
import json
import logging
import uuid
from unittest.mock import MagicMock
import pytest
from PIL import Image
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.logging import log_ai_inference, logger
from app.core.config import settings
from app.services.ai_service import AIService, get_ai_service
from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.schemas import FrameInferenceResult, FaceDetection, BoundingBox


@pytest.fixture
def sample_test_image_bytes() -> bytes:
    """Creates a sample 640x480 RGB JPEG byte stream."""
    img = Image.new("RGB", (640, 480), color=(120, 140, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# -------------------------------------------------------------------------
# Test 1: Direct log_ai_inference Structured Capture
# -------------------------------------------------------------------------

def test_log_ai_inference_captures_all_required_metrics(caplog):
    """
    Verify log_ai_inference produces structured payload containing:
    model, device, image_width, image_height, number_of_faces, processing_time_ms, request_id.
    """
    test_logger = logging.getLogger("test.telemetry")
    req_id = "req-" + str(uuid.uuid4())

    with caplog.at_level(logging.INFO, logger="test.telemetry"):
        telemetry = log_ai_inference(
            logger_instance=test_logger,
            model="scrfd_500m",
            device="CPU",
            image_width=1280,
            image_height=720,
            number_of_faces=2,
            processing_time_ms=34.567,
            request_id=req_id,
        )

    # Validate returned structured dictionary
    assert telemetry["model"] == "scrfd_500m"
    assert telemetry["device"] == "CPU"
    assert telemetry["image_width"] == 1280
    assert telemetry["image_height"] == 720
    assert telemetry["number_of_faces"] == 2
    assert telemetry["processing_time_ms"] == 34.57  # Rounded to 2 decimals
    assert telemetry["request_id"] == req_id

    # Validate output in log record
    assert req_id in caplog.text
    assert "scrfd_500m" in caplog.text
    assert "1280x720" in caplog.text
    assert "faces=2" in caplog.text
    assert "34.57ms" in caplog.text


# -------------------------------------------------------------------------
# Test 2: Sensitive Image Content is NEVER Logged
# -------------------------------------------------------------------------

def test_sensitive_image_content_is_never_logged(caplog):
    """
    Ensures that raw pixel data, byte arrays, or base64 strings are
    strictly filtered out and never emitted into log records.
    """
    test_logger = logging.getLogger("test.privacy")
    secret_binary_content = b"\x89PNG\r\n\x1a\nsecret_biometric_data_never_log_this"

    with caplog.at_level(logging.INFO, logger="test.privacy"):
        telemetry = log_ai_inference(
            logger_instance=test_logger,
            model="scrfd_500m",
            device="CPU",
            image_width=640,
            image_height=480,
            number_of_faces=1,
            processing_time_ms=20.0,
            request_id="privacy-check-001",
            extra_metrics={
                "data": secret_binary_content,
                "raw_bytes": secret_binary_content,
                "safe_field": "valid_metric",
            },
        )

    # Verify secret binary fields were filtered out
    assert "secret_biometric_data_never_log_this" not in caplog.text
    assert "data" not in telemetry
    assert "raw_bytes" not in telemetry
    assert telemetry.get("safe_field") == "valid_metric"


# -------------------------------------------------------------------------
# Test 3: Zero Permanent Disk Persistence for Uploaded Images
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_zero_disk_persistence_on_image_upload(sample_test_image_bytes):
    """
    Validates that image upload processing does NOT save files to disk in storage
    or application directories.
    """
    mock_detector = MagicMock(spec=BaseFaceDetector)
    mock_detector.is_loaded = True
    mock_detector.detect.return_value = FrameInferenceResult(
        image_width=640,
        image_height=480,
        processing_time_ms=25.0,
        device="cpu",
        model_name="scrfd_500m",
        detections=[
            FaceDetection(
                bbox=BoundingBox(x1=50.0, y1=60.0, x2=200.0, y2=220.0),
                confidence=0.97,
            )
        ],
    )

    operational_service = AIService(detector=mock_detector)
    app.dependency_overrides[get_ai_service] = lambda: operational_service

    # Snapshot directory contents before request
    storage_dir = settings.STORAGE_DIR
    files_before = set(os.listdir(storage_dir)) if os.path.exists(storage_dir) else set()

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("volatile_test_image.jpg", sample_test_image_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 200

            # Verify no new files were created in storage directory
            files_after = set(os.listdir(storage_dir)) if os.path.exists(storage_dir) else set()
            new_files = files_after - files_before
            assert len(new_files) == 0, f"Unexpected files saved to disk: {new_files}"
    finally:
        app.dependency_overrides.clear()


# -------------------------------------------------------------------------
# Test 4: End-to-End Inference Logging & Request ID Tracing
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_end_to_end_inference_logging_with_request_id(sample_test_image_bytes, caplog):
    """
    Verify complete request lifecycle:
    - Custom X-Request-ID header is propagated
    - Attached to response header and JSON payload
    - Structured telemetry is logged with exact captured fields
    """
    custom_request_id = "trace-deepvision-" + str(uuid.uuid4())[:8]

    mock_detector = MagicMock(spec=BaseFaceDetector)
    mock_detector.is_loaded = True
    mock_detector.detect.return_value = FrameInferenceResult(
        image_width=640,
        image_height=480,
        processing_time_ms=19.45,
        device="cpu",
        model_name="scrfd_500m",
        detections=[
            FaceDetection(
                bbox=BoundingBox(x1=100.0, y1=100.0, x2=300.0, y2=300.0),
                confidence=0.98,
            )
        ],
    )

    operational_service = AIService(detector=mock_detector)
    app.dependency_overrides[get_ai_service] = lambda: operational_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with caplog.at_level(logging.INFO, logger="deepvision.api.ai"):
                files = {"file": ("input.jpg", sample_test_image_bytes, "image/jpeg")}
                headers = {"X-Request-ID": custom_request_id}
                response = await client.post("/api/v1/ai/detect", files=files, headers=headers)

                assert response.status_code == 200
                data = response.json()

                # Verify Request ID in headers and body
                assert response.headers.get("X-Request-ID") == custom_request_id
                assert data["request_id"] == custom_request_id

                # Verify structured logging captured all required metrics
                assert custom_request_id in caplog.text
                assert "scrfd_500m" in caplog.text
                assert "CPU" in caplog.text
                assert "resolution=640x480" in caplog.text
                assert "faces=1" in caplog.text
                assert "latency=19.45ms" in caplog.text
    finally:
        app.dependency_overrides.clear()
