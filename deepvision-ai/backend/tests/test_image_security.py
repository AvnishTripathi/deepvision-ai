"""
DeepVision AI — Image Ingestion Security Controls Test Suite
Validates the 9 core image safety controls:
1. MIME type validation
2. Actual image content validation (magic byte signatures, anti-spoofing)
3. File size limit enforcement (stream chunk reading with memory caps)
4. Maximum dimensions enforcement (decompression bomb protection)
5. Arbitrary file execution prevention (rejection of binaries & scripts)
6. Untrusted filename sanitization (directory traversal & null byte stripping)
7. Temporary processing isolation (restrictive permissions & auto-cleanup)
8. Zero unnecessary permanent disk storage
9. Zero raw image data logging
"""

import os
import io
import uuid
from unittest.mock import MagicMock
import pytest
from PIL import Image
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import settings
from app.core.image_security import (
    sanitize_filename,
    detect_magic_bytes_format,
    validate_and_decode_image_security,
    secure_temporary_image,
)
from app.core.exceptions import (
    UnsupportedImageError,
    CorruptedImageException,
    EmptyImageException,
    ImageTooLargeException,
    InvalidImageDimensionsException,
)
from app.services.ai_service import AIService, get_ai_service
from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.schemas import FrameInferenceResult, FaceDetection, BoundingBox


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Creates a valid 640x480 RGB JPEG byte stream with genuine JPEG magic bytes."""
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Creates a valid 320x240 RGB PNG byte stream with genuine PNG magic bytes."""
    img = Image.new("RGB", (320, 240), color=(80, 120, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# -------------------------------------------------------------------------
# Security Control 1 & 2: MIME Type & Actual Image Content (Magic Numbers)
# -------------------------------------------------------------------------

def test_magic_bytes_detection():
    """Verify magic byte detector accurately discriminates genuine image headers."""
    # Genuine JPEG: \xFF\xD8\xFF
    assert detect_magic_bytes_format(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01") == "JPEG"

    # Genuine PNG: \x89PNG\r\n\x1a\n
    assert detect_magic_bytes_format(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR") == "PNG"

    # Genuine BMP: BM
    assert detect_magic_bytes_format(b"BM\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00") == "BMP"

    # Fake header (PHP script disguised as image)
    assert detect_magic_bytes_format(b"<?php echo 'malicious payload'; ?>") is None

    # Executable payload (Windows PE header MZ)
    assert detect_magic_bytes_format(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00") is None

    # ELF executable header
    assert detect_magic_bytes_format(b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00") is None


def test_file_type_spoofing_prevented(valid_jpeg_bytes):
    """
    Simulate an attacker uploading a shell script or executable with a fake
    Content-Type: image/jpeg header. The safety pipeline MUST reject it based on
    actual content inspection.
    """
    fake_image_payload = b"#!/bin/bash\nrm -rf /tmp/test\n"

    with pytest.raises(UnsupportedImageError) as exc_info:
        validate_and_decode_image_security(
            data=fake_image_payload,
            declared_content_type="image/jpeg",
            declared_filename="avatar.jpg",
        )
    assert "File header signature does not match" in str(exc_info.value)


@pytest.mark.anyio
async def test_endpoint_rejects_spoofed_mime_type_upload():
    """Verify API endpoint rejects executable disguised as JPEG with HTTP 400."""
    spoofed_payload = b"MZ\x90\x00\x03\x00malicious_windows_executable_code"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("innocent.jpg", spoofed_payload, "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "unsupported_media_type"


# -------------------------------------------------------------------------
# Security Control 3: File Size Limit Enforcement
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_endpoint_enforces_file_size_limit():
    """Verify endpoint rejects payloads exceeding configured limit with HTTP 413."""
    oversized_data = b"x" * (settings.ai_max_image_bytes + 2048)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("large_image.jpg", oversized_data, "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 413
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "payload_too_large"


@pytest.mark.anyio
async def test_endpoint_rejects_zero_byte_empty_file():
    """Verify endpoint rejects empty 0-byte uploads with HTTP 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "empty_file"


# -------------------------------------------------------------------------
# Security Control 4: Maximum Dimensions Enforcement
# -------------------------------------------------------------------------

def test_dimension_bounds_enforced():
    """Verify images exceeding maximum supported dimensions are rejected."""
    # Create 4100x10 image (exceeds 4096px limit)
    img = Image.new("RGB", (4100, 10), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    oversized_bytes = buf.getvalue()

    with pytest.raises(InvalidImageDimensionsException):
        validate_and_decode_image_security(
            data=oversized_bytes,
            declared_content_type="image/png",
            declared_filename="wide.png",
            max_dimension=4096,
        )


# -------------------------------------------------------------------------
# Security Control 5 & 6: Untrusted Filename Sanitization & Path Traversal
# -------------------------------------------------------------------------

def test_filename_sanitization_strips_path_traversal():
    """
    Verify untrusted client filenames cannot escape or execute directory traversal.
    """
    # Linux path traversal
    safe1 = sanitize_filename("../../../etc/passwd.jpg")
    assert ".." not in safe1
    assert "/" not in safe1
    assert safe1.endswith(".jpg")

    # Windows path traversal
    safe2 = sanitize_filename("..\\..\\windows\\system32\\cmd.exe.png")
    assert ".." not in safe2
    assert "\\" not in safe2
    assert safe2.endswith(".png")

    # Null byte injection
    safe3 = sanitize_filename("avatar.jpg\x00.sh")
    assert "\x00" not in safe3
    assert safe3.endswith(".jpg")

    # Empty filename
    safe4 = sanitize_filename("")
    assert safe4.startswith("upload_")
    assert safe4.endswith(".jpg")


# -------------------------------------------------------------------------
# Security Control 7: Temporary Processing Isolation
# -------------------------------------------------------------------------

def test_secure_temporary_image_lifecycle(valid_jpeg_bytes):
    """
    Verify secure_temporary_image context manager creates file with restrictive
    permissions and guarantees immediate cleanup upon exit.
    """
    file_path = None
    with secure_temporary_image(valid_jpeg_bytes, suffix=".jpg") as temp_path:
        file_path = temp_path
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) == len(valid_jpeg_bytes)

    # Immediately after context exit, the file MUST be deleted
    assert not os.path.exists(file_path)


# -------------------------------------------------------------------------
# Security Control 8: Zero Permanent Disk Storage
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_zero_permanent_disk_storage(valid_jpeg_bytes):
    """
    Verify API detection flow operates strictly in-memory and never writes
    the uploaded image to permanent storage directories.
    """
    mock_detector = MagicMock(spec=BaseFaceDetector)
    mock_detector.is_loaded = True
    mock_detector.detect.return_value = FrameInferenceResult(
        image_width=640,
        image_height=480,
        processing_time_ms=22.0,
        device="cpu",
        model_name="scrfd_500m",
        detections=[
            FaceDetection(
                bbox=BoundingBox(x1=100.0, y1=100.0, x2=300.0, y2=300.0),
                confidence=0.96,
            )
        ],
    )

    test_service = AIService(detector=mock_detector)
    app.dependency_overrides[get_ai_service] = lambda: test_service

    storage_dir = settings.STORAGE_DIR
    files_before = set(os.listdir(storage_dir)) if os.path.exists(storage_dir) else set()

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test_memory_only.jpg", valid_jpeg_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 200

            files_after = set(os.listdir(storage_dir)) if os.path.exists(storage_dir) else set()
            assert files_after == files_before
    finally:
        app.dependency_overrides.clear()


# -------------------------------------------------------------------------
# Security Control 9: Never Log Raw Image Data
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_never_log_raw_image_data(valid_jpeg_bytes, caplog):
    """
    Verify application logs never contain raw binary payloads, base64 data,
    or decoded pixel matrices.
    """
    mock_detector = MagicMock(spec=BaseFaceDetector)
    mock_detector.is_loaded = True
    mock_detector.detect.return_value = FrameInferenceResult(
        image_width=640,
        image_height=480,
        processing_time_ms=18.0,
        device="cpu",
        model_name="scrfd_500m",
        detections=[],
    )

    test_service = AIService(detector=mock_detector)
    app.dependency_overrides[get_ai_service] = lambda: test_service

    unique_binary_marker = b"UNIQUE_BINARY_TEST_MARKER_1234567890"
    marker_jpeg_bytes = valid_jpeg_bytes + unique_binary_marker

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with caplog.at_level(10):  # Capture everything down to DEBUG level
                files = {"file": ("safe.jpg", marker_jpeg_bytes, "image/jpeg")}
                response = await client.post("/api/v1/ai/detect", files=files)

                assert response.status_code == 200
                # Raw binary marker MUST NEVER appear in any log record
                assert "UNIQUE_BINARY_TEST_MARKER" not in caplog.text
    finally:
        app.dependency_overrides.clear()
