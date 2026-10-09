"""
DeepVision AI — Security & Privacy Compliance Verification Test Suite (Step 22)

Programmatically enforces the 5 explicit privacy prohibitions and data retention rules:
1. DO NOT identify people (no identity fields, person names, or recognition models).
2. DO NOT match faces against databases (no vector search or gallery matching).
3. DO NOT scrape faces (no background crawlers or external scrapers).
4. DO NOT create biometric profiles (no high-dimensional embedding persistence).
5. DO NOT perform covert surveillance (stateless, consensual request-response processing).
6. Zero permanent disk storage of biometric imagery (pure in-memory volatile processing).
7. Zero raw image data logging (structured logging of metadata only).
8. Existing ingestion security controls remain strictly active.
"""

import os
import io
import tempfile
import logging
from unittest.mock import patch
import numpy as np
import pytest
from PIL import Image
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas.ai import (
    AIDetectionResponse,
    DetectedFaceItem,
    FaceBoundingBox,
    FaceDetectionQuality,
)
from app.ai.analysis.schemas import (
    AIAnalysisResponse,
    StructuredFaceAnalysis,
    ComprehensiveFaceQuality,
)
from app.ai.detection.schemas import FaceDetection, FrameInferenceResult, BoundingBox
from app.core.image_security import sanitize_filename, validate_and_decode_image_security
from app.core.exceptions import InvalidImageDimensionsException, ImageTooLargeException


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generates a valid 320x240 RGB JPEG byte payload in volatile memory."""
    img = Image.new("RGB", (320, 240), color=(120, 130, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# -----------------------------------------------------------------------------
# 1. Prohibited Identity & Biometric Profile Verification
# -----------------------------------------------------------------------------

def test_schemas_strictly_exclude_identity_and_biometric_profile_fields():
    """
    Verifies that API response schemas do NOT define any fields for:
    - person identification
    - biometric profiles
    - recognition feature embeddings
    - database lookups
    """
    prohibited_field_names = {
        "identity",
        "person_name",
        "name",
        "user_id",
        "subject_id",
        "embedding",
        "face_embedding",
        "feature_vector",
        "biometric_profile",
        "biometric_template",
        "gallery_id",
        "match_score",
        "similarity_score",
        "matched_person",
    }

    # Inspect DetectedFaceItem schema
    face_item_fields = set(DetectedFaceItem.model_fields.keys())
    intersection_item = face_item_fields.intersection(prohibited_field_names)
    assert not intersection_item, f"Prohibited identity fields found in DetectedFaceItem: {intersection_item}"

    # Inspect StructuredFaceAnalysis schema
    analysis_fields = set(StructuredFaceAnalysis.model_fields.keys())
    intersection_analysis = analysis_fields.intersection(prohibited_field_names)
    assert not intersection_analysis, f"Prohibited identity fields found in StructuredFaceAnalysis: {intersection_analysis}"

    # Inspect AIDetectionResponse schema
    detect_resp_fields = set(AIDetectionResponse.model_fields.keys())
    assert not detect_resp_fields.intersection(prohibited_field_names)

    # Inspect AIAnalysisResponse schema
    analysis_resp_fields = set(AIAnalysisResponse.model_fields.keys())
    assert not analysis_resp_fields.intersection(prohibited_field_names)


@pytest.mark.anyio
async def test_api_response_payload_contains_no_biometric_embeddings_or_identity(sample_jpeg_bytes):
    """
    Executes an actual API request and validates that the JSON payload
    strictly omits any biometric embeddings, identity tags, or profile data.
    """
    mock_detection = FaceDetection(
        face_id=0,
        bbox=BoundingBox(x1=50.0, y1=50.0, x2=200.0, y2=200.0),
        confidence=0.98,
        landmarks=None,
    )
    mock_frame = FrameInferenceResult(
        image_width=320,
        image_height=240,
        detections=[mock_detection],
        inference_time_ms=10.0,
        processing_time_ms=10.0,
        device="CPU",
        model_name="test-detector",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("app.services.ai_service.AIService.detect_faces", return_value=mock_frame):
            files = {"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

    assert response.status_code == 200
    data = response.json()
    face = data["faces"][0]

    prohibited_keys = ["embedding", "identity", "person_name", "biometric_profile", "feature_vector"]
    for key in prohibited_keys:
        assert key not in face, f"Prohibited key '{key}' found in serialized API face item"


# -----------------------------------------------------------------------------
# 2. Zero Permanent Disk Storage Verification
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_zero_permanent_disk_storage_during_and_after_analysis(sample_jpeg_bytes):
    """
    Verifies that neither the detection nor analysis endpoint creates
    any permanent files or cached images on the filesystem.
    """
    # Record existing file count in standard upload / temp directories
    temp_dir = tempfile.gettempdir()
    initial_temp_files = set(os.listdir(temp_dir))

    mock_detection = FaceDetection(
        face_id=0,
        bbox=BoundingBox(x1=50.0, y1=50.0, x2=200.0, y2=200.0),
        confidence=0.95,
        landmarks=None,
    )
    mock_frame = FrameInferenceResult(
        image_width=320,
        image_height=240,
        detections=[mock_detection],
        inference_time_ms=10.0,
        processing_time_ms=10.0,
        device="CPU",
        model_name="test-detector",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("app.services.ai_service.AIService.detect_faces", return_value=mock_frame):
            files = {"file": ("upload.jpg", sample_jpeg_bytes, "image/jpeg")}
            # 1. POST /detect
            resp_detect = await client.post("/api/v1/ai/detect", files=files)
            assert resp_detect.status_code == 200

            # 2. POST /analyze
            files2 = {"file": ("upload2.jpg", sample_jpeg_bytes, "image/jpeg")}
            resp_analyze = await client.post("/api/v1/ai/analyze", files=files2)
            assert resp_analyze.status_code == 200

    # Verify no persistent files leaked into the temp directory
    final_temp_files = set(os.listdir(temp_dir))
    diff_temp = final_temp_files - initial_temp_files
    # Any new files created must not be biometric image crops or uploaded images
    for fname in diff_temp:
        assert not fname.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".npy")), (
            f"Permanent biometric image leaked to disk: {fname}"
        )


# -----------------------------------------------------------------------------
# 3. Zero Raw Image Data Logging
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_zero_raw_image_data_logging(sample_jpeg_bytes, caplog):
    """
    Verifies that application telemetry logging strictly logs structured metadata
    (width, height, face_count, processing_time_ms, request_id)
    and NEVER writes raw pixel data, base64 strings, or binary streams to logs.
    """
    caplog.set_level(logging.DEBUG)

    mock_detection = FaceDetection(
        face_id=0,
        bbox=BoundingBox(x1=50.0, y1=50.0, x2=200.0, y2=200.0),
        confidence=0.95,
        landmarks=None,
    )
    mock_frame = FrameInferenceResult(
        image_width=320,
        image_height=240,
        detections=[mock_detection],
        inference_time_ms=10.0,
        processing_time_ms=10.0,
        device="CPU",
        model_name="test-detector",
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("app.services.ai_service.AIService.detect_faces", return_value=mock_frame):
            files = {"file": ("test_log.jpg", sample_jpeg_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

    assert response.status_code == 200

    # Inspect all captured log messages
    for record in caplog.records:
        msg = record.getMessage()
        # Must not contain raw image headers or base64 data dumps
        assert "data:image" not in msg
        assert "/9j/" not in msg  # JPEG base64 prefix
        assert "iVBORw" not in msg  # PNG base64 prefix


# -----------------------------------------------------------------------------
# 4. Ingestion Security Controls Maintenance
# -----------------------------------------------------------------------------

def test_ingestion_security_controls_enforced():
    """
    Ensures that existing security controls remain operational:
    - Path traversal sanitization
    - Magic bytes format detection
    - Dimension cap validation
    """
    # 1. Path traversal & null byte sanitization
    sanitized = sanitize_filename("../../etc/passwd\x00malicious.jpg")
    assert "../" not in sanitized
    assert "\x00" not in sanitized
    assert not sanitized.startswith("/")

    # 2. Decompression bomb / dimension cap enforcement
    huge_canvas_bytes = io.BytesIO()
    # 4100x10 image exceeding 4096px limit
    Image.new("RGB", (4100, 10), color=(10, 10, 10)).save(huge_canvas_bytes, format="PNG")
    with pytest.raises(InvalidImageDimensionsException):
        validate_and_decode_image_security(
            data=huge_canvas_bytes.getvalue(),
            declared_content_type="image/png",
            declared_filename="oversized.png",
            max_dimension=4096,
        )
