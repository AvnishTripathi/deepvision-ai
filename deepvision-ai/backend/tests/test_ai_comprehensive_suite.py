"""
DeepVision AI — Comprehensive AI Inference & Pipeline Test Suite
Exhaustively tests the 5 core AI pillars:
1. Configuration (loading valid parameters, handling invalid settings)
2. Device Detection (CPU fallback, zero-crash on unavailable GPU)
3. Image Preprocessing (valid formats, corrupt files, unsupported channels, oversized limits)
4. Detection Schemas (bounding box calculations, coordinate validation, confidence ranges)
5. AI Endpoint (valid uploads, invalid media, oversized payloads, detector unavailability)

Design:
- Completely independent of hardware GPU availability.
- Uses unittest.mock to simulate GPU detection and operational detector states.
- Never returns fabricated detections when detector is unavailable.
"""

import os
import io
import uuid
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import Settings
from app.core.exceptions import (
    CorruptedImageException,
    EmptyImageException,
    ImageTooLargeException,
    InvalidImageDimensionsException,
    UnsupportedImageFormatException,
    DetectorNotConfiguredException,
)
from app.ai.common.types import DeviceType
from app.ai.inference.device import (
    resolve_device,
    get_optimal_providers,
    is_gpu_available,
    is_cuda_available,
    get_available_gpu_providers,
)
from app.ai.detection.preprocessing import (
    ImagePreprocessor,
    ImagePreprocessingConfig,
    load_image,
    letterbox_resize,
    unletterbox_coordinates,
    prepare_input_tensor,
)
from app.ai.detection.schemas import (
    BoundingBox,
    NormalizedBoundingBox,
    FaceBoundingBox,
    DetectedFaceItem,
    LandmarkPoint,
    FaceDetection,
    DetectionResponse,
    FrameInferenceResult,
)
from app.services.ai_service import AIService, get_ai_service
from app.ai.detection.base import BaseFaceDetector


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def synthetic_jpeg_bytes() -> bytes:
    """Generates an in-memory 640x480 RGB JPEG byte stream."""
    img = Image.new("RGB", (640, 480), color=(140, 160, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def synthetic_png_bytes() -> bytes:
    """Generates an in-memory 320x240 RGB PNG byte stream."""
    img = Image.new("RGB", (320, 240), color=(100, 120, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =========================================================================
# 1. Configuration Tests
# =========================================================================

def test_ai_configuration_loads_correctly():
    """Verify Settings loads default AI parameters and computes byte limits."""
    cfg = Settings()
    assert cfg.AI_DEVICE in ("auto", "cpu")
    assert cfg.AI_MODEL_NAME == "scrfd_500m"
    assert cfg.AI_MODEL_VERSION == "1.0.0"
    assert cfg.AI_CONFIDENCE_THRESHOLD == 0.50
    assert cfg.AI_MAX_IMAGE_SIZE_MB == 10
    assert cfg.AI_MAX_IMAGE_DIMENSION == 4096
    assert cfg.AI_ENABLE_GPU is False
    # 10 MB in bytes
    assert cfg.ai_max_image_bytes == 10 * 1024 * 1024


def test_ai_configuration_custom_overrides():
    """Verify Settings accepts valid custom configuration overrides."""
    custom = Settings(
        AI_DEVICE="cpu",
        AI_MODEL_NAME="scrfd_2.5g",
        AI_CONFIDENCE_THRESHOLD=0.75,
        AI_MAX_IMAGE_SIZE_MB=25,
        AI_MAX_IMAGE_DIMENSION=2048,
        AI_ENABLE_GPU=True,
    )
    assert custom.AI_DEVICE == "cpu"
    assert custom.AI_MODEL_NAME == "scrfd_2.5g"
    assert custom.AI_CONFIDENCE_THRESHOLD == 0.75
    assert custom.AI_MAX_IMAGE_SIZE_MB == 25
    assert custom.ai_max_image_bytes == 25 * 1024 * 1024
    assert custom.AI_MAX_IMAGE_DIMENSION == 2048
    assert custom.AI_ENABLE_GPU is True


def test_ai_configuration_invalid_confidence_threshold_handled():
    """Verify Settings rejects confidence thresholds outside [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        Settings(AI_CONFIDENCE_THRESHOLD=1.5)

    with pytest.raises(ValidationError):
        Settings(AI_CONFIDENCE_THRESHOLD=-0.1)


def test_ai_configuration_invalid_max_image_size_handled():
    """Verify Settings rejects non-positive payload size limits."""
    with pytest.raises(ValidationError):
        Settings(AI_MAX_IMAGE_SIZE_MB=0)

    with pytest.raises(ValidationError):
        Settings(AI_MAX_IMAGE_SIZE_MB=-5)


def test_ai_configuration_invalid_max_dimension_handled():
    """Verify Settings rejects unrealistic image dimension bounds (< 64px)."""
    with pytest.raises(ValidationError):
        Settings(AI_MAX_IMAGE_DIMENSION=32)


def test_ai_configuration_device_target_normalization():
    """Verify AI_DEVICE validator normalizes case and handles invalid strings safely."""
    cfg1 = Settings(AI_DEVICE="CPU")
    assert cfg1.AI_DEVICE == "cpu"

    cfg2 = Settings(AI_DEVICE="  CUDA  ")
    assert cfg2.AI_DEVICE == "cuda"

    # Unknown device target normalizes to 'auto'
    cfg3 = Settings(AI_DEVICE="quantum_accelerator")
    assert cfg3.AI_DEVICE == "auto"


# =========================================================================
# 2. Device Detection Tests
# =========================================================================

def test_device_detection_cpu_fallback_works():
    """Verify CPU resolution and execution providers always work reliably."""
    # Explicit CPU request
    assert resolve_device(DeviceType.CPU) == "cpu"
    assert resolve_device("cpu") == "cpu"

    # Disabling GPU forces CPU fallback regardless of hardware
    assert resolve_device("gpu", enable_gpu=False) == "cpu"
    assert resolve_device("auto", enable_gpu=False) == "cpu"

    # CPU provider list
    providers = get_optimal_providers("cpu")
    assert providers == ["CPUExecutionProvider"]


def test_device_detection_unavailable_gpu_does_not_crash():
    """
    Verify requesting GPU acceleration on host without GPU never crashes
    and gracefully falls back to CPU.
    """
    with patch("app.ai.inference.device.is_gpu_available", return_value=False):
        # Auto selection on non-GPU host selects CPU
        assert resolve_device("auto") == "cpu"

        # Explicit GPU request on non-GPU host falls back to CPU without throwing exception
        resolved = resolve_device("gpu")
        assert resolved == "cpu"

        resolved_cuda = resolve_device("cuda")
        assert resolved_cuda == "cpu"


def test_device_detection_simulated_gpu_when_available():
    """Verify GPU resolution when supported execution providers are present."""
    with patch("app.ai.inference.device.is_gpu_available", return_value=True):
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            resolved = resolve_device("auto", enable_gpu=True)
            assert resolved == "gpu"

            providers = get_optimal_providers("gpu")
            assert "CUDAExecutionProvider" in providers
            # CPU fallback provider must always be included at the end
            assert providers[-1] == "CPUExecutionProvider"


def test_device_detection_runtime_provider_failure_resilience():
    """Verify InferenceRuntime falls back to CPU if GPU provider fails to load."""
    from app.ai.inference.runtime import InferenceRuntime

    with patch("onnxruntime.InferenceSession") as mock_session_cls:
        # First call with GPU providers raises, second call with CPU succeeds
        mock_session_cls.side_effect = [
            RuntimeError("CUDA driver not installed"),
            MagicMock(get_inputs=lambda: [MagicMock(name="input")], get_outputs=lambda: [MagicMock(name="output")]),
        ]

        # Use an existing dummy file or mock os.path.exists
        with patch("os.path.exists", return_value=True):
            runtime = InferenceRuntime(model_name="test_model", model_path="dummy.onnx", device="gpu")
            # Should have caught the exception and fallen back to CPU provider
            assert mock_session_cls.call_count == 2
            assert mock_session_cls.call_args[1]["providers"] == ["CPUExecutionProvider"]


# =========================================================================
# 3. Image Preprocessing Tests
# =========================================================================

def test_image_preprocessing_valid_image(synthetic_jpeg_bytes):
    """Verify ingestion, letterboxing, and tensor generation for valid images."""
    preprocessor = ImagePreprocessor(ImagePreprocessingConfig(target_size=(640, 640)))

    # Ingest from raw bytes
    rgb = preprocessor.load_image(synthetic_jpeg_bytes)
    assert rgb.shape == (480, 640, 3)
    assert rgb.dtype == np.uint8

    # Letterbox resize maintaining aspect ratio
    padded, scale, (pad_x, pad_y) = preprocessor.resize(rgb)
    assert padded.shape == (640, 640, 3)
    assert scale == 1.0
    assert pad_x == 0
    assert pad_y == 80  # (640 - 480) / 2 = 80

    # Invert letterbox coordinates
    orig_box = unletterbox_coordinates(100.0, 180.0, 300.0, 380.0, scale, (pad_x, pad_y), (480, 640))
    assert orig_box == (100.0, 100.0, 300.0, 300.0)

    # Tensor preparation
    result = preprocessor.process(rgb)
    assert result.tensor.shape == (1, 3, 640, 640)
    assert result.tensor.dtype == np.float32
    assert result.tensor.min() >= 0.0
    assert result.tensor.max() <= 1.0


def test_image_preprocessing_corrupted_image():
    """Verify preprocessor raises CorruptedImageException on unreadable bytes."""
    corrupted_data = b"NOT_A_VALID_JPEG_HEADER_STREAM_12345"
    with pytest.raises(CorruptedImageException):
        load_image(corrupted_data)


def test_image_preprocessing_empty_image():
    """Verify preprocessor raises EmptyImageException on zero-byte inputs."""
    with pytest.raises(EmptyImageException):
        load_image(b"")

    with pytest.raises(EmptyImageException):
        load_image(np.zeros((0, 0, 3), dtype=np.uint8))


def test_image_preprocessing_unsupported_format():
    """Verify preprocessor raises UnsupportedImageFormatException on bad channels or types."""
    # 2-channel array (unsupported)
    bad_channels = np.zeros((100, 100, 2), dtype=np.uint8)
    with pytest.raises(UnsupportedImageFormatException):
        load_image(bad_channels)

    # 1D array
    with pytest.raises(UnsupportedImageFormatException):
        load_image(np.array([1, 2, 3], dtype=np.uint8))

    # Unsupported object type
    with pytest.raises(UnsupportedImageFormatException):
        load_image({"not": "an_image"})


def test_image_preprocessing_oversized_image():
    """Verify preprocessor raises ImageTooLargeException on oversized bytes or dimensions."""
    config = ImagePreprocessingConfig(max_file_size_bytes=1024, max_dimension=500)
    preprocessor = ImagePreprocessor(config=config)

    # Exceeds max byte size (2048 > 1024)
    with pytest.raises(ImageTooLargeException):
        preprocessor.load_image(b"x" * 2048)

    # Exceeds max pixel dimension (800 > 500)
    oversized_array = np.zeros((800, 400, 3), dtype=np.uint8)
    with pytest.raises(ImageTooLargeException):
        preprocessor.load_image(oversized_array)


# =========================================================================
# 4. Detection Schemas Tests
# =========================================================================

def test_detection_schemas_valid_bounding_boxes():
    """Verify BoundingBox geometry, area calculations, and coordinate normalization."""
    bbox = BoundingBox(x1=50.0, y1=80.0, x2=200.0, y2=260.0)
    assert bbox.width == 150.0
    assert bbox.height == 180.0
    assert bbox.area == 27000.0

    # Normalization to [0.0, 1.0] image space
    norm_box = bbox.to_normalized(image_width=640, image_height=480)
    assert norm_box.x1 == round(50.0 / 640, 4)
    assert norm_box.y1 == round(80.0 / 480, 4)
    assert norm_box.x2 == round(200.0 / 640, 4)
    assert norm_box.y2 == round(260.0 / 480, 4)

    # Invert back to pixel coordinates
    pixel_box = norm_box.to_pixel(image_width=640, image_height=480)
    assert abs(pixel_box.x1 - bbox.x1) < 0.1
    assert abs(pixel_box.y1 - bbox.y1) < 0.1

    # FaceBoundingBox serialization
    dev_box = FaceBoundingBox(x1=50.0, y1=80.0, x2=200.0, y2=260.0)
    dumped = dev_box.model_dump()
    assert dumped == {"x1": 50.0, "y1": 80.0, "x2": 200.0, "y2": 260.0}


def test_detection_schemas_invalid_coordinates():
    """Verify coordinate validators reject out-of-bounds normalized values."""
    # Normalized coordinates must be between 0.0 and 1.0
    with pytest.raises(ValidationError):
        NormalizedBoundingBox(x1=-0.1, y1=0.0, x2=0.5, y2=0.5)

    with pytest.raises(ValidationError):
        NormalizedBoundingBox(x1=0.0, y1=0.0, x2=1.1, y2=0.5)

    # Normalization with zero or negative image dimensions raises ValueError
    bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50)
    with pytest.raises(ValueError):
        bbox.to_normalized(image_width=0, image_height=480)

    with pytest.raises(ValueError):
        bbox.to_normalized(image_width=640, image_height=-10)


def test_detection_schemas_confidence_range():
    """Verify confidence scores are strictly validated within [0.0, 1.0]."""
    valid_box = BoundingBox(x1=10, y1=10, x2=50, y2=50)

    # Valid detection passes
    detection = FaceDetection(bbox=valid_box, confidence=0.88)
    assert detection.confidence == 0.88

    # Confidence > 1.0 raises ValidationError
    with pytest.raises(ValidationError):
        FaceDetection(bbox=valid_box, confidence=1.05)

    # Confidence < 0.0 raises ValidationError
    with pytest.raises(ValidationError):
        FaceDetection(bbox=valid_box, confidence=-0.05)

    # DetectedFaceItem confidence validation
    dev_box = FaceBoundingBox(x1=10, y1=10, x2=50, y2=50)
    with pytest.raises(ValidationError):
        DetectedFaceItem(bbox=dev_box, confidence=1.5)

    # LandmarkPoint confidence validation
    with pytest.raises(ValidationError):
        LandmarkPoint(name="left_eye", x=15.0, y=20.0, confidence=-0.5)


# =========================================================================
# 5. AI Endpoint Tests (POST /api/v1/ai/detect)
# =========================================================================

@pytest.mark.anyio
async def test_ai_endpoint_valid_upload_with_operational_detector(synthetic_jpeg_bytes):
    """
    Verify POST /api/v1/ai/detect with a valid image and operational detector:
    - Returns HTTP 200
    - Matches standard schema: success, model, device, image, processing_time_ms, faces
    - Never fabricates detections
    """
    mock_detector = MagicMock(spec=BaseFaceDetector)
    mock_detector.is_loaded = True  # Operational!
    mock_detector.detect.return_value = FrameInferenceResult(
        image_width=640,
        image_height=480,
        processing_time_ms=28.5,
        device="cpu",
        model_name="scrfd_500m",
        detections=[
            FaceDetection(
                bbox=BoundingBox(x1=120.0, y1=90.0, x2=320.0, y2=340.0),
                confidence=0.975,
            )
        ],
    )

    test_service = AIService(detector=mock_detector)
    app.dependency_overrides[get_ai_service] = lambda: test_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("face_photo.jpg", synthetic_jpeg_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["model"] == "scrfd_500m"
            assert data["device"] == "CPU"
            assert data["image"] == {"width": 640, "height": 480}
            assert data["processing_time_ms"] == 28.5
            assert len(data["faces"]) == 1

            face = data["faces"][0]
            assert face["confidence"] == 0.975
            assert face["bbox"] == {
                "x1": 120.0,
                "y1": 90.0,
                "x2": 320.0,
                "y2": 340.0,
            }
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_ai_endpoint_invalid_upload_types_handled():
    """Verify endpoint rejects unsupported formats, corrupted bytes, and empty files with HTTP 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Plain text file
        res_text = await client.post(
            "/api/v1/ai/detect",
            files={"file": ("readme.txt", b"text payload", "text/plain")},
        )
        assert res_text.status_code == 400
        assert res_text.json()["error"] == "unsupported_media_type"

        # 2. PDF file
        res_pdf = await client.post(
            "/api/v1/ai/detect",
            files={"file": ("doc.pdf", b"%PDF-1.4...", "application/pdf")},
        )
        assert res_pdf.status_code == 400
        assert res_pdf.json()["error"] == "unsupported_media_type"

        # 3. Empty file (0 bytes)
        res_empty = await client.post(
            "/api/v1/ai/detect",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert res_empty.status_code == 400
        assert res_empty.json()["error"] == "empty_file"

        # 4. Corrupted JPEG
        res_corrupt = await client.post(
            "/api/v1/ai/detect",
            files={"file": ("bad.jpg", b"\xFF\xD8\xFFbad_garbage_data", "image/jpeg")},
        )
        assert res_corrupt.status_code == 400
        assert res_corrupt.json()["error"] in ("corrupted_image", "decoding_error")


@pytest.mark.anyio
async def test_ai_endpoint_oversized_upload_handled():
    """Verify endpoint rejects payloads exceeding configured megabyte limit with HTTP 413."""
    oversized_bytes = b"x" * (10 * 1024 * 1024 + 1024)  # > 10MB
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/ai/detect",
            files={"file": ("huge.jpg", oversized_bytes, "image/jpeg")},
        )
        assert response.status_code == 413
        assert response.json()["error"] == "payload_too_large"


@pytest.mark.anyio
async def test_ai_endpoint_detector_unavailable_handled(synthetic_jpeg_bytes):
    """
    Verify endpoint returns HTTP 503 when model weights are not loaded.
    Crucially: NEVER returns fabricated detections.
    """
    # Default unconfigured service has is_operational == False
    unconfigured_service = AIService()
    assert unconfigured_service.is_operational is False

    app.dependency_overrides[get_ai_service] = lambda: unconfigured_service

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test.jpg", synthetic_jpeg_bytes, "image/jpeg")}
            response = await client.post("/api/v1/ai/detect", files=files)

            assert response.status_code == 503
            data = response.json()
            assert data["success"] is False
            assert data["error"] == "detector_not_configured"
            assert "models/face_detection" in data["detail"]
            assert data["model"] == "scrfd_500m"

            # Strict zero fabrication check: no fake faces in body
            assert "faces" not in data or data.get("faces") is None
    finally:
        app.dependency_overrides.clear()
