"""
DeepVision AI — Error Handling & Sanitization Test Suite
Validates that:
- Clean AI-specific exceptions (ModelLoadError, InferenceError, InvalidImageError,
  UnsupportedImageError, AIConfigurationError, DeviceNotAvailableError) map to proper HTTP responses.
- Internal stack traces, raw tracebacks, and internal file paths are NEVER exposed to API users.
- Detailed technical errors and stack traces are logged on the server side (exc_info=True).
"""

import logging
from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

from app.core.exceptions import (
    AIException,
    ModelLoadError,
    InferenceError,
    InvalidImageError,
    UnsupportedImageError,
    AIConfigurationError,
    DeviceNotAvailableError,
    CorruptedImageException,
    EmptyImageException,
    ImageTooLargeException,
)
from app.core.error_handlers import register_exception_handlers


@pytest.fixture
def error_test_app() -> FastAPI:
    """Creates a dedicated FastAPI application instance for error mapping verification."""
    test_app = FastAPI()
    register_exception_handlers(test_app)

    @test_app.get("/trigger/model-load-error")
    def trigger_model_load():
        raise ModelLoadError(
            message="Model weights could not be loaded from /var/secret/models/weights.onnx",
            detail="Install model weights according to docs/model-setup.md",
            model_name="scrfd_500m",
            device="CPU",
        )

    @test_app.get("/trigger/inference-error")
    def trigger_inference():
        # Simulate an internal ONNX runtime buffer overrun / exception
        raise InferenceError(
            message="AI model inference processing failed.",
            detail="Segmentation fault in onnxruntime::SessionState::Execute() at pointer 0x7fff",
            model_name="scrfd_500m",
        )

    @test_app.get("/trigger/invalid-image-error")
    def trigger_invalid_image():
        raise InvalidImageError(message="Image byte stream is truncated or malformed.")

    @test_app.get("/trigger/corrupted-image")
    def trigger_corrupted_image():
        raise CorruptedImageException("Corrupted JPEG header: unexpected EOF.")

    @test_app.get("/trigger/empty-image")
    def trigger_empty_image():
        raise EmptyImageException("Uploaded file is empty (0 bytes).")

    @test_app.get("/trigger/image-too-large")
    def trigger_image_too_large():
        raise ImageTooLargeException("Payload size 15MB exceeds 10MB limit.")

    @test_app.get("/trigger/unsupported-image-error")
    def trigger_unsupported_image():
        raise UnsupportedImageError("Unsupported format 'application/pdf'. Allowed: JPEG, PNG.")

    @test_app.get("/trigger/ai-config-error")
    def trigger_ai_config():
        raise AIConfigurationError(
            message="Invalid AI engine configuration settings.",
            detail="AI_CONFIDENCE_THRESHOLD value 1.5 is outside valid range [0.0, 1.0].",
        )

    @test_app.get("/trigger/device-not-available")
    def trigger_device_not_available():
        raise DeviceNotAvailableError(
            message="Requested AI hardware compute device 'cuda' is not available on this host.",
            detail="No NVIDIA GPU drivers or CUDA execution providers detected.",
            requested_device="cuda",
        )

    @test_app.get("/trigger/unhandled-crash")
    def trigger_unhandled_crash():
        # Simulate a completely unexpected internal Python exception with stack trace
        raise ValueError("Critical internal variable NoneType has no attribute 'weights'")

    return test_app


# -------------------------------------------------------------------------
# Test 1: ModelLoadError Mapping & Sanitization
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_model_load_error_mapped_to_503(error_test_app, caplog):
    """Verify ModelLoadError returns HTTP 503 and logs technical details on server."""
    transport = ASGITransport(app=error_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with caplog.at_level(logging.ERROR):
            response = await client.get("/trigger/model-load-error")

            assert response.status_code == 503
            data = response.json()

            # Sanitized structure
            assert data["success"] is False
            assert data["error"] == "model_load_error"
            assert data["model"] == "scrfd_500m"
            assert data["device"] == "CPU"

            # Must NEVER expose internal file paths or stack traces to client
            assert "/var/secret" not in data.get("detail", "")
            assert "Traceback" not in response.text

            # Technical diagnostics captured in server logs
            assert "AI Internal Server Error" in caplog.text
            assert "ModelLoadError" in caplog.text


# -------------------------------------------------------------------------
# Test 2: InferenceError Mapping & Traceback Shielding
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_inference_error_mapped_to_500_without_stack_trace(error_test_app, caplog):
    """
    Verify InferenceError returns HTTP 500, shields client from C++/CUDA
    memory pointers/tracebacks, and logs full traceback on server side.
    """
    transport = ASGITransport(app=error_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with caplog.at_level(logging.ERROR):
            response = await client.get("/trigger/inference-error")

            assert response.status_code == 500
            data = response.json()

            assert data["success"] is False
            assert data["error"] == "inference_error"
            assert data["message"] == "AI model inference processing failed."

            # Shield internal memory pointers or C++ execution frames
            assert "0x7fff" not in response.text
            assert "Segmentation fault" not in response.text
            assert "Traceback" not in response.text

            # Server log captures full technical info with traceback
            assert "InferenceError" in caplog.text


# -------------------------------------------------------------------------
# Test 3: InvalidImageError & Subclasses Mapping
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_invalid_image_exceptions_mapped_to_400_and_413(error_test_app):
    """Verify InvalidImageError hierarchy maps to HTTP 400 and 413."""
    transport = ASGITransport(app=error_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Base InvalidImageError -> 400
        res1 = await client.get("/trigger/invalid-image-error")
        assert res1.status_code == 400
        assert res1.json()["error"] == "invalid_image"

        # CorruptedImageException -> 400
        res2 = await client.get("/trigger/corrupted-image")
        assert res2.status_code == 400
        assert res2.json()["error"] == "corrupted_image"

        # EmptyImageException -> 400
        res3 = await client.get("/trigger/empty-image")
        assert res3.status_code == 400
        assert res3.json()["error"] == "empty_file"

        # ImageTooLargeException -> 413
        res4 = await client.get("/trigger/image-too-large")
        assert res4.status_code == 413
        assert res4.json()["error"] == "payload_too_large"


# -------------------------------------------------------------------------
# Test 4: UnsupportedImageError Mapping
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_unsupported_image_error_mapped_to_400(error_test_app):
    """Verify UnsupportedImageError returns HTTP 400."""
    transport = ASGITransport(app=error_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/trigger/unsupported-image-error")
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "unsupported_media_type"


# -------------------------------------------------------------------------
# Test 5: AIConfigurationError Mapping
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_ai_configuration_error_mapped_to_500(error_test_app, caplog):
    """Verify AIConfigurationError returns HTTP 500 and logs technical details."""
    transport = ASGITransport(app=error_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with caplog.at_level(logging.ERROR):
            response = await client.get("/trigger/ai-config-error")
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert data["error"] == "ai_configuration_error"
            assert "AIConfigurationError" in caplog.text


# -------------------------------------------------------------------------
# Test 6: DeviceNotAvailableError Mapping
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_device_not_available_error_mapped_to_503(error_test_app, caplog):
    """Verify DeviceNotAvailableError returns HTTP 503 and logs technical details."""
    transport = ASGITransport(app=error_test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with caplog.at_level(logging.ERROR):
            response = await client.get("/trigger/device-not-available")
            assert response.status_code == 503
            data = response.json()
            assert data["success"] is False
            assert data["error"] == "device_not_available"
            assert "DeviceNotAvailableError" in caplog.text


# -------------------------------------------------------------------------
# Test 7: Unhandled Exception Safety Net (Zero Stack Trace Leakage)
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_unhandled_crash_shields_stack_trace_and_logs_server(error_test_app, caplog):
    """
    Verify completely unexpected Python crashes return generic HTTP 500,
    never leak stack traces or variable names to user, and log full traceback on server.
    """
    transport = ASGITransport(app=error_test_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with caplog.at_level(logging.CRITICAL):
            response = await client.get("/trigger/unhandled-crash")

            assert response.status_code == 500
            data = response.json()

            # Client receives generic safe message
            assert data["success"] is False
            assert data["error"] == "internal_server_error"
            assert "unexpected internal server error" in data["message"]

            # Client response must NEVER contain traceback or sensitive identifiers
            assert "Traceback" not in response.text
            assert "NoneType" not in response.text
            assert "weights" not in response.text
            assert "File \"" not in response.text

            # Server log captures full traceback for debugging
            assert "Unhandled Server Exception" in caplog.text
            assert "ValueError" in caplog.text
