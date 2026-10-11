"""
DeepVision AI — Step 7: Production Hardening, Security & Rate Limiting Test Suite
Validates:
- Defensive HTTP security response headers
- Rate limiting on image analysis and video endpoints
- Maximum concurrent video job capacity protection
- Deep upload security and sanitization
- Stack trace isolation and error response sanitization
"""

import io
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.core.config import settings
from app.core.rate_limit import (
    parse_rate_limit,
    InMemorySlidingWindowLimiter,
    _in_memory_limiter,
)
from app.core.exceptions import (
    VideoJobCapacityExceededException,
    VideoTooLargeException,
    UnsupportedImageError,
)
from app.core.image_security import sanitize_filename
from app.core.video_security import sanitize_video_filename
from app.services.video_job_manager import InMemoryVideoJobManager, VideoJobStatus


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# -----------------------------------------------------------------------------
# 1. Defensive HTTP Security Headers
# -----------------------------------------------------------------------------

def test_security_headers_present_on_endpoints(client):
    """Verifies that all responses include defensive HTTP security headers."""
    response = client.get("/")
    assert response.status_code == 200

    # Required defensive headers
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "X-Request-ID" in response.headers


def test_request_id_preserved_or_generated(client):
    """Verifies that client X-Request-ID is honored or a new trace ID is generated."""
    custom_id = "test-trace-uuid-12345"
    response = client.get("/", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


# -----------------------------------------------------------------------------
# 2. Rate Limiting Parser and Logic
# -----------------------------------------------------------------------------

def test_parse_rate_limit_various_units():
    """Validates rate limit string parser for seconds, minutes, hours, days."""
    assert parse_rate_limit("60/minute") == (60, 60)
    assert parse_rate_limit("10/min") == (10, 60)
    assert parse_rate_limit("5/second") == (5, 1)
    assert parse_rate_limit("100/hour") == (100, 3600)
    assert parse_rate_limit("1000/day") == (1000, 86400)
    assert parse_rate_limit("30") == (30, 60)


def test_in_memory_sliding_window_rate_limiter():
    """Tests the in-memory sliding window rate limiter directly."""
    limiter = InMemorySlidingWindowLimiter()
    key = "test_client_1"
    max_req = 3
    window = 10

    # First 3 requests allowed
    allowed1, rem1, retry1 = limiter.is_allowed(key, max_req, window)
    assert allowed1 is True
    assert rem1 == 2

    allowed2, rem2, retry2 = limiter.is_allowed(key, max_req, window)
    assert allowed2 is True
    assert rem2 == 1

    allowed3, rem3, retry3 = limiter.is_allowed(key, max_req, window)
    assert allowed3 is True
    assert rem3 == 0

    # 4th request blocked with retry_after > 0
    allowed4, rem4, retry4 = limiter.is_allowed(key, max_req, window)
    assert allowed4 is False
    assert rem4 == 0
    assert retry4 > 0


def test_endpoint_rate_limiting_exceeded(client):
    """Verifies that exceeding endpoint rate limits returns HTTP 429 with Retry-After header."""
    _in_memory_limiter.reset()

    # Create dummy 1x1 image
    img = Image.new("RGB", (100, 100), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    # Temporarily set rate limit to 2 per minute for testing
    with patch.object(settings, "RATE_LIMIT_IMAGE_ANALYSIS", "2/minute"):
        with patch.object(settings, "RATE_LIMIT_ENABLED", True):
            # 1st request
            r1 = client.post(
                "/api/v1/ai/detect",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
            # 2nd request
            r2 = client.post(
                "/api/v1/ai/detect",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
            # 3rd request should hit rate limit
            r3 = client.post(
                "/api/v1/ai/detect",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )

            assert r3.status_code == 429
            assert "Retry-After" in r3.headers
            data = r3.json()
            assert data["detail"]["error"] == "rate_limit_exceeded"

    _in_memory_limiter.reset()


# -----------------------------------------------------------------------------
# 3. Maximum Concurrent Video Job Protection
# -----------------------------------------------------------------------------

def test_video_job_concurrency_limit_enforced():
    """Verifies that submit_job rejects requests when active jobs reach VIDEO_MAX_CONCURRENT_JOBS."""
    mock_storage = MagicMock()
    mock_video_service = MagicMock()

    manager = InMemoryVideoJobManager(
        storage=mock_storage,
        video_service=mock_video_service,
        max_workers=2,
    )

    with patch.object(settings, "VIDEO_MAX_CONCURRENT_JOBS", 2):
        # Fill capacity with 2 dummy queued jobs
        manager._jobs["job1"] = MagicMock(status=VideoJobStatus.QUEUED)
        manager._jobs["job2"] = MagicMock(status=VideoJobStatus.PROCESSING)

        with pytest.raises(VideoJobCapacityExceededException) as exc_info:
            manager.submit_job(
                input_filepath="fake.mp4",
                filename="fake.mp4",
            )

        assert "Maximum concurrent video processing jobs (2) exceeded" in str(exc_info.value)
        assert exc_info.value.status_code == 429


# -----------------------------------------------------------------------------
# 4. Upload Security & Filename Sanitization
# -----------------------------------------------------------------------------

def test_image_filename_sanitization_removes_traversal():
    """Verifies that filename sanitization completely strips path traversal sequences."""
    traversal_name = "../../../etc/passwd"
    clean = sanitize_filename(traversal_name)
    assert "../" not in clean
    assert clean.endswith(".jpg")

    shell_name = "test;rm -rf /;payload.png"
    clean_shell = sanitize_filename(shell_name)
    assert ";" not in clean_shell
    assert " " not in clean_shell
    assert clean_shell.endswith(".png")


def test_video_filename_sanitization_removes_traversal():
    """Verifies video filename sanitization."""
    traversal_name = "..\\..\\windows\\system32\\calc.exe.mp4"
    clean = sanitize_video_filename(traversal_name)
    assert ".." not in clean
    assert "\\" not in clean
    assert clean.endswith(".mp4")


# -----------------------------------------------------------------------------
# 5. Stack Trace Exposure Prevention
# -----------------------------------------------------------------------------

def test_unhandled_exception_does_not_leak_traceback(client):
    """Verifies that an internal exception returns a sanitized 500 JSON without stack trace."""
    with patch("app.api.v1.ai.read_upload_stream_safely", side_effect=RuntimeError("Database kernel dump /var/secret")):
        img = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        response = client.post(
            "/api/v1/ai/detect",
            files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")},
        )

        assert response.status_code == 500
        data = response.json()
        assert data["success"] is False
        assert "traceback" not in response.text.lower()
        assert "/var/secret" not in response.text
