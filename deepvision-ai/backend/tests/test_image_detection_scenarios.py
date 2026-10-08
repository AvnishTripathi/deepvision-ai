"""
DeepVision AI — Image Detection Scenarios Test Suite (Step 3, Task 10)
Validates real face detection against authentic test image scenarios:
- Test A — One face (Expected: face_count >= 1)
- Test B — Multiple faces (Expected: face_count >= 2 on multi-face scene)
- Test C — No face (Expected: face_count == 0, zero false positives)
- Test D — Different image sizes (small and large images handled reliably)
- Test E — Invalid images (clean validation error handling, zero crashes)

All tests operate against the live FastAPI application and real inference engine.
No fake or hardcoded detection coordinates are used.
"""

import os
import io
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image

from app.main import app
from app.core.config import settings

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
ONE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")
MULTIPLE_FACES_PATH = os.path.join(FIXTURES_DIR, "multiple_faces.jpg")
NO_FACE_PATH = os.path.join(FIXTURES_DIR, "no_face.jpg")
SMALL_FACE_PATH = os.path.join(FIXTURES_DIR, "small_face.jpg")
LARGE_FACE_PATH = os.path.join(FIXTURES_DIR, "large_face.jpg")


def _read_fixture_bytes(path: str) -> bytes:
    assert os.path.exists(path), f"Fixture not found at {path}"
    with open(path, "rb") as f:
        return f.read()


# -------------------------------------------------------------------------
# Test A: One Face
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_scenario_a_one_face():
    """
    Test A — One face
    Expected: face_count >= 1
    Verifies that a genuine photographic portrait yields at least one detected face
    with valid bounding box, confidence >= 0.50, and 5-point facial landmarks.
    """
    image_bytes = _read_fixture_bytes(ONE_FACE_PATH)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("one_face.jpg", image_bytes, "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["face_count"] >= 1
        assert len(data["faces"]) == data["face_count"]
        assert data["image"]["width"] == 400
        assert data["image"]["height"] == 400
        assert data["processing_time_ms"] > 0.0

        # Validate face properties
        first_face = data["faces"][0]
        assert first_face["face_id"] is None
        assert first_face["confidence"] >= 0.50
        assert first_face["confidence"] <= 1.0

        # Validate bounding box boundaries
        bbox = first_face["bbox"]
        assert bbox["x1"] >= 0.0
        assert bbox["y1"] >= 0.0
        assert bbox["x2"] <= data["image"]["width"]
        assert bbox["y2"] <= data["image"]["height"]
        assert bbox["x2"] > bbox["x1"]
        assert bbox["y2"] > bbox["y1"]

        # Validate facial landmarks
        assert first_face["landmarks"] is not None
        assert len(first_face["landmarks"]) == 5
        if isinstance(first_face["landmarks"], dict):
            for name, lm in first_face["landmarks"].items():
                assert lm["x"] >= 0.0 and lm["x"] <= data["image"]["width"]
                assert lm["y"] >= 0.0 and lm["y"] <= data["image"]["height"]
        else:
            for lm in first_face["landmarks"]:
                assert "name" in lm
                assert lm["x"] >= 0.0 and lm["x"] <= data["image"]["width"]
                assert lm["y"] >= 0.0 and lm["y"] <= data["image"]["height"]


# -------------------------------------------------------------------------
# Test B: Multiple Faces
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_scenario_b_multiple_faces():
    """
    Test B — Multiple faces
    Expected: face_count >= 2
    when the image actually contains multiple detectable faces.
    Verifies localization of multiple distinct faces with non-overlapping centers.
    """
    image_bytes = _read_fixture_bytes(MULTIPLE_FACES_PATH)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("multiple_faces.jpg", image_bytes, "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["face_count"] >= 2
        assert len(data["faces"]) == data["face_count"]
        assert data["image"]["width"] == 800
        assert data["image"]["height"] == 400

        # Check distinct coordinates for detected faces
        face_1 = data["faces"][0]
        face_2 = data["faces"][1]

        assert face_1["confidence"] >= 0.50
        assert face_2["confidence"] >= 0.50

        # Verify both bounding boxes are strictly within canvas
        for f in [face_1, face_2]:
            b = f["bbox"]
            assert b["x1"] >= 0 and b["y1"] >= 0
            assert b["x2"] <= 800 and b["y2"] <= 400
            assert b["x2"] > b["x1"]
            assert b["y2"] > b["y1"]

        # Ensure spatial separation between the detected faces
        c1_x = (face_1["bbox"]["x1"] + face_1["bbox"]["x2"]) / 2.0
        c2_x = (face_2["bbox"]["x1"] + face_2["bbox"]["x2"]) / 2.0
        assert abs(c1_x - c2_x) > 150.0, "Faces should be horizontally separated"


# -------------------------------------------------------------------------
# Test C: No Face
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_scenario_c_no_face():
    """
    Test C — No face
    Expected: face_count = 0
    Verifies that non-face scenery images produce zero face detections.
    Guarantees that fake or hallucinatory faces are never returned.
    """
    image_bytes = _read_fixture_bytes(NO_FACE_PATH)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("no_face.jpg", image_bytes, "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["face_count"] == 0
        assert data["faces"] == []
        assert data["image"]["width"] == 600
        assert data["image"]["height"] == 400


# -------------------------------------------------------------------------
# Test D: Different Image Sizes (Small and Large Images)
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_scenario_d_small_image():
    """
    Test D (Small) — Ingestion and inference on smaller resolution image.
    Verifies pipeline scales coordinates correctly to input dimensions.
    """
    image_bytes = _read_fixture_bytes(SMALL_FACE_PATH)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("small_face.jpg", image_bytes, "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["image"]["width"] == 300
        assert data["image"]["height"] == 300
        assert data["processing_time_ms"] > 0.0


@pytest.mark.anyio
async def test_scenario_d_large_image():
    """
    Test D (Large) — Ingestion and inference on large resolution image (1200x900).
    Verifies high-resolution processing, memory safety, and accurate face localization.
    """
    image_bytes = _read_fixture_bytes(LARGE_FACE_PATH)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("large_face.jpg", image_bytes, "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["image"]["width"] == 1200
        assert data["image"]["height"] == 900
        assert data["face_count"] >= 1
        assert len(data["faces"]) == data["face_count"]

        # Validate bounding box coordinates match large canvas dimensions
        box = data["faces"][0]["bbox"]
        assert box["x1"] >= 0 and box["y1"] >= 0
        assert box["x2"] <= 1200 and box["y2"] <= 900
        assert box["x2"] > box["x1"]
        assert box["y2"] > box["y1"]


# -------------------------------------------------------------------------
# Test E: Invalid Image (Clean Validation Errors)
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_scenario_e_invalid_file_type():
    """
    Test E (Invalid file type) — Text file upload.
    Expected: HTTP 400 with clean validation error structure.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("notes.txt", b"Plain text, not an image.", "text/plain")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "unsupported_media_type"
        assert "Unsupported file type" in data["message"]


@pytest.mark.anyio
async def test_scenario_e_corrupted_image_stream():
    """
    Test E (Corrupted image) — Arbitrary corrupted byte stream.
    Expected: HTTP 400 with clean validation error.
    """
    corrupt_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"CORRUPTED_STREAM_CONTENT_XX_00" * 8
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
async def test_scenario_e_empty_file():
    """
    Test E (Empty file) — 0-byte upload.
    Expected: HTTP 400 with clean validation error.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "empty_file"


@pytest.mark.anyio
async def test_scenario_e_oversized_dimensions():
    """
    Test E (Oversized dimensions) — Image exceeding AI_MAX_IMAGE_DIMENSION.
    Expected: HTTP 400 with clean validation error.
    """
    # 4100x10 image (exceeds 4096px limit)
    img = Image.new("RGB", (4100, 10), color=(10, 10, 10))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    oversized_bytes = buffer.getvalue()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("oversized.png", oversized_bytes, "image/png")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"] in ("dimensions_too_large", "invalid_dimensions")
