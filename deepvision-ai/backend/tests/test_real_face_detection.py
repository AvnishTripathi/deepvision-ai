"""
DeepVision AI — Real Face Detection Pipeline Test Suite
Validates real, production-quality face detection using physical model weights:
- Real model loading and operational status
- Accurate face localization on genuine portrait photos
- 5-point facial landmark precision
- Face quality metrics computation on cropped face ROIs
- Zero false positives on non-face scenes (blank images)
- Multi-face localization on composite scenes
- API endpoint integration via POST /api/v1/ai/detect
- Inference latency telemetry under 150ms on CPU
- Model unload and lifecycle management
"""

import os
import io
import time
import numpy as np
import pytest
import cv2
from PIL import Image
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import settings
from app.services.ai_service import AIService, get_ai_service
from app.ai.detection.detector import FaceDetector
from app.ai.detection.schemas import FrameInferenceResult, FaceDetection, BoundingBox, LandmarkPoint


FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_face.jpg")


@pytest.fixture
def sample_portrait_bytes() -> bytes:
    """Loads the real face portrait fixture as byte stream."""
    assert os.path.exists(FIXTURE_PATH), f"Fixture not found at {FIXTURE_PATH}"
    with open(FIXTURE_PATH, "rb") as f:
        return f.read()


@pytest.fixture
def sample_portrait_np(sample_portrait_bytes) -> np.ndarray:
    """Decodes fixture into uint8 RGB NumPy array."""
    img_bgr = cv2.imread(FIXTURE_PATH)
    assert img_bgr is not None, "Failed reading fixture image"
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture
def real_detector() -> FaceDetector:
    """Instantiates a real operational FaceDetector using resolved model path."""
    model_path = settings.resolve_model_path()
    assert model_path is not None, "Physical ONNX model path could not be resolved"
    assert os.path.exists(model_path), f"Physical model file not found at {model_path}"
    detector = FaceDetector(model_path=model_path, device="cpu")
    assert detector.is_loaded is True
    return detector


# -------------------------------------------------------------------------
# Test 1: Real Model Loading & Operational State
# -------------------------------------------------------------------------

def test_real_model_loading():
    """Verify physical model file exists and FaceDetector initializes as operational."""
    model_path = settings.resolve_model_path()
    assert model_path is not None
    assert os.path.exists(model_path)
    assert model_path.endswith(".onnx")

    detector = FaceDetector(model_path=model_path, device="cpu")
    assert detector.is_loaded is True
    assert detector.model_path == model_path


# -------------------------------------------------------------------------
# Test 2: Real Face Detection on Genuine Portrait Photo
# -------------------------------------------------------------------------

def test_real_face_detection_on_portrait_image(real_detector, sample_portrait_np):
    """
    Verify real face detector accurately localizes face, extracts landmarks,
    and measures quality on a real photographic input.
    """
    h, w = sample_portrait_np.shape[:2]
    result: FrameInferenceResult = real_detector.detect(sample_portrait_np, confidence_threshold=0.5)

    assert result.image_width == w
    assert result.image_height == h
    assert result.face_count >= 1
    assert len(result.faces) == result.face_count

    # Inspect the detected face
    face: FaceDetection = result.faces[0]
    assert face.confidence >= 0.5
    assert face.confidence <= 1.0

    # Bounding box bounds check
    bbox: BoundingBox = face.bbox
    assert bbox.x1 >= 0.0
    assert bbox.y1 >= 0.0
    assert bbox.x2 > bbox.x1
    assert bbox.y2 > bbox.y1
    assert bbox.x2 <= w + 10.0  # Tolerant of boundary edge
    assert bbox.y2 <= h + 10.0
    assert bbox.area > 1000.0  # Genuine face occupies substantial area

    # Normalized bounding box check
    assert face.normalized_bbox is not None
    assert 0.0 <= face.normalized_bbox.x1 <= 1.0
    assert 0.0 <= face.normalized_bbox.y1 <= 1.0
    assert 0.0 <= face.normalized_bbox.x2 <= 1.0
    assert 0.0 <= face.normalized_bbox.y2 <= 1.0

    # 5 facial landmarks check
    assert face.landmarks is not None
    assert len(face.landmarks) == 5
    expected_names = {"right_eye", "left_eye", "nose_tip", "mouth_right", "mouth_left"}
    actual_names = {lm.name for lm in face.landmarks}
    assert actual_names == expected_names

    for lm in face.landmarks:
        assert lm.x > 0.0
        assert lm.y > 0.0

    # Quality metrics check
    assert face.quality is not None
    assert face.quality.sharpness > 10.0
    assert 0.0 <= face.quality.brightness <= 255.0
    assert face.quality.contrast > 5.0
    assert 0.0 <= face.quality.overall_quality <= 1.0
    assert face.quality.is_usable is True

    # Performance latency check (CPU inference must be fast)
    assert result.processing_time_ms > 0.0
    assert result.processing_time_ms < 150.0  # Sub-150ms on CPU
    assert result.inference_time_ms > 0.0


# -------------------------------------------------------------------------
# Test 3: Zero False Positives on Blank Non-Face Images
# -------------------------------------------------------------------------

def test_real_face_detection_on_blank_image(real_detector):
    """Verify real detector does NOT hallucinate or fabricate faces on blank backgrounds."""
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = real_detector.detect(blank_image, confidence_threshold=0.5)

    assert result.face_count == 0
    assert result.faces == []
    assert len(result.detections) == 0


def test_real_face_detection_on_random_noise_image(real_detector):
    """Verify real detector does NOT hallucinate faces on random noise."""
    np.random.seed(42)
    noise_image = np.random.randint(0, 256, (400, 400, 3), dtype=np.uint8)
    result = real_detector.detect(noise_image, confidence_threshold=0.6)

    assert result.face_count == 0
    assert result.faces == []


# -------------------------------------------------------------------------
# Test 4: Multi-Face Detection on Composite Scene
# -------------------------------------------------------------------------

def test_real_face_detection_on_multi_face_scene(real_detector, sample_portrait_np):
    """Verify real detector identifies multiple distinct faces in a composite image."""
    # Place two portraits side by side (width = 800, height = 400)
    composite = np.hstack([sample_portrait_np, sample_portrait_np])
    h, w = composite.shape[:2]

    result = real_detector.detect(composite, confidence_threshold=0.5)
    assert result.face_count == 2
    assert len(result.faces) == 2

    face_a = result.faces[0]
    face_b = result.faces[1]

    # Both faces must have non-zero bounding boxes
    assert face_a.bbox.width > 50.0
    assert face_b.bbox.width > 50.0

    # Ensure horizontal separation between the two detected faces
    centers_x = sorted([face_a.bbox.x1 + face_a.bbox.width / 2, face_b.bbox.x1 + face_b.bbox.width / 2])
    assert centers_x[1] - centers_x[0] > 200.0  # Separated across the two halves


# -------------------------------------------------------------------------
# Test 5: End-to-End Live API Endpoint with Real Image Upload
# -------------------------------------------------------------------------

@pytest.mark.anyio
async def test_real_face_detection_api_endpoint(sample_portrait_bytes):
    """
    Verify POST /api/v1/ai/detect processes an uploaded real photo,
    executes real inference, and returns structured JSON conforming to specification.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("portrait.jpg", sample_portrait_bytes, "image/jpeg")}
        response = await client.post("/api/v1/ai/detect", files=files)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "model" in data
        assert "device" in data
        assert data["image"]["width"] == 400
        assert data["image"]["height"] == 400
        assert data["processing_time_ms"] > 0.0

        # Faces array
        assert len(data["faces"]) >= 1
        first_face = data["faces"][0]
        assert first_face["confidence"] >= 0.5
        assert first_face["bbox"]["x1"] > 0
        assert first_face["bbox"]["y1"] > 0
        assert first_face["bbox"]["x2"] > first_face["bbox"]["x1"]
        assert first_face["bbox"]["y2"] > first_face["bbox"]["y1"]

        # Landmarks
        assert first_face["landmarks"] is not None
        assert len(first_face["landmarks"]) == 5


# -------------------------------------------------------------------------
# Test 6: Confidence Threshold Filter on Real Detections
# -------------------------------------------------------------------------

def test_real_face_detection_confidence_cutoff(real_detector, sample_portrait_np):
    """Verify confidence threshold filter correctly excludes detections below threshold."""
    # With normal threshold, face is detected
    res_normal = real_detector.detect(sample_portrait_np, confidence_threshold=0.5)
    assert res_normal.face_count >= 1
    detected_conf = res_normal.faces[0].confidence

    # With impossible threshold higher than detected confidence, returns 0 faces
    res_strict = real_detector.detect(sample_portrait_np, confidence_threshold=detected_conf + 0.1)
    assert res_strict.face_count == 0


# -------------------------------------------------------------------------
# Test 7: Model Lifecycle Unload and Reload
# -------------------------------------------------------------------------

def test_real_detector_lifecycle(real_detector, sample_portrait_np):
    """Verify FaceDetector unloads resources cleanly and can reload without errors."""
    assert real_detector.is_loaded is True

    # Run detection
    res1 = real_detector.detect(sample_portrait_np)
    assert res1.face_count >= 1

    # Unload model
    saved_path = real_detector.model_path
    real_detector.unload()
    assert real_detector.is_loaded is False

    # Reload model
    real_detector.load_model(saved_path, "cpu")
    assert real_detector.is_loaded is True

    # Verify detection works after reload
    res2 = real_detector.detect(sample_portrait_np)
    assert res2.face_count >= 1
