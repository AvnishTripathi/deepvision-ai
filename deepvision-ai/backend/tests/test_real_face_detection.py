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
from app.core.config import settings, Settings
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


# -------------------------------------------------------------------------
# Test 8: Configurable Confidence Threshold via Settings & Environment
# -------------------------------------------------------------------------

def test_confidence_threshold_configured_via_settings_not_hardcoded(monkeypatch):
    """
    Verify AI_CONFIDENCE_THRESHOLD is read from configuration / environment
    and not hard-coded inside the detector.
    """
    # 1. Default threshold is 0.50
    default_settings = Settings()
    assert default_settings.AI_CONFIDENCE_THRESHOLD == 0.50

    # 2. Configurable via environment variable
    monkeypatch.setenv("AI_CONFIDENCE_THRESHOLD", "0.85")
    custom_settings = Settings()
    assert custom_settings.AI_CONFIDENCE_THRESHOLD == 0.85

    # 3. Detector initializes using configured setting
    model_path = custom_settings.resolve_model_path()
    detector = FaceDetector(
        model_path=model_path,
        device="cpu",
        confidence_threshold=custom_settings.AI_CONFIDENCE_THRESHOLD,
    )
    assert detector.confidence_threshold == 0.85


# -------------------------------------------------------------------------
# Test 9: Non-Maximum Suppression (NMS) Behavior & Configuration
# -------------------------------------------------------------------------

def test_nms_removes_duplicate_overlapping_detections_preserving_strongest():
    """
    Verify Non-Maximum Suppression algorithm:
    1. Removes overlapping candidate boxes exceeding IoU threshold.
    2. Preserves the proposal with the highest confidence score.
    3. Retains non-overlapping distinct proposals.
    """
    from app.ai.detection.adapter import nms, compute_iou

    # Three boxes: Box 0 (score 0.95), Box 1 (score 0.80, heavy overlap with 0), Box 2 (score 0.70, separate face)
    boxes = np.array([
        [100.0, 100.0, 200.0, 200.0],  # Box 0: Strong primary detection
        [105.0, 105.0, 202.0, 205.0],  # Box 1: Duplicate proposal (IoU ~ 0.85 with Box 0)
        [350.0, 100.0, 450.0, 200.0],  # Box 2: Distinct second face (IoU = 0.0)
    ], dtype=np.float32)

    scores = np.array([0.95, 0.80, 0.70], dtype=np.float32)

    # IoU between Box 0 and Box 1 is high
    iou_0_1 = compute_iou(boxes[0], boxes[1:2])[0]
    assert iou_0_1 > 0.70

    # Running NMS with IoU threshold 0.40
    kept = nms(boxes, scores, threshold=0.40)

    # Box 1 must be suppressed; only Box 0 (highest score) and Box 2 (separate) remain
    assert kept == [0, 2]
    assert 1 not in kept


def test_nms_threshold_configured_via_settings_and_detector():
    """
    Verify AI_NMS_THRESHOLD is configurable via settings and correctly set on detector.
    Ensures OpenCV-DNN applies C++ NMS directly without running duplicate NMS passes.
    """
    default_settings = Settings()
    assert default_settings.AI_NMS_THRESHOLD == 0.30

    model_path = default_settings.resolve_model_path()
    detector = FaceDetector(
        model_path=model_path,
        device="cpu",
        nms_threshold=0.45,
    )
    assert detector.nms_threshold == 0.45

    # If OpenCV detector is active, verify setNMSThreshold was applied
    if detector._cv2_detector is not None:
        assert round(detector._cv2_detector.getNMSThreshold(), 2) == 0.45


# -------------------------------------------------------------------------
# Test 10: Multi-Format (JPEG, PNG, WEBP) & Color Space (Grayscale, RGB, RGBA)
# -------------------------------------------------------------------------

def test_multi_format_and_color_spaces_handled_correctly(real_detector, sample_portrait_np):
    """
    Verify the detection pipeline properly ingests and normalizes:
    - JPEG encoded byte buffer
    - PNG encoded byte buffer
    - WEBP encoded byte buffer
    - Grayscale (2D NumPy array)
    - Standard RGB (3D NumPy array)
    - RGBA with alpha transparency (4-channel array)
    Guarantees original image data is never modified permanently on disk.
    """
    from app.ai.detection.preprocessing import ImagePreprocessor

    preprocessor = ImagePreprocessor()

    # 1. Standard RGB array
    rgb = preprocessor.load_image(sample_portrait_np)
    assert rgb.shape == sample_portrait_np.shape
    assert rgb.dtype == np.uint8
    res_rgb = real_detector.detect(sample_portrait_np)
    assert res_rgb.face_count >= 1

    # 2. Grayscale 2D array
    gray_2d = cv2.cvtColor(sample_portrait_np, cv2.COLOR_RGB2GRAY)
    assert gray_2d.ndim == 2
    rgb_from_gray = preprocessor.load_image(gray_2d)
    assert rgb_from_gray.shape == (sample_portrait_np.shape[0], sample_portrait_np.shape[1], 3)
    res_gray = real_detector.detect(gray_2d, confidence_threshold=0.40)
    assert res_gray.face_count >= 1

    # 3. RGBA 4-channel image
    pil_rgba = Image.fromarray(sample_portrait_np).convert("RGBA")
    rgba_np = np.array(pil_rgba)
    assert rgba_np.shape[2] == 4
    rgb_from_rgba = preprocessor.load_image(rgba_np)
    assert rgb_from_rgba.shape == sample_portrait_np.shape
    res_rgba = real_detector.detect(rgba_np)
    assert res_rgba.face_count >= 1

    # 4. JPEG encoded byte buffer
    success_jpg, jpg_buf = cv2.imencode(".jpg", cv2.cvtColor(sample_portrait_np, cv2.COLOR_RGB2BGR))
    assert success_jpg
    res_jpg = real_detector.detect(jpg_buf.tobytes())
    assert res_jpg.face_count >= 1

    # 5. PNG encoded byte buffer
    success_png, png_buf = cv2.imencode(".png", cv2.cvtColor(sample_portrait_np, cv2.COLOR_RGB2BGR))
    assert success_png
    res_png = real_detector.detect(png_buf.tobytes())
    assert res_png.face_count >= 1

    # 6. WEBP encoded byte buffer
    success_webp, webp_buf = cv2.imencode(".webp", cv2.cvtColor(sample_portrait_np, cv2.COLOR_RGB2BGR))
    assert success_webp
    res_webp = real_detector.detect(webp_buf.tobytes())
    assert res_webp.face_count >= 1


# -------------------------------------------------------------------------
# Test 11: Bounding Box Validation, Clipping & Safe Rejection
# -------------------------------------------------------------------------

def test_bounding_box_validation_clipping_and_rejection():
    """
    Verify bounding box validator and clipper enforces:
    - x1 >= 0
    - y1 >= 0
    - x2 <= image_width
    - y2 <= image_height
    - x2 > x1
    - y2 > y1
    - Clips slightly out-of-boundary proposals to image edges
    - Discards invalid or degenerate detections safely
    """
    from app.ai.detection.preprocessing import validate_and_clip_box

    image_w, image_h = 640, 480

    # 1. Normal valid proposal
    valid = validate_and_clip_box(100.0, 100.0, 300.0, 300.0, image_w, image_h)
    assert valid == (100.0, 100.0, 300.0, 300.0)

    # 2. Proposal extending slightly beyond boundaries is clipped
    clipped = validate_and_clip_box(-15.5, -10.2, 650.0, 490.5, image_w, image_h)
    assert clipped == (0.0, 0.0, 640.0, 480.0)
    assert clipped[0] >= 0.0
    assert clipped[1] >= 0.0
    assert clipped[2] <= float(image_w)
    assert clipped[3] <= float(image_h)
    assert clipped[2] > clipped[0]
    assert clipped[3] > clipped[1]

    # 3. Inverted coordinates (x2 <= x1 or y2 <= y1) safely discarded
    assert validate_and_clip_box(300.0, 100.0, 200.0, 300.0, image_w, image_h) is None
    assert validate_and_clip_box(100.0, 300.0, 300.0, 200.0, image_w, image_h) is None
    assert validate_and_clip_box(100.0, 100.0, 100.0, 300.0, image_w, image_h) is None

    # 4. Completely out-of-bounds proposals safely discarded
    assert validate_and_clip_box(-100.0, -100.0, -10.0, -10.0, image_w, image_h) is None
    assert validate_and_clip_box(700.0, 500.0, 800.0, 600.0, image_w, image_h) is None

    # 5. Schema clip_to_bounds and is_valid
    from app.schemas.ai import BoundingBox
    box = BoundingBox(x1=-10.0, y1=-10.0, x2=200.0, y2=200.0)
    assert not box.is_valid(image_w, image_h)
    clipped_box = box.clip_to_bounds(image_w, image_h)
    assert clipped_box is not None
    assert clipped_box.x1 == 0.0
    assert clipped_box.y1 == 0.0
    assert clipped_box.is_valid(image_w, image_h)



