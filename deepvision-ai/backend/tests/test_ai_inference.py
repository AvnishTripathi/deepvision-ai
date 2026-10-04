"""
DeepVision AI — AI Engine & Inference Foundation Test Suite
Validates image preprocessing, letterbox mathematics, quality analytics,
Pydantic result schemas, inference abstraction, and end-to-end detection pipeline.
"""

import numpy as np
import cv2
from PIL import Image
import pytest

from app.ai.preprocessing import (
    load_image,
    letterbox_resize,
    unletterbox_coordinates,
    prepare_input_tensor,
)
from app.ai.analysis.quality import (
    compute_sharpness,
    compute_brightness_and_contrast,
    analyze_face_quality,
)
from app.schemas.ai import (
    BoundingBox,
    LandmarkPoint,
    FaceQualityScore,
    FaceDetectionResult,
    FrameInferenceResult,
    PerformanceMetrics,
)
from app.ai.detection.detector import FaceDetector
from app.ai.inference.runtime import InferenceRuntime
from app.ai.inference.device import resolve_device, is_cuda_available, get_device_telemetry


@pytest.fixture
def sample_synthetic_image():
    """Generates a clean synthetic RGB test image with distinct features."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background gradient
    img[:, :, 0] = 30
    img[:, :, 1] = 40
    img[:, :, 2] = 50
    # Simulated face area (oval/rectangle with features)
    cv2.rectangle(img, (200, 120), (440, 360), (220, 180, 150), -1)
    # Eyes
    cv2.circle(img, (270, 200), 15, (50, 40, 30), -1)
    cv2.circle(img, (370, 200), 15, (50, 40, 30), -1)
    # Nose
    cv2.circle(img, (320, 250), 10, (180, 130, 110), -1)
    # Mouth
    cv2.ellipse(img, (320, 310), (40, 15), 0, 0, 180, (160, 50, 50), -1)
    return img


def test_load_image_formats(sample_synthetic_image):
    """Verify load_image accepts NumPy arrays, PIL Images, and bytes."""
    # 1. NumPy array
    rgb1 = load_image(sample_synthetic_image)
    assert rgb1.shape == (480, 640, 3)
    assert rgb1.dtype == np.uint8

    # 2. PIL Image
    pil_img = Image.fromarray(sample_synthetic_image)
    rgb2 = load_image(pil_img)
    assert rgb2.shape == (480, 640, 3)

    # 3. Encoded byte stream
    success, buffer = cv2.imencode(".png", cv2.cvtColor(sample_synthetic_image, cv2.COLOR_RGB2BGR))
    assert success
    rgb3 = load_image(buffer.tobytes())
    assert rgb3.shape == (480, 640, 3)


def test_letterbox_resize_aspect_ratio(sample_synthetic_image):
    """Verify letterbox_resize preserves aspect ratio and outputs target shape."""
    target_size = (640, 640)
    padded, scale, (pad_x, pad_y) = letterbox_resize(sample_synthetic_image, target_size=target_size)

    assert padded.shape == (640, 640, 3)
    # 640x480 scaled into 640x640: scale should be 1.0 (since width is 640)
    # pad_y should be (640 - 480) / 2 = 80
    assert scale == 1.0
    assert pad_x == 0
    assert pad_y == 80


def test_unletterbox_coordinates():
    """Verify unletterbox coordinates inverts scale and padding accurately."""
    orig_shape = (480, 640)
    scale = 1.0
    padding = (0, 80)

    # Bounding box in letterboxed space (height padded by 80)
    box_letterbox = (200.0, 200.0, 440.0, 440.0)
    unpadded = unletterbox_coordinates(
        box_letterbox[0], box_letterbox[1], box_letterbox[2], box_letterbox[3],
        scale, padding, orig_shape
    )

    assert unpadded[0] == 200.0
    assert unpadded[1] == 120.0  # 200 - 80 = 120
    assert unpadded[2] == 440.0
    assert unpadded[3] == 360.0  # 440 - 80 = 360


def test_prepare_input_tensor(sample_synthetic_image):
    """Verify prepare_input_tensor outputs NCHW normalized float32 tensor."""
    tensor, scale, padding = prepare_input_tensor(sample_synthetic_image, target_size=(640, 640))

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_face_quality_analytics(sample_synthetic_image):
    """Verify sharpness, brightness, and contrast analytics discriminate sharp vs blur."""
    sharp_crop = sample_synthetic_image[120:360, 200:440]
    sharp_score = compute_sharpness(sharp_crop)
    brightness, contrast = compute_brightness_and_contrast(sharp_crop)

    assert sharp_score > 0
    assert 0 <= brightness <= 255
    assert contrast > 0

    # Blur the crop and verify sharpness drops significantly
    blurred_crop = cv2.GaussianBlur(sharp_crop, (15, 15), 0)
    blur_score = compute_sharpness(blurred_crop)
    assert blur_score < sharp_score

    # Full quality evaluation
    quality = analyze_face_quality(sharp_crop)
    assert 0.0 <= quality.overall_quality <= 1.0
    assert isinstance(quality.is_usable, bool)


def test_bounding_box_schema():
    """Verify BoundingBox schema and computed fields (width, height, area)."""
    bbox = BoundingBox(x_min=100.0, y_min=50.0, x_max=250.0, y_max=200.0, confidence=0.98)
    assert bbox.width == 150.0
    assert bbox.height == 150.0
    assert bbox.area == 22500.0


def test_face_detector_end_to_end_pipeline(sample_synthetic_image):
    """Verify FaceDetector end-to-end execution, schemas, and latency telemetry."""
    detector = FaceDetector(device="cpu")
    result: FrameInferenceResult = detector.detect(sample_synthetic_image, confidence_threshold=0.5)

    assert result.image_width == 640
    assert result.image_height == 480
    assert result.face_count >= 1
    assert len(result.faces) == result.face_count

    # Verify detected face structure
    face = result.faces[0]
    assert face.face_index == 0
    assert face.bbox.confidence >= 0.5
    assert len(face.landmarks) == 5
    assert face.quality is not None
    assert 0.0 <= face.quality.overall_quality <= 1.0

    # Verify timing metrics
    assert result.preprocessing_time_ms >= 0.0
    assert result.inference_time_ms >= 0.0
    assert result.postprocessing_time_ms >= 0.0
    assert result.total_time_ms > 0.0
    assert result.device_used == "cpu"


def test_device_telemetry_and_resolution():
    """Verify hardware resolution logic and device telemetry."""
    telemetry = get_device_telemetry()
    assert "active_device" in telemetry
    assert "cuda_available" in telemetry
    assert "cpu_cores" in telemetry
    assert telemetry["cpu_cores"] >= 1
    assert telemetry["recommended_threads"] >= 1
    assert "optimal_providers" in telemetry

    # Test device fallback
    assert resolve_device("cpu") == "cpu"
    # When CUDA is not available, auto or cuda gracefully falls back to cpu
    if not is_cuda_available():
        assert resolve_device("cuda") == "cpu"
        assert resolve_device("auto") == "cpu"


def test_ai_engine_package_imports():
    """Verify clean modular structure imports from backend/app/ai packages."""
    import app.ai as ai
    import app.ai.detection as detection
    import app.ai.inference as inference
    import app.ai.analysis as analysis
    import app.ai.tracking as tracking
    import app.ai.recognition as recognition
    import app.ai.common as common

    assert hasattr(ai, "FaceDetector")
    assert hasattr(ai, "InferenceRuntime")
    assert hasattr(ai, "BoundingBox")
    assert hasattr(detection, "FaceDetector")
    assert hasattr(detection, "BaseFaceDetector")
    assert hasattr(detection, "load_image")
    assert hasattr(inference, "InferenceRuntime")
    assert hasattr(inference, "resolve_device")
    assert hasattr(analysis, "analyze_face_quality")
    assert hasattr(common, "ModelArchitecture")

