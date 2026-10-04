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
    ImagePreprocessor,
    ImagePreprocessingConfig,
    PreprocessedImage,
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
    NormalizedBoundingBox,
    LandmarkPoint,
    FaceQualityScore,
    FaceDetection,
    FaceDetectionResult,
    DetectionResponse,
    FrameInferenceResult,
    PerformanceMetrics,
)
from app.ai.detection.detector import FaceDetector
from app.ai.common.types import DeviceType
from app.ai.inference.runtime import InferenceRuntime
from app.ai.inference.device import (
    resolve_device,
    is_gpu_available,
    is_cuda_available,
    get_available_gpu_providers,
    get_optimal_providers,
    get_device_telemetry,
)
from app.core.exceptions import (
    CorruptedImageException,
    EmptyImageException,
    UnsupportedImageFormatException,
    InvalidImageDimensionsException,
    ImageTooLargeException,
)
from app.core.config import Settings


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
    """Verify BoundingBox schema (x1, y1, x2, y2), computed metrics, and normalization."""
    # Test primary x1, y1, x2, y2 coordinates
    bbox = BoundingBox(x1=100.0, y1=80.0, x2=250.0, y2=260.0)
    assert bbox.x1 == 100.0
    assert bbox.y1 == 80.0
    assert bbox.x2 == 250.0
    assert bbox.y2 == 260.0
    assert bbox.width == 150.0
    assert bbox.height == 180.0
    assert bbox.area == 27000.0

    # Test normalization
    norm = bbox.to_normalized(image_width=1000, image_height=1000)
    assert norm.x1 == 0.1
    assert norm.y1 == 0.08
    assert norm.x2 == 0.25
    assert norm.y2 == 0.26
    assert norm.width == 0.15
    assert norm.height == 0.18

    # Convert back to pixel
    restored = norm.to_pixel(image_width=1000, image_height=1000)
    assert restored.x1 == 100.0
    assert restored.y1 == 80.0
    assert restored.x2 == 250.0
    assert restored.y2 == 260.0


def test_face_detection_schema_and_json_serialization():
    """Verify FaceDetection schema fields, defaults, and JSON serialization."""
    detection = FaceDetection(
        face_id=None,
        bbox=BoundingBox(x1=100.0, y1=80.0, x2=250.0, y2=260.0),
        confidence=0.97,
        landmarks=None,
    )
    assert detection.face_id is None
    assert detection.bbox.x1 == 100.0
    assert detection.bbox.y1 == 80.0
    assert detection.bbox.x2 == 250.0
    assert detection.bbox.y2 == 260.0
    assert detection.confidence == 0.97
    assert detection.landmarks is None

    # Verify JSON structure conforms to specification
    dump = detection.model_dump(mode="json")
    assert dump["face_id"] is None
    assert dump["bbox"]["x1"] == 100.0
    assert dump["bbox"]["y1"] == 80.0
    assert dump["bbox"]["x2"] == 250.0
    assert dump["bbox"]["y2"] == 260.0
    assert dump["confidence"] == 0.97
    assert dump["landmarks"] is None


def test_detection_response_schema():
    """Verify DetectionResponse top-level response schema and fields."""
    response = DetectionResponse(
        image_width=640,
        image_height=480,
        processing_time_ms=12.5,
        device="cpu",
        model_name="scrfd-500m",
        detections=[
            FaceDetection(
                face_id=0,
                bbox=BoundingBox(x1=100.0, y1=80.0, x2=250.0, y2=260.0),
                confidence=0.97,
                landmarks=None,
            )
        ],
    )
    assert response.image_width == 640
    assert response.image_height == 480
    assert response.processing_time_ms == 12.5
    assert response.device == "cpu"
    assert response.model_name == "scrfd-500m"
    assert len(response.detections) == 1
    assert response.face_count == 1
    assert response.detections[0].confidence == 0.97


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
    assert face.face_id is None
    assert face.confidence >= 0.5
    assert face.bbox.x1 > 0
    assert face.bbox.y1 > 0
    assert face.landmarks is not None
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
    assert "gpu_available" in telemetry
    assert "cuda_available" in telemetry
    assert "available_gpu_providers" in telemetry
    assert "cpu_cores" in telemetry
    assert telemetry["cpu_cores"] >= 1
    assert telemetry["recommended_threads"] >= 1
    assert "optimal_providers" in telemetry

    # Test device fallback
    assert resolve_device("cpu") == "cpu"
    # When no GPU is available, auto or gpu gracefully falls back to cpu
    if not is_gpu_available():
        assert resolve_device("gpu") == "cpu"
        assert resolve_device("auto") == "cpu"


def test_device_abstraction_auto_cpu_gpu():
    """Verify AUTO, CPU, and GPU determinations via DeviceType and string inputs."""
    # Explicit CPU
    assert resolve_device(DeviceType.CPU) == "cpu"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("CPU") == "cpu"

    # Explicit AUTO - selects GPU if available, else gracefully falls back to CPU
    auto_res = resolve_device(DeviceType.AUTO)
    assert auto_res in ("cpu", "gpu")
    assert resolve_device("auto") == auto_res
    assert resolve_device("AUTO") == auto_res

    # Explicit GPU - selects GPU if available, else gracefully falls back to CPU without crashing
    gpu_res = resolve_device(DeviceType.GPU)
    assert gpu_res in ("cpu", "gpu")
    assert resolve_device("gpu") == gpu_res
    assert resolve_device("GPU") == gpu_res


def test_device_resolution_with_simulated_gpu(monkeypatch):
    """Verify that when a GPU runtime is detected, AUTO and GPU select 'gpu'."""
    monkeypatch.setattr("app.ai.inference.device.is_gpu_available", lambda: True)
    monkeypatch.setattr("app.ai.inference.device.get_available_gpu_providers", lambda: ["CUDAExecutionProvider"])

    assert resolve_device(DeviceType.AUTO) == "gpu"
    assert resolve_device("auto") == "gpu"
    assert resolve_device(DeviceType.GPU) == "gpu"
    assert resolve_device("gpu") == "gpu"
    # CPU must still remain CPU even if GPU is available
    assert resolve_device(DeviceType.CPU) == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_device_resolution_fallback_never_crashes(monkeypatch):
    """Verify that when NO GPU is detected, AUTO and GPU gracefully fall back to CPU."""
    monkeypatch.setattr("app.ai.inference.device.is_gpu_available", lambda: False)
    monkeypatch.setattr("app.ai.inference.device.get_available_gpu_providers", lambda: [])

    assert resolve_device(DeviceType.AUTO) == "cpu"
    assert resolve_device("auto") == "cpu"
    assert resolve_device(DeviceType.GPU) == "cpu"
    assert resolve_device("gpu") == "cpu"
    assert resolve_device("unknown_device_string") == "cpu"


def test_settings_ai_device_configuration():
    """Verify Settings supports typed AI settings and validation rules."""
    s_default = Settings()
    assert s_default.AI_DEVICE == "auto"
    assert s_default.AI_MODEL_NAME == "scrfd_500m"
    assert s_default.AI_MODEL_VERSION == "1.0.0"
    assert s_default.AI_CONFIDENCE_THRESHOLD == 0.50
    assert s_default.AI_MAX_IMAGE_SIZE_MB == 10
    assert s_default.AI_MAX_IMAGE_DIMENSION == 4096
    assert s_default.AI_ENABLE_GPU is False
    assert s_default.ai_max_image_bytes == 10 * 1024 * 1024

    # Custom valid configuration
    s_custom = Settings(
        AI_DEVICE="cpu",
        AI_MODEL_NAME="scrfd_2.5g",
        AI_MODEL_VERSION="2.0.0",
        AI_CONFIDENCE_THRESHOLD=0.65,
        AI_MAX_IMAGE_SIZE_MB=25,
        AI_MAX_IMAGE_DIMENSION=2048,
        AI_ENABLE_GPU="true",
    )
    assert s_custom.AI_DEVICE == "cpu"
    assert s_custom.AI_MODEL_NAME == "scrfd_2.5g"
    assert s_custom.AI_MODEL_VERSION == "2.0.0"
    assert s_custom.AI_CONFIDENCE_THRESHOLD == 0.65
    assert s_custom.AI_MAX_IMAGE_SIZE_MB == 25
    assert s_custom.AI_MAX_IMAGE_DIMENSION == 2048
    assert s_custom.AI_ENABLE_GPU is True

    # Invalid confidence threshold (<0 or >1) rejected
    with pytest.raises(Exception):
        Settings(AI_CONFIDENCE_THRESHOLD=1.5)

    with pytest.raises(Exception):
        Settings(AI_CONFIDENCE_THRESHOLD=-0.1)

    # Invalid image limits rejected
    with pytest.raises(Exception):
        Settings(AI_MAX_IMAGE_SIZE_MB=0)

    with pytest.raises(Exception):
        Settings(AI_MAX_IMAGE_DIMENSION=32)

    # Test resolve_device with enable_gpu=False forcing CPU
    assert resolve_device("gpu", enable_gpu=False) == "cpu"
    assert resolve_device("auto", enable_gpu=False) == "cpu"


def test_optimal_providers_always_contain_cpu_fallback():
    """Verify get_optimal_providers always includes CPUExecutionProvider."""
    cpu_providers = get_optimal_providers("cpu")
    assert "CPUExecutionProvider" in cpu_providers

    gpu_providers = get_optimal_providers("gpu")
    assert "CPUExecutionProvider" in gpu_providers
    assert gpu_providers[-1] == "CPUExecutionProvider"


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
    assert hasattr(ai, "DeviceType")
    assert hasattr(ai, "BoundingBox")
    assert hasattr(detection, "FaceDetector")
    assert hasattr(detection, "BaseFaceDetector")
    assert hasattr(detection, "load_image")
    assert hasattr(inference, "InferenceRuntime")
    assert hasattr(inference, "DeviceType")
    assert hasattr(inference, "resolve_device")
    assert hasattr(inference, "is_gpu_available")
    assert hasattr(analysis, "analyze_face_quality")
    assert hasattr(common, "ModelArchitecture")
    assert hasattr(common, "DeviceType")


def test_base_face_detector_interface_enforces_abstract_methods():
    """Verify BaseFaceDetector cannot be instantiated directly without implementations."""
    from app.ai.detection.base import BaseFaceDetector

    with pytest.raises(TypeError):
        BaseFaceDetector()  # type: ignore


def test_face_detector_lifecycle(sample_synthetic_image):
    """Verify FaceDetector lifecycle: load_model, detect, unload, is_loaded, context manager."""
    from app.ai.detection.detector import FaceDetector
    from app.ai.detection.schemas import FrameInferenceResult

    detector = FaceDetector(device="cpu")
    assert detector.device == "cpu"
    assert not detector.is_loaded

    # 1. detect returns structured FrameInferenceResult
    res: FrameInferenceResult = detector.detect(sample_synthetic_image, confidence_threshold=0.5)
    assert isinstance(res, FrameInferenceResult)
    assert res.face_count >= 1
    assert len(res.faces) == res.face_count

    # 2. load_model
    detector.load_model(model_path=None, device="cpu")
    assert detector.device == "cpu"

    # 3. unload clears state
    detector.unload()
    assert not detector.is_loaded
    assert detector.model_path is None

    # 4. Context manager usage
    with FaceDetector(device="cpu") as managed_detector:
        managed_res = managed_detector.detect(sample_synthetic_image)
        assert isinstance(managed_res, FrameInferenceResult)
    # Exiting context manager triggers unload
    assert not managed_detector.is_loaded


def test_preprocessing_pipeline_corrupted_image():
    """Verify CorruptedImageException is raised on corrupt byte streams or invalid paths."""
    preprocessor = ImagePreprocessor()

    # Random corrupt bytes
    corrupt_bytes = b"NOT_A_VALID_JPEG_OR_PNG_IMAGE_DATA_CORRUPT"
    with pytest.raises(CorruptedImageException):
        preprocessor.load_image(corrupt_bytes)

    # Non-existent file path
    with pytest.raises(CorruptedImageException):
        preprocessor.load_image("non_existent_file_path_xyz123.jpg")


def test_preprocessing_pipeline_empty_image():
    """Verify EmptyImageException is raised on empty byte streams, arrays, or None."""
    preprocessor = ImagePreprocessor()

    # None
    with pytest.raises(EmptyImageException):
        preprocessor.load_image(None)

    # Zero bytes
    with pytest.raises(EmptyImageException):
        preprocessor.load_image(b"")

    # Empty array
    with pytest.raises(EmptyImageException):
        preprocessor.load_image(np.array([]))


def test_preprocessing_pipeline_unsupported_format():
    """Verify UnsupportedImageFormatException on invalid types or array shapes."""
    preprocessor = ImagePreprocessor()

    # Unsupported type (integer or float)
    with pytest.raises(UnsupportedImageFormatException):
        preprocessor.load_image(12345)  # type: ignore

    # Unsupported 5D array
    with pytest.raises(UnsupportedImageFormatException):
        preprocessor.load_image(np.zeros((1, 2, 3, 4, 5)))

    # Unsupported 10-channel array
    with pytest.raises(UnsupportedImageFormatException):
        preprocessor.load_image(np.zeros((100, 100, 10)))


def test_preprocessing_pipeline_image_too_large():
    """Verify ImageTooLargeException on images exceeding maximum dimension limits."""
    # Configure low max_dimension for quick test
    config = ImagePreprocessingConfig(max_dimension=1000)
    preprocessor = ImagePreprocessor(config)

    oversized = np.zeros((1500, 500, 3), dtype=np.uint8)
    with pytest.raises(ImageTooLargeException):
        preprocessor.load_image(oversized)


def test_preprocessing_pipeline_color_conversions():
    """Verify Grayscale and RGBA arrays are properly converted to RGB."""
    preprocessor = ImagePreprocessor()

    # Grayscale (H, W) -> RGB (H, W, 3)
    gray = np.full((100, 100), 128, dtype=np.uint8)
    rgb_from_gray = preprocessor.load_image(gray)
    assert rgb_from_gray.shape == (100, 100, 3)
    assert rgb_from_gray[0, 0, 0] == 128
    assert rgb_from_gray[0, 0, 1] == 128

    # RGBA (H, W, 4) -> RGB (H, W, 3)
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255  # Red channel
    rgba[:, :, 3] = 255  # Alpha
    rgb_from_rgba = preprocessor.load_image(rgba)
    assert rgb_from_rgba.shape == (100, 100, 3)
    assert rgb_from_rgba[0, 0, 0] == 255


def test_preprocessing_pipeline_isolation_and_config(sample_synthetic_image):
    """Verify ImagePreprocessor can be isolated and customized independently of detectors."""
    custom_config = ImagePreprocessingConfig(
        target_size=(320, 320),
        pad_color=(0, 0, 0),
        normalize=True,
    )
    preprocessor = ImagePreprocessor(custom_config)

    res: PreprocessedImage = preprocessor.process(sample_synthetic_image)
    assert isinstance(res, PreprocessedImage)
    assert res.tensor.shape == (1, 3, 320, 320)
    assert res.original_shape == (480, 640)
    assert res.original_image.shape == (480, 640, 3)
    assert 0.0 <= res.tensor.min() <= res.tensor.max() <= 1.0

    # Unletterbox coordinates works isolated from detector
    unpadded = preprocessor.unletterbox_coordinates(
        x1=50, y1=50, x2=150, y2=150,
        scale=res.scale,
        padding=res.padding,
        orig_shape=res.original_shape,
    )
    assert len(unpadded) == 4


