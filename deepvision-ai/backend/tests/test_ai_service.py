"""
DeepVision AI — AI Service Layer Test Suite
Validates application-level AI service orchestration, architectural decoupling,
threshold resolution, payload size limits, and detector lifecycle management.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from app.services.ai_service import AIService, get_ai_service
from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.schemas import DetectionResponse, FrameInferenceResult, FaceDetection, BoundingBox
from app.core.config import Settings
from app.core.exceptions import ImageTooLargeException


@pytest.fixture
def synthetic_face_image():
    """Generates synthetic RGB image array."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:350, 200:450] = [220, 180, 150]  # Face region
    return img


def test_ai_service_detect_faces_end_to_end(synthetic_face_image):
    """Verify AIService.detect_faces produces standardized DetectionResponse."""
    service = AIService()
    response = service.detect_faces(synthetic_face_image)

    assert isinstance(response, DetectionResponse)
    assert response.image_width == 640
    assert response.image_height == 480
    assert response.processing_time_ms > 0
    assert response.device in ("cpu", "gpu")
    assert response.model_name != ""
    assert isinstance(response.detections, list)
    assert response.face_count >= 1

    detection = response.detections[0]
    assert isinstance(detection, FaceDetection)
    assert isinstance(detection.bbox, BoundingBox)
    assert detection.confidence >= 0.5


def test_ai_service_respects_custom_confidence_threshold(synthetic_face_image):
    """Verify AIService applies custom confidence threshold overrides."""
    service = AIService()
    # High threshold should be passed to detector without errors
    response = service.detect_faces(synthetic_face_image, confidence_threshold=0.99)
    assert isinstance(response, DetectionResponse)


def test_ai_service_enforces_payload_size_limit():
    """Verify AIService blocks oversized byte payloads with ImageTooLargeException."""
    custom_config = Settings(AI_MAX_IMAGE_SIZE_MB=1)  # 1 MB max
    service = AIService(config=custom_config)

    # 2 MB payload (exceeds 1 MB limit)
    oversized_bytes = b"x" * (2 * 1024 * 1024)
    with pytest.raises(ImageTooLargeException):
        service.detect_faces(oversized_bytes)


def test_ai_service_architectural_separation(synthetic_face_image):
    """
    Verify strict architectural separation:
    AIService delegates only to the abstract BaseFaceDetector interface,
    never directly touching raw model sessions or tensor runtimes.
    """
    # Create mock detector conforming to BaseFaceDetector contract
    mock_detector = MagicMock(spec=BaseFaceDetector)
    mock_detector.is_loaded = True
    mock_detector.detect.return_value = FrameInferenceResult(
        image_width=640,
        image_height=480,
        processing_time_ms=10.0,
        device="cpu",
        model_name="mock-scrfd",
        detections=[
            FaceDetection(
                face_id=0,
                bbox=BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0),
                confidence=0.98,
                landmarks=None,
            )
        ],
    )

    # Inject mock into service
    service = AIService(detector=mock_detector)
    res = service.detect_faces(synthetic_face_image, confidence_threshold=0.75)

    # Confirm AIService invoked the detector interface with exact parameters
    mock_detector.detect.assert_called_once_with(
        image_input=synthetic_face_image,
        confidence_threshold=0.75,
    )
    assert res.model_name == "mock-scrfd"
    assert len(res.detections) == 1


def test_ai_service_status_telemetry():
    """Verify AIService status reporting telemetry."""
    service = AIService()
    status = service.get_service_status()

    assert status["service"] == "ai_inference"
    assert "model_name" in status
    assert "model_version" in status
    assert "configured_device" in status
    assert "gpu_enabled" in status
    assert "is_model_loaded" in status
    assert "telemetry" in status
    assert "active_device" in status["telemetry"]


def test_ai_service_shutdown_lifecycle():
    """Verify AIService shutdown invokes detector unload."""
    mock_detector = MagicMock(spec=BaseFaceDetector)
    service = AIService(detector=mock_detector)

    service.shutdown()
    mock_detector.unload.assert_called_once()


def test_get_ai_service_singleton():
    """Verify get_ai_service returns persistent singleton instance."""
    s1 = get_ai_service()
    s2 = get_ai_service()
    assert s1 is s2
