"""
DeepVision AI — Configuration Test Suite (Step 19)
Validates strongly typed settings, configurable threshold bounds, feature flags,
and strict zero-secret exposure guarantees.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
from pydantic import ValidationError
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import Settings, settings
from app.services.face_analysis_service import FaceAnalysisService
from app.services.ai_service import AIService, get_ai_service
from app.ai.detection.base import BaseFaceDetector
from app.ai.detection.schemas import FaceDetection, FrameInferenceResult
from app.schemas.ai import BoundingBox, LandmarkPoint


@pytest.fixture
def synthetic_face():
    """640x480 RGB image with textured face region."""
    arr = np.ones((480, 640, 3), dtype=np.uint8) * 128
    np.random.seed(42)
    noise = np.random.randint(-20, 20, (200, 200, 3), dtype=np.int16)
    arr[100:300, 150:350] = np.clip(128 + noise, 0, 255).astype(np.uint8)
    return arr


@pytest.fixture
def sample_detection():
    """Face detection with canonical 5 landmarks."""
    return FaceDetection(
        face_id=1,
        bbox=BoundingBox(x1=150.0, y1=100.0, x2=350.0, y2=300.0),
        confidence=0.98,
        landmarks=[
            LandmarkPoint(name="right_eye", x=200.0, y=170.0),
            LandmarkPoint(name="left_eye", x=300.0, y=170.0),
            LandmarkPoint(name="nose_tip", x=250.0, y=210.0),
            LandmarkPoint(name="mouth_right", x=210.0, y=260.0),
            LandmarkPoint(name="mouth_left", x=290.0, y=260.0),
        ],
    )


# -----------------------------------------------------------------------------
# 1. Typed Configuration & Validation Unit Tests
# -----------------------------------------------------------------------------

def test_settings_typed_defaults():
    """Verifies default values and types for Step 19 settings."""
    assert isinstance(settings.AI_ENABLE_LANDMARKS, bool)
    assert settings.AI_ENABLE_LANDMARKS is True

    assert isinstance(settings.AI_ENABLE_QUALITY_ANALYSIS, bool)
    assert settings.AI_ENABLE_QUALITY_ANALYSIS is True

    assert isinstance(settings.AI_ENABLE_POSE_ANALYSIS, bool)
    assert settings.AI_ENABLE_POSE_ANALYSIS is True

    assert isinstance(settings.FACE_BLUR_THRESHOLD, float)
    assert settings.FACE_BLUR_THRESHOLD == 50.0

    assert isinstance(settings.FACE_BRIGHTNESS_MIN, float)
    assert settings.FACE_BRIGHTNESS_MIN == 60.0

    assert isinstance(settings.FACE_BRIGHTNESS_MAX, float)
    assert settings.FACE_BRIGHTNESS_MAX == 200.0

    assert isinstance(settings.FACE_QUALITY_THRESHOLD, float)
    assert settings.FACE_QUALITY_THRESHOLD == 45.0


def test_settings_boolean_type_coercion():
    """Verifies string flags like 'true'/'false'/'yes'/'no' are coerced to booleans."""
    s = Settings(
        AI_ENABLE_LANDMARKS="false",
        AI_ENABLE_QUALITY_ANALYSIS="0",
        AI_ENABLE_POSE_ANALYSIS="no",
    )
    assert s.AI_ENABLE_LANDMARKS is False
    assert s.AI_ENABLE_QUALITY_ANALYSIS is False
    assert s.AI_ENABLE_POSE_ANALYSIS is False

    s2 = Settings(
        AI_ENABLE_LANDMARKS="true",
        AI_ENABLE_QUALITY_ANALYSIS="1",
        AI_ENABLE_POSE_ANALYSIS="yes",
    )
    assert s2.AI_ENABLE_LANDMARKS is True
    assert s2.AI_ENABLE_QUALITY_ANALYSIS is True
    assert s2.AI_ENABLE_POSE_ANALYSIS is True


def test_settings_threshold_validation_rules():
    """Verifies bounds and range validation on optical and quality thresholds."""
    # Negative blur threshold rejected
    with pytest.raises(ValidationError):
        Settings(FACE_BLUR_THRESHOLD=-5.0)

    # Brightness bounds out of [0, 255] rejected
    with pytest.raises(ValidationError):
        Settings(FACE_BRIGHTNESS_MIN=-1.0)
    with pytest.raises(ValidationError):
        Settings(FACE_BRIGHTNESS_MAX=260.0)

    # Inverted brightness range rejected (MIN >= MAX)
    with pytest.raises(ValidationError):
        Settings(FACE_BRIGHTNESS_MIN=150.0, FACE_BRIGHTNESS_MAX=100.0)

    # Quality threshold out of [0, 100] rejected
    with pytest.raises(ValidationError):
        Settings(FACE_QUALITY_THRESHOLD=-10.0)
    with pytest.raises(ValidationError):
        Settings(FACE_QUALITY_THRESHOLD=105.0)


# -----------------------------------------------------------------------------
# 2. Zero-Secret Exposure Guarantee Tests
# -----------------------------------------------------------------------------

def test_public_ai_config_does_not_expose_secrets():
    """Verifies public configuration helper excludes all secrets."""
    pub_config = settings.get_public_ai_config()

    # Ensure public config includes operational parameters
    assert "enable_landmarks" in pub_config
    assert "enable_quality_analysis" in pub_config
    assert "enable_pose_analysis" in pub_config
    assert "face_blur_threshold" in pub_config
    assert "face_brightness_min" in pub_config
    assert "face_brightness_max" in pub_config
    assert "face_quality_threshold" in pub_config

    # Ensure NO secrets or database URLs are present
    assert "secret_key" not in pub_config
    assert "SECRET_KEY" not in pub_config
    assert "database_url" not in pub_config
    assert "DATABASE_URL" not in pub_config
    assert "redis_url" not in pub_config
    assert "REDIS_URL" not in pub_config


@pytest.mark.anyio
async def test_get_ai_status_endpoint_never_exposes_secrets():
    """
    Verifies GET /api/v1/ai/status includes public config without exposing
    passwords, tokens, or SECRET_KEY.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/ai/status")
        assert response.status_code == 200
        data = response.json()

        assert "config" in data
        cfg = data["config"]
        assert cfg["enable_landmarks"] is True
        assert cfg["face_blur_threshold"] == 50.0

        raw_text = response.text
        assert "SECRET_KEY" not in raw_text
        assert settings.SECRET_KEY not in raw_text
        assert "password" not in raw_text.lower()


# -----------------------------------------------------------------------------
# 3. Pipeline Feature Toggle Behavior Tests
# -----------------------------------------------------------------------------

def test_disabling_landmarks_feature_toggle(synthetic_face, sample_detection):
    """
    When AI_ENABLE_LANDMARKS=False:
    - Landmarks are omitted from analysis.
    - Landmark extraction latency is 0.0.
    - Pose estimation is skipped.
    """
    service = FaceAnalysisService(enable_landmarks=False)
    faces, timing = service.analyze_detected_faces_with_timing(
        image_rgb=synthetic_face,
        detections=[sample_detection],
        image_shape=(480, 640),
    )

    assert len(faces) == 1
    face = faces[0]
    assert face.landmarks is None
    assert face.landmarks_list is None
    assert face.pose is None
    assert timing.landmark_time_ms == 0.0
    assert timing.pose_time_ms == 0.0


def test_disabling_pose_analysis_feature_toggle(synthetic_face, sample_detection):
    """
    When AI_ENABLE_POSE_ANALYSIS=False:
    - Landmarks are extracted normally.
    - Pose is None.
    - Pose latency is 0.0.
    """
    service = FaceAnalysisService(enable_landmarks=True, enable_pose_analysis=False)
    faces, timing = service.analyze_detected_faces_with_timing(
        image_rgb=synthetic_face,
        detections=[sample_detection],
        image_shape=(480, 640),
    )

    assert len(faces) == 1
    face = faces[0]
    assert face.landmarks is not None
    assert face.pose is None
    assert timing.pose_time_ms == 0.0


def test_disabling_quality_analysis_feature_toggle(synthetic_face, sample_detection):
    """
    When AI_ENABLE_QUALITY_ANALYSIS=False:
    - Landmarks and pose are extracted normally.
    - Quality is None.
    - Quality analysis latency is 0.0.
    """
    service = FaceAnalysisService(enable_quality_analysis=False)
    faces, timing = service.analyze_detected_faces_with_timing(
        image_rgb=synthetic_face,
        detections=[sample_detection],
        image_shape=(480, 640),
    )

    assert len(faces) == 1
    face = faces[0]
    assert face.quality is None
    assert timing.quality_analysis_time_ms == 0.0


# -----------------------------------------------------------------------------
# 4. Custom Threshold Impact Tests
# -----------------------------------------------------------------------------

def test_custom_quality_threshold_impact(synthetic_face, sample_detection):
    """
    Verifies that a higher FACE_QUALITY_THRESHOLD changes usability evaluation.
    """
    # Strict quality requirement: score must be >= 99.0
    strict_service = FaceAnalysisService(quality_threshold=99.0)
    faces_strict = strict_service.analyze_detected_faces(
        image_rgb=synthetic_face,
        detections=[sample_detection],
        image_shape=(480, 640),
    )
    assert faces_strict[0].quality is not None
    assert faces_strict[0].quality.is_usable is False

    # Lenient quality requirement: score must be >= 10.0
    lenient_service = FaceAnalysisService(quality_threshold=10.0)
    faces_lenient = lenient_service.analyze_detected_faces(
        image_rgb=synthetic_face,
        detections=[sample_detection],
        image_shape=(480, 640),
    )
    assert faces_lenient[0].quality is not None
    assert faces_lenient[0].quality.is_usable is True
