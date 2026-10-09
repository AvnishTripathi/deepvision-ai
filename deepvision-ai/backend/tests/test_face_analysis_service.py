"""
DeepVision AI — Face Analysis Service Test Suite (Step 15)
Validates dedicated FaceAnalysisService sub-analyzers (Landmarks, Blur, Brightness,
Contrast, Size, Pose, Quality Score) and end-to-end decoupled orchestration.
Architecture:
    API -> AI Service -> Face Detector -> Face Analysis Service
                                            ├── Landmarks
                                            ├── Blur
                                            ├── Brightness
                                            ├── Contrast
                                            ├── Size
                                            ├── Pose
                                            └── Quality Score
"""

import numpy as np
import pytest

from app.services.face_analysis_service import (
    FaceAnalysisService,
    get_face_analysis_service,
)
from app.services.ai_service import AIService
from app.schemas.ai import (
    BoundingBox,
    LandmarkPoint,
    DetectedFaceItem,
    FaceDetectionQuality,
)
from app.ai.detection.schemas import FaceDetection, FrameInferenceResult
from app.ai.analysis.schemas import (
    StructuredFaceAnalysis,
    QualityClassification,
    BlurStatus,
    BrightnessStatus,
)


@pytest.fixture
def analysis_service():
    """Provides FaceAnalysisService instance."""
    return FaceAnalysisService(min_sharpness=50.0)


@pytest.fixture
def sample_face_crop():
    """Generates a realistic synthetic 100x100 RGB face patch with textures."""
    np.random.seed(42)
    crop = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # Add texture for sharpness / contrast
    noise = np.random.randint(-20, 20, (100, 100, 3), dtype=np.int16)
    return np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def sample_5_landmarks():
    """5 standard facial landmark points for a 100x100 face inside a 640x480 frame."""
    return [
        LandmarkPoint(name="right_eye", x=230.0, y=140.0),
        LandmarkPoint(name="left_eye", x=270.0, y=140.0),
        LandmarkPoint(name="nose_tip", x=250.0, y=165.0),
        LandmarkPoint(name="mouth_right", x=235.0, y=190.0),
        LandmarkPoint(name="mouth_left", x=265.0, y=190.0),
    ]


# -----------------------------------------------------------------------------
# 1. Modular Sub-Analyzers Unit Tests
# -----------------------------------------------------------------------------

def test_landmarks_extraction_and_validation(analysis_service, sample_5_landmarks):
    """Test landmark extraction clips coordinates within image bounds."""
    # Test valid landmarks
    lms_dict, lms_list = analysis_service.extract_and_validate_landmarks(
        landmarks=sample_5_landmarks,
        image_shape=(480, 640),
    )
    assert len(lms_list) == 5
    assert "right_eye" in lms_dict
    assert "left_eye" in lms_dict
    assert "nose" in lms_dict

    # Test out-of-bounds coordinates get safely clipped
    oob_landmarks = [
        LandmarkPoint(name="pt1", x=-50.0, y=600.0),
        LandmarkPoint(name="pt2", x=700.0, y=-20.0),
    ]
    _, clipped_list = analysis_service.extract_and_validate_landmarks(
        landmarks=oob_landmarks,
        image_shape=(480, 640),
    )
    assert clipped_list[0].x >= 0.0
    assert clipped_list[0].y <= 480.0
    assert clipped_list[1].x <= 640.0
    assert clipped_list[1].y >= 0.0

    # Test empty or None landmarks
    empty_dict, empty_list = analysis_service.extract_and_validate_landmarks(None, (480, 640))
    assert empty_dict == {}
    assert empty_list == []


def test_blur_analysis(analysis_service, sample_face_crop):
    """Test blur estimation using Laplacian variance."""
    # Textured crop should yield sharpness
    blur_score, blur_status, clarity = analysis_service.analyze_blur(sample_face_crop)
    assert isinstance(blur_score, float)
    assert blur_score >= 0.0
    assert blur_status in ("sharp", "slightly_blurred", "blurry")
    assert round(clarity.sharpness, 1) == blur_score

    # Flat image should yield near zero sharpness and blurry status
    flat_crop = np.full((100, 100, 3), 128, dtype=np.uint8)
    flat_blur_score, flat_status, _ = analysis_service.analyze_blur(flat_crop)
    assert flat_blur_score < 5.0
    assert flat_status == "blurry"


def test_brightness_analysis(analysis_service, sample_face_crop):
    """Test photopic luminance brightness estimation."""
    score, status, illum = analysis_service.analyze_brightness(sample_face_crop)
    assert 100.0 <= score <= 160.0
    assert status == "acceptable"
    assert round(illum.brightness, 1) == score

    # Dark image
    dark_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    dark_score, dark_status, _ = analysis_service.analyze_brightness(dark_crop)
    assert dark_score < 10.0
    assert dark_status == "too_dark"

    # Bright image
    bright_crop = np.full((100, 100, 3), 255, dtype=np.uint8)
    bright_score, bright_status, _ = analysis_service.analyze_brightness(bright_crop)
    assert bright_score > 240.0
    assert bright_status == "too_bright"


def test_contrast_analysis(analysis_service, sample_face_crop):
    """Test contrast assessment from illumination metrics."""
    _, _, illum = analysis_service.analyze_brightness(sample_face_crop)
    contrast_score, contrast_status = analysis_service.analyze_contrast(illum)
    assert isinstance(contrast_score, float)
    assert contrast_status in ("low_contrast", "acceptable", "high_contrast")


def test_size_analysis(analysis_service):
    """Test face size and area ratio calculation."""
    bbox = BoundingBox(x1=100.0, y1=100.0, x2=300.0, y2=300.0)
    size_metrics = analysis_service.analyze_size(bbox, image_shape=(1000, 1000))

    assert size_metrics.face_width == 200.0
    assert size_metrics.face_height == 200.0
    assert size_metrics.face_area == 40000.0
    assert pytest.approx(size_metrics.face_area_ratio, 0.001) == 0.04


def test_pose_estimation(analysis_service, sample_5_landmarks):
    """Test 3D head pose estimation using solvePnP."""
    angles_dict, pose_obj = analysis_service.estimate_pose(sample_5_landmarks, (480, 640))
    assert angles_dict is not None
    assert "yaw" in angles_dict
    assert "pitch" in angles_dict
    assert "roll" in angles_dict
    assert isinstance(angles_dict["yaw"], float)
    assert pose_obj is not None

    # Less than 5 landmarks returns (None, None) gracefully without crashing
    angles_none, pose_none = analysis_service.estimate_pose(sample_5_landmarks[:3], (480, 640))
    assert angles_none is None
    assert pose_none is None


def test_quality_score_calculation(analysis_service, sample_face_crop, sample_5_landmarks):
    """Test transparent composite quality scoring (0 to 100)."""
    bbox = BoundingBox(x1=200.0, y1=100.0, x2=300.0, y2=200.0)
    _, _, clarity = analysis_service.analyze_blur(sample_face_crop)
    _, _, illum = analysis_service.analyze_brightness(sample_face_crop)
    size_metrics = analysis_service.analyze_size(bbox, (480, 640))

    q_score, classification, factors, grade = analysis_service.calculate_quality(
        face_crop=sample_face_crop,
        bbox=bbox,
        image_shape=(480, 640),
        clarity_metrics=clarity,
        illumination_metrics=illum,
        size_metrics=size_metrics,
        landmarks_list=sample_5_landmarks,
        detection_confidence=0.98,
    )

    assert 0.0 <= q_score <= 100.0
    assert classification in ("excellent", "good", "acceptable", "poor")
    assert factors.sharpness >= 0.0
    assert factors.brightness >= 0.0


# -----------------------------------------------------------------------------
# 2. End-to-End Analysis Orchestration & Fault Isolation
# -----------------------------------------------------------------------------

def test_analyze_face_detection_item(analysis_service, sample_5_landmarks):
    """Test processing a single FaceDetection into a DetectedFaceItem."""
    full_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    detection = FaceDetection(
        face_id=1,
        bbox=BoundingBox(x1=200.0, y1=100.0, x2=300.0, y2=200.0),
        confidence=0.97,
        landmarks=sample_5_landmarks,
    )

    item = analysis_service.analyze_face_detection_item(
        image_rgb=full_img,
        detection=detection,
        image_shape=(480, 640),
    )

    assert isinstance(item, DetectedFaceItem)
    assert item.face_id == 1
    assert item.bbox.x1 == 200.0
    assert item.confidence == 0.97
    assert item.quality is not None
    assert isinstance(item.quality, FaceDetectionQuality)
    assert 0.0 <= item.quality.score <= 100.0
    assert item.quality.classification in ("excellent", "good", "acceptable", "poor")
    assert item.pose is not None
    assert "yaw" in item.pose


def test_analyze_detected_faces_fault_isolation(analysis_service, sample_5_landmarks):
    """
    Test batch analysis fault isolation:
    Even if one detection has corrupted/degenerate attributes,
    the service never crashes and safely processes all faces.
    """
    full_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    valid_detection = FaceDetection(
        face_id=1,
        bbox=BoundingBox(x1=200.0, y1=100.0, x2=300.0, y2=200.0),
        confidence=0.95,
        landmarks=sample_5_landmarks,
    )
    degenerate_detection = FaceDetection(
        face_id=2,
        bbox=BoundingBox(x1=-100.0, y1=-100.0, x2=-50.0, y2=-50.0),  # Out of image
        confidence=0.30,
        landmarks=None,
    )

    items = analysis_service.analyze_detected_faces(
        image_rgb=full_img,
        detections=[valid_detection, degenerate_detection],
        image_shape=(480, 640),
    )

    assert len(items) == 2
    assert items[0].face_id == 1
    assert items[1].face_id == 2
    assert items[0].quality.score >= 0.0
    assert items[1].quality.score >= 0.0


def test_analyze_structured_detections(analysis_service, sample_5_landmarks):
    """Test full structured analysis output conforming to AIAnalysisResponse schema."""
    full_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    detection = FaceDetection(
        face_id=1,
        bbox=BoundingBox(x1=200.0, y1=100.0, x2=300.0, y2=200.0),
        confidence=0.95,
        landmarks=sample_5_landmarks,
    )

    structured_list = analysis_service.analyze_structured_detections(
        image_rgb=full_img,
        detections=[detection],
        image_shape=(480, 640),
    )

    assert len(structured_list) == 1
    analysis = structured_list[0]
    assert isinstance(analysis, StructuredFaceAnalysis)
    assert analysis.face_id == 1
    assert analysis.quality is not None
    assert analysis.pose is not None
    assert analysis.clarity is not None
    assert analysis.illumination is not None
    assert analysis.size is not None


# -----------------------------------------------------------------------------
# 3. AI Service Integration & Singleton Tests
# -----------------------------------------------------------------------------

def test_ai_service_detect_and_analyze_faces_integration():
    """Verify AIService.detect_and_analyze_faces cleanly coordinates detector and analysis service."""
    service = AIService()
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[100:350, 200:450] = [220, 180, 150]  # Face region

    face_items, detection_result = service.detect_and_analyze_faces(
        image_input=test_img,
        confidence_threshold=0.5,
    )

    assert isinstance(face_items, list)
    assert isinstance(detection_result, FrameInferenceResult)
    if face_items:
        face = face_items[0]
        assert isinstance(face, DetectedFaceItem)
        assert face.quality is not None
        assert 0.0 <= face.quality.score <= 100.0


def test_get_face_analysis_service_singleton():
    """Verify get_face_analysis_service returns persistent singleton instance."""
    s1 = get_face_analysis_service()
    s2 = get_face_analysis_service()
    assert s1 is s2
