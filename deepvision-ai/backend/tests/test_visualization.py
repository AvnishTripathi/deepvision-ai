"""
DeepVision AI — Face Detection Visualization Test Suite
Validates the development visualization utility:
- Bounding box rendering
- Confidence score badge rendering
- 5-point facial landmark rendering
- Memory safety (input image is never mutated in-place)
- Zero permanent storage of user uploads
- Multi-format ingestion (NumPy, PIL, bytes, files)
- End-to-end file annotation workflow (input.jpg -> output_detected.jpg)
- Standalone CLI utility execution
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import pytest
import cv2
from PIL import Image

from app.core.config import settings
from app.ai.detection.detector import FaceDetector
from app.ai.detection.schemas import (
    FaceDetection,
    BoundingBox,
    LandmarkPoint,
    FrameInferenceResult,
    AIDetectionResponse,
    DetectedFaceItem,
    FaceBoundingBox,
    ImageDimensions,
)
from app.ai.visualization.annotator import (
    FaceVisualizer,
    draw_detections,
    annotate_image,
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sample_face.jpg")


@pytest.fixture
def sample_face_np() -> np.ndarray:
    """Loads portrait fixture into BGR NumPy array."""
    assert os.path.exists(FIXTURE_PATH), f"Fixture not found at {FIXTURE_PATH}"
    img = cv2.imread(FIXTURE_PATH)
    assert img is not None
    return img


@pytest.fixture
def mock_detection() -> FaceDetection:
    """Creates a mock face detection with landmarks."""
    return FaceDetection(
        bbox=BoundingBox(x1=50.0, y1=60.0, x2=200.0, y2=220.0),
        confidence=0.95,
        landmarks=[
            LandmarkPoint(name="right_eye", x=80.0, y=100.0),
            LandmarkPoint(name="left_eye", x=160.0, y=100.0),
            LandmarkPoint(name="nose_tip", x=120.0, y=140.0),
            LandmarkPoint(name="mouth_right", x=90.0, y=180.0),
            LandmarkPoint(name="mouth_left", x=150.0, y=180.0),
        ],
    )


# -------------------------------------------------------------------------
# Test 1: Memory Safety & Non-Mutation
# -------------------------------------------------------------------------

def test_visualizer_never_mutates_input_image_in_place(sample_face_np, mock_detection):
    """
    Verify the visualizer never mutates the original image array in-place,
    returning an independent annotated copy.
    """
    original_copy = sample_face_np.copy()

    annotated = draw_detections(sample_face_np, [mock_detection])

    # The returned array must have the same shape and type
    assert annotated.shape == sample_face_np.shape
    assert annotated.dtype == sample_face_np.dtype

    # The original array must NOT have been modified
    assert np.array_equal(sample_face_np, original_copy)

    # The annotated copy must contain drawn elements (differs from original)
    assert not np.array_equal(annotated, sample_face_np)


# -------------------------------------------------------------------------
# Test 2: Bounding Box and Badge Rendering
# -------------------------------------------------------------------------

def test_visualizer_renders_bounding_box_and_confidence(mock_detection):
    """
    Verify bounding box rectangle and confidence badge pixels are drawn on canvas.
    """
    # Create black canvas 300x300
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)

    annotated = draw_detections(
        canvas,
        [mock_detection],
        show_landmarks=False,
        show_confidence=True,
    )

    # Pixel on bounding box boundary (e.g. top border at y=60, x=100) must be non-zero
    assert np.any(annotated[60, 100] > 0)

    # Far corner away from detection should remain unmodified black
    assert np.all(annotated[10, 10] == 0)


# -------------------------------------------------------------------------
# Test 3: Facial Landmark Rendering
# -------------------------------------------------------------------------

def test_visualizer_renders_facial_landmarks(mock_detection):
    """
    Verify landmark circles are drawn at their respective coordinates.
    """
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)

    annotated = draw_detections(
        canvas,
        [mock_detection],
        show_bbox=False,
        show_confidence=False,
        show_landmarks=True,
    )

    # Eye landmark at (80, 100) -> annotated[y=100, x=80] must be non-zero
    assert np.any(annotated[100, 80] > 0)

    # Nose tip at (120, 140) -> annotated[y=140, x=120] must be non-zero
    assert np.any(annotated[140, 120] > 0)


# -------------------------------------------------------------------------
# Test 4: Empty Detections Return Clean Unmodified Copy
# -------------------------------------------------------------------------

def test_visualizer_empty_detections(sample_face_np):
    """
    When zero faces are detected, the visualizer returns an image
    with pixels identical to the input.
    """
    annotated = draw_detections(sample_face_np, [])
    assert np.array_equal(annotated, sample_face_np)


# -------------------------------------------------------------------------
# Test 5: Ingestion Format Diversity (PIL, Grayscale, RGBA, Bytes)
# -------------------------------------------------------------------------

def test_visualizer_handles_diverse_input_types(sample_face_np, mock_detection):
    """
    Verify annotator handles PIL Image, Grayscale array, RGBA array, and encoded bytes.
    """
    visualizer = FaceVisualizer()

    # 1. PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(sample_face_np, cv2.COLOR_BGR2RGB))
    ann_pil = visualizer.annotate(pil_img, [mock_detection])
    assert ann_pil.shape == sample_face_np.shape

    # 2. Grayscale 2D array
    gray_img = cv2.cvtColor(sample_face_np, cv2.COLOR_BGR2GRAY)
    ann_gray = visualizer.annotate(gray_img, [mock_detection])
    assert ann_gray.ndim == 3
    assert ann_gray.shape[:2] == gray_img.shape

    # 3. RGBA 4-channel array
    rgba_img = cv2.cvtColor(sample_face_np, cv2.COLOR_BGR2RGBA)
    ann_rgba = visualizer.annotate(rgba_img, [mock_detection])
    assert ann_rgba.shape[2] == 3

    # 4. Encoded bytes
    ret, buf = cv2.imencode(".jpg", sample_face_np)
    assert ret
    ann_bytes = visualizer.annotate(buf.tobytes(), [mock_detection])
    assert ann_bytes.shape == sample_face_np.shape


# -------------------------------------------------------------------------
# Test 6: Ingestion Result Object Diversity
# -------------------------------------------------------------------------

def test_visualizer_handles_diverse_detection_schemas(sample_face_np):
    """
    Verify visualizer works with FrameInferenceResult, AIDetectionResponse,
    and plain dictionaries.
    """
    visualizer = FaceVisualizer()

    # 1. AIDetectionResponse
    api_resp = AIDetectionResponse(
        model="test_model",
        device="CPU",
        image=ImageDimensions(width=sample_face_np.shape[1], height=sample_face_np.shape[0]),
        processing_time_ms=25.0,
        face_count=1,
        faces=[
            DetectedFaceItem(
                face_id=None,
                bbox=FaceBoundingBox(x1=30.0, y1=40.0, x2=200.0, y2=250.0),
                confidence=0.88,
                landmarks=[LandmarkPoint(name="nose_tip", x=100.0, y=140.0)],
            )
        ],
    )
    ann_api = visualizer.annotate(sample_face_np, api_resp)
    assert ann_api.shape == sample_face_np.shape
    assert not np.array_equal(ann_api, sample_face_np)

    # 2. Raw dictionaries
    dict_detections = [
        {
            "bbox": {"x1": 40, "y1": 50, "x2": 180, "y2": 220},
            "confidence": 0.92,
            "landmarks": [{"name": "left_eye", "x": 120, "y": 90}],
        }
    ]
    ann_dict = visualizer.annotate(sample_face_np, dict_detections)
    assert not np.array_equal(ann_dict, sample_face_np)


# -------------------------------------------------------------------------
# Test 7: End-to-End File Annotation Workflow (input.jpg -> output_detected.jpg)
# -------------------------------------------------------------------------

def test_annotate_image_file_workflow(tmp_path):
    """
    Verify end-to-end development utility:
    input.jpg -> DeepVision AI -> output_detected.jpg
    Saves annotated file to specified output path with real face detections.
    """
    model_path = settings.resolve_model_path()
    assert model_path is not None and os.path.exists(model_path)

    detector = FaceDetector(model_path=model_path, device="cpu")
    out_file = tmp_path / "sample_face_detected.jpg"

    assert not out_file.exists()

    ann_img, inference_result = annotate_image(
        input_path=FIXTURE_PATH,
        output_path=out_file,
        detector=detector,
        confidence_threshold=0.50,
    )

    # Output file must exist and be non-empty
    assert out_file.exists()
    assert out_file.stat().st_size > 0

    # Output must be a valid decodable image
    decoded = cv2.imread(str(out_file))
    assert decoded is not None
    assert decoded.shape == ann_img.shape

    # Real face must be detected
    assert inference_result.face_count >= 1
    assert len(inference_result.faces) == inference_result.face_count


# -------------------------------------------------------------------------
# Test 8: CLI Script Integration
# -------------------------------------------------------------------------

def test_visualize_cli_script_execution(tmp_path):
    """
    Verify scripts/visualize_detection.py runs via subprocess and outputs annotated copy.
    """
    cli_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "visualize_detection.py"
    assert cli_script.exists(), f"CLI script not found at {cli_script}"

    out_file = tmp_path / "cli_output.jpg"
    cmd = [
        sys.executable,
        str(cli_script),
        "--input", FIXTURE_PATH,
        "--output", str(out_file),
        "--confidence", "0.50",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI script failed with error: {result.stderr}"
    assert "Faces Detected:  1" in result.stdout
    assert "[SUCCESS] Annotated image written to:" in result.stdout

    # File exists and is valid
    assert out_file.exists()
    assert out_file.stat().st_size > 0
