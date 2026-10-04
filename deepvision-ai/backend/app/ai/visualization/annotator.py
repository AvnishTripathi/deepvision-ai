"""
DeepVision AI — Face Detection Visualization & Annotation Tool
Development and testing utility for rendering detected face bounding boxes,
confidence badges, and facial landmarks onto image frames.

Designed strictly for development, testing, and debugging. Does not permanently
store or persist user uploads. Operates purely in volatile memory unless an
explicit output file path is provided by the developer.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any, Dict
import numpy as np
import cv2
from PIL import Image

from app.core.config import settings
from app.schemas.ai import (
    FaceDetection,
    BoundingBox,
    FaceBoundingBox,
    LandmarkPoint,
    FrameInferenceResult,
    AIDetectionResponse,
    DetectedFaceItem,
)


# Default landmark palette (BGR color tuples)
DEFAULT_LANDMARK_COLORS = {
    "right_eye": (255, 191, 0),     # Deep Sky Blue / Cyan (BGR)
    "left_eye": (255, 191, 0),      # Deep Sky Blue / Cyan (BGR)
    "nose_tip": (0, 215, 255),      # Gold / Amber (BGR)
    "mouth_right": (0, 0, 255),     # Crimson Red (BGR)
    "mouth_left": (0, 0, 255),      # Crimson Red (BGR)
}


class FaceVisualizer:
    """
    Renders bounding boxes, confidence badges, and facial landmarks
    onto image frames for development verification.
    """

    def __init__(
        self,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),       # Green (BGR)
        text_color: Tuple[int, int, int] = (255, 255, 255),   # White (BGR)
        badge_bg_color: Optional[Tuple[int, int, int]] = (0, 180, 0), # Dark Green (BGR)
        landmark_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        landmark_radius: Optional[int] = None,
        line_thickness: Optional[int] = None,
        font_scale: Optional[float] = None,
        show_bbox: bool = True,
        show_confidence: bool = True,
        show_landmarks: bool = True,
    ):
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.badge_bg_color = badge_bg_color or bbox_color
        self.landmark_colors = landmark_colors or DEFAULT_LANDMARK_COLORS
        self.landmark_radius = landmark_radius
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.show_bbox = show_bbox
        self.show_confidence = show_confidence
        self.show_landmarks = show_landmarks

    def annotate(
        self,
        image: Union[np.ndarray, Image.Image, bytes, str, Path],
        detections: Union[
            List[FaceDetection],
            FrameInferenceResult,
            AIDetectionResponse,
            List[DetectedFaceItem],
            List[Dict[str, Any]],
        ],
        return_rgb: bool = False,
    ) -> np.ndarray:
        """
        Draws face detections on an image copy.

        Args:
            image: Input image (NumPy BGR/RGB array, PIL Image, bytes, or file path).
            detections: Detection results (list of FaceDetection/DetectedFaceItem/dicts,
                        or FrameInferenceResult / AIDetectionResponse).
            return_rgb: If True, returns image in RGB format; if False, returns BGR.

        Returns:
            np.ndarray: Annotated image copy (input image is never modified in-place).
        """
        canvas = self._prepare_bgr_canvas(image)
        h, w = canvas.shape[:2]

        # Calculate dynamic rendering scales based on image resolution
        thickness = self.line_thickness or max(1, int(round(min(h, w) / 350)))
        font_scale = self.font_scale or max(0.4, min(1.2, min(h, w) / 800.0))
        landmark_r = self.landmark_radius or max(2, int(round(min(h, w) / 200)))

        # Normalize detections list
        face_list = self._extract_face_list(detections)

        for item in face_list:
            bbox, confidence, landmarks = self._unpack_detection(item)

            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(round(c)) for c in bbox]

            # 1. Draw Bounding Box
            if self.show_bbox:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), self.bbox_color, thickness)

            # 2. Draw Confidence Badge
            if self.show_confidence and confidence is not None:
                label_text = f"{confidence * 100:.1f}%" if confidence <= 1.0 else f"{confidence:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
                )

                # Position badge above bbox, or inside if near top border
                badge_y2 = y1 if (y1 - text_h - 6) >= 0 else min(h, y1 + text_h + 8)
                badge_y1 = max(0, badge_y2 - text_h - 6)
                badge_x1 = max(0, x1)
                badge_x2 = min(w, badge_x1 + text_w + 8)

                # Draw filled background rectangle for text contrast
                cv2.rectangle(
                    canvas,
                    (badge_x1, badge_y1),
                    (badge_x2, badge_y2),
                    self.badge_bg_color,
                    -1,
                )

                # Draw label text
                text_org_x = badge_x1 + 4
                text_org_y = badge_y2 - 4
                cv2.putText(
                    canvas,
                    label_text,
                    (text_org_x, text_org_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    self.text_color,
                    max(1, thickness - 1),
                    cv2.LINE_AA,
                )

            # 3. Draw Facial Landmarks
            if self.show_landmarks and landmarks:
                for idx, lm in enumerate(landmarks):
                    lx, ly, lm_name = self._unpack_landmark(lm, idx)
                    if lx is None or ly is None:
                        continue
                    pt = (int(round(lx)), int(round(ly)))
                    lm_color = self.landmark_colors.get(lm_name, (0, 255, 255))
                    
                    # Outer dark border circle for visibility on bright regions
                    cv2.circle(canvas, pt, landmark_r + 1, (0, 0, 0), -1, cv2.LINE_AA)
                    # Colored landmark point
                    cv2.circle(canvas, pt, landmark_r, lm_color, -1, cv2.LINE_AA)

        if return_rgb:
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return canvas

    def _prepare_bgr_canvas(self, image: Union[np.ndarray, Image.Image, bytes, str, Path]) -> np.ndarray:
        """Converts diverse image inputs into an independent BGR uint8 NumPy array."""
        if isinstance(image, (str, Path)):
            path_str = str(image)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image file not found: {path_str}")
            img_bgr = cv2.imread(path_str)
            if img_bgr is None:
                raise ValueError(f"Failed decoding image file: {path_str}")
            return img_bgr.copy()

        if isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Failed decoding image bytes.")
            return img_bgr

        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image, np.ndarray):
            # Guarantee a fresh copy so original array is never modified
            img_copy = image.copy()
            if img_copy.ndim == 2:
                return cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
            if img_copy.ndim == 3:
                if img_copy.shape[2] == 4:
                    return cv2.cvtColor(img_copy, cv2.COLOR_RGBA2BGR)
                if img_copy.shape[2] == 3:
                    # Assumed BGR if already uint8; return fresh copy
                    return img_copy
            raise ValueError(f"Unsupported image array shape: {image.shape}")

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _extract_face_list(self, detections: Any) -> List[Any]:
        """Extracts list of face items from various result types."""
        if hasattr(detections, "faces"):
            return list(detections.faces)
        if hasattr(detections, "detections"):
            return list(detections.detections)
        if isinstance(detections, list):
            return detections
        return []

    def _unpack_detection(
        self, item: Any
    ) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[float], Optional[List[Any]]]:
        """Extracts (x1, y1, x2, y2), confidence, and landmarks from a detection item."""
        bbox = None
        confidence = None
        landmarks = None

        if isinstance(item, (FaceDetection, DetectedFaceItem)):
            b = item.bbox
            bbox = (b.x1, b.y1, b.x2, b.y2)
            confidence = item.confidence
            landmarks = item.landmarks
        elif isinstance(item, dict):
            confidence = item.get("confidence")
            landmarks = item.get("landmarks")
            raw_box = item.get("bbox")
            if isinstance(raw_box, dict):
                bbox = (
                    float(raw_box.get("x1", raw_box.get("x_min", 0.0))),
                    float(raw_box.get("y1", raw_box.get("y_min", 0.0))),
                    float(raw_box.get("x2", raw_box.get("x_max", 0.0))),
                    float(raw_box.get("y2", raw_box.get("y_max", 0.0))),
                )
            elif isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                bbox = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
        elif hasattr(item, "bbox"):
            b = getattr(item, "bbox")
            if hasattr(b, "x1"):
                bbox = (b.x1, b.y1, b.x2, b.y2)
            elif isinstance(b, (list, tuple)) and len(b) == 4:
                bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            confidence = getattr(item, "confidence", None)
            landmarks = getattr(item, "landmarks", None)

        return bbox, confidence, landmarks

    def _unpack_landmark(self, lm: Any, index: int) -> Tuple[Optional[float], Optional[float], str]:
        """Extracts x, y, and landmark name from a landmark object or tuple."""
        if isinstance(lm, LandmarkPoint):
            return lm.x, lm.y, lm.name
        if isinstance(lm, dict):
            return (
                float(lm.get("x", 0.0)),
                float(lm.get("y", 0.0)),
                lm.get("name", f"point_{index}"),
            )
        if hasattr(lm, "x") and hasattr(lm, "y"):
            return getattr(lm, "x"), getattr(lm, "y"), getattr(lm, "name", f"point_{index}")
        if isinstance(lm, (list, tuple)) and len(lm) >= 2:
            return float(lm[0]), float(lm[1]), f"point_{index}"
        return None, None, f"point_{index}"


def draw_detections(
    image: Union[np.ndarray, Image.Image, bytes, str, Path],
    detections: Union[
        List[FaceDetection],
        FrameInferenceResult,
        AIDetectionResponse,
        List[DetectedFaceItem],
        List[Dict[str, Any]],
    ],
    show_bbox: bool = True,
    show_confidence: bool = True,
    show_landmarks: bool = True,
    return_rgb: bool = False,
) -> np.ndarray:
    """
    Convenience function to draw detections onto an image copy.

    Args:
        image: Source image (array, PIL, bytes, or file path).
        detections: Structured detections.
        show_bbox: Whether to render face bounding boxes.
        show_confidence: Whether to render confidence percentage badge.
        show_landmarks: Whether to render 5-point facial landmarks.
        return_rgb: Whether to return RGB array instead of default BGR.

    Returns:
        np.ndarray: Annotated image copy.
    """
    visualizer = FaceVisualizer(
        show_bbox=show_bbox,
        show_landmarks=show_landmarks,
        show_confidence=show_confidence,
    )
    return visualizer.annotate(image, detections, return_rgb=return_rgb)


def annotate_image(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    detector: Optional[Any] = None,
    confidence_threshold: Optional[float] = None,
    show_landmarks: bool = True,
) -> Tuple[np.ndarray, FrameInferenceResult]:
    """
    Development utility: loads input image, executes real face detection,
    and produces an annotated copy saved to output_path.

    Example:
        input.jpg -> DeepVision AI -> output_detected.jpg

    Args:
        input_path: Path to the input image file.
        output_path: Destination path for the annotated image (optional).
                     If None, defaults to '<input_stem>_detected<ext>'.
        detector: Optional FaceDetector instance. If None, initialized from default settings.
        confidence_threshold: Detection confidence cutoff (defaults to settings.AI_CONFIDENCE_THRESHOLD).
        show_landmarks: Whether to render facial landmarks.

    Returns:
        Tuple[np.ndarray, FrameInferenceResult]: Annotated image array and inference result.
    """
    input_p = Path(input_path)
    if not input_p.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Resolve detector
    if detector is None:
        from app.ai.detection.detector import FaceDetector
        model_path = settings.resolve_model_path()
        detector = FaceDetector(
            model_path=model_path,
            device=settings.AI_DEVICE,
            confidence_threshold=confidence_threshold or settings.AI_CONFIDENCE_THRESHOLD,
        )

    # Ingest image with OpenCV
    img_bgr = cv2.imread(str(input_p))
    if img_bgr is None:
        raise ValueError(f"Could not read input image: {input_path}")

    # Convert to RGB for inference
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Execute real model inference
    inference_result: FrameInferenceResult = detector.detect(
        img_rgb,
        confidence_threshold=confidence_threshold,
    )

    # Annotate on fresh BGR copy
    visualizer = FaceVisualizer(show_landmarks=show_landmarks)
    annotated_bgr = visualizer.annotate(img_bgr, inference_result)

    # Write output if requested
    if output_path is not None:
        out_p = Path(output_path)
    else:
        out_p = input_p.parent / f"{input_p.stem}_detected{input_p.suffix}"

    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), annotated_bgr)

    return annotated_bgr, inference_result
