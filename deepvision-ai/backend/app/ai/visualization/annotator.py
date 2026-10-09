"""
DeepVision AI — Face Detection Visualization & Annotation Tool (Step 17)
Development and testing utility for rendering detected face bounding boxes,
confidence badges, 5-point facial landmarks, quality scores, status classifications,
and 3D head pose orientation indicators onto image frames.

Designed strictly for development, testing, and debugging. Does not permanently
store or persist user uploads. Operates purely in volatile memory unless an
explicit output file path is provided by the developer.

Visual Concept:
┌─────────────────────────┐
│ Face 1                  │
│ Confidence: 97%         │
│ Quality: 91/100         │
│ Status: Excellent       │
│ Pose: Y:+3.1 P:-1.2 R:+0.5│
│                         │
│     •       •           │
│        •                │
│      •   •              │
└─────────────────────────┘
"""

import os
import math
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
    LandmarkCoordinate,
    FrameInferenceResult,
    AIDetectionResponse,
    DetectedFaceItem,
    FaceDetectionQuality,
)
from app.ai.analysis.schemas import (
    StructuredFaceAnalysis,
    AIAnalysisResponse,
    QualityClassification,
    HeadPoseEstimation,
)


# Default landmark palette (BGR color tuples)
DEFAULT_LANDMARK_COLORS = {
    "right_eye": (255, 191, 0),     # Deep Sky Blue / Cyan (BGR)
    "left_eye": (255, 191, 0),      # Deep Sky Blue / Cyan (BGR)
    "nose_tip": (0, 215, 255),      # Gold / Amber (BGR)
    "nose": (0, 215, 255),          # Gold / Amber (BGR)
    "mouth_right": (0, 69, 255),    # Coral Red (BGR)
    "right_mouth": (0, 69, 255),    # Coral Red (BGR)
    "mouth_left": (0, 69, 255),     # Coral Red (BGR)
    "left_mouth": (0, 69, 255),     # Coral Red (BGR)
}

# Quality classification status colors (BGR tuples)
STATUS_COLORS = {
    "excellent": (46, 204, 113),    # Emerald Green (BGR)
    "good": (50, 205, 50),          # Lime Green (BGR)
    "acceptable": (0, 215, 255),    # Gold / Amber (BGR)
    "poor": (50, 50, 240),          # Crimson Red (BGR)
}


class FaceVisualizer:
    """
    Renders bounding boxes, confidence badges, facial landmarks,
    face quality scores, quality classification, and 3D head pose orientation
    onto image frames for development verification.
    """

    def __init__(
        self,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),       # Default Green (BGR)
        text_color: Tuple[int, int, int] = (255, 255, 255),   # White (BGR)
        badge_bg_color: Optional[Tuple[int, int, int]] = None,
        landmark_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
        landmark_radius: Optional[int] = None,
        line_thickness: Optional[int] = None,
        font_scale: Optional[float] = None,
        show_bbox: bool = True,
        show_confidence: bool = True,
        show_landmarks: bool = True,
        show_quality: bool = True,
        show_pose: bool = True,
        show_face_id: bool = True,
        show_pose_axes: bool = True,
        status_color_coding: bool = True,
        analyze_on_the_fly: bool = True,
    ):
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.badge_bg_color = badge_bg_color or (25, 25, 25)  # Dark Charcoal background
        self.landmark_colors = landmark_colors or DEFAULT_LANDMARK_COLORS
        self.landmark_radius = landmark_radius
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.show_bbox = show_bbox
        self.show_confidence = show_confidence
        self.show_landmarks = show_landmarks
        self.show_quality = show_quality
        self.show_pose = show_pose
        self.show_face_id = show_face_id
        self.show_pose_axes = show_pose_axes
        self.status_color_coding = status_color_coding
        self.analyze_on_the_fly = analyze_on_the_fly

    def annotate(
        self,
        image: Union[np.ndarray, Image.Image, bytes, str, Path],
        detections: Union[
            List[FaceDetection],
            FrameInferenceResult,
            AIDetectionResponse,
            AIAnalysisResponse,
            List[DetectedFaceItem],
            List[StructuredFaceAnalysis],
            List[Dict[str, Any]],
        ],
        return_rgb: bool = False,
    ) -> np.ndarray:
        """
        Draws face detections with quality scores, classifications, pose,
        and landmarks onto an image copy.

        Args:
            image: Input image (NumPy BGR/RGB array, PIL Image, bytes, or file path).
            detections: Detection results (list of FaceDetection/DetectedFaceItem/StructuredFaceAnalysis/dicts,
                        or FrameInferenceResult / AIDetectionResponse / AIAnalysisResponse).
            return_rgb: If True, returns image in RGB format; if False, returns BGR.

        Returns:
            np.ndarray: Annotated image copy (input image is never modified in-place).
        """
        canvas = self._prepare_bgr_canvas(image)
        h, w = canvas.shape[:2]

        # Calculate dynamic rendering scales based on image resolution
        thickness = self.line_thickness or max(1, int(round(min(h, w) / 380)))
        font_scale = self.font_scale or max(0.38, min(0.9, min(h, w) / 900.0))
        landmark_r = self.landmark_radius or max(2, int(round(min(h, w) / 220)))

        # Normalize detections list
        face_list = self._extract_face_list(detections)

        for idx, item in enumerate(face_list, start=1):
            (
                bbox,
                confidence,
                landmarks,
                face_id,
                q_score,
                q_class,
                pose,
            ) = self._unpack_detection(item, default_index=idx)

            if bbox is None:
                continue

            x1, y1, x2, y2 = [int(round(c)) for c in bbox]

            # On-the-fly analysis if quality or pose is missing and requested
            if self.analyze_on_the_fly and (
                (self.show_quality and (q_score is None or q_class is None))
                or (self.show_pose and pose is None)
            ):
                q_score, q_class, pose = self._compute_on_the_fly_analytics(
                    canvas=canvas,
                    bbox=(x1, y1, x2, y2),
                    landmarks=landmarks,
                    confidence=confidence or 0.95,
                    existing_q_score=q_score,
                    existing_q_class=q_class,
                    existing_pose=pose,
                )

            # Determine primary status color
            status_color = self.bbox_color
            if self.status_color_coding and q_class:
                class_key = str(q_class).lower()
                status_color = STATUS_COLORS.get(class_key, self.bbox_color)

            # 1. Draw Bounding Box
            if self.show_bbox:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), status_color, thickness)

            # 2. Draw Multi-Line Information Card
            has_badge_content = (
                self.show_face_id
                or (self.show_confidence and confidence is not None)
                or (self.show_quality and (q_score is not None or q_class is not None))
                or (self.show_pose and pose is not None)
            )

            if has_badge_content:
                self._draw_info_card(
                    canvas=canvas,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    face_id=face_id,
                    confidence=confidence,
                    quality_score=q_score,
                    quality_class=q_class,
                    pose=pose,
                    status_color=status_color,
                    font_scale=font_scale,
                    thickness=thickness,
                )

            # 3. Draw Facial Landmarks
            normalized_landmarks = self._normalize_landmarks(landmarks)
            nose_point = None

            if self.show_landmarks and normalized_landmarks:
                for lx, ly, lm_name in normalized_landmarks:
                    pt = (int(round(lx)), int(round(ly)))
                    if lm_name in ("nose", "nose_tip"):
                        nose_point = pt

                    lm_color = self.landmark_colors.get(lm_name, (0, 215, 255))
                    # Outer dark border circle for visibility on bright regions
                    cv2.circle(canvas, pt, landmark_r + 1, (0, 0, 0), -1, cv2.LINE_AA)
                    # Colored landmark point
                    cv2.circle(canvas, pt, landmark_r, lm_color, -1, cv2.LINE_AA)

            # 4. Optional Pose Orientation Arrow / Direction Vector
            if self.show_pose_axes and pose and nose_point:
                self._draw_pose_direction(
                    canvas=canvas,
                    nose_pt=nose_point,
                    pose=pose,
                    bbox_w=x2 - x1,
                    thickness=thickness,
                )

        if return_rgb:
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return canvas

    def _draw_info_card(
        self,
        canvas: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        face_id: Any,
        confidence: Optional[float],
        quality_score: Optional[float],
        quality_class: Optional[str],
        pose: Optional[Dict[str, float]],
        status_color: Tuple[int, int, int],
        font_scale: float,
        thickness: int,
    ) -> None:
        """Renders structured multi-line card displaying face metadata above/inside bbox."""
        h, w = canvas.shape[:2]
        pad = 6
        text_thick = max(1, thickness - 1)

        card_lines: List[Tuple[str, Tuple[int, int, int]]] = []

        # Line 1: Face Header (e.g. "Face 1")
        if self.show_face_id:
            label = f"Face {face_id}" if face_id is not None else "Face"
            card_lines.append((label, status_color))

        # Line 2: Confidence (e.g. "Confidence: 97%")
        if self.show_confidence and confidence is not None:
            pct = int(round(confidence * 100)) if confidence <= 1.0 else int(round(confidence))
            card_lines.append((f"Confidence: {pct}%", self.text_color))

        # Line 3: Quality Score (e.g. "Quality: 91/100")
        if self.show_quality and quality_score is not None:
            q_val = int(round(quality_score)) if quality_score > 1.0 else int(round(quality_score * 100))
            card_lines.append((f"Quality: {q_val}/100", self.text_color))

        # Line 4: Status Classification (e.g. "Status: Excellent")
        if self.show_quality and quality_class is not None:
            status_text = str(quality_class).capitalize()
            card_lines.append((f"Status: {status_text}", status_color))

        # Line 5: 3D Head Pose Orientation (e.g. "Pose: Y:+3.1 P:-1.2 R:+0.5")
        if self.show_pose and pose is not None:
            yaw = pose.get("yaw", 0.0)
            pitch = pose.get("pitch", 0.0)
            roll = pose.get("roll", 0.0)
            card_lines.append((f"Pose: Y:{yaw:+.1f} P:{pitch:+.1f} R:{roll:+.1f}", (220, 220, 220)))

        if not card_lines:
            return

        # Measure text dimensions
        line_metrics = [
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)[0]
            for text, _ in card_lines
        ]
        max_line_w = max(lw for lw, lh in line_metrics)
        max_line_h = max(lh for lw, lh in line_metrics)
        line_height = int(round(max_line_h * 1.5))

        card_w = max_line_w + (pad * 2)
        card_h = (len(card_lines) * line_height) + pad + 2

        # Position card: prefer above bbox, else place inside bbox top
        card_x1 = max(0, x1)
        card_x2 = min(w, card_x1 + card_w)

        if (y1 - card_h) >= 0:
            card_y1 = y1 - card_h
            card_y2 = y1
        else:
            card_y1 = min(h - card_h, y1)
            card_y2 = card_y1 + card_h

        # Draw solid dark background card
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), self.badge_bg_color, -1)

        # Draw accent top border in status color
        cv2.line(canvas, (card_x1, card_y1), (card_x2, card_y1), status_color, max(2, thickness))

        # Draw left border accent
        cv2.line(canvas, (card_x1, card_y1), (card_x1, card_y2), status_color, max(1, thickness - 1))

        # Draw text lines
        cur_y = card_y1 + pad + max_line_h
        for text, color in card_lines:
            cv2.putText(
                canvas,
                text,
                (card_x1 + pad, cur_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                text_thick,
                cv2.LINE_AA,
            )
            cur_y += line_height

    def _draw_pose_direction(
        self,
        canvas: np.ndarray,
        nose_pt: Tuple[int, int],
        pose: Dict[str, float],
        bbox_w: int,
        thickness: int,
    ) -> None:
        """Renders subtle 2D orientation pointer from nose indicating 3D head gaze."""
        yaw = pose.get("yaw", 0.0)
        pitch = pose.get("pitch", 0.0)

        # Length proportional to face size
        vec_len = max(12.0, float(bbox_w) * 0.28)
        rad_yaw = math.radians(yaw)
        rad_pitch = math.radians(pitch)

        # Yaw rotates horizontally (negative = turned left, positive = turned right)
        dx = int(round(math.sin(rad_yaw) * vec_len))
        # Pitch rotates vertically (negative = tilted down, positive = tilted up)
        dy = int(round(-math.sin(rad_pitch) * vec_len))

        nx, ny = nose_pt
        end_pt = (nx + dx, ny + dy)

        # Draw arrowed direction vector in Amber
        cv2.arrowedLine(
            canvas,
            (nx, ny),
            end_pt,
            (0, 215, 255),
            max(1, thickness),
            tipLength=0.35,
        )

    def _compute_on_the_fly_analytics(
        self,
        canvas: np.ndarray,
        bbox: Tuple[int, int, int, int],
        landmarks: Any,
        confidence: float,
        existing_q_score: Optional[float],
        existing_q_class: Optional[str],
        existing_pose: Optional[Dict[str, float]],
    ) -> Tuple[Optional[float], Optional[str], Optional[Dict[str, float]]]:
        """Calculates quality score, status, and pose on-the-fly for raw detections."""
        q_score = existing_q_score
        q_class = existing_q_class
        pose = existing_pose

        try:
            from app.services.face_analysis_service import get_face_analysis_service
            analysis_svc = get_face_analysis_service()
            h, w = canvas.shape[:2]
            x1, y1, x2, y2 = bbox

            crop_x1 = max(0, x1)
            crop_y1 = max(0, y1)
            crop_x2 = min(w, x2)
            crop_y2 = min(h, y2)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                return q_score, q_class, pose

            face_crop_bgr = canvas[crop_y1:crop_y2, crop_x1:crop_x2]
            face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

            # Extract landmarks
            lms_dict, lms_list = analysis_svc.extract_and_validate_landmarks(landmarks, (h, w))

            # Quality calculation
            if q_score is None:
                b_box = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
                size_m = analysis_svc.analyze_size(b_box, (h, w))
                _, _, clarity = analysis_svc.analyze_blur(face_crop_rgb)
                _, _, illum = analysis_svc.analyze_brightness(face_crop_rgb)
                score_100, class_str, _, _ = analysis_svc.calculate_quality(
                    face_crop=face_crop_rgb,
                    bbox=b_box,
                    image_shape=(h, w),
                    clarity_metrics=clarity,
                    illumination_metrics=illum,
                    size_metrics=size_m,
                    landmarks_list=lms_list,
                    detection_confidence=confidence,
                )
                q_score = score_100
                q_class = class_str

            # Pose calculation
            if pose is None and lms_list and len(lms_list) >= 5:
                angles_dict, _ = analysis_svc.estimate_pose(lms_list, (h, w))
                pose = angles_dict

        except Exception:
            pass

        return q_score, q_class, pose

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
            img_copy = image.copy()
            if img_copy.ndim == 2:
                return cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
            if img_copy.ndim == 3:
                if img_copy.shape[2] == 4:
                    return cv2.cvtColor(img_copy, cv2.COLOR_RGBA2BGR)
                if img_copy.shape[2] == 3:
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
        self, item: Any, default_index: int = 1
    ) -> Tuple[
        Optional[Tuple[float, float, float, float]],
        Optional[float],
        Any,
        Any,
        Optional[float],
        Optional[str],
        Optional[Dict[str, float]],
    ]:
        """Extracts bbox, confidence, landmarks, face_id, quality_score, status, and pose."""
        bbox = None
        confidence = None
        landmarks = None
        face_id = default_index
        quality_score = None
        quality_class = None
        pose = None

        # 1. DetectedFaceItem / StructuredFaceAnalysis / FaceDetection objects
        if isinstance(item, (FaceDetection, DetectedFaceItem, StructuredFaceAnalysis)):
            b = item.bbox
            if hasattr(b, "x1"):
                bbox = (b.x1, b.y1, b.x2, b.y2)
            confidence = getattr(item, "confidence", None)
            landmarks = getattr(item, "landmarks", None) or getattr(item, "landmarks_list", None)
            if getattr(item, "face_id", None) is not None:
                face_id = item.face_id

            # Quality extraction
            if hasattr(item, "quality") and item.quality is not None:
                q = item.quality
                if hasattr(q, "score") and q.score is not None:
                    quality_score = float(q.score)
                elif hasattr(q, "quality_score") and q.quality_score is not None:
                    quality_score = float(q.quality_score)
                elif hasattr(q, "overall_quality") and q.overall_quality is not None:
                    quality_score = float(q.overall_quality) * 100.0

                if hasattr(q, "classification") and q.classification is not None:
                    c = q.classification
                    quality_class = c.value if hasattr(c, "value") else str(c)

            if quality_score is None and getattr(item, "quality_score", None) is not None:
                quality_score = float(item.quality_score)
            if quality_class is None and getattr(item, "classification", None) is not None:
                c = item.classification
                quality_class = c.value if hasattr(c, "value") else str(c)

            # Pose extraction
            raw_pose = getattr(item, "pose", None)
            if raw_pose is not None:
                if isinstance(raw_pose, dict):
                    pose = {
                        "yaw": float(raw_pose.get("yaw", 0.0)),
                        "pitch": float(raw_pose.get("pitch", 0.0)),
                        "roll": float(raw_pose.get("roll", 0.0)),
                    }
                elif hasattr(raw_pose, "to_angles_dict"):
                    pose = raw_pose.to_angles_dict()
                elif hasattr(raw_pose, "yaw"):
                    pose = {
                        "yaw": float(raw_pose.yaw),
                        "pitch": float(raw_pose.pitch),
                        "roll": float(raw_pose.roll),
                    }

        # 2. Dictionary input
        elif isinstance(item, dict):
            if item.get("face_id") is not None:
                face_id = item.get("face_id")
            confidence = item.get("confidence")
            landmarks = item.get("landmarks") or item.get("landmarks_list")

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

            # Quality
            raw_q = item.get("quality")
            if isinstance(raw_q, dict):
                quality_score = raw_q.get("score", raw_q.get("quality_score"))
                quality_class = raw_q.get("classification", raw_q.get("status"))
            if quality_score is None:
                quality_score = item.get("quality_score")
            if quality_class is None:
                quality_class = item.get("classification", item.get("status"))

            # Pose
            raw_p = item.get("pose")
            if isinstance(raw_p, dict):
                pose = {
                    "yaw": float(raw_p.get("yaw", 0.0)),
                    "pitch": float(raw_p.get("pitch", 0.0)),
                    "roll": float(raw_p.get("roll", 0.0)),
                }

        # 3. Generic object with bbox attribute
        elif hasattr(item, "bbox"):
            b = getattr(item, "bbox")
            if hasattr(b, "x1"):
                bbox = (b.x1, b.y1, b.x2, b.y2)
            elif isinstance(b, (list, tuple)) and len(b) == 4:
                bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            confidence = getattr(item, "confidence", None)
            landmarks = getattr(item, "landmarks", None)
            if getattr(item, "face_id", None) is not None:
                face_id = item.face_id

        return bbox, confidence, landmarks, face_id, quality_score, quality_class, pose

    def _normalize_landmarks(self, landmarks: Any) -> List[Tuple[float, float, str]]:
        """Converts diverse landmark formats (dict, list of points/tuples) into List[(x, y, name)]."""
        if not landmarks:
            return []

        result: List[Tuple[float, float, str]] = []

        if isinstance(landmarks, dict):
            for name, val in landmarks.items():
                if isinstance(val, dict):
                    result.append((float(val.get("x", 0.0)), float(val.get("y", 0.0)), str(name)))
                elif hasattr(val, "x") and hasattr(val, "y"):
                    result.append((float(val.x), float(val.y), str(name)))
                elif isinstance(val, (list, tuple)) and len(val) >= 2:
                    result.append((float(val[0]), float(val[1]), str(name)))
            return result

        if isinstance(landmarks, (list, tuple)):
            for idx, lm in enumerate(landmarks):
                lx, ly, name = self._unpack_landmark_single(lm, idx)
                if lx is not None and ly is not None:
                    result.append((lx, ly, name))
            return result

        return []

    def _unpack_landmark_single(self, lm: Any, index: int) -> Tuple[Optional[float], Optional[float], str]:
        """Extracts x, y, and landmark name from a single landmark element."""
        if isinstance(lm, (LandmarkPoint, LandmarkCoordinate)):
            return float(lm.x), float(lm.y), getattr(lm, "name", f"point_{index}")
        if isinstance(lm, dict):
            return (
                float(lm.get("x", 0.0)),
                float(lm.get("y", 0.0)),
                lm.get("name", f"point_{index}"),
            )
        if hasattr(lm, "x") and hasattr(lm, "y"):
            return float(getattr(lm, "x")), float(getattr(lm, "y")), getattr(lm, "name", f"point_{index}")
        if isinstance(lm, (list, tuple)) and len(lm) >= 2:
            return float(lm[0]), float(lm[1]), f"point_{index}"
        return None, None, f"point_{index}"


def draw_detections(
    image: Union[np.ndarray, Image.Image, bytes, str, Path],
    detections: Union[
        List[FaceDetection],
        FrameInferenceResult,
        AIDetectionResponse,
        AIAnalysisResponse,
        List[DetectedFaceItem],
        List[StructuredFaceAnalysis],
        List[Dict[str, Any]],
    ],
    show_bbox: bool = True,
    show_confidence: bool = True,
    show_landmarks: bool = True,
    show_quality: bool = True,
    show_pose: bool = True,
    show_face_id: bool = True,
    show_pose_axes: bool = True,
    status_color_coding: bool = True,
    return_rgb: bool = False,
) -> np.ndarray:
    """
    Convenience function to draw detections with quality scores, classifications,
    pose, and landmarks onto an image copy in volatile memory.

    Args:
        image: Source image (array, PIL, bytes, or file path).
        detections: Structured detections.
        show_bbox: Whether to render face bounding boxes.
        show_confidence: Whether to render confidence percentage badge.
        show_landmarks: Whether to render 5-point facial landmarks.
        show_quality: Whether to render quality score (0-100) and status.
        show_pose: Whether to render pose orientation angles.
        show_face_id: Whether to render face index/ID header.
        show_pose_axes: Whether to render direction arrow from nose.
        status_color_coding: Whether to tint bounding box based on status.
        return_rgb: Whether to return RGB array instead of default BGR.

    Returns:
        np.ndarray: Annotated image copy.
    """
    visualizer = FaceVisualizer(
        show_bbox=show_bbox,
        show_confidence=show_confidence,
        show_landmarks=show_landmarks,
        show_quality=show_quality,
        show_pose=show_pose,
        show_face_id=show_face_id,
        show_pose_axes=show_pose_axes,
        status_color_coding=status_color_coding,
    )
    return visualizer.annotate(image, detections, return_rgb=return_rgb)


def annotate_image(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    detector: Optional[Any] = None,
    confidence_threshold: Optional[float] = None,
    show_landmarks: bool = True,
    show_quality: bool = True,
    show_pose: bool = True,
) -> Tuple[np.ndarray, FrameInferenceResult]:
    """
    Development utility: loads input image, executes real face detection,
    computes face analytics, and produces an annotated copy saved to output_path.

    Example:
        input.jpg -> DeepVision AI -> output_detected.jpg

    Args:
        input_path: Path to the input image file.
        output_path: Destination path for the annotated image (optional).
                     If None, defaults to '<input_stem>_detected<ext>'.
        detector: Optional FaceDetector instance. If None, initialized from default settings.
        confidence_threshold: Detection confidence cutoff (defaults to settings.AI_CONFIDENCE_THRESHOLD).
        show_landmarks: Whether to render facial landmarks.
        show_quality: Whether to render quality score and status.
        show_pose: Whether to render 3D head pose orientation.

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
    visualizer = FaceVisualizer(
        show_landmarks=show_landmarks,
        show_quality=show_quality,
        show_pose=show_pose,
        analyze_on_the_fly=True,
    )
    annotated_bgr = visualizer.annotate(img_bgr, inference_result)

    # Write output if requested
    if output_path is not None:
        out_p = Path(output_path)
    else:
        out_p = input_p.parent / f"{input_p.stem}_detected{input_p.suffix}"

    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), annotated_bgr)

    return annotated_bgr, inference_result
