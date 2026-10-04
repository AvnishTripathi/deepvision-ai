"""
DeepVision AI — Model Output Adapters
Translates raw model tensor outputs (logits, anchor offsets, landmark regressions)
into standardized domain schemas (FaceDetection, BoundingBox, LandmarkPoint).
Decouples specific model head architectures (SCRFD, YuNet, YOLO) from the detector.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from app.ai.detection.schemas import (
    BoundingBox,
    NormalizedBoundingBox,
    LandmarkPoint,
    FaceDetection,
)
from app.ai.detection.preprocessing import unletterbox_coordinates


class BaseModelAdapter(ABC):
    """Abstract interface for decoding raw model tensors into FaceDetection objects."""

    @abstractmethod
    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple[int, int],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> List[FaceDetection]:
        """
        Decodes raw model tensors.
        
        Args:
            outputs: List of raw output tensors from model forward-pass.
            scale: Letterbox scale applied during preprocessing.
            padding: (pad_x, pad_y) applied during preprocessing.
            orig_shape: (height, width) of input image before preprocessing.
            confidence_threshold: Score cutoff for candidate proposals.
            nms_threshold: IoU threshold for Non-Maximum Suppression.
            
        Returns:
            List[FaceDetection]: Decoded face detections in original image coordinates.
        """
        pass


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Computes Intersection-over-Union (IoU) between a box and an array of boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection

    return np.where(union > 0, intersection / union, 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.4) -> List[int]:
    """Applies Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes."""
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        ious = compute_iou(boxes[i], boxes[order[1:]])
        inds = np.where(ious <= threshold)[0]
        order = order[inds + 1]

    return keep


class SCRFDAdapter(BaseModelAdapter):
    """
    Model output adapter for SCRFD (Sample and Computation Redistribution for Efficient Face Detection).
    Decodes multi-stride anchor feature pyramids (strides 8, 16, 32), score logits,
    bounding box distance offsets, and 5-point facial landmark coordinates.
    """

    def __init__(self, strides: Tuple[int, ...] = (8, 16, 32)):
        self.strides = strides

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple[int, int],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> List[FaceDetection]:
        """
        Decodes raw SCRFD ONNX output arrays.
        SCRFD outputs 9 tensors: 3 strides x [scores, bboxes, landmarks].
        """
        orig_h, orig_w = orig_shape
        all_boxes: List[np.ndarray] = []
        all_scores: List[float] = []
        all_kpss: List[Optional[np.ndarray]] = []

        # If model outputs conform to standard SCRFD 6 or 9 head layout
        if len(outputs) in (6, 9):
            has_kps = len(outputs) == 9
            stride_count = len(self.strides)

            for idx, stride in enumerate(self.strides):
                score_tensor = outputs[idx]
                bbox_tensor = outputs[idx + stride_count]
                kps_tensor = outputs[idx + stride_count * 2] if has_kps else None

                # Flatten batch dimension
                if score_tensor.ndim == 3 and score_tensor.shape[0] == 1:
                    score_tensor = score_tensor[0]
                if bbox_tensor.ndim == 3 and bbox_tensor.shape[0] == 1:
                    bbox_tensor = bbox_tensor[0]
                if kps_tensor is not None and kps_tensor.ndim == 3 and kps_tensor.shape[0] == 1:
                    kps_tensor = kps_tensor[0]

                # Scores thresholding
                scores = score_tensor.flatten()
                pos_indices = np.where(scores >= confidence_threshold)[0]

                if len(pos_indices) == 0:
                    continue

                for pos in pos_indices:
                    all_scores.append(float(scores[pos]))
                    # Decode bbox (scaled distance offsets * stride)
                    box = bbox_tensor[pos] * stride if bbox_tensor.ndim > 1 else bbox_tensor
                    all_boxes.append(box[:4])
                    if kps_tensor is not None:
                        all_kpss.append(kps_tensor[pos] * stride)
                    else:
                        all_kpss.append(None)

        if not all_boxes:
            return []

        boxes_arr = np.array(all_boxes, dtype=np.float32)
        scores_arr = np.array(all_scores, dtype=np.float32)

        # Apply Non-Maximum Suppression
        keep_indices = nms(boxes_arr, scores_arr, threshold=nms_threshold)

        detections: List[FaceDetection] = []
        for rank, i in enumerate(keep_indices):
            raw_box = boxes_arr[i]
            x1, y1, x2, y2 = unletterbox_coordinates(
                float(raw_box[0]), float(raw_box[1]),
                float(raw_box[2]), float(raw_box[3]),
                scale, padding, orig_shape
            )
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            norm_bbox = bbox.to_normalized(orig_w, orig_h)

            landmarks: Optional[List[LandmarkPoint]] = None
            if all_kpss[i] is not None:
                kps = all_kpss[i].reshape(-1, 2)
                names = ("left_eye", "right_eye", "nose_tip", "mouth_left", "mouth_right")
                landmarks = []
                for pt_idx, (kx, ky) in enumerate(kps[:5]):
                    rx, ry, _, _ = unletterbox_coordinates(float(kx), float(ky), float(kx), float(ky), scale, padding, orig_shape)
                    landmarks.append(LandmarkPoint(
                        name=names[pt_idx] if pt_idx < len(names) else f"pt_{pt_idx}",
                        x=rx,
                        y=ry,
                        confidence=round(scores_arr[i], 4),
                    ))

            detections.append(FaceDetection(
                face_id=rank,
                bbox=bbox,
                confidence=round(float(scores_arr[i]), 4),
                landmarks=landmarks,
                normalized_bbox=norm_bbox,
                metadata={"stride_decoded": True},
            ))

        return detections


class BaselineAdapter(BaseModelAdapter):
    """
    Standard fallback adapter when physical weights have not yet been placed on disk.
    Allows end-to-end integration testing of the pipeline without third-party binary downloads.
    """

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple[int, int],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> List[FaceDetection]:
        # Emits baseline detection if threshold satisfied
        orig_h, orig_w = orig_shape
        if 0.95 < confidence_threshold:
            return []

        x1 = round(orig_w * 0.25, 2)
        y1 = round(orig_h * 0.20, 2)
        x2 = round(orig_w * 0.75, 2)
        y2 = round(orig_h * 0.80, 2)

        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        norm_bbox = bbox.to_normalized(orig_w, orig_h)
        box_w = x2 - x1
        box_h = y2 - y1

        landmarks = [
            LandmarkPoint(name="left_eye", x=round(x1 + box_w * 0.32, 2), y=round(y1 + box_h * 0.38, 2), confidence=0.96),
            LandmarkPoint(name="right_eye", x=round(x1 + box_w * 0.68, 2), y=round(y1 + box_h * 0.38, 2), confidence=0.96),
            LandmarkPoint(name="nose_tip", x=round(x1 + box_w * 0.50, 2), y=round(y1 + box_h * 0.55, 2), confidence=0.94),
            LandmarkPoint(name="mouth_left", x=round(x1 + box_w * 0.35, 2), y=round(y1 + box_h * 0.72, 2), confidence=0.92),
            LandmarkPoint(name="mouth_right", x=round(x1 + box_w * 0.65, 2), y=round(y1 + box_h * 0.72, 2), confidence=0.92),
        ]

        return [
            FaceDetection(
                face_id=None,
                bbox=bbox,
                confidence=0.95,
                landmarks=landmarks,
                normalized_bbox=norm_bbox,
                metadata={"adapter": "baseline"},
            )
        ]
