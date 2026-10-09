"""
DeepVision AI — ByteTrack Multi-Face Tracking Engine (Step 5.4, 5.5)
Implements high-performance, CPU-optimized ByteTrack multi-object tracking
tailored specifically for face tracking across video frames.

CORE ARCHITECTURAL DESIGN:
1. Reuses existing OpenCV YuNet face detections without secondary model dependencies.
2. Two-stage BYTE association associates high-confidence detections first,
   then recovers occluded or motion-blurred faces using low-confidence detections.
3. 2D Constant-Velocity Kalman Filter provides smooth bounding box predictions
   and coasts through temporary missed frames / occlusions.
4. Strictly adheres to Step 5.3 Identity Rule: track_id represents temporary visual
   continuity only. Zero biometric embeddings, zero face recognition, zero database matching.
"""

import time
import logging
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np

from app.schemas.ai import FaceDetection, BoundingBox, LandmarkPoint
from app.ai.tracking.base import BaseFaceTracker
from app.ai.tracking.schemas import (
    TrackState,
    TrackResult,
    TrackMetadata,
    TrackingSummary,
)

logger = logging.getLogger("deepvision.tracking")


# =============================================================================
# Kalman Filter Motion Model for 2D Bounding Boxes
# State: [cx, cy, s, r, vx, vy, vs, vr]^T
# cx, cy: Bounding box center
# s: Area (w * h)
# r: Aspect ratio (w / h)
# =============================================================================

class KalmanBoxTracker:
    """
    Kalman filter motion model estimating bounding box trajectory and velocity.
    Operates in state space [cx, cy, s, r, vx, vy, vs, vr].
    """

    def __init__(self, bbox: BoundingBox, initial_confidence: float = 1.0, landmarks: Optional[List[Any]] = None):
        # 8-dimensional state vector
        self.state = np.zeros(8, dtype=np.float32)
        
        # State covariance matrix P
        self.P = np.diag([10.0, 10.0, 50.0, 1.0, 1000.0, 1000.0, 1000.0, 1.0]).astype(np.float32)
        
        # Process noise covariance Q
        self.Q = np.diag([1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.0001, 0.0001]).astype(np.float32)
        
        # Measurement noise covariance R (measurement: [cx, cy, s, r])
        self.R = np.diag([1.0, 1.0, 10.0, 0.01]).astype(np.float32)
        
        # Initialize state with first measurement
        cx, cy, s, r = self._bbox_to_z(bbox)
        self.state[:4] = [cx, cy, s, r]
        
        # Tracklet metadata
        self.track_id: Optional[int] = None
        self.state_enum: TrackState = TrackState.TENTATIVE
        self.age: int = 1
        self.hits: int = 1
        self.missed_frames: int = 0
        self.start_frame: int = 0
        self.last_frame: int = 0
        
        # Confidence smoothing
        self.latest_detection_confidence: Optional[float] = initial_confidence
        self.track_confidence: float = initial_confidence
        self.confidence_history: List[float] = [initial_confidence]
        
        # Landmarks cache
        self.landmarks = landmarks

    @staticmethod
    def _bbox_to_z(bbox: BoundingBox) -> Tuple[float, float, float, float]:
        """Converts BoundingBox [x1, y1, x2, y2] to measurement [cx, cy, s, r]."""
        w = max(1.0, bbox.width)
        h = max(1.0, bbox.height)
        cx = bbox.x1 + w / 2.0
        cy = bbox.y1 + h / 2.0
        s = w * h
        r = w / h
        return float(cx), float(cy), float(s), float(r)

    @staticmethod
    def _z_to_bbox(cx: float, cy: float, s: float, r: float) -> BoundingBox:
        """Converts state [cx, cy, s, r] back to BoundingBox [x1, y1, x2, y2]."""
        s = max(1.0, s)
        r = max(0.01, r)
        w = np.sqrt(s * r)
        h = s / w
        x1 = round(float(cx - w / 2.0), 2)
        y1 = round(float(cy - h / 2.0), 2)
        x2 = round(float(cx + w / 2.0), 2)
        y2 = round(float(cy + h / 2.0), 2)
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def predict(self) -> BoundingBox:
        """
        Advances the state vector via constant-velocity motion prediction.
        x' = x + v * dt (dt = 1 frame)
        """
        # Center coordinates move with velocity
        self.state[0] += self.state[4]
        self.state[1] += self.state[5]
        self.state[2] += self.state[6]
        self.state[3] += self.state[7]
        
        # Covariance prediction: P' = F P F^T + Q
        # For identity velocity transition, add velocity uncertainty to position
        self.P[:4, :4] += self.P[4:, 4:] + self.Q[:4, :4]
        self.P[4:, 4:] += self.Q[4:, 4:]
        
        self.age += 1
        return self.get_bbox()

    def update(
        self,
        bbox: BoundingBox,
        confidence: float,
        landmarks: Optional[List[Any]] = None,
        frame_index: int = 0,
    ) -> None:
        """
        Updates the Kalman filter state with a matched detection.
        """
        z = np.array(self._bbox_to_z(bbox), dtype=np.float32)
        
        # Innovation y = z - H x'
        y = z - self.state[:4]
        
        # Innovation covariance S = H P H^T + R
        S = self.P[:4, :4] + self.R
        
        # Kalman gain K = P H^T S^-1
        try:
            K = np.linalg.solve(S.T, self.P[:, :4].T).T
        except np.linalg.LinAlgError:
            K = np.zeros((8, 4), dtype=np.float32)
        
        # State update: x = x' + K y
        self.state += K @ y
        
        # Covariance update: P = (I - K H) P'
        I_KH = np.eye(8, dtype=np.float32)
        I_KH[:, :4] -= K
        self.P = I_KH @ self.P
        
        # Ensure aspect ratio and area remain physically plausible
        self.state[2] = max(1.0, self.state[2])
        self.state[3] = max(0.05, min(10.0, self.state[3]))
        
        # Lifecycle counters
        self.hits += 1
        self.missed_frames = 0
        self.last_frame = frame_index
        self.latest_detection_confidence = float(confidence)
        self.confidence_history.append(float(confidence))
        
        # Exponential moving average for track confidence
        alpha = 0.3
        self.track_confidence = round(alpha * float(confidence) + (1.0 - alpha) * self.track_confidence, 4)
        
        if landmarks is not None:
            self.landmarks = landmarks

    def mark_missed(self) -> None:
        """Increments missed frames counter when no detection matches."""
        self.missed_frames += 1
        self.latest_detection_confidence = None
        # Decay track confidence slightly during coasting
        self.track_confidence = max(0.1, round(self.track_confidence * 0.95, 4))

    def get_bbox(self) -> BoundingBox:
        """Returns the current estimated BoundingBox."""
        return self._z_to_bbox(self.state[0], self.state[1], self.state[2], self.state[3])

    def get_velocity(self) -> Tuple[float, float]:
        """Returns center velocity (vx, vy) in pixels/frame."""
        return (round(float(self.state[4]), 2), round(float(self.state[5]), 2))


# =============================================================================
# Bounding Box Geometry & Linear Assignment
# =============================================================================

def compute_iou_matrix(boxes_a: List[BoundingBox], boxes_b: List[BoundingBox]) -> np.ndarray:
    """
    Computes Intersection-over-Union (IoU) matrix between two lists of BoundingBoxes.
    Returns: NxM numpy array of IoU values in range [0.0, 1.0].
    """
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Convert to vectorized arrays [N, 4] and [M, 4]
    a = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes_a], dtype=np.float32)
    b = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes_b], dtype=np.float32)

    # Intersection coordinates
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Areas
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    union = np.maximum(union, 1e-6)

    return (intersection / union).astype(np.float32)


def linear_assignment(
    cost_matrix: np.ndarray,
    cost_threshold: float = 0.7,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solves linear sum assignment for bipartite matching.
    Uses scipy.optimize.linear_sum_assignment if available; otherwise falls back
    to a fast, greedy matching algorithm with identical signature.
    
    Args:
        cost_matrix: Cost matrix (e.g. 1.0 - IoU) of shape (num_tracks, num_dets).
        cost_threshold: Maximum allowable cost for an assigned pair.
        
    Returns:
        matches: List of (track_idx, det_idx) pairs.
        unmatched_tracks: List of unassigned track indices.
        unmatched_dets: List of unassigned detection indices.
    """
    num_tracks, num_dets = cost_matrix.shape
    if num_tracks == 0:
        return [], [], list(range(num_dets))
    if num_dets == 0:
        return [], list(range(num_tracks)), []

    # Attempt optimal Hungarian matching via SciPy if present
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ImportError:
        # Fast, robust greedy bipartite matching fallback
        flat_indices = np.argsort(cost_matrix, axis=None)
        assigned_rows = set()
        assigned_cols = set()
        row_ind = []
        col_ind = []
        for idx in flat_indices:
            r = int(idx // num_dets)
            c = int(idx % num_dets)
            if r not in assigned_rows and c not in assigned_cols:
                row_ind.append(r)
                col_ind.append(c)
                assigned_rows.add(r)
                assigned_cols.add(c)
        row_ind = np.array(row_ind, dtype=int)
        col_ind = np.array(col_ind, dtype=int)

    matches: List[Tuple[int, int]] = []
    unmatched_tracks = set(range(num_tracks))
    unmatched_dets = set(range(num_dets))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= cost_threshold:
            matches.append((int(r), int(c)))
            unmatched_tracks.discard(int(r))
            unmatched_dets.discard(int(c))

    return matches, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))


# =============================================================================
# ByteTrack Multi-Face Tracker Implementation
# =============================================================================

class ByteTrackFaceTracker(BaseFaceTracker):
    """
    ByteTrack implementation optimized for video face tracking.
    
    Adheres strictly to the architectural rule:
    Video Frame -> Face Detector -> Bounding Boxes -> Tracker -> Track IDs -> Face Analysis
    """

    def __init__(
        self,
        high_detection_threshold: float = 0.50,
        low_detection_threshold: float = 0.15,
        match_threshold_first: float = 0.70,   # Max cost (1 - IoU) for high-confidence match
        match_threshold_second: float = 0.50,  # Max cost for secondary low-confidence match
        max_time_lost: int = 30,               # Maximum frames to coast lost track (~1s at 30 fps)
        min_hits: int = 3,                     # Consecutive hits required to confirm track
    ):
        self.high_detection_threshold = high_detection_threshold
        self.low_detection_threshold = low_detection_threshold
        self.match_threshold_first = match_threshold_first
        self.match_threshold_second = match_threshold_second
        self.max_time_lost = max_time_lost
        self.min_hits = min_hits

        # Tracker state
        self._next_track_id: int = 1
        self._tracked_stracks: List[KalmanBoxTracker] = []
        self._lost_stracks: List[KalmanBoxTracker] = []
        self._retired_tracks_history: List[KalmanBoxTracker] = []
        self._total_frames: int = 0
        self._peak_simultaneous_faces: int = 0
        self._track_lost_events: int = 0

    @property
    def active_track_count(self) -> int:
        """Count of confirmed active tracks."""
        return sum(1 for t in self._tracked_stracks if t.state_enum == TrackState.CONFIRMED)

    @property
    def total_unique_tracks(self) -> int:
        """Total sequential track IDs assigned during this session."""
        return self._next_track_id - 1

    @property
    def lost_track_count(self) -> int:
        """Total count of track loss events observed during this session."""
        return self._track_lost_events

    def reset(self) -> None:
        """Resets tracker state, clearing all active tracks and resetting track IDs."""
        self._next_track_id = 1
        self._tracked_stracks.clear()
        self._lost_stracks.clear()
        self._retired_tracks_history.clear()
        self._total_frames = 0
        self._peak_simultaneous_faces = 0
        self._track_lost_events = 0
        logger.debug("ByteTrackFaceTracker state reset successfully.")

    def _parse_detections(
        self,
        detections: Union[List[FaceDetection], List[Dict[str, Any]], np.ndarray],
    ) -> Tuple[List[BoundingBox], List[float], List[Optional[List[Any]]]]:
        """
        Parses diverse detection formats into standardized lists:
        (bboxes, confidences, landmarks).
        """
        bboxes: List[BoundingBox] = []
        confidences: List[float] = []
        landmarks_list: List[Optional[List[Any]]] = []

        if isinstance(detections, np.ndarray):
            # Shape Nx5: [x1, y1, x2, y2, score]
            for row in detections:
                if len(row) >= 5:
                    bboxes.append(BoundingBox(x1=float(row[0]), y1=float(row[1]), x2=float(row[2]), y2=float(row[3])))
                    confidences.append(float(row[4]))
                    landmarks_list.append(None)
            return bboxes, confidences, landmarks_list

        for det in detections:
            if isinstance(det, FaceDetection):
                bboxes.append(det.bbox)
                confidences.append(det.confidence)
                landmarks_list.append(det.landmarks)
            elif isinstance(det, dict):
                bbox_data = det.get("bbox")
                if isinstance(bbox_data, dict):
                    bbox = BoundingBox(**bbox_data)
                elif isinstance(bbox_data, (list, tuple)) and len(bbox_data) == 4:
                    bbox = BoundingBox(x1=bbox_data[0], y1=bbox_data[1], x2=bbox_data[2], y2=bbox_data[3])
                else:
                    continue
                bboxes.append(bbox)
                confidences.append(float(det.get("confidence", 1.0)))
                landmarks_list.append(det.get("landmarks"))
            elif hasattr(det, "bbox") and hasattr(det, "confidence"):
                bboxes.append(det.bbox)
                confidences.append(float(det.confidence))
                landmarks_list.append(getattr(det, "landmarks", None))

        return bboxes, confidences, landmarks_list

    def update(
        self,
        detections: Union[List[FaceDetection], List[Dict[str, Any]], np.ndarray],
        frame: Optional[np.ndarray] = None,
        frame_index: int = 0,
        timestamp_ms: Optional[float] = None,
    ) -> List[TrackResult]:
        """
        Executes two-stage BYTE association for the current frame detections.
        """
        self._total_frames += 1
        bboxes, confidences, landmarks_list = self._parse_detections(detections)

        # -------------------------------------------------------------
        # 1. Partition detections into High-Score and Low-Score sets
        # -------------------------------------------------------------
        high_det_indices = []
        low_det_indices = []

        for idx, conf in enumerate(confidences):
            if conf >= self.high_detection_threshold:
                high_det_indices.append(idx)
            elif conf >= self.low_detection_threshold:
                low_det_indices.append(idx)

        # -------------------------------------------------------------
        # 2. Kalman Prediction for all active and lost tracks
        # -------------------------------------------------------------
        all_pool: List[KalmanBoxTracker] = self._tracked_stracks + self._lost_stracks
        for track in all_pool:
            track.predict()

        # -------------------------------------------------------------
        # 3. Stage 1: Match active/lost tracks with High-Score detections
        # -------------------------------------------------------------
        pool_boxes = [t.get_bbox() for t in all_pool]
        high_boxes = [bboxes[i] for i in high_det_indices]

        iou_matrix_1 = compute_iou_matrix(pool_boxes, high_boxes)
        cost_matrix_1 = 1.0 - iou_matrix_1

        matches_1, unmatched_pool_idx_1, unmatched_high_idx = linear_assignment(
            cost_matrix_1, cost_threshold=self.match_threshold_first
        )

        matched_tracks_1: List[KalmanBoxTracker] = []
        for track_idx, high_idx in matches_1:
            track = all_pool[track_idx]
            original_det_idx = high_det_indices[high_idx]
            track.update(
                bbox=bboxes[original_det_idx],
                confidence=confidences[original_det_idx],
                landmarks=landmarks_list[original_det_idx],
                frame_index=frame_index,
            )
            # Promote to confirmed if min_hits reached
            if track.state_enum == TrackState.TENTATIVE and track.hits >= self.min_hits:
                track.state_enum = TrackState.CONFIRMED
                if track.track_id is None:
                    track.track_id = self._next_track_id
                    self._next_track_id += 1
            elif track.state_enum == TrackState.LOST:
                track.state_enum = TrackState.CONFIRMED

            matched_tracks_1.append(track)

        # -------------------------------------------------------------
        # 4. Stage 2: Match remaining tracks with Low-Score detections
        # (Only evaluate confirmed tracks; ignore tentative in stage 2)
        # -------------------------------------------------------------
        remaining_pool = [
            all_pool[idx]
            for idx in unmatched_pool_idx_1
            if all_pool[idx].state_enum == TrackState.CONFIRMED
        ]
        low_boxes = [bboxes[i] for i in low_det_indices]

        matched_tracks_2: List[KalmanBoxTracker] = []
        unmatched_after_stage_2: List[KalmanBoxTracker] = []

        if low_boxes and remaining_pool:
            rem_pool_boxes = [t.get_bbox() for t in remaining_pool]
            iou_matrix_2 = compute_iou_matrix(rem_pool_boxes, low_boxes)
            cost_matrix_2 = 1.0 - iou_matrix_2

            matches_2, unmatched_rem_idx, _ = linear_assignment(
                cost_matrix_2, cost_threshold=self.match_threshold_second
            )

            for track_idx, low_idx in matches_2:
                track = remaining_pool[track_idx]
                original_det_idx = low_det_indices[low_idx]
                track.update(
                    bbox=bboxes[original_det_idx],
                    confidence=confidences[original_det_idx],
                    landmarks=landmarks_list[original_det_idx],
                    frame_index=frame_index,
                )
                track.state_enum = TrackState.CONFIRMED
                matched_tracks_2.append(track)

            for idx in unmatched_rem_idx:
                unmatched_after_stage_2.append(remaining_pool[idx])
        else:
            unmatched_after_stage_2 = remaining_pool

        # Include remaining unmatched tentative tracks from stage 1
        for idx in unmatched_pool_idx_1:
            track = all_pool[idx]
            if track.state_enum == TrackState.TENTATIVE:
                unmatched_after_stage_2.append(track)

        # -------------------------------------------------------------
        # 5. Handle unmatched tracks (Coast or transition to Lost/Deleted)
        # -------------------------------------------------------------
        for track in unmatched_after_stage_2:
            track.mark_missed()
            if track.state_enum == TrackState.CONFIRMED:
                track.state_enum = TrackState.LOST
                self._track_lost_events += 1
            elif track.state_enum == TrackState.TENTATIVE:
                track.state_enum = TrackState.DELETED

        # -------------------------------------------------------------
        # 6. Initialize new tracks from unmatched High-Score detections
        # -------------------------------------------------------------
        new_tracks: List[KalmanBoxTracker] = []
        for high_idx in unmatched_high_idx:
            orig_idx = high_det_indices[high_idx]
            new_tracker = KalmanBoxTracker(
                bbox=bboxes[orig_idx],
                initial_confidence=confidences[orig_idx],
                landmarks=landmarks_list[orig_idx],
            )
            new_tracker.start_frame = frame_index
            new_tracker.last_frame = frame_index
            
            # If min_hits is 1, promote immediately
            if self.min_hits <= 1:
                new_tracker.state_enum = TrackState.CONFIRMED
                new_tracker.track_id = self._next_track_id
                self._next_track_id += 1
            else:
                new_tracker.state_enum = TrackState.TENTATIVE

            new_tracks.append(new_tracker)

        # -------------------------------------------------------------
        # 7. Update active pools and retire expired tracks
        # -------------------------------------------------------------
        self._tracked_stracks = [
            t for t in (matched_tracks_1 + matched_tracks_2 + new_tracks)
            if t.state_enum in (TrackState.CONFIRMED, TrackState.TENTATIVE)
        ]

        self._lost_stracks = []
        for t in unmatched_after_stage_2:
            if t.state_enum == TrackState.LOST:
                if t.missed_frames <= self.max_time_lost:
                    self._lost_stracks.append(t)
                else:
                    t.state_enum = TrackState.DELETED
                    self._retired_tracks_history.append(t)
            elif t.state_enum == TrackState.DELETED:
                self._retired_tracks_history.append(t)

        # -------------------------------------------------------------
        # 8. Build output TrackResult list for confirmed tracks
        # -------------------------------------------------------------
        output_results: List[TrackResult] = []
        for track in self._tracked_stracks:
            if track.state_enum == TrackState.CONFIRMED and track.track_id is not None:
                output_results.append(
                    TrackResult(
                        track_id=track.track_id,
                        bbox=track.get_bbox(),
                        detection_confidence=track.latest_detection_confidence,
                        track_confidence=track.track_confidence,
                        age=track.age,
                        hits=track.hits,
                        missed_frames=track.missed_frames,
                        state=track.state_enum,
                        landmarks=track.landmarks,
                        velocity=track.get_velocity(),
                    )
                )

        # Update peak simultaneous faces observed in any single frame
        self._peak_simultaneous_faces = max(self._peak_simultaneous_faces, len(output_results))

        return output_results

    def get_summary(self) -> TrackingSummary:
        """Compiles aggregate tracking statistics across the completed video stream."""
        all_tracks = self._tracked_stracks + self._lost_stracks + self._retired_tracks_history
        confirmed_tracks = [t for t in all_tracks if t.track_id is not None]

        metadata_list: List[TrackMetadata] = []
        durations = []

        for t in confirmed_tracks:
            total_f = max(1, t.last_frame - t.start_frame + 1)
            durations.append(total_f)
            avg_conf = (
                sum(t.confidence_history) / len(t.confidence_history)
                if t.confidence_history
                else t.track_confidence
            )
            metadata_list.append(
                TrackMetadata(
                    track_id=t.track_id,  # type: ignore
                    start_frame=t.start_frame,
                    end_frame=t.last_frame,
                    total_frames=total_f,
                    total_hits=t.hits,
                    average_confidence=round(float(avg_conf), 4),
                )
            )

        avg_dur = float(np.mean(durations)) if durations else 0.0

        return TrackingSummary(
            total_frames_processed=self._total_frames,
            total_unique_tracks=self.total_unique_tracks,
            average_track_duration_frames=round(avg_dur, 2),
            peak_simultaneous_faces=self._peak_simultaneous_faces,
            track_lost_count=self._track_lost_events,
            tracks=metadata_list,
        )
