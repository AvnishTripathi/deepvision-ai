"""
DeepVision AI — Base Multi-Face Tracker Interface (Step 5.5)
Defines the abstract contract for multi-face tracking engines.
Ensures tracking logic is completely decoupled from the face detector.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
import numpy as np

from app.schemas.ai import FaceDetection, BoundingBox
from app.ai.tracking.schemas import TrackResult, TrackMetadata, TrackingSummary


class BaseFaceTracker(ABC):
    """
    Abstract interface for multi-face tracking algorithms.
    
    Architecture Contract:
    Video Frame -> Face Detector -> Bounding Boxes -> Tracker -> Track IDs -> Face Analysis
    """

    @abstractmethod
    def update(
        self,
        detections: Union[List[FaceDetection], List[Dict[str, Any]], np.ndarray],
        frame: Optional[np.ndarray] = None,
        frame_index: int = 0,
        timestamp_ms: Optional[float] = None,
    ) -> List[TrackResult]:
        """
        Updates tracker state using detections from the current video frame.
        
        Args:
            detections: List of FaceDetection objects, dictionaries with bbox/confidence,
                        or Nx5 numpy array [x1, y1, x2, y2, score].
            frame: Optional RGB or BGR frame image (for visual tracking or optical flow if used).
            frame_index: Sequential frame number (0-based).
            timestamp_ms: Frame presentation timestamp in milliseconds.
            
        Returns:
            List[TrackResult]: List of confirmed, active face tracks with persistent temporary track_id.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets tracker state, clears all active tracks, and resets sequential track IDs.
        Must be called between different video files or streams to ensure temporary IDs
        remain strictly session-scoped.
        """
        pass

    @property
    @abstractmethod
    def active_track_count(self) -> int:
        """Count of currently active, tracked faces."""
        pass

    @property
    @abstractmethod
    def total_unique_tracks(self) -> int:
        """Total count of unique temporary track IDs assigned during this session."""
        pass

    @property
    @abstractmethod
    def lost_track_count(self) -> int:
        """Total count of track loss events observed during this session."""
        pass

    @abstractmethod
    def get_summary(self) -> TrackingSummary:
        """Returns aggregate lifecycle summary metrics for all tracks observed during the session."""
        pass
