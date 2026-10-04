"""
DeepVision AI — Base Multi-Face Tracker Interface
Prepares architectural contract for video frame sequence tracking,
trajectory management, and track ID persistence.
"""

from abc import ABC, abstractmethod
from typing import List
from app.schemas.ai import FaceDetectionResult


class BaseFaceTracker(ABC):
    """Abstract interface for multi-object / facial tracking algorithms."""

    @abstractmethod
    def update(self, detections: List[FaceDetectionResult], frame_timestamp: float) -> List[FaceDetectionResult]:
        """
        Associates frame detections with existing tracklets using IoU / Kalman motion models.
        Assigns consistent track_id to each face detection across frames.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets tracker state between different video streams or sessions."""
        pass

