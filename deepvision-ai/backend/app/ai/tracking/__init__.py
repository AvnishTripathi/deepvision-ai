"""
DeepVision AI — Multi-Face Tracking Module (Step 5)
Provides real-time multi-face tracking across video frames with persistent temporary track IDs,
Kalman filter motion prediction, and ByteTrack association.
"""

from app.ai.tracking.base import BaseFaceTracker
from app.ai.tracking.tracker import ByteTrackFaceTracker, KalmanBoxTracker
from app.ai.tracking.schemas import (
    TrackState,
    TrackResult,
    TrackedFrameResult,
    TrackMetadata,
    TrackingSummary,
)
from app.ai.tracking.factory import get_face_tracker

__all__ = [
    "BaseFaceTracker",
    "ByteTrackFaceTracker",
    "KalmanBoxTracker",
    "TrackState",
    "TrackResult",
    "TrackedFrameResult",
    "TrackMetadata",
    "TrackingSummary",
    "get_face_tracker",
]
