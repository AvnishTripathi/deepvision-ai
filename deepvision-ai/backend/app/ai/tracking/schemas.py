"""
DeepVision AI — Multi-Face Tracking Data Schemas
Defines strongly typed Pydantic v2 data models for temporary video tracklets,
tracking metrics, and frame tracking outputs.

CRITICAL IDENTITY RULE (Step 5.3):
Track IDs (e.g., track_id = 1, 2, 3) represent temporary visual continuity
across nearby video frames only. They NEVER represent person identity,
biometric templates, or database profiles.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict, computed_field

from app.schemas.ai import BoundingBox, LandmarkPoint


class TrackState(str, Enum):
    """Lifecycle states of a video tracklet."""
    TENTATIVE = "tentative"     # Newly detected, awaiting confirmation over min_hits
    CONFIRMED = "confirmed"     # Stable active track with assigned track_id
    LOST = "lost"               # Occluded or temporarily missed, coasting via Kalman prediction
    DELETED = "deleted"         # Permanently retired tracklet


class TrackResult(BaseModel):
    """
    Structured result for a tracked face within a video frame.
    
    Example:
    {
      "track_id": 7,
      "bbox": {"x1": 100.0, "y1": 120.0, "x2": 300.0, "y2": 350.0},
      "detection_confidence": 0.96,
      "track_confidence": 0.91,
      "age": 25,
      "hits": 24,
      "missed_frames": 0
    }
    """
    track_id: int = Field(..., description="Temporary sequential track identifier (scoped to video session)")
    bbox: BoundingBox = Field(..., description="Estimated or detected face bounding box coordinates")
    detection_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence of detector in current frame (null if coasted/occluded)"
    )
    track_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Smoothed tracking confidence score"
    )
    age: int = Field(..., ge=1, description="Total frame lifetime of this tracklet")
    hits: int = Field(..., ge=1, description="Number of frames where track matched an active detection")
    missed_frames: int = Field(0, ge=0, description="Consecutive frames without matching detection")
    state: TrackState = Field(TrackState.CONFIRMED, description="Current tracking lifecycle state")
    landmarks: Optional[List[LandmarkPoint]] = Field(default=None, description="Facial landmarks if available")
    velocity: Optional[Tuple[float, float]] = Field(default=None, description="Center velocity vector (vx, vy) in px/frame")

    model_config = ConfigDict(populate_by_name=True)

    @computed_field
    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @computed_field
    @property
    def is_coasted(self) -> bool:
        """True if position was estimated by Kalman motion model rather than direct detector hit."""
        return self.detection_confidence is None or self.missed_frames > 0


class TrackedFrameResult(BaseModel):
    """Container for all confirmed face tracks in a specific video frame."""
    frame_index: int = Field(..., ge=0, description="0-indexed sequence number of the video frame")
    timestamp_ms: Optional[float] = Field(default=None, ge=0.0, description="Frame presentation timestamp in milliseconds")
    tracks: List[TrackResult] = Field(default_factory=list, description="List of confirmed active face tracks")
    detection_time_ms: Optional[float] = Field(default=None, description="Frame face detection latency (ms)")
    tracking_time_ms: Optional[float] = Field(default=None, description="Frame tracking association latency (ms)")

    model_config = ConfigDict(populate_by_name=True)

    @computed_field
    @property
    def track_count(self) -> int:
        return len(self.tracks)


class TrackMetadata(BaseModel):
    """Aggregate lifecycle metrics for an individual persistent tracklet."""
    track_id: int = Field(..., description="Temporary track identifier")
    start_frame: int = Field(..., description="First frame where track appeared")
    end_frame: int = Field(..., description="Last frame where track was active")
    total_frames: int = Field(..., description="Total lifespan frames (end_frame - start_frame + 1)")
    total_hits: int = Field(..., description="Total frames with confirmed detector hits")
    average_confidence: float = Field(..., description="Mean detection confidence over track lifetime")
    duration_seconds: Optional[float] = Field(default=None, description="Observed duration in seconds")

    model_config = ConfigDict(populate_by_name=True)


class TrackingSummary(BaseModel):
    """High-level tracking summary metrics across an entire processed video."""
    total_frames_processed: int = Field(..., ge=0, description="Total video frames processed")
    total_unique_tracks: int = Field(..., ge=0, description="Total count of unique temporary track IDs assigned")
    average_track_duration_frames: float = Field(0.0, description="Mean duration of tracks in frames")
    peak_simultaneous_faces: int = Field(0, ge=0, description="Maximum simultaneous active faces observed in a single frame")
    track_lost_count: int = Field(0, ge=0, description="Total count of track loss / temporary occlusion events")
    tracks: List[TrackMetadata] = Field(default_factory=list, description="Per-track summary lifecycle metadata")

    model_config = ConfigDict(populate_by_name=True)
