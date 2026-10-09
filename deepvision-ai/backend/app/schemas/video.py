"""
DeepVision AI — Video Processing & Tracking Schemas (Step 5.8)
Provides strongly typed Pydantic v2 schemas for video metadata, processing options,
frame tracking results, and aggregate video processing metrics.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict, model_validator

from app.schemas.ai import BoundingBox, LandmarkPoint
from app.ai.tracking.schemas import TrackResult, TrackingSummary, TrackMetadata


class VideoMetadata(BaseModel):
    """
    Reusable video stream and container metadata structure.
    Strict rule: If metadata is unavailable or unreliable, handle safely; do not invent metadata.
    """
    width: int = Field(..., gt=0, description="Video frame pixel width")
    height: int = Field(..., gt=0, description="Video frame pixel height")
    fps: float = Field(..., gt=0.0, description="Video frame rate in frames per second")
    frame_count: int = Field(..., ge=0, description="Total declared frame count in video stream")
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Calculated or reported video duration in seconds (frame_count / fps)"
    )
    codec: Optional[str] = Field(
        default=None,
        description="FourCC video container compression codec (e.g., 'avc1', 'mp4v', 'XVID')"
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def calculate_duration_if_reliable(cls, data: Any) -> Any:
        if isinstance(data, dict):
            fps = float(data.get("fps", 0.0))
            frame_count = int(data.get("frame_count", 0))
            duration = data.get("duration_seconds")

            # Calculate duration safely if fps > 0 and frame_count > 0, and not already provided
            if duration is None and fps > 0.1 and frame_count > 0:
                calc_dur = round(frame_count / fps, 3)
                data["duration_seconds"] = calc_dur
        return data


class VideoProcessingOptions(BaseModel):
    """Configuration options for a video processing session (Step 5.11)."""
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold override (default: from settings)"
    )
    process_every_n_frames: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Legacy alias for detection_interval"
    )
    detection_interval: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Run face detection every N frames (default: 1; on intermediate frames tracks are coasted via Kalman prediction)"
    )
    analysis_interval: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Run expensive face quality/pose analysis every N frames (default: 5)"
    )
    max_fps: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=120.0,
        description="Optional maximum processing FPS target for frame sampling"
    )
    enable_analysis: bool = Field(
        default=True,
        description="Whether to execute face optics/quality/pose analysis on tracked faces"
    )
    annotate_video: bool = Field(
        default=True,
        description="Whether to generate an annotated video file with tracks and HUD"
    )
    max_frames: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional limit on maximum frames to process (useful for previewing or testing)"
    )

    model_config = ConfigDict(populate_by_name=True)


class VideoFrameTrackResult(BaseModel):
    """Tracking and detection result for an individual video frame."""
    frame_index: int = Field(..., ge=0, description="Frame sequence index (0-based)")
    timestamp_ms: Optional[float] = Field(default=None, description="Frame presentation timestamp (ms)")
    tracks: List[TrackResult] = Field(default_factory=list, description="Active confirmed tracks in this frame")
    detection_count: int = Field(0, description="Raw face detections in this frame")
    processing_time_ms: Optional[float] = Field(default=None, description="Total frame processing latency (ms)")

    model_config = ConfigDict(populate_by_name=True)


class VideoProcessingMetrics(BaseModel):
    """Telemetry and timing metrics for a video processing job (Step 5.13)."""
    total_frames: int = Field(..., description="Total frames declared/extracted in video")
    processed_frames: int = Field(..., description="Frames processed through pipeline")
    total_processing_time: float = Field(..., description="Total processing wall-clock time (ms)")
    average_frame_processing_time: float = Field(..., description="Average processing time per frame (ms)")
    average_fps: float = Field(..., description="Average throughput processing speed (FPS)")
    peak_simultaneous_faces: int = Field(0, description="Maximum number of simultaneous active faces in any single frame")
    unique_track_count: int = Field(..., description="Total count of unique temporary track IDs assigned")
    track_lost_count: int = Field(0, description="Total count of times tracks entered lost/occluded state or terminated")
    total_faces_detected: int = Field(0, description="Cumulative detections across all processed frames")
    detection_time_ms: float = Field(0.0, description="Cumulative detection inference latency (ms)")
    tracking_time_ms: float = Field(0.0, description="Cumulative tracking association latency (ms)")
    analysis_time_ms: float = Field(0.0, description="Cumulative face optics/quality analysis latency (ms)")
    annotation_time_ms: float = Field(0.0, description="Cumulative video annotation & encoding latency (ms)")

    # Legacy alias properties for backwards compatibility
    @property
    def total_frames_extracted(self) -> int:
        return self.total_frames

    @property
    def total_frames_processed(self) -> int:
        return self.processed_frames

    @property
    def elapsed_time_ms(self) -> float:
        return self.total_processing_time

    @property
    def fps_throughput(self) -> float:
        return self.average_fps

    @property
    def total_unique_tracks(self) -> int:
        return self.unique_track_count

    model_config = ConfigDict(populate_by_name=True)


class VideoProcessingResult(BaseModel):
    """Top-level structured result for a completed video processing request."""
    success: bool = Field(True, description="Pipeline execution success status")
    metadata: VideoMetadata = Field(..., description="Video stream properties and dimensions")
    tracking_summary: TrackingSummary = Field(..., description="Track lifecycle summary across the video")
    metrics: VideoProcessingMetrics = Field(..., description="Granular performance telemetry and throughput")
    annotated_video_path: Optional[str] = Field(
        default=None,
        description="Path to output annotated video file (if annotation was requested)"
    )
    frame_results: Optional[List[VideoFrameTrackResult]] = Field(
        default=None,
        description="Optional frame-by-frame track details (omitted by default for large videos)"
    )

    model_config = ConfigDict(populate_by_name=True)
