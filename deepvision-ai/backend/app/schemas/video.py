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
    """Configuration options for a video processing session."""
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
        description="Process every Nth frame (1 = every frame, 2 = half-rate, etc.)"
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
    """Telemetry and timing metrics for a video processing job."""
    total_frames_extracted: int = Field(..., description="Total frames read from video stream")
    total_frames_processed: int = Field(..., description="Frames passed through face detection & tracking")
    total_faces_detected: int = Field(..., description="Cumulative detections across all processed frames")
    total_unique_tracks: int = Field(..., description="Total unique temporary track IDs assigned")
    elapsed_time_ms: float = Field(..., description="Total wall-clock execution time (ms)")
    fps_throughput: float = Field(..., description="Processing speed in frames per second")
    detection_time_ms: float = Field(0.0, description="Cumulative detection inference latency (ms)")
    tracking_time_ms: float = Field(0.0, description="Cumulative tracking association latency (ms)")
    analysis_time_ms: float = Field(0.0, description="Cumulative face optics/quality analysis latency (ms)")
    annotation_time_ms: float = Field(0.0, description="Cumulative video annotation & encoding latency (ms)")

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
