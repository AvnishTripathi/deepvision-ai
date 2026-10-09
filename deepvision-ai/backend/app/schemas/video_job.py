"""
DeepVision AI — Video Processing Job Schemas (Step 5.16, 5.17, 5.18)
Defines asynchronous job lifecycle schemas and API response models.
Strict privacy & security rule: Never expose raw server filesystem paths to clients.
"""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class VideoJobStatus(str, Enum):
    """Lifecycle statuses for asynchronous video jobs."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoJobCreateResponse(BaseModel):
    """Initial response returned upon video upload and job enqueueing."""
    job_id: str = Field(..., description="Unique UUID tracking identifier for the asynchronous video job")
    status: VideoJobStatus = Field(VideoJobStatus.QUEUED, description="Initial queued status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "a9c79e60-478e-4a6f-9988-842247ba269a",
                "status": "queued",
            }
        }
    )


class VideoJobStatusResponse(BaseModel):
    """Polled status response detailing progress, metrics, and completion state."""
    job_id: str = Field(..., description="Unique job identifier")
    status: VideoJobStatus = Field(..., description="Current job lifecycle state")
    progress: int = Field(0, ge=0, le=100, description="Processing completion percentage (0-100)")
    processed_frames: int = Field(0, ge=0, description="Frames processed through pipeline so far")
    total_frames: int = Field(0, ge=0, description="Total declared frames in video stream")
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Comprehensive tracking & performance telemetry upon completion"
    )
    error: Optional[str] = Field(default=None, description="Error explanation if job failed")
    output_media: Optional[str] = Field(
        default=None,
        description="Download URI endpoint for annotated video artifact (never raw server filesystem path)"
    )
    created_at: Optional[datetime] = Field(default=None, description="Job creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Job execution start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "a9c79e60-478e-4a6f-9988-842247ba269a",
                "status": "completed",
                "progress": 100,
                "processed_frames": 1000,
                "total_frames": 1000,
                "metrics": {
                    "unique_track_count": 4,
                    "peak_simultaneous_faces": 3,
                    "average_fps": 18.4,
                    "total_processing_time": 5434.8,
                },
                "output_media": "/api/v1/video/jobs/a9c79e60-478e-4a6f-9988-842247ba269a/download",
            }
        }
    )
