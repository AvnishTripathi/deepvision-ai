"""
DeepVision AI — Video Face Detection & Tracking API Endpoints (Step 5.17, 5.18, 5.19)
Provides asynchronous video processing endpoints:
- POST /api/v1/video/analyze: Upload video and enqueue processing job
- GET  /api/v1/video/jobs/{job_id}: Poll job status, progress percentage, and tracking metrics
- GET  /api/v1/video/jobs/{job_id}/download: Securely download annotated video artifact
"""

import os
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.core.exceptions import EntityNotFoundException, CorruptedVideoException
from app.core.video_security import save_upload_video_safely, probe_and_validate_video
from app.schemas.video import VideoProcessingOptions
from app.schemas.video_job import (
    VideoJobCreateResponse,
    VideoJobStatusResponse,
)
from app.services.video_job_manager import (
    BaseVideoJobManager,
    get_video_job_manager,
)
from app.core.rate_limit import rate_limit_video_analysis

logger = logging.getLogger("deepvision.api.video")

router = APIRouter(prefix="/video", tags=["Video Tracking"])


@router.post(
    "/analyze",
    response_model=VideoJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Video for Face Detection & Tracking",
    dependencies=[Depends(rate_limit_video_analysis)],
    description="""
    Uploads a video file, validates container readability, and enqueues an asynchronous
    face detection and multi-face tracking processing job.
    
    Returns a unique `job_id` and initial status `queued`.
    Never exposes internal server filesystem paths to the client.
    """,
)
async def analyze_video(
    file: UploadFile = File(..., description="Video container file (MP4, AVI, MOV, MKV, WebM)"),
    confidence_threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Detection threshold override"),
    detection_interval: int = Query(1, ge=1, le=30, description="Detection frame interval (1 = every frame)"),
    analysis_interval: int = Query(5, ge=1, le=60, description="Face quality analysis frame interval"),
    annotate_video: bool = Query(True, description="Whether to render annotated video artifact"),
    enable_analysis: bool = Query(True, description="Whether to compute quality and optics diagnostics"),
    job_manager: BaseVideoJobManager = Depends(get_video_job_manager),
) -> VideoJobCreateResponse:
    """Uploads and initiates asynchronous video tracking job."""
    # 1. Safely stream upload bytes to temporary file (enforcing payload limits)
    temp_path, bytes_read = await save_upload_video_safely(file)

    try:
        # 2. Validate video container structure and stream readability
        metadata = probe_and_validate_video(temp_path)
        logger.info(
            "Received valid video upload: %s (%dx%d, %d frames)",
            file.filename,
            metadata.width,
            metadata.height,
            metadata.frame_count,
        )

        # 3. Build options
        opts = VideoProcessingOptions(
            confidence_threshold=confidence_threshold,
            detection_interval=detection_interval,
            analysis_interval=analysis_interval,
            annotate_video=annotate_video,
            enable_analysis=enable_analysis,
        )

        # 4. Enqueue job via decoupled job manager
        response = job_manager.submit_job(
            input_filepath=temp_path,
            filename=file.filename or "video.mp4",
            options=opts,
        )

        return response

    except Exception:
        # Clean up temporary upload file if validation or enqueueing fails
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


@router.get(
    "/jobs/{job_id}",
    response_model=VideoJobStatusResponse,
    summary="Get Video Processing Job Status",
    description="""
    Polls the processing status, progress percentage (0-100), processed frame counts,
    and aggregate tracking metrics for an asynchronous video processing job.
    """,
)
def get_job_status(
    job_id: str,
    job_manager: BaseVideoJobManager = Depends(get_video_job_manager),
) -> VideoJobStatusResponse:
    """Returns current status and metrics for the specified job."""
    status_response = job_manager.get_job_status(job_id)
    if not status_response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video job '{job_id}' not found.",
        )
    return status_response


@router.post(
    "/jobs/{job_id}/cancel",
    summary="Cancel Video Processing Job",
    description="""
    Requests cancellation of an ongoing or queued video processing job.
    Cleanly releases all system resources, halts processing, and marks job cancelled.
    """,
)
def cancel_job(
    job_id: str,
    job_manager: BaseVideoJobManager = Depends(get_video_job_manager),
) -> dict:
    """Cancels an active or queued video processing job."""
    success, message = job_manager.cancel_job(job_id)
    if not success:
        if "not found" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=message,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": message,
    }


@router.get(
    "/jobs/{job_id}/download",
    summary="Download Annotated Video Artifact",
    description="""
    Securely streams the generated annotated video file for a completed job.
    Never exposes internal filesystem paths.
    """,
)
def download_annotated_video(
    job_id: str,
    job_manager: BaseVideoJobManager = Depends(get_video_job_manager),
) -> FileResponse:
    """Streams the annotated video artifact if the job is completed."""
    status_response = job_manager.get_job_status(job_id)
    if not status_response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video job '{job_id}' not found.",
        )

    if status_response.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video job '{job_id}' is not yet completed (current status: '{status_response.status}').",
        )

    artifact_path = job_manager.get_output_artifact_path(job_id)
    if not artifact_path or not os.path.exists(artifact_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated video artifact not found or was not requested during job submission.",
        )

    return FileResponse(
        path=artifact_path,
        media_type="video/mp4",
        filename=f"deepvision_annotated_{job_id[:8]}.mp4",
    )

