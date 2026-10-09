"""
DeepVision AI — Asynchronous Video Job Manager & Queue Foundation (Step 5.16, 5.20)
Provides a decoupled job execution manager.
Supports in-memory background worker execution with thread-safe state management,
ready for migration to Redis / Celery / RQ / Arq without changing API controller contracts.
"""

import os
import uuid
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor

from app.core.storage import get_storage_backend, BaseStorageBackend
from app.schemas.video import VideoProcessingOptions, VideoProcessingResult
from app.schemas.video_job import (
    VideoJobStatus,
    VideoJobCreateResponse,
    VideoJobStatusResponse,
)
from app.core.exceptions import VideoProcessingCancelledException
from app.services.video_processing_service import (
    VideoProcessingService,
    get_video_processing_service,
)

logger = logging.getLogger("deepvision.service.video_jobs")


class VideoJobRecord:
    """Internal representation of a video processing job."""

    def __init__(
        self,
        job_id: str,
        input_filename: str,
        input_storage_key: str,
        options: VideoProcessingOptions,
    ):
        self.job_id = job_id
        self.status = VideoJobStatus.QUEUED
        self.input_filename = input_filename
        self.input_storage_key = input_storage_key
        self.output_storage_key: Optional[str] = None
        self.options = options

        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        self.progress = 0
        self.processed_frames = 0
        self.total_frames = 0
        self.metrics: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self._cancellation_event = threading.Event()

    def request_cancellation(self) -> None:
        """Signals background worker to terminate processing loop."""
        self._cancellation_event.set()

    def is_cancelled(self) -> bool:
        """Returns True if cancellation was requested."""
        return self._cancellation_event.is_set()

    def to_status_response(self) -> VideoJobStatusResponse:
        """Converts internal record to client-safe response (no server paths)."""
        output_media = (
            f"/api/v1/video/jobs/{self.job_id}/download"
            if self.status == VideoJobStatus.COMPLETED and self.output_storage_key
            else None
        )
        return VideoJobStatusResponse(
            job_id=self.job_id,
            status=self.status,
            progress=self.progress,
            processed_frames=self.processed_frames,
            total_frames=self.total_frames,
            metrics=self.metrics,
            error=self.error,
            output_media=output_media,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )


class BaseVideoJobManager(ABC):
    """Abstract interface for video job queue and lifecycle managers."""

    @abstractmethod
    def submit_job(
        self,
        input_filepath: str,
        filename: str,
        options: Optional[VideoProcessingOptions] = None,
    ) -> VideoJobCreateResponse:
        """Enqueues a new video processing job."""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[VideoJobStatusResponse]:
        """Returns client-safe status response for a given job."""
        pass

    @abstractmethod
    def get_output_artifact_path(self, job_id: str) -> Optional[str]:
        """Returns physical path to annotated video artifact for downloading."""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> tuple[bool, str]:
        """Attempts to cancel job. Returns (success, message)."""
        pass


class InMemoryVideoJobManager(BaseVideoJobManager):
    """
    Thread-safe, asynchronous in-memory video job manager.
    Executes jobs via a background thread pool executor.
    """

    def __init__(
        self,
        storage: Optional[BaseStorageBackend] = None,
        video_service: Optional[VideoProcessingService] = None,
        max_workers: int = 2,
    ):
        self._storage = storage or get_storage_backend()
        self._video_service = video_service or get_video_processing_service()
        self._jobs: Dict[str, VideoJobRecord] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="VideoJobWorker")

    def submit_job(
        self,
        input_filepath: str,
        filename: str,
        options: Optional[VideoProcessingOptions] = None,
    ) -> VideoJobCreateResponse:
        job_id = str(uuid.uuid4())
        opts = options or VideoProcessingOptions()

        # Store input video into media storage
        input_key = f"videos/{job_id}_{filename}"
        self._storage.store_file(input_filepath, input_key)

        record = VideoJobRecord(
            job_id=job_id,
            input_filename=filename,
            input_storage_key=input_key,
            options=opts,
        )

        with self._lock:
            self._jobs[job_id] = record

        # Dispatch asynchronous task to thread pool
        self._executor.submit(self._execute_job, job_id)
        logger.info("Enqueued video processing job: %s for %s", job_id, filename)

        return VideoJobCreateResponse(job_id=job_id, status=VideoJobStatus.QUEUED)

    def get_job_status(self, job_id: str) -> Optional[VideoJobStatusResponse]:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None
            return record.to_status_response()

    def get_output_artifact_path(self, job_id: str) -> Optional[str]:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record or record.status != VideoJobStatus.COMPLETED or not record.output_storage_key:
                return None
            return self._storage.get_local_path(record.output_storage_key)

    def cancel_job(self, job_id: str) -> tuple[bool, str]:
        """Attempts to cancel an active or queued job."""
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return False, f"Video job '{job_id}' not found."

            if record.status == VideoJobStatus.COMPLETED:
                return False, "Cannot cancel a completed job."

            if record.status in (VideoJobStatus.CANCELLED, VideoJobStatus.FAILED):
                return False, f"Job is already in '{record.status.value}' state."

            if record.status == VideoJobStatus.QUEUED:
                record.status = VideoJobStatus.CANCELLED
                record.completed_at = datetime.now(timezone.utc)
                record.error = "Job cancelled before execution began."
                return True, "Queued job was cancelled successfully."

            # If currently PROCESSING:
            record.request_cancellation()
            record.status = VideoJobStatus.CANCELLED
            record.completed_at = datetime.now(timezone.utc)
            record.error = "Job cancellation requested by client."
            return True, "Job cancellation initiated successfully."

    def _execute_job(self, job_id: str) -> None:
        """Background execution worker method."""
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            if record.status == VideoJobStatus.CANCELLED:
                return
            record.status = VideoJobStatus.PROCESSING
            record.started_at = datetime.now(timezone.utc)

        input_path = self._storage.get_local_path(record.input_storage_key)
        if not input_path or not os.path.exists(input_path):
            with self._lock:
                record.status = VideoJobStatus.FAILED
                record.error = "Input video file could not be found in storage."
                record.completed_at = datetime.now(timezone.utc)
            return

        # Progress tracking callback (Step 5.20)
        last_reported_pct = -1

        def progress_callback(cur_frame: int, total_f: int, pct: float):
            nonlocal last_reported_pct
            int_pct = int(round(pct))
            # Throttle updates: update every 5% progress or at completion to avoid lock contention
            if int_pct >= last_reported_pct + 5 or cur_frame >= total_f:
                last_reported_pct = int_pct
                with self._lock:
                    record.processed_frames = cur_frame
                    record.total_frames = total_f
                    record.progress = min(100, int_pct)

        try:
            result: VideoProcessingResult = self._video_service.process_video_file(
                video_input_path=input_path,
                options=record.options,
                progress_callback=progress_callback,
                is_cancelled=record.is_cancelled,
            )

            # Store generated output video
            output_key = None
            if result.annotated_video_path and os.path.exists(result.annotated_video_path):
                output_key = f"outputs/{job_id}_annotated.mp4"
                self._storage.store_file(result.annotated_video_path, output_key)
                try:
                    os.remove(result.annotated_video_path)
                except OSError:
                    pass

            with self._lock:
                # Do not overwrite if cancellation intervened
                if record.status != VideoJobStatus.CANCELLED:
                    record.status = VideoJobStatus.COMPLETED
                    record.completed_at = datetime.now(timezone.utc)
                    record.progress = 100
                    record.total_frames = result.metadata.frame_count
                    record.processed_frames = result.metrics.processed_frames
                    record.output_storage_key = output_key
                    record.metrics = {
                        "unique_track_count": result.metrics.unique_track_count,
                        "peak_simultaneous_faces": result.metrics.peak_simultaneous_faces,
                        "average_fps": result.metrics.average_fps,
                        "total_frames": result.metrics.total_frames,
                        "processed_frames": result.metrics.processed_frames,
                        "total_processing_time": result.metrics.total_processing_time,
                        "average_frame_processing_time": result.metrics.average_frame_processing_time,
                        "track_lost_count": result.metrics.track_lost_count,
                        "total_faces_detected": result.metrics.total_faces_detected,
                    }

            logger.info("Video job %s completed in %.2fms", job_id, result.metrics.total_processing_time)

        except VideoProcessingCancelledException as exc:
            logger.info("Video job %s successfully stopped on cancellation: %s", job_id, exc)
            with self._lock:
                record.status = VideoJobStatus.CANCELLED
                record.error = "Job cancelled by client request."
                record.completed_at = datetime.now(timezone.utc)

        except Exception as exc:
            logger.exception("Video processing job %s failed: %s", job_id, exc)
            with self._lock:
                record.status = VideoJobStatus.FAILED
                record.error = "An unexpected error occurred during video processing."
                record.completed_at = datetime.now(timezone.utc)


# Singleton job manager
_video_job_manager: Optional[BaseVideoJobManager] = None


def get_video_job_manager() -> BaseVideoJobManager:
    """Returns application-level VideoJobManager singleton."""
    global _video_job_manager
    if _video_job_manager is None:
        _video_job_manager = InMemoryVideoJobManager()
    return _video_job_manager

