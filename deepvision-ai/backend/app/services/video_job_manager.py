"""
DeepVision AI — Asynchronous Video Job Manager & Queue Architecture (Step 5.16, 7.7, 7.8)
Provides a decoupled job execution manager supporting:
1. RedisQueueVideoJobManager (Production Redis queue & background worker architecture)
2. InMemoryVideoJobManager (Thread-safe development/test execution fallback)

Preferred Architecture:
  Frontend -> FastAPI -> Create Job -> Redis Queue -> Worker -> AI Processing -> Storage -> Job Status
"""

import os
import json
import uuid
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor

from app.core.storage import get_storage_backend, BaseStorageBackend
from app.core.redis import get_redis_client
from app.schemas.video import VideoProcessingOptions, VideoProcessingResult
from app.schemas.video_job import (
    VideoJobStatus,
    VideoJobCreateResponse,
    VideoJobStatusResponse,
)
from app.core.config import settings
from app.core.exceptions import (
    VideoProcessingCancelledException,
    VideoJobCapacityExceededException,
)
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
        self.current_faces: Optional[int] = None
        self.active_tracks: Optional[int] = None
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
            current_faces=self.current_faces,
            active_tracks=self.active_tracks,
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
        max_concurrent = getattr(settings, "VIDEO_MAX_CONCURRENT_JOBS", 4)

        with self._lock:
            active_count = sum(
                1 for j in self._jobs.values()
                if j.status in (VideoJobStatus.QUEUED, VideoJobStatus.PROCESSING)
            )
            if active_count >= max_concurrent:
                logger.warning(
                    "Video job submission rejected: active jobs count (%d) reached maximum limit (%d)",
                    active_count,
                    max_concurrent,
                )
                raise VideoJobCapacityExceededException(
                    f"Server busy: Maximum concurrent video processing jobs ({max_concurrent}) exceeded. Please retry once active jobs complete."
                )

        job_id = str(uuid.uuid4())
        opts = options or VideoProcessingOptions()

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

        last_reported_pct = -1

        def progress_callback(cur_frame: int, total_f: int, pct: float):
            nonlocal last_reported_pct
            int_pct = int(round(pct))
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

            output_key = None
            if result.annotated_video_path and os.path.exists(result.annotated_video_path):
                output_key = f"outputs/{job_id}_annotated.mp4"
                self._storage.store_file(result.annotated_video_path, output_key)
                try:
                    os.remove(result.annotated_video_path)
                except OSError:
                    pass

            with self._lock:
                if record.status != VideoJobStatus.CANCELLED:
                    record.status = VideoJobStatus.COMPLETED
                    record.completed_at = datetime.now(timezone.utc)
                    record.progress = 100
                    record.total_frames = result.metadata.frame_count
                    record.processed_frames = result.metrics.processed_frames
                    record.output_storage_key = output_key
                    tracks_summary = [
                        {
                            "track_id": t.track_id,
                            "frames_active": t.total_frames,
                            "start_frame": t.start_frame,
                            "end_frame": t.end_frame,
                            "avg_confidence": round(t.average_confidence, 2),
                        }
                        for t in (result.tracking_summary.tracks if result.tracking_summary else [])
                    ]
                    record.metrics = {
                        "duration_seconds": result.metadata.duration_seconds or (result.metadata.frame_count / max(1.0, result.metadata.fps)),
                        "fps": result.metadata.fps,
                        "unique_track_count": result.metrics.unique_track_count,
                        "unique_tracks_count": result.metrics.unique_track_count,
                        "peak_simultaneous_faces": result.metrics.peak_simultaneous_faces,
                        "average_fps": result.metrics.average_fps,
                        "average_processing_fps": result.metrics.average_fps,
                        "total_frames": result.metrics.total_frames,
                        "processed_frames": result.metrics.processed_frames,
                        "total_processing_time": result.metrics.total_processing_time,
                        "processing_time_seconds": round(result.metrics.total_processing_time / 1000.0, 2),
                        "average_frame_processing_time": result.metrics.average_frame_processing_time,
                        "track_lost_count": result.metrics.track_lost_count,
                        "total_faces_detected": result.metrics.total_faces_detected,
                        "tracks_summary": tracks_summary,
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


class RedisVideoJobManager(BaseVideoJobManager):
    """
    Production Redis Queue & Worker Video Job Manager.
    Architecture:
      FastAPI -> Create Job -> Redis Queue -> Worker Thread/Process -> AI Processing -> Storage -> Job Status
    """

    QUEUE_KEY = "deepvision:queue:video_jobs"
    ACTIVE_SET_KEY = "deepvision:jobs:active"
    JOB_PREFIX = "deepvision:job:"

    def __init__(
        self,
        storage: Optional[BaseStorageBackend] = None,
        video_service: Optional[VideoProcessingService] = None,
        max_workers: int = 2,
    ):
        self._storage = storage or get_storage_backend()
        self._video_service = video_service or get_video_processing_service()
        self._redis = get_redis_client()
        self._worker_running = True
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="RedisVideoWorker")

        # Start daemon background queue consumer if Redis is available
        if self._redis:
            for _ in range(max_workers):
                self._executor.submit(self._worker_loop)

    def _job_key(self, job_id: str) -> str:
        return f"{self.JOB_PREFIX}{job_id}"

    def submit_job(
        self,
        input_filepath: str,
        filename: str,
        options: Optional[VideoProcessingOptions] = None,
    ) -> VideoJobCreateResponse:
        if not self._redis:
            raise RuntimeError("Redis client is not available.")

        max_concurrent = getattr(settings, "VIDEO_MAX_CONCURRENT_JOBS", 4)
        active_count = self._redis.scard(self.ACTIVE_SET_KEY) or 0

        if active_count >= max_concurrent:
            logger.warning(
                "Redis video job submission rejected: active jobs count (%d) reached limit (%d)",
                active_count,
                max_concurrent,
            )
            raise VideoJobCapacityExceededException(
                f"Server busy: Maximum concurrent video processing jobs ({max_concurrent}) exceeded. Please retry once active jobs complete."
            )

        job_id = str(uuid.uuid4())
        opts = options or VideoProcessingOptions()

        input_key = f"videos/{job_id}_{filename}"
        self._storage.store_file(input_filepath, input_key)

        now_iso = datetime.now(timezone.utc).isoformat()
        job_data = {
            "job_id": job_id,
            "status": VideoJobStatus.QUEUED.value,
            "input_filename": filename,
            "input_storage_key": input_key,
            "output_storage_key": "",
            "options": opts.model_dump_json(),
            "progress": 0,
            "processed_frames": 0,
            "total_frames": 0,
            "metrics": "",
            "error": "",
            "is_cancelled": 0,
            "created_at": now_iso,
            "started_at": "",
            "completed_at": "",
        }

        # Store in Redis Hash and Push onto Queue
        pipeline = self._redis.pipeline()
        pipeline.hset(self._job_key(job_id), mapping=job_data)
        pipeline.sadd(self.ACTIVE_SET_KEY, job_id)
        pipeline.rpush(self.QUEUE_KEY, job_id)
        pipeline.execute()

        logger.info("Enqueued video processing job to Redis: %s for %s", job_id, filename)
        return VideoJobCreateResponse(job_id=job_id, status=VideoJobStatus.QUEUED)

    def get_job_status(self, job_id: str) -> Optional[VideoJobStatusResponse]:
        if not self._redis:
            return None

        raw = self._redis.hgetall(self._job_key(job_id))
        if not raw:
            return None

        status_enum = VideoJobStatus(raw.get("status", VideoJobStatus.QUEUED.value))
        output_storage_key = raw.get("output_storage_key") or None
        output_media = (
            f"/api/v1/video/jobs/{job_id}/download"
            if status_enum == VideoJobStatus.COMPLETED and output_storage_key
            else None
        )

        metrics = None
        if raw.get("metrics"):
            try:
                metrics = json.loads(raw["metrics"])
            except json.JSONDecodeError:
                pass

        created_at = datetime.fromisoformat(raw["created_at"]) if raw.get("created_at") else datetime.now(timezone.utc)
        started_at = datetime.fromisoformat(raw["started_at"]) if raw.get("started_at") else None
        completed_at = datetime.fromisoformat(raw["completed_at"]) if raw.get("completed_at") else None

        current_faces = int(raw["current_faces"]) if raw.get("current_faces") else None
        active_tracks = int(raw["active_tracks"]) if raw.get("active_tracks") else None

        return VideoJobStatusResponse(
            job_id=job_id,
            status=status_enum,
            progress=int(raw.get("progress", 0)),
            processed_frames=int(raw.get("processed_frames", 0)),
            total_frames=int(raw.get("total_frames", 0)),
            current_faces=current_faces,
            active_tracks=active_tracks,
            metrics=metrics,
            error=raw.get("error") or None,
            output_media=output_media,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
        )

    def get_output_artifact_path(self, job_id: str) -> Optional[str]:
        if not self._redis:
            return None

        raw = self._redis.hgetall(self._job_key(job_id))
        if not raw or raw.get("status") != VideoJobStatus.COMPLETED.value:
            return None
        output_key = raw.get("output_storage_key")
        if not output_key:
            return None
        return self._storage.get_local_path(output_key)

    def cancel_job(self, job_id: str) -> tuple[bool, str]:
        if not self._redis:
            return False, "Redis client unavailable."

        raw = self._redis.hgetall(self._job_key(job_id))
        if not raw:
            return False, f"Video job '{job_id}' not found."

        current_status = raw.get("status")
        if current_status == VideoJobStatus.COMPLETED.value:
            return False, "Cannot cancel a completed job."
        if current_status in (VideoJobStatus.CANCELLED.value, VideoJobStatus.FAILED.value):
            return False, f"Job is already in '{current_status}' state."

        now_iso = datetime.now(timezone.utc).isoformat()
        pipeline = self._redis.pipeline()
        pipeline.hset(
            self._job_key(job_id),
            mapping={
                "is_cancelled": 1,
                "status": VideoJobStatus.CANCELLED.value,
                "completed_at": now_iso,
                "error": "Job cancelled by client request.",
            },
        )
        pipeline.srem(self.ACTIVE_SET_KEY, job_id)
        pipeline.execute()
        return True, "Job cancellation initiated successfully."

    def _worker_loop(self) -> None:
        """Daemon worker loop popping and executing jobs from the Redis queue."""
        import time
        while self._worker_running:
            try:
                # BLPOP with 2-second timeout
                item = self._redis.blpop(self.QUEUE_KEY, timeout=2)
                if not item:
                    continue
                _, job_id = item
                self._process_single_job(job_id)
            except Exception as exc:
                if self._worker_running:
                    logger.warning("Redis worker loop error: %s", exc)
                    time.sleep(1)

    def _process_single_job(self, job_id: str) -> None:
        """Processes a single dequeued job."""
        raw = self._redis.hgetall(self._job_key(job_id))
        if not raw or int(raw.get("is_cancelled", 0)) == 1:
            self._redis.srem(self.ACTIVE_SET_KEY, job_id)
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        self._redis.hset(
            self._job_key(job_id),
            mapping={"status": VideoJobStatus.PROCESSING.value, "started_at": now_iso},
        )

        input_key = raw.get("input_storage_key")
        input_path = self._storage.get_local_path(input_key)
        if not input_path or not os.path.exists(input_path):
            self._redis.hset(
                self._job_key(job_id),
                mapping={
                    "status": VideoJobStatus.FAILED.value,
                    "error": "Input video file could not be found in storage.",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self._redis.srem(self.ACTIVE_SET_KEY, job_id)
            return

        opts = VideoProcessingOptions()
        if raw.get("options"):
            try:
                opts = VideoProcessingOptions.model_validate_json(raw["options"])
            except Exception:
                pass

        last_reported_pct = -1

        def progress_callback(cur_frame: int, total_f: int, pct: float):
            nonlocal last_reported_pct
            int_pct = int(round(pct))
            if int_pct >= last_reported_pct + 5 or cur_frame >= total_f:
                last_reported_pct = int_pct
                self._redis.hset(
                    self._job_key(job_id),
                    mapping={
                        "processed_frames": cur_frame,
                        "total_frames": total_f,
                        "progress": min(100, int_pct),
                    },
                )

        def is_cancelled() -> bool:
            c = self._redis.hget(self._job_key(job_id), "is_cancelled")
            return int(c or 0) == 1

        try:
            result = self._video_service.process_video_file(
                video_input_path=input_path,
                options=opts,
                progress_callback=progress_callback,
                is_cancelled=is_cancelled,
            )

            output_key = None
            if result.annotated_video_path and os.path.exists(result.annotated_video_path):
                output_key = f"outputs/{job_id}_annotated.mp4"
                self._storage.store_file(result.annotated_video_path, output_key)
                try:
                    os.remove(result.annotated_video_path)
                except OSError:
                    pass

            # Only mark completed if not cancelled during execution
            if not is_cancelled():
                tracks_summary = [
                    {
                        "track_id": t.track_id,
                        "frames_active": t.total_frames,
                        "start_frame": t.start_frame,
                        "end_frame": t.end_frame,
                        "avg_confidence": round(t.average_confidence, 2),
                    }
                    for t in (result.tracking_summary.tracks if result.tracking_summary else [])
                ]
                metrics_dict = {
                    "duration_seconds": result.metadata.duration_seconds or (result.metadata.frame_count / max(1.0, result.metadata.fps)),
                    "fps": result.metadata.fps,
                    "unique_track_count": result.metrics.unique_track_count,
                    "unique_tracks_count": result.metrics.unique_track_count,
                    "peak_simultaneous_faces": result.metrics.peak_simultaneous_faces,
                    "average_fps": result.metrics.average_fps,
                    "average_processing_fps": result.metrics.average_fps,
                    "total_frames": result.metrics.total_frames,
                    "processed_frames": result.metrics.processed_frames,
                    "total_processing_time": result.metrics.total_processing_time,
                    "processing_time_seconds": round(result.metrics.total_processing_time / 1000.0, 2),
                    "average_frame_processing_time": result.metrics.average_frame_processing_time,
                    "track_lost_count": result.metrics.track_lost_count,
                    "total_faces_detected": result.metrics.total_faces_detected,
                    "tracks_summary": tracks_summary,
                }
                self._redis.hset(
                    self._job_key(job_id),
                    mapping={
                        "status": VideoJobStatus.COMPLETED.value,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "progress": 100,
                        "total_frames": result.metadata.frame_count,
                        "processed_frames": result.metrics.processed_frames,
                        "output_storage_key": output_key or "",
                        "metrics": json.dumps(metrics_dict),
                    },
                )
            logger.info("Redis video job %s completed in %.2fms", job_id, result.metrics.total_processing_time)

        except VideoProcessingCancelledException as exc:
            logger.info("Redis video job %s stopped on cancellation: %s", job_id, exc)
            self._redis.hset(
                self._job_key(job_id),
                mapping={
                    "status": VideoJobStatus.CANCELLED.value,
                    "error": "Job cancelled by client request.",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as exc:
            logger.exception("Redis video job %s failed: %s", job_id, exc)
            self._redis.hset(
                self._job_key(job_id),
                mapping={
                    "status": VideoJobStatus.FAILED.value,
                    "error": "An unexpected error occurred during video processing.",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        finally:
            self._redis.srem(self.ACTIVE_SET_KEY, job_id)


# Singleton job manager
_video_job_manager: Optional[BaseVideoJobManager] = None


def get_video_job_manager() -> BaseVideoJobManager:
    """
    Returns application-level VideoJobManager singleton.
    Uses RedisVideoJobManager if Redis is configured and accessible,
    falling back seamlessly to InMemoryVideoJobManager otherwise.
    """
    global _video_job_manager
    if _video_job_manager is None:
        redis_client = get_redis_client()
        if redis_client:
            try:
                redis_client.ping()
                logger.info("Initializing RedisVideoJobManager with connected Redis instance.")
                _video_job_manager = RedisVideoJobManager()
            except Exception as exc:
                logger.warning("Redis ping failed (%s), falling back to InMemoryVideoJobManager.", exc)
                _video_job_manager = InMemoryVideoJobManager()
        else:
            _video_job_manager = InMemoryVideoJobManager()

    return _video_job_manager

