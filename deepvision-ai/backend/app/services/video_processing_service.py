"""
DeepVision AI — Production Video Face Detection & Multi-Face Tracking Service (Step 5.9, 5.10)
Provides end-to-end streaming video processing:
- Video Ingestion & Security Validation
- Video Stream Metadata Extraction
- Streaming Frame Extraction (zero full-video memory buffering)
- Multi-Face Detection via existing YuNet FaceDetector
- Multi-Face Tracking via decoupled ByteTrackFaceTracker
- Face Quality & Optics Diagnostics via existing FaceAnalysisService
- Frame-by-Frame Video Annotation & VideoWriter Encoding
- Comprehensive Tracking & Performance Telemetry
"""

import os
import time
import uuid
import tempfile
import logging
from typing import Optional, List, Dict, Any, Generator, Tuple
import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import (
    CorruptedVideoException,
    EmptyVideoException,
    VideoTooLargeException,
    UnsupportedVideoError,
    VideoProcessingCancelledException,
)
from app.core.video_security import (
    probe_and_validate_video,
    sanitize_video_filename,
)
from app.schemas.ai import FaceDetection, BoundingBox
from app.schemas.video import (
    VideoMetadata,
    VideoProcessingOptions,
    VideoFrameTrackResult,
    VideoProcessingMetrics,
    VideoProcessingResult,
)
from app.ai.detection.detector import FaceDetector
from app.ai.detection.base import BaseFaceDetector
from app.ai.tracking.base import BaseFaceTracker
from app.ai.tracking.factory import get_face_tracker
from app.ai.tracking.schemas import TrackResult, TrackingSummary
from app.services.face_analysis_service import FaceAnalysisService, get_face_analysis_service
from app.ai.visualization.annotator import FaceVisualizer

logger = logging.getLogger("deepvision.service.video")


class VideoProcessingService:
    """
    High-performance, streaming video face detection and tracking service.
    
    Architecture:
    Video Upload -> Validation -> Video Metadata -> Streaming Frame Extraction ->
    Face Detector -> Face Tracking -> Face Analysis -> Annotation -> Output Video -> Processing Metrics
    """

    def __init__(
        self,
        detector: Optional[BaseFaceDetector] = None,
        tracker: Optional[BaseFaceTracker] = None,
        analysis_service: Optional[FaceAnalysisService] = None,
        visualizer: Optional[FaceVisualizer] = None,
    ):
        # 1. Reuse existing face detector
        self._detector = detector or FaceDetector(
            model_path=settings.resolve_model_path(),
            device=settings.AI_DEVICE,
        )
        # 2. Reuse ByteTrack face tracker
        self._tracker = tracker or get_face_tracker(
            tracker_name="bytetrack",
            high_detection_threshold=settings.AI_CONFIDENCE_THRESHOLD,
            low_detection_threshold=0.15,
            min_hits=2,
            max_time_lost=25,
        )
        # 3. Reuse existing face analysis service
        self._analysis_service = analysis_service or get_face_analysis_service()
        # 4. Visualizer for frame annotations
        self._visualizer = visualizer or FaceVisualizer(
            show_bbox=True,
            show_confidence=True,
            show_landmarks=True,
            show_quality=True,
            show_pose=True,
            show_face_id=True,
            status_color_coding=True,
            analyze_on_the_fly=False,  # Analysis performed explicitly in pipeline
        )

    @property
    def detector(self) -> BaseFaceDetector:
        return self._detector

    @property
    def tracker(self) -> BaseFaceTracker:
        return self._tracker

    @property
    def analysis_service(self) -> FaceAnalysisService:
        return self._analysis_service

    def process_video_file(
        self,
        video_input_path: str,
        output_dir: Optional[str] = None,
        options: Optional[VideoProcessingOptions] = None,
        progress_callback: Optional[Any] = None,
        is_cancelled: Optional[Any] = None,
    ) -> VideoProcessingResult:
        """
        Processes a video file completely through the streaming detection and tracking pipeline.
        
        Args:
            video_input_path: Path to valid video container on disk.
            output_dir: Optional directory for saving the annotated video artifact.
            options: Video processing options (frame stride, max frames, annotation flag, etc.).
            progress_callback: Optional callable receiving (current_frame, total_frames, percent).
            is_cancelled: Optional callable returning True if client requested job cancellation.
            
        Returns:
            VideoProcessingResult: Complete telemetry, metadata, tracking summary, and artifact paths.
        """
        opts = options or VideoProcessingOptions()
        wall_start_time = time.perf_counter()

        # -------------------------------------------------------------
        # 1. Validation & Stream Metadata
        # -------------------------------------------------------------
        metadata: VideoMetadata = probe_and_validate_video(video_input_path)
        logger.info(
            "Starting video processing: %s (%dx%d @ %.1ffps, %d frames)",
            video_input_path,
            metadata.width,
            metadata.height,
            metadata.fps,
            metadata.frame_count,
        )

        # Initialize a fresh session tracker configured with job-specific thresholds
        session_tracker = get_face_tracker(
            tracker_name="bytetrack",
            high_detection_threshold=(
                opts.confidence_threshold
                if opts.confidence_threshold is not None
                else settings.AI_CONFIDENCE_THRESHOLD
            ),
            low_detection_threshold=0.15,
            min_hits=1 if (metadata.frame_count <= 5 or opts.process_every_n_frames > 1) else 2,
            max_time_lost=25,
        )

        # -------------------------------------------------------------
        # 2. Initialize Video Reader & Writer
        # -------------------------------------------------------------
        cap = cv2.VideoCapture(video_input_path)
        if not cap.isOpened():
            raise CorruptedVideoException(f"Failed to open video stream at {video_input_path}")

        # Setup annotated video output writer if requested
        video_writer: Optional[cv2.VideoWriter] = None
        output_video_path: Optional[str] = None

        if opts.annotate_video:
            target_dir = output_dir or os.path.join(tempfile.gettempdir(), "deepvision_annotated_videos")
            os.makedirs(target_dir, exist_ok=True)
            output_video_path = os.path.join(
                target_dir,
                f"annotated_{uuid.uuid4().hex[:10]}_{sanitize_video_filename(os.path.basename(video_input_path))}"
            )
            # Try mp4v FourCC first, with fallback to XVID or MJPG
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # Effective output fps accounts for frame stride
            effective_fps = max(1.0, metadata.fps / opts.process_every_n_frames)
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                effective_fps,
                (metadata.width, metadata.height),
            )
            if not video_writer.isOpened():
                # Fallback FourCC
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_writer = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    effective_fps,
                    (metadata.width, metadata.height),
                )

        # Resolve processing intervals (Step 5.11)
        det_interval = max(1, opts.detection_interval or opts.process_every_n_frames or settings.VIDEO_DETECTION_INTERVAL)
        ana_interval = max(1, opts.analysis_interval or settings.VIDEO_ANALYSIS_INTERVAL)
        track_analysis_cache: Dict[int, Dict[str, Any]] = {}

        # -------------------------------------------------------------
        # 3. Streaming Frame-by-Frame Processing Loop
        # (Does NOT load the entire video into RAM)
        # -------------------------------------------------------------
        frame_idx = 0
        processed_frames_count = 0
        total_detections_count = 0

        total_detection_time = 0.0
        total_tracking_time = 0.0
        total_analysis_time = 0.0
        total_annotation_time = 0.0

        frame_results: List[VideoFrameTrackResult] = []
        max_limit = opts.max_frames or metadata.frame_count

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break

                # Step 5.21: Cancellation check
                if is_cancelled and is_cancelled():
                    logger.info("Video processing loop received cancellation request at frame %d.", frame_idx)
                    if output_video_path and os.path.exists(output_video_path):
                        try:
                            os.remove(output_video_path)
                        except OSError:
                            pass
                    raise VideoProcessingCancelledException(f"Video job was cancelled by client at frame {frame_idx}.")

                if frame_idx >= max_limit:
                    break

                processed_frames_count += 1
                frame_ts_ms = round((frame_idx / metadata.fps) * 1000.0, 2) if metadata.fps > 0 else None

                # Convert BGR to RGB for model inference
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # --- A. Face Detection (Interval-Gated) ---
                run_detector_this_frame = (frame_idx % det_interval) == 0
                frame_det_ms = 0.0

                if run_detector_this_frame:
                    det_t0 = time.perf_counter()
                    det_response = self._detector.detect(
                        image_input=frame_rgb,
                        confidence_threshold=opts.confidence_threshold,
                    )
                    det_t1 = time.perf_counter()
                    frame_det_ms = (det_t1 - det_t0) * 1000.0
                    total_detection_time += frame_det_ms
                    raw_detections = det_response.detections
                    total_detections_count += len(raw_detections)
                else:
                    # Intermediate frame: detector skipped for throughput; tracker coasts via Kalman
                    raw_detections = []

                # --- B. Multi-Face Tracking (ByteTrack) ---
                track_t0 = time.perf_counter()
                active_tracks: List[TrackResult] = session_tracker.update(
                    detections=raw_detections,
                    frame=frame_rgb,
                    frame_index=frame_idx,
                    timestamp_ms=frame_ts_ms,
                )
                track_t1 = time.perf_counter()
                frame_track_ms = (track_t1 - track_t0) * 1000.0
                total_tracking_time += frame_track_ms

                # --- C. Face Analysis on Confirmed Tracks (Interval-Gated & Cached) ---
                frame_analysis_ms = 0.0
                run_analysis_this_frame = (frame_idx % ana_interval) == 0

                if opts.enable_analysis and active_tracks:
                    ana_t0 = time.perf_counter()
                    for trk in active_tracks:
                        tid = trk.track_id
                        if run_analysis_this_frame or tid not in track_analysis_cache:
                            # Recompute optical & geometric quality diagnostics
                            x1 = max(0, int(trk.bbox.x1))
                            y1 = max(0, int(trk.bbox.y1))
                            x2 = min(metadata.width, int(trk.bbox.x2))
                            y2 = min(metadata.height, int(trk.bbox.y2))
                            if x2 > x1 and y2 > y1:
                                face_crop = frame_rgb[y1:y2, x1:x2]
                                try:
                                    _, _, clarity = self._analysis_service.analyze_blur(face_crop)
                                    _, _, illum = self._analysis_service.analyze_brightness(face_crop)
                                    size_m = self._analysis_service.analyze_size(trk.bbox, (metadata.height, metadata.width))
                                    q_score, q_class, factors, grade = self._analysis_service.calculate_quality(
                                        face_crop=face_crop,
                                        bbox=trk.bbox,
                                        image_shape=(metadata.height, metadata.width),
                                        clarity_metrics=clarity,
                                        illumination_metrics=illum,
                                        size_metrics=size_m,
                                        detection_confidence=trk.detection_confidence or trk.track_confidence,
                                    )
                                    track_analysis_cache[tid] = {
                                        "quality_score": q_score,
                                        "classification": q_class,
                                    }
                                except Exception as err:
                                    logger.debug("Face analysis skipped on track %d: %s", tid, err)

                        # Apply cached diagnostics to track representation
                        if tid in track_analysis_cache:
                            cached = track_analysis_cache[tid]
                            trk_dict = getattr(trk, "__dict__", {})
                            trk_dict["quality_score"] = cached.get("quality_score")
                            trk_dict["classification"] = cached.get("classification")

                    ana_t1 = time.perf_counter()
                    frame_analysis_ms = (ana_t1 - ana_t0) * 1000.0
                    total_analysis_time += frame_analysis_ms

                # --- D. Frame Annotation & Video Encoding ---
                if opts.annotate_video and video_writer is not None:
                    ann_t0 = time.perf_counter()
                    annotated_bgr = self._visualizer.annotate(
                        image=frame_bgr,
                        detections=active_tracks,
                        return_rgb=False,
                    )
                    # Overlay persistent session HUD (Frame, FPS, Active Tracks, Peak Tracks)
                    hud_text = (
                        f"Frame: {frame_idx}/{metadata.frame_count} | "
                        f"Active: {len(active_tracks)} | "
                        f"Peak: {session_tracker.get_summary().peak_simultaneous_faces} | "
                        f"Unique: {session_tracker.total_unique_tracks}"
                    )
                    cv2.putText(
                        annotated_bgr,
                        hud_text,
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    video_writer.write(annotated_bgr)
                    ann_t1 = time.perf_counter()
                    total_annotation_time += (ann_t1 - ann_t0) * 1000.0

                # Record frame track result
                frame_results.append(
                    VideoFrameTrackResult(
                        frame_index=frame_idx,
                        timestamp_ms=frame_ts_ms,
                        tracks=active_tracks,
                        detection_count=len(raw_detections),
                        processing_time_ms=round(frame_det_ms + frame_track_ms + frame_analysis_ms, 2),
                    )
                )

                # Progress callback
                if progress_callback:
                    pct = round((frame_idx + 1) / max(1, metadata.frame_count) * 100.0, 1)
                    progress_callback(frame_idx + 1, metadata.frame_count, pct)

                frame_idx += 1

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()

        # -------------------------------------------------------------
        # 4. Metrics & Summary Compilation (Step 5.13)
        # -------------------------------------------------------------
        wall_end_time = time.perf_counter()
        elapsed_ms = round((wall_end_time - wall_start_time) * 1000.0, 2)
        fps_throughput = round((processed_frames_count / (elapsed_ms / 1000.0)), 2) if elapsed_ms > 0 else 0.0
        avg_frame_time = round(elapsed_ms / max(1, processed_frames_count), 2)

        tracking_summary: TrackingSummary = session_tracker.get_summary()

        metrics = VideoProcessingMetrics(
            total_frames=metadata.frame_count,
            processed_frames=processed_frames_count,
            total_processing_time=elapsed_ms,
            average_frame_processing_time=avg_frame_time,
            average_fps=fps_throughput,
            peak_simultaneous_faces=tracking_summary.peak_simultaneous_faces,
            unique_track_count=tracking_summary.total_unique_tracks,
            track_lost_count=tracking_summary.track_lost_count,
            total_faces_detected=total_detections_count,
            detection_time_ms=round(total_detection_time, 2),
            tracking_time_ms=round(total_tracking_time, 2),
            analysis_time_ms=round(total_analysis_time, 2),
            annotation_time_ms=round(total_annotation_time, 2),
        )

        logger.info(
            "Video processing completed: %d frames processed in %.2fms (%.1f fps throughput). Unique tracks: %d",
            processed_frames_count,
            elapsed_ms,
            fps_throughput,
            tracking_summary.total_unique_tracks,
        )

        return VideoProcessingResult(
            success=True,
            metadata=metadata,
            tracking_summary=tracking_summary,
            metrics=metrics,
            annotated_video_path=output_video_path,
            frame_results=frame_results if processed_frames_count <= 200 else None,
        )


# Singleton accessor
_video_processing_service: Optional[VideoProcessingService] = None


def get_video_processing_service() -> VideoProcessingService:
    """Returns application-level VideoProcessingService singleton."""
    global _video_processing_service
    if _video_processing_service is None:
        _video_processing_service = VideoProcessingService()
    return _video_processing_service
