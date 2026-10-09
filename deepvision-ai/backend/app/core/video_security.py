"""
DeepVision AI — Video Security, Validation & Ingestion Controls (Step 5.7)
Provides robust, defense-in-depth safety validation for uploaded video files:
- Container extension & MIME type verification
- Memory-safe streaming disk writing with hard payload caps
- OpenCV video readability & stream decoding verification
- Frame rate, resolution, frame count, and duration boundary checks
- Clean rejection of corrupt, truncated, or zero-byte video payloads
- Ephemeral temporary file lifecycle management
"""

import os
import re
import uuid
import tempfile
import logging
from typing import Tuple, Optional, Generator
from contextlib import contextmanager
import cv2
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import (
    UnsupportedVideoError,
    CorruptedVideoException,
    EmptyVideoException,
    VideoTooLargeException,
    VideoDurationExceededException,
)
from app.schemas.video import VideoMetadata

logger = logging.getLogger("deepvision.security.video")

# Permitted video file extensions
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Permitted video MIME types
ALLOWED_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/x-msvideo",
    "video/avi",
    "video/quicktime",
    "video/x-matroska",
    "video/webm",
    "application/octet-stream",  # Often sent by generic upload clients
}

# Maximum video resolution bounds (4096 x 4096)
MAX_VIDEO_WIDTH = 4096
MAX_VIDEO_HEIGHT = 4096

# Maximum processing duration (10 minutes = 600 seconds by default)
MAX_VIDEO_DURATION_SECONDS = 600.0


def sanitize_video_filename(filename: Optional[str]) -> str:
    """Sanitizes filename, stripping path traversal characters and non-alphanumeric symbols."""
    if not filename:
        return f"video_{uuid.uuid4().hex[:8]}.mp4"
    name = os.path.basename(filename)
    name = re.sub(r"[^\w\.\-]", "_", name)
    name = re.sub(r"\.{2,}", ".", name)
    return name or f"video_{uuid.uuid4().hex[:8]}.mp4"


def validate_video_extension_and_mime(filename: str, content_type: Optional[str] = None) -> str:
    """
    Validates declared file extension and MIME type.
    Returns: Normalized lowercase extension (e.g. '.mp4').
    """
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise UnsupportedVideoError(
            f"Unsupported video extension '{ext}'. Allowed extensions: {sorted(list(ALLOWED_VIDEO_EXTENSIONS))}."
        )

    if content_type:
        clean_mime = content_type.lower().split(";")[0].strip()
        if clean_mime not in ALLOWED_VIDEO_MIME_TYPES:
            logger.warning(
                "Video uploaded with unverified MIME type '%s' (filename: %s)",
                clean_mime,
                filename,
            )

    return ext


async def save_upload_video_safely(
    upload_file: UploadFile,
    max_bytes: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Streams uploaded video data to an ephemeral temporary file on disk.
    Enforces chunked reading to prevent memory exhaustion on large files (up to MAX_VIDEO_SIZE_MB).
    
    Returns:
        Tuple[str, int]: (temp_file_path, total_bytes_written)
    """
    max_allowed = max_bytes or (settings.MAX_VIDEO_SIZE_MB * 1024 * 1024)
    filename = sanitize_video_filename(upload_file.filename)
    validate_video_extension_and_mime(filename, upload_file.content_type)

    temp_dir = os.path.join(tempfile.gettempdir(), "deepvision_videos")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}_{filename}")

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1 MB streaming chunks

    try:
        with open(temp_path, "wb") as f_out:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_allowed:
                    raise VideoTooLargeException(
                        f"Video file size exceeds maximum permitted limit of {settings.MAX_VIDEO_SIZE_MB}MB."
                    )
                f_out.write(chunk)

        if total_bytes == 0:
            raise EmptyVideoException("Uploaded video file is empty (0 bytes).")

        logger.debug("Safely saved uploaded video: %s (%d bytes)", temp_path, total_bytes)
        return temp_path, total_bytes

    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def probe_and_validate_video(
    video_path: str,
    max_duration_seconds: float = MAX_VIDEO_DURATION_SECONDS,
) -> VideoMetadata:
    """
    Validates physical video readability using OpenCV VideoCapture.
    Extracts stream dimensions, frame rate, frame count, and duration.
    
    Raises:
        CorruptedVideoException: If file cannot be decoded or has zero valid frames.
        VideoDurationExceededException: If declared duration exceeds limit.
    """
    if not os.path.exists(video_path):
        raise CorruptedVideoException(f"Video file does not exist at {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise CorruptedVideoException("Failed to open video container. Video is corrupted or codec is unsupported.")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Decode FourCC codec string
        codec_str = None
        if fourcc_int > 0:
            codec_chars = [chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]
            codec_str = "".join(codec_chars).strip()

        max_w = getattr(settings, "VIDEO_MAX_WIDTH", MAX_VIDEO_WIDTH)
        max_h = getattr(settings, "VIDEO_MAX_HEIGHT", MAX_VIDEO_HEIGHT)
        max_f = getattr(settings, "VIDEO_MAX_FRAME_COUNT", 18000)
        max_dur = max_duration_seconds or getattr(settings, "VIDEO_MAX_DURATION_SECONDS", MAX_VIDEO_DURATION_SECONDS)

        # Check for zero or negative dimensions
        if width <= 0 or height <= 0:
            raise CorruptedVideoException(f"Invalid video dimensions: {width}x{height}px.")

        if width > max_w or height > max_h:
            raise CorruptedVideoException(
                f"Video resolution {width}x{height}px exceeds maximum allowed limit of {max_w}x{max_h}px."
            )

        if frame_count > max_f:
            raise VideoDurationExceededException(
                f"Video frame count ({frame_count}) exceeds maximum allowed limit of {max_f} frames."
            )

        # Validate readability by reading the very first frame
        ret, first_frame = cap.read()
        if not ret or first_frame is None or first_frame.size == 0:
            raise CorruptedVideoException("Video contains zero readable frames or corrupted video track.")

        # Handle unreliable or missing FPS
        if fps <= 0.1 or fps > 240.0:
            fps = 25.0  # Safe standard fallback if container omits FPS header

        # Handle frame count
        if frame_count < 1:
            frame_count = 1

        duration_sec = round(frame_count / fps, 3) if (fps > 0 and frame_count > 0) else None

        if duration_sec is not None and duration_sec > max_dur:
            raise VideoDurationExceededException(
                f"Video duration ({duration_sec:.1f}s) exceeds maximum allowed duration of {max_dur}s."
            )

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration_sec,
            codec=codec_str,
        )

    finally:
        cap.release()


@contextmanager
def ephemeral_video_file(filepath: str) -> Generator[str, None, None]:
    """Context manager ensuring temporary video file is deleted upon exit."""
    try:
        yield filepath
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.debug("Cleaned up ephemeral video file: %s", filepath)
            except OSError as err:
                logger.warning("Could not delete ephemeral video %s: %s", filepath, err)
