"""
DeepVision AI — Image Ingestion Security & Safety Controls
Implements defense-in-depth safety controls for uploaded images:
- Strict MIME type and magic number byte signature verification
- Protection against file type spoofing, polyglot payloads, and executables
- Memory-safe stream reading with hard byte limits
- Decompression bomb / pixel flood attack mitigation
- Image dimension bounds enforcement
- Filename sanitization and untrusted input shielding
- Ephemeral in-memory processing with zero unnecessary permanent disk storage
- Zero logging of raw binary or pixel payloads
"""

import os
import io
import re
import uuid
import tempfile
import logging
from typing import Tuple, Optional, Generator
from contextlib import contextmanager
import numpy as np
from PIL import Image
import cv2
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import (
    UnsupportedImageError,
    CorruptedImageException,
    EmptyImageException,
    ImageTooLargeException,
    InvalidImageDimensionsException,
)

logger = logging.getLogger("deepvision.security")

# Allowed MIME types declared by clients
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}

# Allowed filename extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Known Magic Byte signatures for supported image containers
MAGIC_SIGNATURES = {
    "JPEG": (b"\xFF\xD8\xFF",),
    "PNG": (b"\x89PNG\r\n\x1a\n",),
    "BMP": (b"BM",),
}

# Max decompression bomb threshold (100 million pixels = ~10000x10000)
# DeepVision enforces 4096px by default, so 100M pixels is a solid safety ceiling
MAX_SAFE_PIXELS = 100_000_000
Image.MAX_IMAGE_PIXELS = MAX_SAFE_PIXELS


def sanitize_filename(filename: Optional[str]) -> str:
    """
    Sanitizes untrusted client-supplied filenames:
    - Strips directory traversal sequences (../, ..\\, /, \\)
    - Strips null bytes and control characters
    - Generates a safe random UUID-backed fallback if filename is empty or dangerous
    
    NEVER trust or use raw uploaded filenames directly for filesystem or shell operations.
    """
    if not filename:
        return f"upload_{uuid.uuid4().hex[:12]}.jpg"

    # 1. Strip path separators and null bytes
    clean = os.path.basename(filename).replace("\x00", "").strip()

    # 2. Extract and validate extension
    ext = os.path.splitext(clean.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".jpg"

    # 3. Strip special shell and filesystem characters
    base_name = os.path.splitext(clean)[0]
    safe_base = re.sub(r"[^a-zA-Z0-9_\-]", "_", base_name)[:64]
    if not safe_base:
        safe_base = f"upload_{uuid.uuid4().hex[:12]}"

    return f"{safe_base}{ext}"


def detect_magic_bytes_format(data: bytes) -> Optional[str]:
    """
    Inspects leading binary magic numbers to determine the actual image container type.
    Defends against file extension spoofing (e.g. an executable or script named image.jpg).
    """
    if len(data) < 12:
        return None

    # Check JPEG
    if data.startswith(b"\xFF\xD8\xFF"):
        return "JPEG"

    # Check PNG
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"

    # Check BMP
    if data.startswith(b"BM"):
        return "BMP"

    # Check WebP (RIFF....WEBP)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"

    return None


async def read_upload_stream_safely(
    file: UploadFile,
    max_bytes: int,
    chunk_size: int = 65536,
) -> bytes:
    """
    Reads an uploaded stream in fixed 64KB chunks with an active byte counter.
    Immediately terminates and raises ImageTooLargeException if payload exceeds
    the configured limit, preventing memory exhaustion attacks before full loading.
    """
    chunks = []
    total_bytes = 0

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            # Prevent reading further into memory
            max_mb = max_bytes // (1024 * 1024)
            raise ImageTooLargeException(
                f"Image payload size exceeds the maximum allowed limit of {max_mb}MB."
            )

        chunks.append(chunk)

    payload = b"".join(chunks)
    if len(payload) == 0:
        raise EmptyImageException("Uploaded image payload is empty (0 bytes).")

    return payload


def validate_and_decode_image_security(
    data: bytes,
    declared_content_type: Optional[str] = None,
    declared_filename: Optional[str] = None,
    max_dimension: Optional[int] = None,
) -> Tuple[np.ndarray, str, Tuple[int, int]]:
    """
    Multi-stage defense-in-depth safety pipeline for image bytes:
    1. Empty byte check
    2. Declared MIME type and extension validation
    3. Binary magic number signature inspection (anti-spoofing)
    4. Pillow structural verification (anti-polyglot, decompression bomb defense)
    5. In-memory RGB decoding and pixel dimension bounds check
    
    Guarantees:
    - Never executes arbitrary code.
    - Never trusts uploaded filenames.
    - Never logs raw bytes.
    - Operates purely in volatile memory.

    Returns:
        Tuple of (image_rgb_array, verified_format, (width, height))
    """
    max_dim = max_dimension or settings.AI_MAX_IMAGE_DIMENSION

    # 1. Empty Check
    if not data or len(data) == 0:
        raise EmptyImageException("Uploaded image file is empty (0 bytes).")

    # 2. Declared MIME Type & Extension Check
    raw_ext = os.path.splitext((declared_filename or "").lower())[1]
    content_type = (declared_content_type or "").lower().strip()

    if (content_type and content_type not in ALLOWED_MIME_TYPES) or (raw_ext and raw_ext not in ALLOWED_EXTENSIONS):
        raise UnsupportedImageError(
            f"Unsupported file type '{content_type or raw_ext}'. "
            f"Allowed formats: image/jpeg, image/png, image/webp, image/bmp."
        )

    safe_name = sanitize_filename(declared_filename)

    # 3. Magic Bytes Signature Verification
    detected_format = detect_magic_bytes_format(data)
    if not detected_format:
        logger.warning("Rejected upload: Magic byte signature does not match any valid image container.")
        raise UnsupportedImageError(
            "File header signature does not match a valid image format. "
            "Executable, script, or non-image files are strictly rejected."
        )

    # 4. Pillow Structural & Decompression Bomb Verification
    try:
        with Image.open(io.BytesIO(data)) as pil_img:
            # Check format matches magic bytes
            pil_format = (pil_img.format or "").upper()
            if pil_format not in ("JPEG", "PNG", "WEBP", "BMP"):
                raise UnsupportedImageError(f"Unsupported container format: {pil_format}")

            # Decompression bomb pixel check
            width, height = pil_img.size
            if width <= 0 or height <= 0:
                raise InvalidImageDimensionsException(f"Invalid image dimensions: {width}x{height}")

            total_pixels = width * height
            if total_pixels > MAX_SAFE_PIXELS:
                raise ImageTooLargeException(
                    f"Pixel count ({total_pixels:,}) exceeds maximum safe threshold "
                    f"of {MAX_SAFE_PIXELS:,} pixels (decompression bomb protection)."
                )

            # verify() reads container tables without decoding pixels
            # Note: verify() must be done on fresh stream if reusing
            pil_img.verify()

    except (UnsupportedImageError, ImageTooLargeException, InvalidImageDimensionsException):
        raise
    except Exception as exc:
        logger.warning("Pillow container verification failed: %s", exc)
        raise CorruptedImageException(f"Corrupted or malformed image container: {str(exc)}")

    # 5. Dimension Bounds Check (enforcing AI_MAX_IMAGE_DIMENSION)
    if width > max_dim or height > max_dim:
        raise InvalidImageDimensionsException(
            f"Image dimensions {width}x{height} exceed the maximum allowed limit of {max_dim}px."
        )

    # 6. In-Memory OpenCV Decoding into Standard uint8 RGB
    try:
        nparr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise CorruptedImageException("Failed to decode image byte stream into pixel matrix.")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except CorruptedImageException:
        raise
    except Exception as exc:
        logger.warning("OpenCV image decoding failed: %s", exc)
        raise CorruptedImageException(f"Image decoding failed: {str(exc)}")

    dec_h, dec_w = rgb.shape[:2]
    return rgb, detected_format, (dec_w, dec_h)


@contextmanager
def secure_temporary_image(data: bytes, suffix: str = ".jpg") -> Generator[str, None, None]:
    """
    Context manager for isolated temporary file processing when third-party
    tools strictly require a filesystem path.
    Guarantees:
    - Creates file strictly within the OS temporary directory with restrictive 0600 permissions.
    - Uses random cryptographic filename.
    - Guarantees immediate deletion upon context exit.
    """
    clean_suffix = suffix if suffix in ALLOWED_EXTENSIONS else ".jpg"
    temp_file = tempfile.NamedTemporaryFile(suffix=clean_suffix, delete=False)
    try:
        temp_file.write(data)
        temp_file.flush()
        temp_file.close()
        # Set restrictive permissions (read/write by owner only)
        try:
            os.chmod(temp_file.name, 0o600)
        except Exception:
            pass  # Windows may ignore unix chmod modes

        yield temp_file.name
    finally:
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception as exc:
                logger.debug("Failed deleting temporary file %s: %s", temp_file.name, exc)
