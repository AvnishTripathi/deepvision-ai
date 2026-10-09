"""
DeepVision AI — Application Exceptions
Domain-specific exceptions for error handling across the platform.
Provides clean AI-specific exception hierarchies with mapped HTTP statuses
and structured error codes for sanitized client responses.
"""

from typing import Optional, Dict, Any


class DeepVisionException(Exception):
    """Base exception for all DeepVision errors."""

    def __init__(
        self,
        message: str = "An unexpected DeepVision error occurred.",
        detail: Optional[str] = None,
        error_code: str = "deepvision_error",
        status_code: int = 500,
    ):
        self.message = message
        self.detail = detail or message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)


# -----------------------------------------------------------------------------
# AI Engine & Computer Vision Exceptions
# -----------------------------------------------------------------------------

class AIException(DeepVisionException):
    """Base exception for all AI inference, model lifecycle, and image errors."""

    def __init__(
        self,
        message: str = "An error occurred in the AI processing engine.",
        detail: Optional[str] = None,
        error_code: str = "ai_error",
        status_code: int = 500,
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            detail=detail,
            error_code=error_code,
            status_code=status_code,
        )
        self.extra = extra or {}


class ModelLoadError(AIException):
    """
    Raised when loading model weights, initializing ONNX Runtime sessions,
    or reading model artifacts fails.
    """

    def __init__(
        self,
        message: str = "AI model could not be loaded or is not configured.",
        detail: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            detail=detail or message,
            error_code="model_load_error",
            status_code=503,
            extra={"model_name": model_name, "device": device},
        )
        self.model_name = model_name
        self.device = device


class DetectorNotConfiguredException(ModelLoadError):
    """
    Raised when the AI face detector is invoked before model weights are configured.
    """

    def __init__(
        self,
        message: str = "AI face detection model is not operational. The detector is not configured.",
        detail: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            detail=detail or message,
            model_name=model_name,
            device=device,
        )
        self.error_code = "detector_not_configured"


class InferenceError(AIException):
    """
    Raised during model forward-pass execution, tensor transformation,
    or parsing model outputs.
    """

    def __init__(
        self,
        message: str = "AI model inference processing failed.",
        detail: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            detail=detail or message,
            error_code="inference_error",
            status_code=500,
            extra={"model_name": model_name},
        )
        self.model_name = model_name


class ModelInferenceException(InferenceError):
    """Legacy alias for backward compatibility."""
    pass


class InvalidImageError(AIException):
    """
    Raised when an input image cannot be processed due to corruption,
    empty payload, or invalid dimensions.
    """

    def __init__(
        self,
        message: str = "Invalid image file provided.",
        detail: Optional[str] = None,
        error_code: str = "invalid_image",
        status_code: int = 400,
    ):
        super().__init__(
            message=message,
            detail=detail,
            error_code=error_code,
            status_code=status_code,
        )


class ImagePreprocessingException(InvalidImageError):
    """Base exception for image ingestion and preprocessing pipeline failures."""
    pass


class CorruptedImageException(InvalidImageError):
    """Raised when image bytes or files cannot be parsed due to data corruption."""

    def __init__(self, message: str = "Corrupted or unreadable image bytes; failed to decode."):
        super().__init__(
            message=message,
            error_code="corrupted_image",
            status_code=400,
        )


class EmptyImageException(InvalidImageError):
    """Raised when the image is empty (zero bytes or zero height/width)."""

    def __init__(self, message: str = "Uploaded image file is empty (0 bytes)."):
        super().__init__(
            message=message,
            error_code="empty_file",
            status_code=400,
        )


class InvalidImageDimensionsException(InvalidImageError):
    """Raised when image dimensions are non-positive or exceed max bounds."""

    def __init__(self, message: str = "Image dimensions are invalid or exceed supported bounds."):
        super().__init__(
            message=message,
            error_code="invalid_dimensions",
            status_code=400,
        )


class ImageTooLargeException(InvalidImageDimensionsException):
    """Raised when image exceeds maximum supported pixel dimensions or file size limits."""

    def __init__(self, message: str = "Image exceeds maximum allowed size or resolution limits."):
        super().__init__(message=message)
        self.error_code = "payload_too_large"
        self.status_code = 413


class UnsupportedImageError(AIException):
    """
    Raised when an image format or MIME type is not supported.
    """

    def __init__(
        self,
        message: str = "Unsupported image format. Allowed formats: JPEG, PNG, WebP, BMP.",
        detail: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            detail=detail,
            error_code="unsupported_media_type",
            status_code=400,
        )


class UnsupportedImageFormatException(UnsupportedImageError):
    """Raised when image format or color channels cannot be processed."""

    def __init__(self, message: str = "Unsupported image color format or channels."):
        super().__init__(message=message)
        self.error_code = "unsupported_format"


class AIConfigurationError(AIException):
    """
    Raised when AI hyperparameters, runtime devices, or model configurations
    are invalid or inconsistent.
    """

    def __init__(
        self,
        message: str = "Invalid AI engine configuration settings.",
        detail: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            detail=detail,
            error_code="ai_configuration_error",
            status_code=500,
        )


class DeviceNotAvailableError(AIException):
    """
    Raised when a requested hardware compute acceleration device (e.g. CUDA/GPU)
    is not available on the host environment.
    """

    def __init__(
        self,
        message: str = "Requested AI hardware compute device is not available on this host.",
        detail: Optional[str] = None,
        requested_device: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            detail=detail,
            error_code="device_not_available",
            status_code=503,
            extra={"requested_device": requested_device},
        )
        self.requested_device = requested_device



# -----------------------------------------------------------------------------
# Video Processing Exceptions (Step 5)
# -----------------------------------------------------------------------------

class VideoException(AIException):
    """Base exception for all video ingestion, decoding, and processing errors."""

    def __init__(
        self,
        message: str = "An error occurred during video processing.",
        detail: Optional[str] = None,
        error_code: str = "video_error",
        status_code: int = 400,
    ):
        super().__init__(
            message=message,
            detail=detail,
            error_code=error_code,
            status_code=status_code,
        )


class UnsupportedVideoError(VideoException):
    """Raised when an uploaded video container format or MIME type is unsupported."""

    def __init__(self, message: str = "Unsupported video format. Supported: MP4, AVI, MOV, MKV, WebM."):
        super().__init__(message=message, error_code="unsupported_video_format", status_code=400)


class CorruptedVideoException(VideoException):
    """Raised when a video file is corrupt, unreadable, or missing valid video streams."""

    def __init__(self, message: str = "Corrupted or unreadable video file; cannot decode video frames."):
        super().__init__(message=message, error_code="corrupted_video", status_code=400)


class EmptyVideoException(VideoException):
    """Raised when the video file is empty (0 bytes)."""

    def __init__(self, message: str = "Uploaded video file is empty (0 bytes)."):
        super().__init__(message=message, error_code="empty_file", status_code=400)


class VideoTooLargeException(VideoException):
    """Raised when a video exceeds maximum file size limits."""

    def __init__(self, message: str = "Video file exceeds maximum allowed size limits."):
        super().__init__(message=message, error_code="payload_too_large", status_code=413)


class VideoDurationExceededException(VideoException):
    """Raised when a video duration exceeds the maximum permitted limit."""

    def __init__(self, message: str = "Video duration exceeds maximum allowed processing duration."):
        super().__init__(message=message, error_code="video_duration_exceeded", status_code=400)


class VideoProcessingCancelledException(VideoException):
    """Raised when a video processing job is cancelled by client request."""

    def __init__(self, message: str = "Video processing job was cancelled."):
        super().__init__(message=message, error_code="job_cancelled", status_code=200)


# -----------------------------------------------------------------------------
# Storage & Database Exceptions
# -----------------------------------------------------------------------------

class DatabaseConnectionException(DeepVisionException):
    """Raised when the database connection fails."""

    def __init__(self, message: str = "Database connection failed."):
        super().__init__(message=message, error_code="database_connection_error", status_code=500)


class CacheConnectionException(DeepVisionException):
    """Raised when the cache connection fails."""

    def __init__(self, message: str = "Cache connection failed."):
        super().__init__(message=message, error_code="cache_connection_error", status_code=500)


class EntityNotFoundException(DeepVisionException):
    """Raised when a requested resource is not found."""

    def __init__(self, message: str = "Requested resource not found."):
        super().__init__(message=message, error_code="not_found", status_code=404)


class MediaProcessingException(DeepVisionException):
    """Raised when media decoding, transcoding, or reading fails."""

    def __init__(self, message: str = "Media processing failed."):
        super().__init__(message=message, error_code="media_processing_error", status_code=400)
