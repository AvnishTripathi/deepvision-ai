"""
DeepVision AI — Application Exceptions
Domain-specific exceptions for error handling across the platform.
"""


class DeepVisionException(Exception):
    """Base exception for all DeepVision errors."""
    def __init__(self, message: str = "An unexpected DeepVision error occurred."):
        self.message = message
        super().__init__(self.message)


class DatabaseConnectionException(DeepVisionException):
    """Raised when the database connection fails."""
    pass


class CacheConnectionException(DeepVisionException):
    """Raised when the cache connection fails."""
    pass


class EntityNotFoundException(DeepVisionException):
    """Raised when a requested resource is not found."""
    pass


class ModelInferenceException(DeepVisionException):
    """Raised during model execution or pipeline processing errors."""
    pass


class MediaProcessingException(DeepVisionException):
    """Raised when media decoding, transcoding, or reading fails."""
    pass


class ImagePreprocessingException(DeepVisionException):
    """Base exception for all image ingestion and preprocessing errors."""
    pass


class CorruptedImageException(ImagePreprocessingException):
    """Raised when image bytes or files cannot be parsed due to data corruption."""
    pass


class EmptyImageException(ImagePreprocessingException):
    """Raised when the image is empty (zero bytes or zero height/width)."""
    pass


class UnsupportedImageFormatException(ImagePreprocessingException):
    """Raised when the image format or color channels cannot be processed."""
    pass


class InvalidImageDimensionsException(ImagePreprocessingException):
    """Raised when image dimensions are invalid (e.g. non-positive or corrupted shape)."""
    pass


class ImageTooLargeException(InvalidImageDimensionsException):
    """Raised when image exceeds maximum supported pixel dimensions or memory limits."""
    pass
