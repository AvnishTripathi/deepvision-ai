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
