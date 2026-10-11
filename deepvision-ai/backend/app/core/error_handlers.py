"""
DeepVision AI — Centralized Exception & Error Handling
Maps AI-specific domain exceptions and system failures to standard HTTP responses.
Guarantees that internal stack traces, system paths, and memory dumps are NEVER
exposed to API clients, while recording full technical diagnostics in server logs.

Consistent Error Response Schema:
{
  "success": false,
  "error": "unsupported_media_type",
  "message": "...",
  "request_id": "uuid-trace-id",
  "error_detail": {
    "code": "UNSUPPORTED_MEDIA_TYPE",
    "message": "..."
  }
}
"""

import logging
from typing import Optional
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    DeepVisionException,
    AIException,
    ModelLoadError,
    InferenceError,
    InvalidImageError,
    UnsupportedImageError,
    AIConfigurationError,
    DeviceNotAvailableError,
)

logger = logging.getLogger("deepvision.error_handler")


def _get_request_id(request: Request) -> Optional[str]:
    """Extracts correlated request ID from request state or client headers."""
    if hasattr(request, "state") and hasattr(request.state, "request_id"):
        return str(request.state.request_id)
    return request.headers.get("X-Request-ID")


def register_exception_handlers(app: FastAPI) -> None:
    """Registers standardized, sanitized exception handlers on the FastAPI application."""

    @app.exception_handler(AIException)
    async def ai_exception_handler(request: Request, exc: AIException) -> JSONResponse:
        """
        Handles all AI engine and computer vision domain exceptions.
        Logs detailed technical diagnostics on server-side without leaking internals to clients.
        """
        status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        error_code = getattr(exc, "error_code", "ai_error")
        req_id = _get_request_id(request)

        # 1. Server-side detailed logging
        if status_code >= 500:
            logger.error(
                "AI Internal Server Error [%s %s] [request_id=%s] — %s: %s (detail: %s)",
                request.method,
                request.url.path,
                req_id or "-",
                exc.__class__.__name__,
                exc.message,
                exc.detail,
                exc_info=True,  # Captures full traceback in server logs
            )
        else:
            logger.warning(
                "AI Client Validation Error [%s %s] [request_id=%s] — %s: %s",
                request.method,
                request.url.path,
                req_id or "-",
                exc.__class__.__name__,
                exc.message,
            )

        # 2. Build sanitized client response (zero stack trace exposure)
        payload = {
            "success": False,
            "error": error_code,
            "message": exc.message,
            "request_id": req_id,
            "error_detail": {
                "code": error_code.upper(),
                "message": exc.message,
            },
        }

        # Include contextual guidance for model configuration in dev
        if isinstance(exc, ModelLoadError):
            if exc.detail:
                payload["detail"] = exc.detail
            if exc.model_name:
                payload["model"] = exc.model_name
            if exc.device:
                payload["device"] = exc.device

        return JSONResponse(
            status_code=status_code,
            content=payload,
        )

    @app.exception_handler(DeepVisionException)
    async def deepvision_exception_handler(request: Request, exc: DeepVisionException) -> JSONResponse:
        """Handles general platform-level DeepVision exceptions."""
        status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        error_code = getattr(exc, "error_code", "deepvision_error")
        req_id = _get_request_id(request)

        if status_code >= 500:
            logger.error(
                "DeepVision System Exception [%s %s] [request_id=%s] — %s: %s",
                request.method,
                request.url.path,
                req_id or "-",
                exc.__class__.__name__,
                exc.message,
                exc_info=True,
            )
        else:
            logger.warning(
                "DeepVision Warning [%s %s] [request_id=%s] — %s: %s",
                request.method,
                request.url.path,
                req_id or "-",
                exc.__class__.__name__,
                exc.message,
            )

        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error": error_code,
                "message": exc.message,
                "request_id": req_id,
                "error_detail": {
                    "code": error_code.upper(),
                    "message": exc.message,
                },
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Global safety net for any unexpected Python exceptions.
        Guarantees that raw Python stack traces or internal environment variables
        are never leaked in API responses.
        """
        req_id = _get_request_id(request)
        logger.critical(
            "Unhandled Server Exception [%s %s] [request_id=%s] — %s: %s",
            request.method,
            request.url.path,
            req_id or "-",
            exc.__class__.__name__,
            str(exc),
            exc_info=True,  # Detailed server traceback
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "internal_server_error",
                "message": "An unexpected internal server error occurred while processing the request.",
                "request_id": req_id,
                "error_detail": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected internal server error occurred while processing the request.",
                },
            },
        )

