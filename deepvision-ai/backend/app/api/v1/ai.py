"""
DeepVision AI — AI Inference & Face Detection API Router
Provides development and production endpoints for face detection, image analysis,
and AI engine runtime status telemetry with hardened image safety controls.
"""

import os
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, Request, UploadFile, File, Query, Depends

from app.core.config import settings
from app.core.logging import log_ai_inference
from app.core.image_security import (
    read_upload_stream_safely,
    validate_and_decode_image_security,
    sanitize_filename,
)
from app.core.exceptions import (
    UnsupportedImageError,
    EmptyImageException,
    ImageTooLargeException,
    InvalidImageDimensionsException,
    DetectorNotConfiguredException,
    InferenceError,
)
from app.schemas.ai import (
    AIDetectionResponse,
    AIDetectionErrorResponse,
    FaceBoundingBox,
    DetectedFaceItem,
    ImageDimensions,
)
from app.services.ai_service import AIService, get_ai_service

logger = logging.getLogger("deepvision.api.ai")

router = APIRouter(prefix="/ai", tags=["AI Detection"])


@router.post(
    "/detect",
    response_model=AIDetectionResponse,
    responses={
        200: {"description": "Face detection analysis completed successfully"},
        400: {"description": "Invalid, corrupted, or unsupported image input"},
        413: {"description": "Image payload size exceeds configured limits"},
        500: {"description": "Internal model evaluation error (sanitized)"},
        503: {
            "model": AIDetectionErrorResponse,
            "description": "Detector model is not configured or operational",
        },
    },
    summary="Execute Face Detection Inference",
    description=(
        "Analyzes an uploaded image and returns detected faces, bounding boxes, "
        "confidence scores, and inference telemetry. If the AI detector model is not "
        "configured or operational, returns a structured development error response."
    ),
)
async def detect_faces(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze for face detection"),
    confidence_threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Optional minimum confidence threshold override"
    ),
    ai_service: AIService = Depends(get_ai_service),
):
    """
    Ingests and validates image upload with defense-in-depth safety controls:
    - MIME type & binary magic number verification
    - Stream chunk reading with hard byte caps (mitigates memory exhaustion DoS)
    - Untrusted filename sanitization
    - Decompression bomb / pixel flood attack mitigation
    - Hard image dimension enforcement
    - Pure in-memory processing (zero unnecessary disk persistence)
    - Zero raw image data logging
    """
    # 1. Resolve request tracing identifier & sanitize untrusted filename
    request_id = getattr(request.state, "request_id", None) or request.headers.get("X-Request-ID") or str(uuid.uuid4())
    safe_filename = sanitize_filename(file.filename)

    # 2. Memory-Safe Stream Reading with Hard Payload Cap
    content = await read_upload_stream_safely(
        file=file,
        max_bytes=settings.ai_max_image_bytes,
    )

    # 3. Comprehensive Image Safety Validation & In-Memory RGB Decoding
    # Inspects declared MIME, checks magic bytes, verifies container structure,
    # and enforces resolution bounds strictly in volatile memory.
    image_np, verified_format, (w, h) = validate_and_decode_image_security(
        data=content,
        declared_content_type=file.content_type,
        declared_filename=file.filename,
        max_dimension=settings.AI_MAX_IMAGE_DIMENSION,
    )

    # 4. Operational Detector Check
    # Never return fabricated detections. If the model is not operational,
    # raise DetectorNotConfiguredException mapped to HTTP 503.
    if not ai_service.is_operational:
        logger.info(
            "Detection request received but detector is not operational [request_id=%s, model=%s, device=%s]",
            request_id,
            settings.AI_MODEL_NAME,
            settings.AI_DEVICE,
        )
        raise DetectorNotConfiguredException(
            message=(
                f"AI face detection model is not operational. The detector model weights "
                f"for '{settings.AI_MODEL_NAME}' are not configured or loaded into the inference engine. "
                f"Please place model weights in 'models/face_detection/' as documented in docs/model-setup.md."
            ),
            detail=(
                f"AI face detection model is not operational. The detector model weights "
                f"for '{settings.AI_MODEL_NAME}' are not configured or loaded into the inference engine. "
                f"Please place model weights in 'models/face_detection/' as documented in docs/model-setup.md."
            ),
            model_name=settings.AI_MODEL_NAME,
            device=settings.AI_DEVICE.upper(),
        )

    # 5. Execute Real Model Inference (Real detections only — never fabricated)
    try:
        detection_result = ai_service.detect_faces(
            image_input=image_np,
            confidence_threshold=confidence_threshold,
            request_id=request_id,
        )
    except DetectorNotConfiguredException:
        raise
    except Exception as exc:
        logger.error(
            "Inference execution failed on model %s [request_id=%s]: %s",
            settings.AI_MODEL_NAME,
            request_id,
            exc,
            exc_info=True,
        )
        raise InferenceError(
            message="An error occurred during AI inference processing.",
            detail=str(exc),
            model_name=settings.AI_MODEL_NAME,
        )

    # 6. Format Real Structured Output
    face_items = [
        DetectedFaceItem(
            bbox=FaceBoundingBox(
                x1=round(d.bbox.x1, 2),
                y1=round(d.bbox.y1, 2),
                x2=round(d.bbox.x2, 2),
                y2=round(d.bbox.y2, 2),
            ),
            confidence=round(d.confidence, 4),
            landmarks=d.landmarks,
        )
        for d in detection_result.detections
    ]

    # 7. Structured Telemetry Logging (Guaranteed zero image content logged)
    log_ai_inference(
        logger_instance=logger,
        model=detection_result.model_name or settings.AI_MODEL_NAME,
        device=str(detection_result.device).upper(),
        image_width=w,
        image_height=h,
        number_of_faces=len(face_items),
        processing_time_ms=detection_result.processing_time_ms,
        request_id=request_id,
    )

    return AIDetectionResponse(
        success=True,
        model=detection_result.model_name or settings.AI_MODEL_NAME,
        device=str(detection_result.device).upper(),
        image=ImageDimensions(width=w, height=h),
        processing_time_ms=round(detection_result.processing_time_ms, 2),
        faces=face_items,
        request_id=request_id,
    )


@router.get(
    "/status",
    summary="Get AI Inference Engine Status",
    description="Returns detector operational readiness, hardware device telemetry, and configuration.",
)
def get_ai_status(ai_service: AIService = Depends(get_ai_service)):
    """Provides operational readiness telemetry for monitoring and frontend health probes."""
    return {
        "status": "ready" if ai_service.is_operational else "pending_model_configuration",
        "is_operational": ai_service.is_operational,
        "model_name": settings.AI_MODEL_NAME,
        "model_version": settings.AI_MODEL_VERSION,
        "configured_device": settings.AI_DEVICE,
        "confidence_threshold": settings.AI_CONFIDENCE_THRESHOLD,
        "max_image_size_mb": settings.AI_MAX_IMAGE_SIZE_MB,
        "max_image_dimension": settings.AI_MAX_IMAGE_DIMENSION,
        "telemetry": ai_service.get_service_status(),
    }
