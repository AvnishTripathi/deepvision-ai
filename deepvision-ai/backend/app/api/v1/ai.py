"""
DeepVision AI — AI Inference & Face Detection API Router
Provides development and production endpoints for face detection, image analysis,
and AI engine runtime status telemetry.
"""

import os
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Query, Depends

from app.core.config import settings
from app.core.exceptions import (
    UnsupportedImageError,
    EmptyImageException,
    ImageTooLargeException,
    InvalidImageDimensionsException,
    DetectorNotConfiguredException,
    InferenceError,
)
from app.ai.detection.preprocessing import load_image
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

ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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
    file: UploadFile = File(..., description="Image file to analyze for face detection"),
    confidence_threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Optional minimum confidence threshold override"
    ),
    ai_service: AIService = Depends(get_ai_service),
):
    """
    Ingests and validates image upload, enforces payload and dimension limits,
    verifies detector operational state, and executes face detection inference.
    All errors raise domain-specific AI exceptions mapped to sanitized HTTP responses.
    """
    filename = file.filename or ""
    content_type = (file.content_type or "").lower().strip()
    ext = os.path.splitext(filename.lower())[1]

    # 1. File Type Validation (MIME type & extension)
    if content_type not in ALLOWED_IMAGE_MIME_TYPES and ext not in ALLOWED_EXTENSIONS:
        logger.warning("Rejected file with invalid type: content_type='%s', ext='%s'", content_type, ext)
        raise UnsupportedImageError(
            message=(
                f"Unsupported file type '{content_type or ext}'. "
                f"Allowed formats: image/jpeg, image/png, image/webp, image/bmp."
            )
        )

    # 2. File Size & Payload Stream Reading
    try:
        content = await file.read()
    except Exception as exc:
        logger.error("Failed reading uploaded file stream: %s", exc, exc_info=True)
        raise EmptyImageException("Failed to read uploaded file payload stream.")

    if len(content) == 0:
        raise EmptyImageException("Uploaded file is empty (0 bytes).")

    if len(content) > settings.ai_max_image_bytes:
        raise ImageTooLargeException(
            f"Image payload size ({len(content)} bytes) exceeds configured limit "
            f"of {settings.AI_MAX_IMAGE_SIZE_MB}MB ({settings.ai_max_image_bytes} bytes)."
        )

    # 3. Corrupted Image Decoding & Dimension Verification
    # load_image automatically raises CorruptedImageException, EmptyImageException,
    # or InvalidImageDimensionsException if byte stream is corrupted.
    image_np = load_image(content)

    # 4. Explicit Image Dimensions Limit Check
    h, w = image_np.shape[:2]
    if h > settings.AI_MAX_IMAGE_DIMENSION or w > settings.AI_MAX_IMAGE_DIMENSION:
        raise InvalidImageDimensionsException(
            f"Image resolution {w}x{h} exceeds maximum allowed dimension "
            f"of {settings.AI_MAX_IMAGE_DIMENSION}px."
        )

    # 5. Operational Detector Check
    # Never return fabricated detections. If the model is not operational,
    # raise DetectorNotConfiguredException mapped to HTTP 503.
    if not ai_service.is_operational:
        logger.info(
            "Detection request received but detector is not operational (model: %s, device: %s)",
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

    # 6. Execute Real Model Inference (Real detections only — never fabricated)
    try:
        detection_result = ai_service.detect_faces(
            image_input=image_np,
            confidence_threshold=confidence_threshold,
        )
    except DetectorNotConfiguredException:
        raise
    except Exception as exc:
        logger.error("Inference execution failed on model %s: %s", settings.AI_MODEL_NAME, exc, exc_info=True)
        raise InferenceError(
            message="An error occurred during AI inference processing.",
            detail=str(exc),
            model_name=settings.AI_MODEL_NAME,
        )

    # 7. Format Real Structured Output
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

    return AIDetectionResponse(
        success=True,
        model=detection_result.model_name or settings.AI_MODEL_NAME,
        device=str(detection_result.device).upper(),
        image=ImageDimensions(width=w, height=h),
        processing_time_ms=round(detection_result.processing_time_ms, 2),
        faces=face_items,
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
