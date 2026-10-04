"""
DeepVision AI — Logging Configuration & Structured Telemetry
Configures structured application logging and provides high-performance,
privacy-preserving telemetry recording for AI inference pipelines.
Ensures zero logging of raw image buffers, pixels, or sensitive payloads.
"""

import sys
import json
import logging
from typing import Optional, Dict, Any
from app.core.config import settings


class StructuredJsonFormatter(logging.Formatter):
    """
    JSON log formatter for production log aggregation systems (e.g. Datadog, CloudWatch, ELK).
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured telemetry if attached
        if hasattr(record, "telemetry") and isinstance(record.telemetry, dict):
            log_record["telemetry"] = record.telemetry

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_logging() -> None:
    """Configures application-wide logging."""
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Suppress verbose third-party logs
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


logger = logging.getLogger("deepvision")
inference_logger = logging.getLogger("deepvision.ai.inference")


def log_ai_inference(
    logger_instance: logging.Logger,
    model: str,
    device: str,
    image_width: int,
    image_height: int,
    number_of_faces: int,
    processing_time_ms: float,
    request_id: str,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Emits structured telemetry around AI inference forward-passes.
    
    Guarantees:
    - Captures: model, device, image_width, image_height, number_of_faces, processing_time_ms, request_id.
    - Privacy: NEVER logs raw image bytes, pixels, base64 payloads, or biometric embeddings.
    - In-Memory Only: No filesystem persistence of image data.
    
    Returns:
        Dict[str, Any]: The structured telemetry payload that was logged.
    """
    telemetry = {
        "event": "ai_inference",
        "request_id": str(request_id),
        "model": str(model),
        "device": str(device).upper(),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "number_of_faces": int(number_of_faces),
        "processing_time_ms": round(float(processing_time_ms), 2),
    }

    if extra_metrics:
        # Filter out any potential byte or binary fields from extra_metrics
        safe_extra = {
            k: v for k, v in extra_metrics.items()
            if not isinstance(v, (bytes, bytearray)) and k not in ("image", "bytes", "data", "raw")
        }
        telemetry.update(safe_extra)

    # Log in structured JSON format
    logger_instance.info(
        "AI Inference Completed | request_id=%s model=%s device=%s resolution=%dx%d faces=%d latency=%.2fms | json=%s",
        telemetry["request_id"],
        telemetry["model"],
        telemetry["device"],
        telemetry["image_width"],
        telemetry["image_height"],
        telemetry["number_of_faces"],
        telemetry["processing_time_ms"],
        json.dumps(telemetry),
        extra={"telemetry": telemetry},
    )

    return telemetry
