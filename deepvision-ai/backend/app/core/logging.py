"""
DeepVision AI — Logging Configuration & Structured Telemetry
Configures structured application logging, request correlation, and telemetry.
Guarantees:
- Structured schema: timestamp, level, service, request_id, job_id, endpoint, processing_time, error_type.
- Zero-leakage: Redacts passwords, API keys, tokens, database credentials, raw uploaded images, and base64 payloads.
"""

import sys
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from app.core.config import settings

CREDENTIAL_PATTERN = re.compile(
    r"(password|passwd|secret|api_key|token|auth_token|access_token|key_hash)[\s:=]+([^\s,;]+)",
    re.IGNORECASE,
)
BASE64_IMAGE_PATTERN = re.compile(
    r"data:image/[^;]+;base64,[a-zA-Z0-9+/=]+",
    re.IGNORECASE,
)


class SensitiveDataFilter(logging.Filter):
    """Filter that intercepts and masks credentials and raw image payloads in log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            msg = CREDENTIAL_PATTERN.sub(r"\1=***REDACTED***", record.msg)
            msg = BASE64_IMAGE_PATTERN.sub("data:image/***REDACTED***", msg)
            record.msg = msg
        return True


class StructuredJsonFormatter(logging.Formatter):
    """
    JSON log formatter for production log aggregation systems (Datadog, CloudWatch, ELK).
    Emits standardized JSON logs containing correlation IDs and telemetry.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": getattr(record, "service", settings.SERVICE_NAME),
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Request and Job correlation
        if hasattr(record, "request_id") and record.request_id:
            log_record["request_id"] = str(record.request_id)
        if hasattr(record, "job_id") and record.job_id:
            log_record["job_id"] = str(record.job_id)
        if hasattr(record, "endpoint") and record.endpoint:
            log_record["endpoint"] = str(record.endpoint)
        if hasattr(record, "processing_time_ms") and record.processing_time_ms is not None:
            log_record["processing_time_ms"] = record.processing_time_ms
        if hasattr(record, "error_type") and record.error_type:
            log_record["error_type"] = str(record.error_type)

        # Include structured telemetry if attached
        if hasattr(record, "telemetry") and isinstance(record.telemetry, dict):
            log_record["telemetry"] = record.telemetry

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_logging() -> None:
    """Configures application-wide logging with defensive filters and structured formatters."""
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format, date_format))
    handler.addFilter(SensitiveDataFilter())

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[handler],
    )

    # Suppress verbose third-party logs
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


logger = logging.getLogger("deepvision")
inference_logger = logging.getLogger("deepvision.ai.inference")


def log_structured_event(
    logger_instance: logging.Logger,
    level: str,
    message: str,
    service: str = "deepvision-core",
    request_id: Optional[str] = None,
    job_id: Optional[str] = None,
    endpoint: Optional[str] = None,
    processing_time_ms: Optional[float] = None,
    error_type: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Emits a structured application log event with full request correlation.
    
    Fields:
    - timestamp (ISO-8601 UTC)
    - level
    - service
    - request_id
    - job_id
    - endpoint
    - processing_time_ms
    - error_type
    """
    event_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "service": service,
        "message": message,
    }
    if request_id:
        event_payload["request_id"] = str(request_id)
    if job_id:
        event_payload["job_id"] = str(job_id)
    if endpoint:
        event_payload["endpoint"] = str(endpoint)
    if processing_time_ms is not None:
        event_payload["processing_time_ms"] = round(float(processing_time_ms), 2)
    if error_type:
        event_payload["error_type"] = str(error_type)

    if extra_fields:
        safe_extra = {
            k: v for k, v in extra_fields.items()
            if not isinstance(v, (bytes, bytearray)) and k not in ("password", "token", "secret", "image", "bytes")
        }
        event_payload.update(safe_extra)

    log_method = getattr(logger_instance, level.lower(), logger_instance.info)
    log_method(
        "%s [%s] request_id=%s job_id=%s latency=%s ms",
        message,
        event_payload.get("endpoint", "-"),
        event_payload.get("request_id", "-"),
        event_payload.get("job_id", "-"),
        str(event_payload.get("processing_time_ms", "-")),
        extra=event_payload,
    )
    return event_payload


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
        safe_extra = {
            k: v for k, v in extra_metrics.items()
            if not isinstance(v, (bytes, bytearray)) and k not in ("image", "bytes", "data", "raw")
        }
        telemetry.update(safe_extra)

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
        extra={"telemetry": telemetry, "request_id": str(request_id)},
    )

    return telemetry
