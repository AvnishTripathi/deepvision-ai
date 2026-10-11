"""
DeepVision AI — Health, Liveness & Readiness Probes
Provides standardized Kubernetes / cloud-native health check probes:
- GET /api/v1/health       -> Base application health probe
- GET /api/v1/health/live  -> Liveness probe (process is running)
- GET /api/v1/health/ready -> Readiness probe (DB, Redis, AI model availability)
"""

from typing import Dict, Any
from fastapi import APIRouter, Response, status

from app.core.config import settings
from app.core.database import check_db_connection
from app.core.redis import check_redis_connection
from app.services.ai_service import get_ai_service

router = APIRouter(tags=["Health"])


@router.get("/health")
def get_health() -> Dict[str, Any]:
    """Basic service health check probe (maintained for backward compatibility)."""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": settings.APP_VERSION,
    }


@router.get("/health/live")
def get_liveness() -> Dict[str, Any]:
    """
    Kubernetes / Docker Liveness Probe.
    Verifies that the application process is running and event loop is responsive.
    Executes with zero remote I/O overhead.
    """
    return {
        "status": "alive",
        "service": settings.SERVICE_NAME,
        "version": settings.APP_VERSION,
    }


@router.get("/health/ready")
def get_readiness(response: Response) -> Dict[str, Any]:
    """
    Kubernetes / Docker Readiness Probe.
    Verifies that critical dependencies required to serve requests are operational:
    - PostgreSQL / SQLite database connectivity
    - Redis cache / queue connectivity (if configured)
    - AI face detection model operational status
    """
    db_check = check_db_connection()
    redis_check = check_redis_connection()

    ai_service = get_ai_service()
    ai_status = "operational" if ai_service.is_operational else "uninitialized"

    # Database is critical; Redis is non-fatal if disabled
    db_ready = db_check.get("status") in ("connected", "ok")
    redis_ready = redis_check.get("status") in ("connected", "disabled")
    ai_ready = ai_service.is_operational

    is_ready = db_ready and redis_ready and ai_ready

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if is_ready else "not_ready",
        "service": settings.SERVICE_NAME,
        "version": settings.APP_VERSION,
        "dependencies": {
            "database": db_check,
            "redis": redis_check,
            "ai_engine": {
                "status": ai_status,
                "model_name": getattr(ai_service.config, "AI_MODEL_NAME", "yunet"),
                "device": getattr(ai_service.config, "AI_DEVICE", "CPU"),
            },
        },
    }
