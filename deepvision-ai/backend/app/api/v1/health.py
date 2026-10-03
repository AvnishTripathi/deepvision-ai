from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health")
def get_health():
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": settings.APP_VERSION
    }
