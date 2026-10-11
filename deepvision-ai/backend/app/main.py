"""
DeepVision AI — Application Entry Point
Clean, scalable, production-hardened FastAPI service for DeepVision AI.
"""

import uuid
from contextlib import asynccontextmanager
from typing import Callable, Awaitable, Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.database import init_db
from app.core.error_handlers import register_exception_handlers
from app.api.v1 import api_router
from app.services.ai_service import init_ai_service, shutdown_ai_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize application logging & database connection
    setup_logging()
    init_db()

    # 2. Application startup: Pre-load AI model once and keep detector available
    ai_service = init_ai_service()
    if ai_service.is_operational:
        app.state.ai_ready = True
    else:
        app.state.ai_ready = False

    yield

    # 3. Application shutdown: Cleanly unload AI model and release hardware resources
    shutdown_ai_service()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Advanced AI-Powered Face Detection, Video Analysis & Real-Time Vision Platform",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
    lifespan=lifespan,
)

# Register centralized exception handlers (sanitizes responses, prevents stack trace leakage)
register_exception_handlers(app)


# Security Headers & Tracing Middleware (Step 7.3)
@app.middleware("http")
async def security_headers_and_tracing_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attaches a unique trace ID and sets hardened security HTTP headers on all responses."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)

    # Tracing header
    response.headers["X-Request-ID"] = request_id

    # Defensive HTTP Security Headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response


# Trusted Host Middleware (Step 7.3)
if settings.ALLOWED_HOSTS and settings.ALLOWED_HOSTS != ["*"]:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )


# CORS middleware configured via typed settings
cors_has_wildcard = "*" in settings.CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=not cors_has_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
def root() -> Dict[str, Any]:
    return {
        "service": settings.SERVICE_NAME,
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "online",
        "documentation": "/docs" if not settings.is_production else "disabled in production",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

