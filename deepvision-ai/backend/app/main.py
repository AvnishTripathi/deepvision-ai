"""
DeepVision AI — Application Entry Point
Clean, scalable FastAPI service for DeepVision AI.
"""

import uuid
from contextlib import asynccontextmanager
from typing import Callable, Awaitable, Dict, Any
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.database import init_db
from app.core.error_handlers import register_exception_handlers
from app.api.v1 import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize application logging & database connection
    setup_logging()
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Advanced AI-Powered Face Detection, Video Analysis & Real-Time Vision Platform",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Register centralized exception handlers (sanitizes responses, prevents stack trace leakage)
register_exception_handlers(app)


# Request ID tracing middleware
@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attaches a unique trace ID to each request and propagates it in response headers."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# CORS middleware configured via typed settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
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
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
