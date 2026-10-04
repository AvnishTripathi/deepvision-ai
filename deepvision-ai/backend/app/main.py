"""
DeepVision AI — Application Entry Point
Clean, scalable FastAPI service for DeepVision AI.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
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
def root():
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
