"""
DeepVision AI — Application Entry Point
FastAPI service orchestrating endpoints, middleware, and lifecycle hooks.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.core.exceptions import DeepVisionException


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup hook
    setup_logging()
    logger.info("Initializing %s v%s in %s mode...", settings.PROJECT_NAME, settings.PROJECT_VERSION, settings.APP_ENV)
    yield
    # Shutdown hook
    logger.info("Shutting down %s...", settings.PROJECT_NAME)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Advanced AI-Powered Face Detection, Video Analysis & Real-Time Vision Platform",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DeepVisionException)
async def deepvision_exception_handler(request: Request, exc: DeepVisionException):
    return JSONResponse(
        status_code=400,
        content={"success": False, "error": exc.__class__.__name__, "message": exc.message}
    )


@app.get("/", tags=["Root"])
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "environment": settings.APP_ENV
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
