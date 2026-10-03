import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from prometheus_client import make_asgi_app
import structlog
import os

from database import create_all_tables
from routers import auth_router, detect_router, history_router

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up DeepFaceGuard backend")
    create_all_tables()
    os.makedirs("./static", exist_ok=True)
    yield
    # Shutdown
    logger.info("Shutting down DeepFaceGuard backend")

app = FastAPI(title="DeepVision AI API", version="1.0.0-step1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        "request",
        method=request.method,
        url=str(request.url.path),
        status_code=response.status_code,
        duration=f"{process_time:.4f}s"
    )
    return response

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=404, content={"message": "Not Found"})

@app.exception_handler(422)
async def custom_422_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=422, content={"message": "Validation Error", "details": str(exc)})

@app.exception_handler(500)
async def custom_500_handler(request: Request, exc: Exception):
    logger.error("internal_error", error=str(exc))
    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(auth_router.router)
app.include_router(detect_router.router)
app.include_router(history_router.router)

# Health & System Status router
@app.get("/health", tags=["health"])
def health_check():
    from database import check_database_connection
    from redis_client import check_redis_connection
    from config import settings
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "environment": settings.APP_ENV,
        "database": check_database_connection(),
        "redis": check_redis_connection()
    }

# Create static dir if it doesn't exist
os.makedirs("c:\\Users\\abhis\\Music\\Deep Face Detection\\backend\\static", exist_ok=True)
app.mount("/static", StaticFiles(directory="c:\\Users\\abhis\\Music\\Deep Face Detection\\backend\\static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
