import os
import uuid
import json
import base64
import time
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

import schemas, models, auth
from database import get_db
from config import settings

# Dummy ml_engine implementation for now
class MLEngine:
    @staticmethod
    def detect_image(image_path_or_bytes):
        return {
            "overall_score": 0.85,
            "label": "fake",
            "faces": [
                {
                    "face_index": 0,
                    "bbox": [10, 10, 100, 100],
                    "deepfake_score": 0.9,
                    "label": "fake",
                    "confidence": 0.95
                }
            ],
            "processing_time_ms": 120,
            "model_version": "v1.0"
        }

import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Try real ML engine; fallback to stub
try:
    from ml_engine.detector import detect_image as _ml_detect_image
    from ml_engine.detector import detect_video_frames as _ml_detect_video
    _ML_AVAILABLE = True
except ImportError as e:
    _ML_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"ml_engine not available, using stub: {e}")

router = APIRouter(prefix="/api/v1/detect", tags=["detect"])

@router.post("/image")
async def detect_image_api(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(auth.get_optional_user)
):
    """Upload an image and get deepfake detection results."""
    # Validate file type
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid image type: {file.content_type}. Allowed: {settings.ALLOWED_IMAGE_TYPES}")

    # Read and validate size
    content = await file.read()
    if len(content) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Image too large. Max {settings.MAX_IMAGE_SIZE_MB}MB")

    # Save file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(content)

    # Create DB job
    job_id = str(uuid.uuid4())
    job = models.DetectionJob(
        id=job_id,
        user_id=current_user.id if current_user else None,
        type="image",
        status="processing",
        original_filename=file.filename,
        file_path=file_path
    )
    db.add(job)
    db.commit()

    start_time = time.time()
    try:
        if _ML_AVAILABLE:
            result_data = await _ml_detect_image(file_path)
        else:
            result_data = MLEngine.detect_image(file_path)
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    proc_time = int((time.time() - start_time) * 1000)

    # Save result
    job.status = "done"
    job.completed_at = datetime.utcnow()
    job.processing_time_ms = proc_time
    job.result_json = json.dumps(result_data)
    db.commit()

    # Map result to response schema
    faces = result_data.get("faces", [])
    return {
        "job_id": job.id,
        "status": "done",
        "overall_score": result_data.get("overall_score", 0.0),
        "label": result_data.get("overall_label", result_data.get("label", "REAL")),
        "face_count": result_data.get("face_count", len(faces)),
        "faces": faces,
        "heatmap_base64": result_data.get("heatmap_base64"),
        "signal_chart_base64": result_data.get("signal_chart_base64"),
        "processing_time_ms": proc_time,
        "model_version": result_data.get("model_version", "stub-1.0"),
    }


import asyncio
from fastapi import BackgroundTasks

async def _process_video_background(job_id: str, file_path: str):
    """Background task to extract frames and analyze video via ML engine."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        if _ML_AVAILABLE:
            result_data = await _ml_detect_video(file_path, sample_fps=2)
        else:
            result_data = {
                "status": "completed",
                "overall_score": 0.45,
                "overall_label": "SUSPICIOUS",
                "frame_count_analyzed": 10,
                "timeline": []
            }
        
        job = db.query(models.DetectionJob).filter(models.DetectionJob.id == job_id).first()
        if job:
            job.status = "done"
            job.completed_at = datetime.utcnow()
            job.processing_time_ms = result_data.get("processing_time_ms", 1500)
            job.result_json = json.dumps(result_data)
            db.commit()
    except Exception as e:
        job = db.query(models.DetectionJob).filter(models.DetectionJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()


@router.post("/video", response_model=schemas.VideoDetectResponse)
async def detect_video_api(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(auth.get_optional_user)
):
    if file.content_type not in settings.ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid video type. Allowed: {settings.ALLOWED_VIDEO_TYPES}")
        
    job_id = str(uuid.uuid4())
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    filename = f"{job_id}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    job = models.DetectionJob(
        id=job_id,
        user_id=current_user.id if current_user else None,
        type="video",
        status="processing",
        original_filename=file.filename,
        file_path=file_path
    )
    db.add(job)
    db.commit()
    
    # Process video frames asynchronously in background task
    background_tasks.add_task(_process_video_background, job_id, file_path)
    
    return schemas.VideoDetectResponse(
        job_id=job_id,
        status="processing",
        message="Video analysis scheduled in background."
    )


@router.post("/url")
async def detect_url_api(
    payload: schemas.UrlDetectRequest,
    db: Session = Depends(get_db),
    current_user: Optional[models.User] = Depends(auth.get_optional_user)
):
    """Download an image from a URL and analyze it for deepfake content."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(payload.url)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: HTTP {resp.status_code}")
            content = resp.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image from URL: {str(e)}")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(content)

    job_id = str(uuid.uuid4())
    job = models.DetectionJob(
        id=job_id,
        user_id=current_user.id if current_user else None,
        type="image",
        status="processing",
        original_filename=payload.url,
        file_path=file_path
    )
    db.add(job)
    db.commit()

    start_time = time.time()
    try:
        if _ML_AVAILABLE:
            result_data = await _ml_detect_image(file_path)
        else:
            result_data = MLEngine.detect_image(file_path)
    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    proc_time = int((time.time() - start_time) * 1000)

    job.status = "done"
    job.completed_at = datetime.utcnow()
    job.processing_time_ms = proc_time
    job.result_json = json.dumps(result_data)
    db.commit()

    faces = result_data.get("faces", [])
    return {
        "job_id": job.id,
        "status": "done",
        "overall_score": result_data.get("overall_score", 0.0),
        "label": result_data.get("overall_label", result_data.get("label", "REAL")),
        "face_count": result_data.get("face_count", len(faces)),
        "faces": faces,
        "heatmap_base64": result_data.get("heatmap_base64"),
        "signal_chart_base64": result_data.get("signal_chart_base64"),
        "processing_time_ms": proc_time,
        "model_version": result_data.get("model_version", "v1.0"),
    }


@router.get("/status/{job_id}", response_model=schemas.JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(models.DetectionJob).filter(models.DetectionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return schemas.JobStatusResponse(
        job_id=job.id,
        status=job.status,
        error_message=job.error_message,
        completed_at=job.completed_at
    )

@router.get("/result/{job_id}", response_model=schemas.DetectionResult)
def get_job_result(job_id: str, db: Session = Depends(get_db)):
    job = db.query(models.DetectionJob).filter(models.DetectionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail="Job not done yet")
        
    result_data = json.loads(job.result_json) if job.result_json else {}
    label = result_data.get("overall_label") or result_data.get("label") or "REAL"
    score = result_data.get("overall_score", 0.0)
    faces = result_data.get("faces", [])
    
    return schemas.DetectionResult(
        job_id=job.id,
        status=job.status,
        overall_score=score,
        label=label,
        face_count=result_data.get("face_count", len(faces)),
        faces=faces,
        heatmap_base64=result_data.get("heatmap_base64"),
        signal_chart_base64=result_data.get("signal_chart_base64"),
        processing_time_ms=job.processing_time_ms or 0,
        model_version=result_data.get("model_version", "v1.0")
    )

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Real-time ultra-low-latency deepfake detection via WebSocket.
    Client sends: {"frame": "<base64_encoded_jpeg>"}
    Server returns: {"faces": [...], "overall_score": float, "label": str, "processing_time_ms": int}
    """
    await websocket.accept()
    import io
    from PIL import Image
    import sys

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        from ml_engine.live_analyzer import detect_frame_in_memory
        _use_fast_live = True
    except Exception as e:
        _use_fast_live = False

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if "frame" not in message:
                await websocket.send_json({"error": "No frame provided"})
                continue

            start_time = time.time()
            try:
                frame_bytes = base64.b64decode(message["frame"])
                pil_image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

                if _use_fast_live:
                    result = await detect_frame_in_memory(pil_image)
                else:
                    result = {"status": "ok", "faces": [], "overall_score": 0.0, "label": "REAL"}

            except Exception as e:
                result = {
                    "status": "error",
                    "error": f"Frame decode error: {str(e)}",
                    "faces": [],
                    "overall_score": 0.0,
                    "label": "UNKNOWN"
                }

            proc_time = int((time.time() - start_time) * 1000)
            result["processing_time_ms"] = proc_time
            await websocket.send_json(result)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
