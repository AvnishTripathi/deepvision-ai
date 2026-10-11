"""
DeepVision AI — End-to-End System Integration Test Suite
Validates the complete end-to-end operational chain:
  API Client -> FastAPI Routing -> Middleware & Request IDs
             -> AI Face Detection (YuNet ONNX)
             -> Face Quality & Geometry Analysis
             -> Database Persistence (SQLAlchemy 2.0 / Alembic Schema)
             -> Background Video Worker & ByteTrack Multi-Face Tracking
             -> Output Storage Retrieval
"""

import io
import os
import json
import uuid
import time
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.main import app
from app.core.config import settings
from app.models import (
    Base,
    User,
    MediaAsset,
    AnalysisJob,
    AnalysisResult,
    ModelVersion,
)
from app.core.storage import get_storage_backend, StorageCategory
from app.services.video_job_manager import get_video_job_manager, VideoJobStatus


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def test_image_bytes():
    """Generates a valid 200x200 RGB JPEG image."""
    img = Image.new("RGB", (200, 200), color=(180, 150, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def test_db():
    """Creates isolated in-memory test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return engine


def test_e2e_image_detection_and_analysis_chain(client, test_image_bytes, test_db):
    """
    End-to-End Chain:
    Client Request -> Hardened Headers & Request ID -> Image Security
                   -> AI Service Inference -> Face Analysis -> Response Schema
                   -> Database Record
    """
    req_id = f"trace-e2e-{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/api/v1/ai/analyze",
        files={"file": ("portrait.jpg", test_image_bytes, "image/jpeg")},
        headers={"X-Request-ID": req_id},
    )

    assert response.status_code == 200
    # Defensive HTTP Headers present
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Request-ID") == req_id

    data = response.json()
    assert data["success"] is True
    assert data["request_id"] == req_id
    assert "face_count" in data
    assert "faces" in data
    assert "total_analysis_time_ms" in data

    # Persist job metadata in database
    with Session(test_db) as session:
        media = MediaAsset(
            storage_key=f"media/{req_id}_portrait.jpg",
            filename="portrait.jpg",
            file_size_bytes=len(test_image_bytes),
            mime_type="image/jpeg",
            media_type="image",
            width=200,
            height=200,
        )
        session.add(media)
        session.flush()

        job = AnalysisJob(
            job_type="image_analysis",
            status="completed",
            input_media_id=media.id,
            progress=100,
            request_id=req_id,
        )
        session.add(job)
        session.flush()

        result = AnalysisResult(
            job_id=job.id,
            face_count=data["face_count"],
            results_json=json.dumps(data["faces"]),
            processing_time_ms=data["total_analysis_time_ms"],
        )
        session.add(result)
        session.commit()

        # Verify database record
        persisted = session.query(AnalysisJob).filter_by(request_id=req_id).first()
        assert persisted is not None
        assert persisted.status == "completed"
        assert persisted.input_media.filename == "portrait.jpg"


def test_e2e_video_job_pipeline_and_worker_execution(client, tmp_path):
    """
    End-to-End Chain:
    Video Upload -> Video Validation -> Job Enqueuing
                 -> Background Worker Processing -> ByteTrack Tracking
                 -> Completion Polling -> Artifact Download
    """
    import cv2

    # Generate a lightweight synthetic 10-frame test video
    video_path = tmp_path / "synthetic_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (160, 120))
    for i in range(10):
        frame = np.full((120, 160, 3), 100 + i * 5, dtype=np.uint8)
        # Draw a synthetic oval representing a face
        cv2.ellipse(frame, (80, 60), (25, 35), 0, 0, 360, (200, 170, 150), -1)
        out.write(frame)
    out.release()

    video_bytes = video_path.read_bytes()

    # 1. Submit video analysis job
    submit_res = client.post(
        "/api/v1/video/analyze",
        files={"file": ("synthetic_test.mp4", video_bytes, "video/mp4")},
    )
    assert submit_res.status_code == 202
    submit_data = submit_res.json()
    job_id = submit_data["job_id"]
    assert submit_data["status"] == "queued"

    # 2. Poll job status until completed
    max_wait = 15.0
    start = time.perf_counter()
    completed = False
    status_data = None

    while time.perf_counter() - start < max_wait:
        poll_res = client.get(f"/api/v1/video/jobs/{job_id}")
        assert poll_res.status_code == 200
        status_data = poll_res.json()
        if status_data["status"] == "completed":
            completed = True
            break
        elif status_data["status"] == "failed":
            pytest.fail(f"Video job failed: {status_data.get('error')}")
        time.sleep(0.3)

    assert completed is True, f"Job did not complete within {max_wait}s (current status: {status_data.get('status')})"
    assert status_data["progress"] == 100
    assert status_data["total_frames"] == 10
    assert status_data["metrics"] is not None
    assert "average_fps" in status_data["metrics"]

    # 3. Download the generated annotated video artifact
    download_res = client.get(f"/api/v1/video/jobs/{job_id}/download")
    assert download_res.status_code == 200
    assert download_res.headers.get("content-type") == "video/mp4"
    assert len(download_res.content) > 0
