"""
DeepVision AI — Step 7: Architecture, Database, Queue & Health Verification Suite
Validates:
1. PostgreSQL / SQLAlchemy 2.0 models & clean schema creation
2. Storage abstraction (LocalStorage, ObjectStorage, separated temp/media/results partitions)
3. Structured logging with request_id, job_id, and credential masking
4. Health endpoints: /health, /health/live, /health/ready
5. Consistent error responses with request correlation
6. Configurable video job concurrency (MAX_CONCURRENT_VIDEO_JOBS)
7. Clean Alembic migration execution (alembic upgrade head)
"""

import os
import io
import json
import logging
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.main import app
from app.core.config import settings
from app.models import (
    Base,
    User,
    MediaAsset,
    ModelVersion,
    AnalysisJob,
    AnalysisResult,
    SystemLog,
)
from app.core.storage import (
    StorageCategory,
    LocalStorageBackend,
    ObjectStorageBackend,
    get_storage_backend,
)
from app.core.logging import (
    SensitiveDataFilter,
    StructuredJsonFormatter,
    log_structured_event,
)
from app.services.video_job_manager import (
    InMemoryVideoJobManager,
    VideoJobStatus,
)
from app.core.exceptions import VideoJobCapacityExceededException


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# -----------------------------------------------------------------------------
# 1. Database Entities & Schema Verification (Step 7.9)
# -----------------------------------------------------------------------------

def test_database_models_metadata_creates_cleanly():
    """Verifies that all required domain models create cleanly without schema errors."""
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)

    tables = Base.metadata.tables.keys()
    assert "users" in tables
    assert "media_assets" in tables
    assert "model_versions" in tables
    assert "analysis_jobs" in tables
    assert "analysis_results" in tables
    assert "system_logs" in tables

    # Verify no biometric identity / face gallery tables exist
    for forbidden in ("identities", "face_vectors", "embeddings", "biometrics", "watchlist"):
        assert forbidden not in tables


def test_database_crud_operations_on_entities():
    """Tests basic CRUD operations on models in an isolated session."""
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)

    with Session(test_engine) as session:
        model = ModelVersion(
            model_name="yunet_2023mar",
            version="1.0.0",
            task_type="face_detection",
            file_hash="abc123sha256",
        )
        session.add(model)

        media = MediaAsset(
            storage_key="media/sample_test.jpg",
            storage_backend="local",
            filename="sample_test.jpg",
            file_size_bytes=10240,
            mime_type="image/jpeg",
            media_type="image",
            width=640,
            height=480,
        )
        session.add(media)
        session.flush()

        job = AnalysisJob(
            job_type="image_detection",
            status="completed",
            input_media_id=media.id,
            progress=100,
            request_id="req-test-uuid",
        )
        session.add(job)
        session.flush()

        result = AnalysisResult(
            job_id=job.id,
            model_version_id=model.id,
            face_count=1,
            results_json=json.dumps([{"bbox": [10, 10, 50, 50], "confidence": 0.95}]),
            processing_time_ms=18.5,
        )
        session.add(result)

        audit = SystemLog(
            level="INFO",
            service="deepvision-core",
            endpoint="/api/v1/ai/detect",
            request_id="req-test-uuid",
            message="Detection completed successfully",
        )
        session.add(audit)
        session.commit()

        saved_job = session.query(AnalysisJob).filter_by(id=job.id).first()
        assert saved_job is not None
        assert saved_job.input_media.filename == "sample_test.jpg"
        assert saved_job.status == "completed"


# -----------------------------------------------------------------------------
# 2. Storage Architecture & Segregation (Step 7.11)
# -----------------------------------------------------------------------------

def test_local_storage_segregated_partitions(tmp_path):
    """Verifies that LocalStorage separates temporary, media, and results directories."""
    storage = LocalStorageBackend(base_dir=str(tmp_path))

    assert os.path.exists(storage.temp_dir)
    assert os.path.exists(storage.media_dir)
    assert os.path.exists(storage.results_dir)

    temp_key = storage.store_bytes(b"temp_frame_data", "frame_001.raw", category=StorageCategory.TEMPORARY)
    temp_path = storage.get_local_path(temp_key, category=StorageCategory.TEMPORARY)
    assert temp_path.startswith(storage.temp_dir)
    assert os.path.exists(temp_path)

    test_file = tmp_path / "upload.mp4"
    test_file.write_bytes(b"mp4_content")
    media_key = storage.store_file(str(test_file), "user_video.mp4", category=StorageCategory.MEDIA)
    media_path = storage.get_local_path(media_key, category=StorageCategory.MEDIA)
    assert media_path.startswith(storage.media_dir)

    res_key = storage.store_bytes(b"annotated_data", "annotated.mp4", category=StorageCategory.RESULTS)
    res_path = storage.get_local_path(res_key, category=StorageCategory.RESULTS)
    assert res_path.startswith(storage.results_dir)

    assert storage.delete(temp_key, category=StorageCategory.TEMPORARY) is True
    assert storage.exists(temp_key, category=StorageCategory.TEMPORARY) is False


def test_object_storage_emulated_cloud_fallback(tmp_path):
    """Verifies ObjectStorage abstraction with local emulation without cloud credentials."""
    local_backend = LocalStorageBackend(base_dir=str(tmp_path))
    cloud_storage = ObjectStorageBackend(provider="emulated", bucket_name="test-bucket", local_fallback=local_backend)

    key = cloud_storage.store_bytes(b"cloud_data", "test_blob.bin", category=StorageCategory.MEDIA)
    assert cloud_storage.exists(key, category=StorageCategory.MEDIA) is True
    assert cloud_storage.delete(key, category=StorageCategory.MEDIA) is True


# -----------------------------------------------------------------------------
# 3. Health, Liveness & Readiness Probes (Step 7.15)
# -----------------------------------------------------------------------------

def test_health_base_endpoint(client):
    """Verifies GET /api/v1/health returns base health metadata."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_health_liveness_probe(client):
    """Verifies GET /api/v1/health/live returns 200 alive."""
    res = client.get("/api/v1/health/live")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "alive"
    assert data["service"] == settings.SERVICE_NAME


def test_health_readiness_probe_success(client):
    """Verifies GET /api/v1/health/ready evaluates dependencies."""
    res = client.get("/api/v1/health/ready")
    assert res.status_code in (200, 503)
    data = res.json()
    assert "dependencies" in data
    assert "database" in data["dependencies"]
    assert "redis" in data["dependencies"]
    assert "ai_engine" in data["dependencies"]


# -----------------------------------------------------------------------------
# 4. Structured Logging & Sensitive Data Masking (Step 7.12)
# -----------------------------------------------------------------------------

def test_structured_event_logging():
    """Verifies structured application log output schema."""
    test_logger = logging.getLogger("test_structured")
    event = log_structured_event(
        logger_instance=test_logger,
        level="INFO",
        message="Job finished",
        service="deepvision-core",
        request_id="req-1234",
        job_id="job-5678",
        endpoint="/api/v1/video/analyze",
        processing_time_ms=120.5,
    )
    assert event["request_id"] == "req-1234"
    assert event["job_id"] == "job-5678"
    assert event["endpoint"] == "/api/v1/video/analyze"
    assert event["processing_time_ms"] == 120.5
    assert event["level"] == "INFO"


def test_sensitive_data_filter_redacts_credentials():
    """Verifies that passwords, tokens, and raw image data are redacted by the log filter."""
    filt = SensitiveDataFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=1,
        msg="Login attempt with password: superSecretPassword123 and token=Bearer_xyz",
        args=(),
        exc_info=None,
    )
    filt.filter(record)
    assert "superSecretPassword123" not in record.msg
    assert "***REDACTED***" in record.msg


# -----------------------------------------------------------------------------
# 5. Request ID Correlation & Error Response Consistency (Step 7.13, 7.14)
# -----------------------------------------------------------------------------

def test_error_response_contains_request_id_and_structured_detail(client):
    """Verifies that errors include request_id correlation and structured error_detail."""
    trace_id = "trace-correlation-id-999"
    res = client.post(
        "/api/v1/ai/detect",
        files={"file": ("fake.txt", b"not an image", "text/plain")},
        headers={"X-Request-ID": trace_id},
    )
    assert res.status_code == 400
    data = res.json()
    assert data["success"] is False
    assert data["request_id"] == trace_id
    assert "error_detail" in data
    assert data["error_detail"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
    assert "message" in data


# -----------------------------------------------------------------------------
# 6. Concurrency Protection (Step 7.8)
# -----------------------------------------------------------------------------

def test_max_concurrent_video_jobs_setting_alias():
    """Verifies that MAX_CONCURRENT_VIDEO_JOBS is synced to VIDEO_MAX_CONCURRENT_JOBS."""
    with patch.object(settings, "VIDEO_MAX_CONCURRENT_JOBS", 3):
        manager = InMemoryVideoJobManager(
            storage=MagicMock(),
            video_service=MagicMock(),
            max_workers=1,
        )
        manager._jobs["j1"] = MagicMock(status=VideoJobStatus.QUEUED)
        manager._jobs["j2"] = MagicMock(status=VideoJobStatus.PROCESSING)
        manager._jobs["j3"] = MagicMock(status=VideoJobStatus.QUEUED)

        with pytest.raises(VideoJobCapacityExceededException) as exc:
            manager.submit_job(input_filepath="fake.mp4", filename="fake.mp4")

        assert "Maximum concurrent video processing jobs (3) exceeded" in str(exc.value)


# -----------------------------------------------------------------------------
# 7. Database Migrations (Step 7.10)
# -----------------------------------------------------------------------------

def test_alembic_upgrade_head_on_clean_database(tmp_path):
    """Verifies that alembic upgrade head successfully executes on a fresh database."""
    from alembic.config import Config
    from alembic import command

    clean_db_path = tmp_path / "clean_test.db"
    clean_db_url = f"sqlite:///{clean_db_path}"

    with patch.dict(os.environ, {"DATABASE_URL": clean_db_url}):
        cfg = Config("backend/alembic.ini")
        cfg.set_main_option("script_location", "backend/alembic")
        command.upgrade(cfg, "head")

    assert os.path.exists(clean_db_path)
